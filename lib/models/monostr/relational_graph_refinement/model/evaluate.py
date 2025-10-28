import os
import glob
import numpy as np


def load_label_file(path):
    """KITTI label file을 읽어서 Car 항목만 추출.
    반환: ndarray [N, 15] with
      [x,y,z,w,h,l,ry,x1,y1,x2,y2,trunc,occ,score,height]
    - score는 GT에는 없으므로 1.0으로 채움
    - height는 2D 박스 높이(y2-y1)
    """
    boxes = []
    if not os.path.exists(path):
        return np.zeros((0, 15))
    with open(path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split()
            if not items:
                continue
            cls = items[0]
            if cls != "Car":
                continue
            trunc = float(items[1]) if len(items) > 1 else 0.0
            occ = int(items[2]) if len(items) > 2 else 0
            x1, y1, x2, y2 = map(float, items[4:8])
            h, w, l = map(float, items[8:11])
            x, y, z, ry = map(float, items[11:15])
            score = float(items[15]) if len(items) > 15 else 1.0
            height = y2 - y1
            boxes.append([x, y, z, w, h, l, ry, x1, y1, x2, y2, trunc, occ, score, height])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 15), dtype=np.float32)


def _bev_polygon(box7):
    x, y, z, w, h, l, ry = box7
    cosa, sina = np.cos(ry), np.sin(ry)
    dx, dz = w / 2.0, l / 2.0
    corners = np.array([[dx, dz], [dx, -dz], [-dx, -dz], [-dx, dz]], dtype=np.float32)
    rot = np.array([[cosa, -sina], [sina, cosa]], dtype=np.float32)
    corners = corners @ rot.T
    corners += np.array([x, z], dtype=np.float32)
    return corners


def _poly_area(poly):
    # Shoelace formula for simple polygon area
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _poly_intersection_area(p1, p2):
    try:
        from shapely.geometry import Polygon
        inter = Polygon(p1).intersection(Polygon(p2)).area
        return float(inter)
    except Exception:
        # Fallback: no intersection without shapely
        return 0.0


def bev_iou(box_a7, box_b7):
    pa = _bev_polygon(box_a7)
    pb = _bev_polygon(box_b7)
    inter = _poly_intersection_area(pa, pb)
    area_a = _poly_area(pa)
    area_b = _poly_area(pb)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def evaluate(pred_dir, gt_dir, iou_thresh=0.7):
    """간단한 BEV IoU 매칭 기반 평가. 반환 dict: precision, recall, TP, FP, FN."""
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    all_tp, all_fp, all_fn = 0, 0, 0

    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, base)
        gt_boxes = load_label_file(gt_path)
        pred_boxes = load_label_file(pred_path)

        matched_gt = set()
        tp, fp = 0, 0
        for pb in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for gi, gb in enumerate(gt_boxes):
                iou = bev_iou(pb[:7], gb[:7])
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_iou >= iou_thresh and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)
        all_tp += tp
        all_fp += fp
        all_fn += fn

    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)

    return {"precision": precision, "recall": recall, "TP": all_tp, "FP": all_fp, "FN": all_fn}


# ---------------- KITTI-style AP (R40) for BEV and 3D ----------------

def _height_overlap_y(box_a7, box_b7):
    ya, ha = box_a7[1], box_a7[4]
    yb, hb = box_b7[1], box_b7[4]
    min_a, max_a = ya - ha / 2.0, ya + ha / 2.0
    min_b, max_b = yb - hb / 2.0, yb + hb / 2.0
    return max(0.0, min(max_a, max_b) - max(min_a, min_b))


def iou_3d_upright(box_a7, box_b7):
    inter_bev = bev_iou(box_a7, box_b7)
    # Convert bev IOU to intersection area by backing out areas
    pa = _bev_polygon(box_a7)
    pb = _bev_polygon(box_b7)
    area_a = _poly_area(pa)
    area_b = _poly_area(pb)
    inter_area = inter_bev * (area_a + area_b) / (1 + inter_bev) if inter_bev > 0 else 0.0
    h_ov = _height_overlap_y(box_a7, box_b7)
    inter_vol = inter_area * h_ov
    vol_a = area_a * box_a7[4]
    vol_b = area_b * box_b7[4]
    union = vol_a + vol_b - inter_vol
    return inter_vol / (union + 1e-6)


def _filter_difficulty(gt_boxes, difficulty: str):
    # difficulty based on truncation, occlusion, and 2D bbox height
    if difficulty == 'easy':
        cond = (gt_boxes[:, 12] <= 0.15) & (gt_boxes[:, 13] == 0) & (gt_boxes[:, 14] >= 25)  # Relaxed height condition
    elif difficulty == 'moderate':
        cond = (gt_boxes[:, 12] <= 0.3) & (gt_boxes[:, 13] <= 1) & (gt_boxes[:, 14] >= 25)
    elif difficulty == 'hard':
        cond = (gt_boxes[:, 12] <= 0.5) & (gt_boxes[:, 13] <= 2) & (gt_boxes[:, 14] >= 25)
    else:
        cond = np.ones((gt_boxes.shape[0],), dtype=bool)
    return gt_boxes[cond]


def _compute_ap_r40(tp, fp, num_gt):
    if num_gt == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (num_gt + 1e-6)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
    # Interpolated precision
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    # 40 recall points
    recall_levels = np.linspace(0.0, 1.0, 40)
    ap = 0.0
    for r in recall_levels:
        p = precision[recall >= r].max() if np.any(recall >= r) else 0.0
        ap += p
    ap /= len(recall_levels)
    return ap * 100.0  # percentage


def _eval_one(pred_dir, gt_dir, iou_fn, iou_thresh, difficulty):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))
    # Collect predictions across all images
    recs = []  # (score, is_tp as we will fill later)
    num_gt_total = 0
    for gt_path in gt_files:
        base = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, base)
        gts = load_label_file(gt_path)
        gts = _filter_difficulty(gts, difficulty)
        num_gt_total += len(gts)
        preds = load_label_file(pred_path)
        # For matching, sort predictions by score desc
        order = np.argsort(-preds[:, 14]) if len(preds) > 0 else np.array([], dtype=int)
        preds = preds[order]
        used = np.zeros((len(gts),), dtype=bool)
        for pb in preds:
            best_iou, best_idx = 0.0, -1
            for gi, gb in enumerate(gts):
                if used[gi]:
                    continue
                iou = iou_fn(pb[:7], gb[:7])
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            is_tp = 1 if (best_iou >= iou_thresh and best_idx >= 0) else 0
            if is_tp:
                used[best_idx] = True
            recs.append((pb[14], is_tp))

    if len(recs) == 0:
        return 0.0
    # Sort by score
    recs.sort(key=lambda x: -x[0])
    tp = np.array([r[1] for r in recs], dtype=np.int32)
    fp = 1 - tp
    return _compute_ap_r40(tp, fp, num_gt_total)


def evaluate_ap_kitti(pred_dir, gt_dir, iou_thresh=0.7):
    """Return dict with AP_R40 for BEV and 3D at specified IoU threshold for Car class.
    { 'bev': {'easy': x, 'moderate': y, 'hard': z}, '3d': {...} }
    """
    out = {'bev': {}, '3d': {}}
    for diff in ['easy', 'moderate', 'hard']:
        ap_bev = _eval_one(pred_dir, gt_dir, bev_iou, iou_thresh, diff)
        ap_3d = _eval_one(pred_dir, gt_dir, iou_3d_upright, iou_thresh, diff)
        out['bev'][diff] = ap_bev
        out['3d'][diff] = ap_3d
    return out
