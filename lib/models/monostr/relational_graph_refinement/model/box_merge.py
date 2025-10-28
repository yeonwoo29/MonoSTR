import os
import glob
import numpy as np
from shapely.geometry import Polygon

# ------------------------
# BEV IoU 계산 함수
# ------------------------
def bev_polygon(box):
    """3D box → BEV polygon (x,z plane)."""
    x, y, z, w, h, l, ry = box
    cosa, sina = np.cos(ry), np.sin(ry)
    dx, dz = w/2, l/2
    corners = np.array([
        [ dx,  dz],
        [ dx, -dz],
        [-dx, -dz],
        [-dx,  dz]
    ])
    rot = np.array([[cosa, -sina],[sina, cosa]])
    corners = corners @ rot.T
    corners += np.array([x, z])
    return Polygon(corners)

def bev_iou(box1, box2):
    """두 3D box의 BEV IoU 계산"""
    poly1, poly2 = bev_polygon(box1), bev_polygon(box2)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / (union + 1e-6)

# ------------------------
# KITTI I/O
# ------------------------
def load_kitti_pred_file(path):
    detections = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls != 'Car':  # Car만 처리
                continue
            x1, y1, x2, y2 = map(float, parts[4:8])
            h, w, l = map(float, parts[8:11])
            x, y, z, ry = map(float, parts[11:15])
            score = float(parts[15]) if len(parts) > 15 else 1.0
            detections.append({
                'cls': cls,
                'box2d': [x1, y1, x2, y2],
                'box3d': [x, y, z, w, h, l, ry],
                'score': score
            })
    return detections

def save_kitti_lines(path, detections):
    with open(path, 'w') as f:
        for det in detections:
            x1, y1, x2, y2 = det['box2d']
            x, y, z, w, h, l, ry = det['box3d']
            score = det['score']
            obj_class = 'Car'
            truncated, occluded, alpha = 0.00, 0, -1.67
            line = f"{obj_class} {truncated:.2f} {occluded} {alpha:.2f} " \
                   f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} " \
                   f"{h:.2f} {w:.2f} {l:.2f} " \
                   f"{x:.2f} {y:.2f} {z:.2f} {ry:.2f} {score:.4f}"
            f.write(line + "\n")

# ------------------------
# 겹침 처리 (score 낮은 박스 제거)
# ------------------------
def suppress_overlaps(detections, iou_thresh=0.5):
    """
    detections: [{'cls','box2d','box3d','score'}, ...]
    IoU ≥ iou_thresh 이면 score 낮은 것 제거
    """
    detections = sorted(detections, key=lambda d: d['score'], reverse=True)
    keep = []

    while detections:
        best = detections.pop(0)  # 현재 가장 score 높은 박스
        keep.append(best)
        survivors = []
        for det in detections:
            iou = bev_iou(best['box3d'], det['box3d'])
            if iou < iou_thresh:
                survivors.append(det)  # 겹치지 않으면 유지
            # 겹치면 score 낮은게 제거됨 (best가 이미 더 높음)
        detections = survivors

    return keep

# ------------------------
# 폴더 단위 처리
# ------------------------
def merge_folder_nms(input_dir, output_dir, iou_thresh=0.5):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = sorted(glob.glob(os.path.join(input_dir, '*.txt')))
    for in_path in txt_files:
        image_id = os.path.splitext(os.path.basename(in_path))[0]
        detections = load_kitti_pred_file(in_path)
        if not detections:
            out_path = os.path.join(output_dir, image_id + '.txt')
            open(out_path, 'a').close()
            continue

        filtered = suppress_overlaps(detections, iou_thresh=iou_thresh)
        out_path = os.path.join(output_dir, image_id + '.txt')
        save_kitti_lines(out_path, filtered)

# ------------------------
# 실행 예시
# ------------------------
if __name__ == "__main__":
    input_dir = "../dataset/3d_output_val"
    output_dir = "../dataset/merge_output_val"
    merge_folder_nms(input_dir, output_dir, iou_thresh=0.3)
    print(f"✅ 겹침 제거 완료: {output_dir}")
