import os
import torch
import math
import numpy as np

def angle_error_deg(pred, target):
    diff = pred - target
    diff = (diff + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
    return abs(diff * 180.0 / math.pi)

def parse_kitti_label_line(line):
    parts = line.strip().split()
    cls = parts[0]
    alpha = float(parts[3])
    bbox = [float(x) for x in parts[4:8]]  # 2D bbox
    dims = [float(x) for x in parts[8:11]]  # h, w, l
    loc = [float(x) for x in parts[11:14]]  # x, y, z
    ry = float(parts[14])                   # rotation_y
    return {"cls": cls, "alpha": alpha, "bbox": bbox, "dims": dims, "loc": loc, "ry": ry}

def load_kitti_txt(path, valid_ids):
    data = {}
    for fname in os.listdir(path):
        if not fname.endswith(".txt"):
            continue
        image_id = os.path.splitext(fname)[0]
        if image_id not in valid_ids:
            continue
        with open(os.path.join(path, fname), "r") as f:
            lines = f.readlines()
        objects = [parse_kitti_label_line(line) for line in lines]
        data[image_id] = objects
    return data

# --------------- 3D IoU 계산 함수 ---------------
def box3d_iou(box1, box2):
    # box = {dims[h,w,l], loc[x,y,z], ry}
    # 여기서는 단순화를 위해 KITTI 공식 코드 대신, BEV IoU + height overlap 방식으로 근사
    import shapely.geometry as geom
    import shapely.affinity as aff

    w1, h1, l1 = box1["dims"][1], box1["dims"][0], box1["dims"][2]
    x1, y1, z1 = box1["loc"]
    ry1 = box1["ry"]

    w2, h2, l2 = box2["dims"][1], box2["dims"][0], box2["dims"][2]
    x2, y2, z2 = box2["loc"]
    ry2 = box2["ry"]

    # BEV box (x,z 평면)
    rect1 = geom.box(-l1/2, -w1/2, l1/2, w1/2)
    rect2 = geom.box(-l2/2, -w2/2, l2/2, w2/2)

    rect1 = aff.rotate(rect1, -ry1*180/math.pi, use_radians=False)
    rect2 = aff.rotate(rect2, -ry2*180/math.pi, use_radians=False)

    rect1 = aff.translate(rect1, xoff=x1, yoff=z1)
    rect2 = aff.translate(rect2, xoff=x2, yoff=z2)

    inter_area = rect1.intersection(rect2).area
    union_area = rect1.union(rect2).area

    # height overlap
    y1_min, y1_max = y1 - h1, y1
    y2_min, y2_max = y2 - h2, y2
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    inter_vol = inter_area * inter_h
    vol1 = l1 * w1 * h1
    vol2 = l2 * w2 * h2
    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol

# ------------------------------------------------

def evaluate(gt_dir, pred_dir, val_list_path):
    with open(val_list_path, "r") as f:
        valid_ids = set(line.strip() for line in f.readlines())

    gt_data = load_kitti_txt(gt_dir, valid_ids)
    pred_data = load_kitti_txt(pred_dir, valid_ids)

    total_error, count = 0.0, 0

    for image_id in valid_ids:
        if image_id not in gt_data or image_id not in pred_data:
            continue
        gts = gt_data[image_id]
        preds = pred_data[image_id]

        matched_pred = set()
        for gt in gts:
            best_iou, best_pred = 0, None
            for i, pred in enumerate(preds):
                if i in matched_pred:
                    continue
                iou = box3d_iou(gt, pred)
                if iou > best_iou:
                    best_iou, best_pred = iou, (i, pred)
            if best_iou >= 0.1 and best_pred is not None: #####
                idx, pred = best_pred
                matched_pred.add(idx)

                error = angle_error_deg(pred["ry"], gt["ry"])
                total_error += error
                count += 1

    mean_error = total_error / count if count > 0 else 0
    print(f"Mean Angular Error (MAE): {mean_error:.2f} degrees over {count} matched objects (IoU>=0.1)")

if __name__ == "__main__":
    gt_dir = "./label_2"
    pred_dir = "../outputs_best/data"
    val_list_path = "./ImageSets/val.txt"
    evaluate(gt_dir, pred_dir, val_list_path)
