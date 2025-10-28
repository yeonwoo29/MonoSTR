import torch
import json
import os
import numpy as np
from box_merge import load_kitti_pred_file

def parse_kitti_label_file(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != 'Car':
                continue
            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            ry = float(parts[14])
            boxes.append([x, y, z, w, h, l, ry])
    return boxes

def debug_sample():
    # Load one sample from train data
    json_path = "../dataset/keypoints_with_theta_train.json"
    label_dir = "../dataset/label_2"
    merge_output_dir = "../dataset/merge_output"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Take first sample
    obj = data[0]
    image_id = obj["image_id"].replace(".png", "")
    print(f"Sample image_id: {image_id}")
    
    # Check original bbox3d
    print(f"Original bbox3d: {obj['bbox3d']}")
    
    # Check merged bbox3d
    merge_path = os.path.join(merge_output_dir, f"{image_id}.txt")
    if os.path.exists(merge_path):
        detections = load_kitti_pred_file(merge_path)
        if len(detections) > 0:
            merged_bbox = detections[0]["box3d"]
            print(f"Merged bbox3d: {merged_bbox}")
        else:
            print("No merged detections found")
    else:
        print("No merged file found")
    
    # Check keypoints and theta
    print(f"Keypoints shape: {len(obj['keypoints'])}")
    print(f"Theta: {obj['theta']}")
    
    # Check GT boxes
    label_path = os.path.join(label_dir, f"{image_id}.txt")
    gt_boxes = parse_kitti_label_file(label_path)
    print(f"GT boxes count: {len(gt_boxes)}")
    for i, gt in enumerate(gt_boxes):
        print(f"GT {i}: {gt}")
    
    # Check feature construction
    bbox3d = torch.tensor(merged_bbox if os.path.exists(merge_path) and len(load_kitti_pred_file(merge_path)) > 0 else obj["bbox3d"], dtype=torch.float32)
    keypoints = torch.tensor(obj["keypoints"], dtype=torch.float32).flatten()
    theta = torch.tensor([obj["theta"]], dtype=torch.float32)
    
    feature = torch.cat([bbox3d, keypoints, theta], dim=0)
    print(f"Feature shape: {feature.shape}")
    print(f"Feature first 10 values: {feature[:10]}")

if __name__ == "__main__":
    debug_sample()
