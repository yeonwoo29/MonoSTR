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

def debug_gt_matching():
    # Check a few samples to see GT matching quality
    json_path = "../dataset/keypoints_with_theta_train.json"
    label_dir = "../dataset/label_2"
    merge_output_dir = "../dataset/merge_output"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check first 5 samples
    for i in range(min(5, len(data))):
        obj = data[i]
        image_id = obj["image_id"].replace(".png", "")
        print(f"\n=== Sample {i+1}: {image_id} ===")
        
        # Get merged bbox
        merge_path = os.path.join(merge_output_dir, f"{image_id}.txt")
        if os.path.exists(merge_path):
            detections = load_kitti_pred_file(merge_path)
            if len(detections) > 0:
                merged_bbox = detections[0]["box3d"]
                print(f"Merged bbox: {merged_bbox}")
            else:
                print("No merged detections")
                continue
        else:
            print("No merged file")
            continue
        
        # Get GT boxes
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        gt_boxes = parse_kitti_label_file(label_path)
        print(f"GT boxes count: {len(gt_boxes)}")
        
        # Check distances
        merged_center = np.array(merged_bbox[:3])
        for j, gt in enumerate(gt_boxes):
            gt_center = np.array(gt[:3])
            dist = np.linalg.norm(gt_center - merged_center)
            print(f"  GT {j}: center={gt_center}, dist={dist:.3f}")
            print(f"    GT box: {gt}")
        
        # Find best match
        min_dist = float('inf')
        best_gt = None
        for gt in gt_boxes:
            gt_center = np.array(gt[:3])
            dist = np.linalg.norm(gt_center - merged_center)
            if dist < min_dist:
                min_dist = dist
                best_gt = gt
        
        if best_gt is not None:
            print(f"Best match: dist={min_dist:.3f}, GT={best_gt}")
        else:
            print("No GT found!")

if __name__ == "__main__":
    debug_gt_matching()
