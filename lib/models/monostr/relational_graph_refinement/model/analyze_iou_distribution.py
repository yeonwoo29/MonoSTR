import os
import numpy as np
from evaluate import load_label_file, bev_iou, iou_3d_upright

def analyze_iou_distribution():
    """Analyze IoU distribution between predictions and Easy GT"""
    merge_output_dir = "../dataset/merge_output"
    label_dir = "../dataset/label_2"
    val_ids_path = "../dataset/ImageSets/val.txt"
    
    # Load val image IDs
    with open(val_ids_path, 'r') as f:
        val_ids = set(line.strip() for line in f if line.strip())
    
    print("=== IoU Distribution Analysis ===")
    
    all_bev_ious = []
    all_3d_ious = []
    easy_gt_count = 0
    total_matches = 0
    
    # Analyze first 10 val images
    sample_ids = list(val_ids)[:10]
    print(f"Sample IDs: {sample_ids}")
    
    for image_id in sample_ids:
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        pred_path = os.path.join(merge_output_dir, f"{image_id}.txt")
        
        if not os.path.exists(label_path):
            print(f"  No label file: {label_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"  No pred file: {pred_path}")
            continue
            
        gt_boxes = load_label_file(label_path)
        pred_boxes = load_label_file(pred_path)
        
        # Filter Easy GT
        easy_gt_boxes = []
        for gt in gt_boxes:
            truncated = gt[12]
            occluded = gt[13]
            height = gt[14]
            if truncated <= 0.15 and occluded == 0 and height >= 40:
                easy_gt_boxes.append(gt)
        
        easy_gt_count += len(easy_gt_boxes)
        
        if len(easy_gt_boxes) == 0 or len(pred_boxes) == 0:
            continue
        
        print(f"\nImage {image_id}: {len(easy_gt_boxes)} Easy GT, {len(pred_boxes)} predictions")
        
        # Find best IoU for each Easy GT
        for gt in easy_gt_boxes:
            best_bev_iou = 0.0
            best_3d_iou = 0.0
            
            for pred in pred_boxes:
                bev_iou_val = bev_iou(pred[:7], gt[:7])
                iou_3d_val = iou_3d_upright(pred[:7], gt[:7])
                
                if bev_iou_val > best_bev_iou:
                    best_bev_iou = bev_iou_val
                if iou_3d_val > best_3d_iou:
                    best_3d_iou = iou_3d_val
            
            all_bev_ious.append(best_bev_iou)
            all_3d_ious.append(best_3d_iou)
            total_matches += 1
            
            print(f"  Easy GT: BEV IoU={best_bev_iou:.3f}, 3D IoU={best_3d_iou:.3f}")
    
    if all_bev_ious:
        print(f"\n=== Summary ===")
        print(f"Total Easy GT analyzed: {easy_gt_count}")
        print(f"Total matches found: {total_matches}")
        print(f"BEV IoU - Mean: {np.mean(all_bev_ious):.3f}, Max: {np.max(all_bev_ious):.3f}, Min: {np.min(all_bev_ious):.3f}")
        print(f"3D IoU  - Mean: {np.mean(all_3d_ious):.3f}, Max: {np.max(all_3d_ious):.3f}, Min: {np.min(all_3d_ious):.3f}")
        
        # Count IoU thresholds
        bev_above_05 = sum(1 for iou in all_bev_ious if iou >= 0.5)
        bev_above_07 = sum(1 for iou in all_bev_ious if iou >= 0.7)
        iou3d_above_05 = sum(1 for iou in all_3d_ious if iou >= 0.5)
        iou3d_above_07 = sum(1 for iou in all_3d_ious if iou >= 0.7)
        
        print(f"BEV IoU >= 0.5: {bev_above_05}/{len(all_bev_ious)} ({bev_above_05/len(all_bev_ious)*100:.1f}%)")
        print(f"BEV IoU >= 0.7: {bev_above_07}/{len(all_bev_ious)} ({bev_above_07/len(all_bev_ious)*100:.1f}%)")
        print(f"3D IoU >= 0.5: {iou3d_above_05}/{len(all_3d_ious)} ({iou3d_above_05/len(all_3d_ious)*100:.1f}%)")
        print(f"3D IoU >= 0.7: {iou3d_above_07}/{len(all_3d_ious)} ({iou3d_above_07/len(all_3d_ious)*100:.1f}%)")

if __name__ == "__main__":
    analyze_iou_distribution()
