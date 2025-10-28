import os
import numpy as np
from evaluate import load_label_file, bev_iou, iou_3d_upright

def simple_iou_test():
    """Simple test to check IoU calculation"""
    merge_output_dir = "../dataset/merge_output"
    label_dir = "../dataset/label_2"
    
    # Test with a specific image
    image_id = "000003"  # We know this has Easy GT from earlier analysis
    
    label_path = os.path.join(label_dir, f"{image_id}.txt")
    pred_path = os.path.join(merge_output_dir, f"{image_id}.txt")
    
    print(f"Testing image {image_id}")
    print(f"Label path: {label_path}")
    print(f"Pred path: {pred_path}")
    print(f"Label exists: {os.path.exists(label_path)}")
    print(f"Pred exists: {os.path.exists(pred_path)}")
    
    if os.path.exists(label_path) and os.path.exists(pred_path):
        gt_boxes = load_label_file(label_path)
        pred_boxes = load_label_file(pred_path)
        
        print(f"GT boxes shape: {gt_boxes.shape}")
        print(f"Pred boxes shape: {pred_boxes.shape}")
        
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            print(f"First GT box: {gt_boxes[0][:7]}")
            print(f"First pred box: {pred_boxes[0][:7]}")
            
            # Calculate IoU
            bev_iou_val = bev_iou(pred_boxes[0][:7], gt_boxes[0][:7])
            iou_3d_val = iou_3d_upright(pred_boxes[0][:7], gt_boxes[0][:7])
            
            print(f"BEV IoU: {bev_iou_val:.3f}")
            print(f"3D IoU: {iou_3d_val:.3f}")

if __name__ == "__main__":
    simple_iou_test()

