import os
import numpy as np
from evaluate import load_label_file

def check_easy_conditions():
    """Check Easy filtering conditions"""
    label_dir = "../dataset/label_2"
    val_ids_path = "../dataset/ImageSets/val.txt"
    
    # Load val image IDs
    with open(val_ids_path, 'r') as f:
        val_ids = set(line.strip() for line in f if line.strip())
    
    print("=== Easy Filtering Conditions Analysis ===")
    
    total_cars = 0
    easy_by_our_criteria = 0
    easy_by_kitti_criteria = 0
    
    # Check first 20 val images
    sample_ids = list(val_ids)[:20]
    
    for image_id in sample_ids:
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        if not os.path.exists(label_path):
            continue
            
        gt_boxes = load_label_file(label_path)
        total_cars += len(gt_boxes)
        
        for gt in gt_boxes:
            truncated = gt[12]
            occluded = gt[13]
            height = gt[14]  # 2D bbox height
            
            # Our criteria (from analyze_difficulty.py)
            our_easy = (occluded == 0 and truncated < 0.15)
            
            # KITTI criteria (from evaluate.py)
            kitti_easy = (truncated <= 0.15) and (occluded == 0) and (height >= 40)
            
            if our_easy:
                easy_by_our_criteria += 1
            if kitti_easy:
                easy_by_kitti_criteria += 1
                
            if our_easy and not kitti_easy:
                print(f"Image {image_id}: Our Easy but not KITTI Easy - height={height:.1f}")
    
    print(f"\n=== Summary ===")
    print(f"Total cars analyzed: {total_cars}")
    print(f"Easy by our criteria: {easy_by_our_criteria} ({easy_by_our_criteria/total_cars*100:.1f}%)")
    print(f"Easy by KITTI criteria: {easy_by_kitti_criteria} ({easy_by_kitti_criteria/total_cars*100:.1f}%)")
    
    if easy_by_kitti_criteria == 0:
        print("❌ No Easy objects by KITTI criteria!")
        print("This explains why Easy AP = 0")
    else:
        print("✅ Easy objects exist by KITTI criteria")

if __name__ == "__main__":
    check_easy_conditions()

