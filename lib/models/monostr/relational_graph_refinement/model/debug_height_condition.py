import os
import numpy as np
from evaluate import load_label_file

def debug_height_condition():
    """Debug the height condition specifically"""
    label_dir = "../dataset/label_2"
    val_ids_path = "../dataset/ImageSets/val.txt"
    
    # Load val image IDs
    with open(val_ids_path, 'r') as f:
        val_ids = set(line.strip() for line in f if line.strip())
    
    print("=== Height Condition Debug ===")
    
    total_cars = 0
    height_stats = []
    
    # Check first 10 val images
    sample_ids = list(val_ids)[:10]
    
    for image_id in sample_ids:
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        if not os.path.exists(label_path):
            continue
            
        gt_boxes = load_label_file(label_path)
        total_cars += len(gt_boxes)
        
        for i, gt in enumerate(gt_boxes):
            truncated = gt[12]
            occluded = gt[13]
            height = gt[14]  # 2D bbox height
            
            height_stats.append(height)
            
            # Check each condition separately
            cond1 = truncated <= 0.15
            cond2 = occluded == 0
            cond3 = height >= 25
            
            if cond1 and cond2 and not cond3:
                print(f"Image {image_id}, GT {i}: truncated={truncated:.3f}, occluded={occluded}, height={height:.1f}")
                print(f"  Conditions: trunc<=0.15={cond1}, occ==0={cond2}, height>=25={cond3}")
    
    if height_stats:
        print(f"\n=== Height Statistics ===")
        print(f"Total cars: {total_cars}")
        print(f"Height - Min: {np.min(height_stats):.1f}, Max: {np.max(height_stats):.1f}, Mean: {np.mean(height_stats):.1f}")
        print(f"Height >= 25: {sum(1 for h in height_stats if h >= 25)}/{len(height_stats)} ({sum(1 for h in height_stats if h >= 25)/len(height_stats)*100:.1f}%)")
        print(f"Height >= 40: {sum(1 for h in height_stats if h >= 40)}/{len(height_stats)} ({sum(1 for h in height_stats if h >= 40)/len(height_stats)*100:.1f}%)")

if __name__ == "__main__":
    debug_height_condition()

