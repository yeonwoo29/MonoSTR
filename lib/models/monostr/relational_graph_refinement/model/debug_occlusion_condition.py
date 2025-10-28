import os
import numpy as np
from evaluate import load_label_file

def debug_occlusion_condition():
    """Debug the occlusion condition specifically"""
    label_dir = "../dataset/label_2"
    val_ids_path = "../dataset/ImageSets/val.txt"
    
    # Load val image IDs
    with open(val_ids_path, 'r') as f:
        val_ids = set(line.strip() for line in f if line.strip())
    
    print("=== Occlusion Condition Debug ===")
    
    total_cars = 0
    occlusion_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # Check first 20 val images
    sample_ids = list(val_ids)[:20]
    
    for image_id in sample_ids:
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        if not os.path.exists(label_path):
            continue
            
        gt_boxes = load_label_file(label_path)
        total_cars += len(gt_boxes)
        
        for i, gt in enumerate(gt_boxes):
            truncated = gt[12]
            occluded = int(gt[13])
            height = gt[14]
            
            occlusion_counts[occluded] += 1
            
            # Check each condition separately
            cond1 = truncated <= 0.15
            cond2 = occluded == 0
            cond3 = height >= 25
            
            if cond1 and cond3 and not cond2:
                print(f"Image {image_id}, GT {i}: truncated={truncated:.3f}, occluded={occluded}, height={height:.1f}")
                print(f"  Conditions: trunc<=0.15={cond1}, occ==0={cond2}, height>=25={cond3}")
    
    print(f"\n=== Occlusion Statistics ===")
    print(f"Total cars: {total_cars}")
    for occ, count in occlusion_counts.items():
        print(f"Occlusion {occ}: {count} ({count/total_cars*100:.1f}%)")
    
    print(f"\nEasy condition breakdown:")
    print(f"Occlusion == 0: {occlusion_counts[0]} ({occlusion_counts[0]/total_cars*100:.1f}%)")

if __name__ == "__main__":
    debug_occlusion_condition()

