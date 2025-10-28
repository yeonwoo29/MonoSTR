import os
import tempfile
import shutil
from evaluate import evaluate_ap_kitti

def debug_easy_ap():
    """Debug why Easy AP is 0"""
    merge_output_dir = "../dataset/merge_output"
    label_dir = "../dataset/label_2"
    val_ids_path = "../dataset/ImageSets/val.txt"
    
    # Load val image IDs
    with open(val_ids_path, 'r') as f:
        val_ids = set(line.strip() for line in f if line.strip())
    
    print("=== Easy AP Debug Analysis ===")
    
    # Test different IoU thresholds
    iou_thresholds = [0.3, 0.5, 0.7]
    
    for iou_thresh in iou_thresholds:
        print(f"\n--- IoU Threshold: {iou_thresh} ---")
        
        with tempfile.TemporaryDirectory(prefix=f"easy_debug_{iou_thresh}_") as tmp_dir:
            # Copy only val files
            for filename in os.listdir(merge_output_dir):
                if filename.endswith('.txt'):
                    image_id = filename.replace('.txt', '')
                    if image_id in val_ids:
                        src = os.path.join(merge_output_dir, filename)
                        dst = os.path.join(tmp_dir, filename)
                        shutil.copy2(src, dst)
            
            # Evaluate with current IoU threshold
            ap = evaluate_ap_kitti(tmp_dir, label_dir, iou_thresh=iou_thresh)
            
            print(f"Easy AP: {ap['3d']['easy']:.4f}")
            print(f"Moderate AP: {ap['3d']['moderate']:.4f}")
            print(f"Hard AP: {ap['3d']['hard']:.4f}")
    
    # Analyze a few specific samples
    print("\n=== Sample Analysis ===")
    sample_ids = list(val_ids)[:5]  # First 5 val samples
    
    for image_id in sample_ids:
        print(f"\nImage {image_id}:")
        
        # Check GT
        label_path = os.path.join(label_dir, f"{image_id}.txt")
        easy_gt_count = 0
        total_gt_count = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts[0] != 'Car':
                        continue
                    total_gt_count += 1
                    truncated = float(parts[1])
                    occluded = int(parts[2])
                    if occluded == 0 and truncated < 0.15:
                        easy_gt_count += 1
                        print(f"  Easy GT: truncated={truncated:.3f}, occluded={occluded}")
        
        # Check predictions
        pred_path = os.path.join(merge_output_dir, f"{image_id}.txt")
        pred_count = 0
        if os.path.exists(pred_path):
            with open(pred_path, 'r') as f:
                for line in f:
                    if line.strip():
                        pred_count += 1
        
        print(f"  GT: {total_gt_count} total, {easy_gt_count} easy")
        print(f"  Predictions: {pred_count}")

if __name__ == "__main__":
    debug_easy_ap()

