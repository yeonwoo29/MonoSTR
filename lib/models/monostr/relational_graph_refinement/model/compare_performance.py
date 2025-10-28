import os
import tempfile
import shutil
from evaluate import evaluate_ap_kitti

def compare_performance():
    """Compare 3d_output vs merge_output performance on VAL data only"""
    three_d_output_dir = "../dataset/3d_output"
    merge_output_dir = "../dataset/merge_output"
    label_dir = "../dataset/label_2"
    val_ids_path = "../dataset/ImageSets/val.txt"
    
    # Load val image IDs
    with open(val_ids_path, 'r') as f:
        val_ids = set(line.strip() for line in f if line.strip())
    
    print("=== Performance Comparison: 3d_output vs merge_output ===")
    print("IoU Threshold: 0.7")
    print()
    
    # Test 3d_output
    print("Testing 3d_output performance (VAL only)...")
    with tempfile.TemporaryDirectory(prefix="3d_output_test_") as tmp_dir:
        # Copy only val files from 3d_output to temp dir
        for filename in os.listdir(three_d_output_dir):
            if filename.endswith('.txt'):
                image_id = filename.replace('.txt', '')
                if image_id in val_ids:
                    src = os.path.join(three_d_output_dir, filename)
                    dst = os.path.join(tmp_dir, filename)
                    shutil.copy2(src, dst)
        
        # Evaluate 3d_output
        ap_3d = evaluate_ap_kitti(tmp_dir, label_dir)
        
        print("3d_output AP_R40 Results:")
        print(f"BEV - Easy: {ap_3d['bev']['easy']:.2f}, Moderate: {ap_3d['bev']['moderate']:.2f}, Hard: {ap_3d['bev']['hard']:.2f}")
        print(f"3D  - Easy: {ap_3d['3d']['easy']:.2f}, Moderate: {ap_3d['3d']['moderate']:.2f}, Hard: {ap_3d['3d']['hard']:.2f}")
    
    print()
    
    # Test merge_output
    print("Testing merge_output performance (VAL only)...")
    with tempfile.TemporaryDirectory(prefix="merge_output_test_") as tmp_dir:
        # Copy only val files from merge_output to temp dir
        for filename in os.listdir(merge_output_dir):
            if filename.endswith('.txt'):
                image_id = filename.replace('.txt', '')
                if image_id in val_ids:
                    src = os.path.join(merge_output_dir, filename)
                    dst = os.path.join(tmp_dir, filename)
                    shutil.copy2(src, dst)
        
        # Evaluate merge_output
        ap_merge = evaluate_ap_kitti(tmp_dir, label_dir)
        
        print("merge_output AP_R40 Results:")
        print(f"BEV - Easy: {ap_merge['bev']['easy']:.2f}, Moderate: {ap_merge['bev']['moderate']:.2f}, Hard: {ap_merge['bev']['hard']:.2f}")
        print(f"3D  - Easy: {ap_merge['3d']['easy']:.2f}, Moderate: {ap_merge['3d']['moderate']:.2f}, Hard: {ap_merge['3d']['hard']:.2f}")
    
    print()
    print("=== Summary ===")
    print("3D AP_R40 Improvement (merge_output - 3d_output):")
    print(f"Easy: {ap_merge['3d']['easy'] - ap_3d['3d']['easy']:+.2f}")
    print(f"Moderate: {ap_merge['3d']['moderate'] - ap_3d['3d']['moderate']:+.2f}")
    print(f"Hard: {ap_merge['3d']['hard'] - ap_3d['3d']['hard']:+.2f}")
    
    # Analyze why Easy AP is 0
    print("\n=== Easy AP Analysis ===")
    analyze_easy_ap_zero(val_ids, label_dir, merge_output_dir)

def analyze_easy_ap_zero(val_ids, label_dir, merge_output_dir):
    """Analyze why Easy AP is 0"""
    easy_gt_count = 0
    easy_pred_count = 0
    total_gt_count = 0
    total_pred_count = 0
    
    # Count Easy GT objects in val set
    for image_id in val_ids:
        label_path = os.path.join(label_dir, f"{image_id}.txt")
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
    
    # Count predictions in val set
    for image_id in val_ids:
        pred_path = os.path.join(merge_output_dir, f"{image_id}.txt")
        if os.path.exists(pred_path):
            with open(pred_path, 'r') as f:
                for line in f:
                    if line.strip():
                        total_pred_count += 1
    
    print(f"VAL set statistics:")
    print(f"Total GT objects: {total_gt_count}")
    print(f"Easy GT objects: {easy_gt_count} ({easy_gt_count/total_gt_count*100:.1f}%)")
    print(f"Total predictions: {total_pred_count}")
    
    if easy_gt_count == 0:
        print("❌ No Easy GT objects in VAL set!")
    else:
        print("✅ Easy GT objects exist in VAL set")
        print("Possible reasons for Easy AP = 0:")
        print("1. Predictions don't match Easy GT objects")
        print("2. IoU threshold 0.7 is too high for Easy objects")
        print("3. Predictions are too far from Easy GT objects")

if __name__ == "__main__":
    compare_performance()
