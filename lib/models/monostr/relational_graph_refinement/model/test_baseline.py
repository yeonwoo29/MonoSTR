import os
import tempfile
import shutil
from evaluate import evaluate_ap_kitti

def test_baseline():
    """Test baseline performance by copying merge_output directly"""
    merge_output_dir = "../dataset/merge_output"
    label_dir = "../dataset/label_2"
    
    # Create temporary directory
    with tempfile.TemporaryDirectory(prefix="baseline_test_") as tmp_dir:
        # Copy all files from merge_output to temp dir
        for filename in os.listdir(merge_output_dir):
            if filename.endswith('.txt'):
                src = os.path.join(merge_output_dir, filename)
                dst = os.path.join(tmp_dir, filename)
                shutil.copy2(src, dst)
        
        # Evaluate baseline
        print("Testing baseline performance (merge_output without refinement)...")
        ap = evaluate_ap_kitti(tmp_dir, label_dir)
        
        print("Baseline AP_R40 Results:")
        print(f"BEV - Easy: {ap['bev']['easy']:.2f}, Moderate: {ap['bev']['moderate']:.2f}, Hard: {ap['bev']['hard']:.2f}")
        print(f"3D  - Easy: {ap['3d']['easy']:.2f}, Moderate: {ap['3d']['moderate']:.2f}, Hard: {ap['3d']['hard']:.2f}")

if __name__ == "__main__":
    test_baseline()
