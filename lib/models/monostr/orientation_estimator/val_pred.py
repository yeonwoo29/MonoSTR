import os
import torch
import math

def angle_error_deg(pred, target):
    error_rad = torch.abs(pred - target)
    error_rad = torch.remainder(error_rad + math.pi, 2 * math.pi) - math.pi
    return torch.abs(error_rad * 180.0 / math.pi)

def load_theta_from_txt(path, valid_ids):
    thetas = {}
    for fname in os.listdir(path):
        if not fname.endswith(".txt"):
            continue
        image_id = os.path.splitext(fname)[0]
        if image_id not in valid_ids:
            continue
        with open(os.path.join(path, fname), "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        # KITTI 형식: class, trunc, occ, alpha, ...
        alpha = float(lines[0].split()[3])
        thetas[image_id] = alpha
    return thetas

def evaluate(gt_dir, pred_dir, val_list_path):
    # val.txt에서 사용할 이미지 id 로드
    with open(val_list_path, "r") as f:
        valid_ids = set(line.strip() for line in f.readlines())

    gt_thetas = load_theta_from_txt(gt_dir, valid_ids)
    pred_thetas = load_theta_from_txt(pred_dir, valid_ids)

    total_error, count = 0.0, 0
    for image_id in valid_ids:
        if image_id not in gt_thetas or image_id not in pred_thetas:
            continue
        gt_theta = gt_thetas[image_id]
        pred_theta = pred_thetas[image_id]

        gt_tensor = torch.tensor(gt_theta)
        pred_tensor = torch.tensor(pred_theta)

        error = angle_error_deg(pred_tensor, gt_tensor)
        total_error += error.item()
        count += 1

    mean_error = total_error / count if count > 0 else 0
    print(f"Mean Angular Error (MAE): {mean_error:.2f} degrees over {count} samples")

if __name__ == "__main__":
    gt_dir = "./label_2"          # 전체 label_2 txt
    pred_dir = "./3d_output"      # 전체 output txt
    val_list_path = "./ImageSets/val.txt"
    evaluate(gt_dir, pred_dir, val_list_path)