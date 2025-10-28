import torch
import torch.nn as nn
import numpy as np
import json
import math
from torch.utils.data import Dataset, DataLoader
from theta_regression import KeypointThetaDataset, MLPThetaPredictor 

def denormalize_theta(theta):
    return theta * 2 * math.pi - math.pi

def angle_error_deg(pred, target):
    error_rad = torch.abs(pred - target)
    error_rad = torch.remainder(error_rad + math.pi, 2 * math.pi) - math.pi 
    return torch.abs(error_rad * 180.0 / math.pi)

def validate(model_path, val_json, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    val_dataset = KeypointThetaDataset(val_json)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Load model
    model = MLPThetaPredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_deg_error = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            print(pred, y)
            pred_angle = denormalize_theta(pred)
            gt_angle = denormalize_theta(y)

            angle_errors = angle_error_deg(pred_angle, gt_angle)
            total_deg_error += angle_errors.sum().item()
            total_samples += x.size(0)

    mean_deg_error = total_deg_error / total_samples
    print(f"Mean Angular Error (MAE): {mean_deg_error:.2f} degrees")

if __name__ == "__main__":
    model_path = "mlp_theta_predictor.pth"
    val_json = "./keypoints_with_theta_val.json"
    validate(model_path, val_json)
