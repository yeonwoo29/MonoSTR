import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
class KeypointThetaDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.data = [item for item in self.data if item["theta"] is not None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        keypoints = np.array(item["keypoints"], dtype=np.float32).copy()  # (12, 2)
        keypoints[:, 0] /= 1242
        keypoints[:, 1] /= 375
        keypoints_flat = keypoints.flatten()  # (24,)

        key_left, key_top, key_right, key_bottom = np.array(item["crop_bbox"], dtype=np.float32)
        key_width = (key_right - key_left)
        key_height = (key_bottom - key_top)
        box_ratio = key_height / key_width

        key_front_x = (keypoints[0, 0] + keypoints[1, 0] + keypoints[4, 0] + 
                    keypoints[5, 0] + keypoints[8, 0] + keypoints[9, 0]) / 6
        key_back_x = (keypoints[2, 0] + keypoints[3, 0] + keypoints[6, 0] + 
                    keypoints[7, 0] + keypoints[10, 0] + keypoints[11, 0]) / 6
        key_front_y = (keypoints[0, 1] + keypoints[1, 1] + keypoints[4, 1] + 
                    keypoints[5, 1] + keypoints[8, 1] + keypoints[9, 1]) / 6
        key_back_y = (keypoints[2, 1] + keypoints[3, 1] + keypoints[6, 1] + 
                    keypoints[7, 1] + keypoints[10, 1] + keypoints[11, 1]) / 6

        features = np.concatenate([keypoints_flat, [box_ratio, key_front_x, key_front_y, key_back_x, key_back_y]])  # (29,)

        theta = np.array(item["theta"], dtype=np.float32)
        theta = theta /3.14

        return torch.tensor(features, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32)


class MLPThetaPredictor(nn.Module):
    def __init__(self, input_dim=29, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            # nn.GELU(),
            # nn.Tanh(),
            # nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, x):
        return (self.model(x).squeeze(1) -0.5) * 2

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    
    
def denormalize_theta(theta):
    return theta * 3.14


# def angle_error_deg(pred, target):
#     pred_rad = torch.abs(pred * 3.14)
#     target_rad = torch.abs(target * 3.14)
#     error_rad = pred_rad - target_rad
#     return torch.abs(error_rad * 180.0 / 3.14)  
def angle_error_deg(pred, target):
    diff = pred - target
    # wrap difference to [-pi, pi]
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return torch.abs(diff * 180.0 / math.pi)

def angle_loss(pred, target, reduction='mean'):
    diff = pred - target
    diff = (diff + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
    loss = torch.abs(diff)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss  # no reduction

def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-2):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    before_val_loss = 10
    train_losses, val_losses = [], []
    degree_set = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            # loss = criterion(torch.abs(pred), torch.abs(y))
            # loss = criterion(torch.sin(pred), torch.sin(y)) + criterion(torch.cos(pred), torch.cos(y))
            loss = angle_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train MSE: {avg_loss:.4f}", end='')
        scheduler.step()
        # Validation
        model.eval()
        val_loss = 0
        confirm_i = 0
        total_deg_error = 0
        total_samples = 0     
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                # loss = criterion(torch.abs(pred), torch.abs(y))
                # loss = criterion(torch.sin(pred), torch.sin(y)) + criterion(torch.cos(pred), torch.cos(y))
                loss = angle_loss(pred, y)
                val_loss += loss.item() * x.size(0)
                    
                pred_angle = denormalize_theta(pred)
                gt_angle = denormalize_theta(y)

                angle_errors = angle_error_deg(pred_angle, gt_angle)
                total_deg_error += angle_errors.sum().item()
                total_samples += x.size(0)

        mean_deg_error = total_deg_error / total_samples
        degree_set.append(mean_deg_error)
        print(f"\n Mean Angular Error (MAE): {mean_deg_error:.2f} degrees", end='')                

        val_loss /= len(val_loader.dataset)
        print(f" - Val MSE: {val_loss:.4f}")
        if val_loss <= before_val_loss:
            before_val_loss = val_loss
            save_path = "mlp_theta_predictor.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        train_losses.append(avg_loss)
        val_losses.append(val_loss)
    min_deg = min(degree_set)
    min_epoch = degree_set.index(min_deg) + 1
    print(f"\nBest MAE: {min_deg:.2f} degrees at epoch {min_epoch}")
    return train_losses, val_losses


if __name__ == "__main__":
    train_json = "/media/hong/CC101550101542BE/keypoints/keypoints_with_theta_train.json"
    val_json = "/media/hong/CC101550101542BE/keypoints/keypoints_with_theta_val.json"

    train_dataset = KeypointThetaDataset(train_json)
    val_dataset = KeypointThetaDataset(val_json)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPThetaPredictor().to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    plot_losses(train_losses, val_losses)
