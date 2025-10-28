####################################### 겹침처리 추가 필요!!!!!!!!!!!

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple

# Add path to import helper from box_merge
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from box_merge import load_kitti_pred_file  # reuse KITTI reader
from evaluate import evaluate  # returns metrics (we'll ensure it returns)

# --------------- KITTI GT 로드 함수 ---------------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
def parse_kitti_label_file(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != 'Car':
                continue
            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            ry = float(parts[14])
            boxes.append([x, y, z, w, h, l, ry])
    return boxes

# --------------- Dataset 정의 ---------------
class GraphRefineDataset(Dataset):
    def __init__(self, json_path: str, label_dir: str, merge_output_dir: str, ids_filter_path: str = None):
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Filter by ids list if provided
        allow_ids = None
        if ids_filter_path and os.path.exists(ids_filter_path):
            with open(ids_filter_path, 'r') as f:
                allow_ids = set(line.strip() for line in f if line.strip())
            data = [obj for obj in data if obj.get("image_id", "").replace(".png", "") in allow_ids]
        # Keep only samples that have a non-empty merged output file
        kept = []
        for obj in data:
            image_id = obj.get("image_id", "").replace(".png", "")
            merged_path = os.path.join(merge_output_dir, f"{image_id}.txt")
            if os.path.exists(merged_path) and os.path.getsize(merged_path) > 0:
                # Also check if the file has actual detections
                detections = load_kitti_pred_file(merged_path)
                if len(detections) > 0:
                    kept.append(obj)
        self.data = kept
        self.label_dir = label_dir
        self.merge_output_dir = merge_output_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        image_id = obj["image_id"].replace(".png", "")

        # Get initial bbox from merged output (required)
        merged_path = os.path.join(self.merge_output_dir, f"{image_id}.txt")
        if not os.path.exists(merged_path):
            raise ValueError(f"No merged output found for {image_id}")
        
        detections = load_kitti_pred_file(merged_path)
        if len(detections) == 0:
            raise ValueError(f"No detections in merged output for {image_id}")
        
        init_bbox = detections[0]["box3d"]  # take first merged box

        bbox3d = torch.tensor(init_bbox, dtype=torch.float32)
        keypoints = torch.tensor(obj["keypoints"], dtype=torch.float32).flatten()
        theta = torch.tensor([obj["theta"]], dtype=torch.float32)
        crop_bbox = torch.tensor(obj["crop_bbox"], dtype=torch.float32)

        feature = torch.cat([bbox3d, keypoints, theta], dim=0)  # (32,)
        center = bbox3d[:3]  # x, y, z

        # Load GT box from label_2 and match by nearest center
        label_path = os.path.join(self.label_dir, f"{image_id}.txt")
        gt_boxes = parse_kitti_label_file(label_path)

        min_dist = float('inf')
        matched_gt = None
        for gt in gt_boxes:
            gt_center = np.array(gt[:3])
            dist = np.linalg.norm(gt_center - center.numpy())
            if dist < min_dist:
                min_dist = dist
                matched_gt = gt

        if matched_gt is None:
            matched_gt = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]

        gt_bbox3d = torch.tensor(matched_gt, dtype=torch.float32)

        return {
            "feature": feature,
            "center": center,
            "gt_bbox3d": gt_bbox3d,
            "init_bbox3d": bbox3d,
            "crop_bbox": crop_bbox,
            "image_id": obj["image_id"]
        }

# --------------- Graph Utility 함수 ---------------
def build_distance_adj_matrix(pos, threshold=3.0):
    N = pos.size(0)
    dists = torch.cdist(pos, pos)
    adj = (dists < threshold).float()
    adj.fill_diagonal_(0)
    deg = adj.sum(1, keepdim=True)
    adj = adj / (deg + 1e-6)
    return adj

# --------------- GCN Layer 정의 ---------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        agg = torch.matmul(adj, x)
        return F.relu(self.linear(agg))

# --------------- GNN 모델 정의 ---------------
class GraphRefinementModule(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, out_dim=7):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat, center):
        adj = build_distance_adj_matrix(center, threshold=3.0)
        x = F.relu(self.linear1(feat))
        x = F.relu(self.linear2(x))
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        delta = self.regressor(x)
        return delta

# --------------- KITTI 저장 함수 ---------------
def save_kitti_format(refined_bbox, crop_bbox, image_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    obj_class = "Car"
    truncated, occluded, alpha = 0.00, 0, -1.67
    x1, y1, x2, y2 = crop_bbox.tolist()
    x, y, z, w, h, l, ry = refined_bbox.tolist()
    line = f"{obj_class} {truncated:.2f} {occluded} {alpha:.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}"

    save_path = os.path.join(save_dir, image_id.replace(".png", ".txt"))
    with open(save_path, 'a') as f:
        f.write(line + "\n")

# --------------- Collate 함수 ---------------
def collate_fn(batch):
    return {
        "feature": torch.stack([b["feature"] for b in batch]),
        "center": torch.stack([b["center"] for b in batch]),
        "gt_bbox3d": torch.stack([b["gt_bbox3d"] for b in batch]),  # ← 추가
        "init_bbox3d": torch.stack([b["init_bbox3d"] for b in batch]),
        "crop_bbox": [b["crop_bbox"] for b in batch],
        "image_id": [b["image_id"] for b in batch]
    }


run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(".." if os.path.basename(os.getcwd()) == "model" else ".", "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"training_{run_timestamp}.csv")

# 로그 파일 헤더 작성
with open(log_path, "w") as f:
    f.write(
        "epoch,avg_loss,reg_loss,ang_loss,prec@0.7,rec@0.7,bev_easy,bev_moderate,bev_hard,3d_easy,3d_moderate,3d_hard,best\n"
    )

def angle_diff_wrap(pred, gt):
    return torch.atan2(torch.sin(pred - gt), torch.cos(pred - gt))


# --------------- 학습 + 저장 파이프라인 ---------------
if __name__ == "__main__":
    # Paths (relative to repo root)
    dataset_root = os.path.join(".." if os.path.basename(os.getcwd()) == "model" else ".", "dataset")
    train_json = os.path.join(dataset_root, "keypoints_with_theta_train.json")
    val_json = os.path.join(dataset_root, "keypoints_with_theta_val.json")
    label_dir = os.path.join(dataset_root, "label_2")
    merge_output_dir = os.path.join(dataset_root, "merge_output")
    train_ids = os.path.join(dataset_root, "ImageSets", "train.txt")
    val_ids = os.path.join(dataset_root, "ImageSets", "val.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets and loaders
    train_dataset = GraphRefineDataset(train_json, label_dir, merge_output_dir, train_ids)
    val_dataset = GraphRefineDataset(val_json, label_dir, merge_output_dir, val_ids)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = GraphRefinementModule().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Lower learning rate

    best_f1 = -1.0
    latest_ckpt = os.path.join(log_dir, "graph_refine_latest.pth")
    best_ckpt = os.path.join(log_dir, "graph_refine_best.pth")

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_reg_loss = 0.0
        running_ang_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            feat = batch["feature"].to(device)
            center = batch["center"].to(device)
            target = batch["gt_bbox3d"].to(device)
            init = batch["init_bbox3d"].to(device)

            delta = model(feat, center)
            refined = init + delta

            resid_lin = refined[:, :6] - target[:, :6]
            resid_ang = angle_diff_wrap(refined[:, 6], target[:, 6])

            reg_loss = F.smooth_l1_loss(resid_lin, torch.zeros_like(resid_lin))
            ang_loss = F.smooth_l1_loss(resid_ang, torch.zeros_like(resid_ang))
            loss = reg_loss + 1.0 * ang_loss  # Reduce angle loss weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_reg_loss += reg_loss.item()
            running_ang_loss += ang_loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        avg_reg_loss = running_reg_loss / max(len(train_loader), 1)
        avg_ang_loss = running_ang_loss / max(len(train_loader), 1)

        # Save latest checkpoint
        torch.save(model.state_dict(), latest_ckpt)

        # Evaluate on val set: generate preds (temp), run evaluate, then remove preds
        model.eval()
        import tempfile, shutil
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix=f"val_preds_e{epoch+1}_", dir=log_dir)
        preds_dir = tmp_dir_obj.name
        with torch.no_grad():
            for sample in tqdm(val_dataset, desc="Val Infer"):
                feat = sample["feature"].unsqueeze(0).to(device)
                center = sample["center"].unsqueeze(0).to(device)
                delta = model(feat, center)
                refined = sample["init_bbox3d"].to(device) + delta.squeeze(0)
                refined_cpu = refined.detach().cpu()
                save_kitti_format(refined_cpu, sample["crop_bbox"], sample["image_id"], preds_dir)

        # Evaluate using provided evaluator at two IoU thresholds
        m70 = evaluate(preds_dir, label_dir, iou_thresh=0.7)
        m50 = evaluate(preds_dir, label_dir, iou_thresh=0.5)
        # KITTI-style AP_R40
        from evaluate import evaluate_ap_kitti
        ap = evaluate_ap_kitti(preds_dir, label_dir)
        precision = m70.get("precision", 0.0)
        recall = m70.get("recall", 0.0)
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)

        # Save best checkpoint
        is_best = False
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_ckpt)
            is_best = True

        # Log to file
        with open(log_path, "a") as f:
            f.write(
                f"{epoch+1},{avg_loss:.6f},{avg_reg_loss:.6f},{avg_ang_loss:.6f},{precision:.6f},{recall:.6f},"
                f"{ap['bev']['easy']:.4f},{ap['bev']['moderate']:.4f},{ap['bev']['hard']:.4f},"
                f"{ap['3d']['easy']:.4f},{ap['3d']['moderate']:.4f},{ap['3d']['hard']:.4f},"
                f"{'best' if is_best else ''}\n"
            )

        # Console style summary resembling KITTI logs (not official AP)
        print(f"Val@0.70 -> P:{m70['precision']:.4f} R:{m70['recall']:.4f} TP:{m70['TP']} FP:{m70['FP']} FN:{m70['FN']}")
        print(f"Val@0.50 -> P:{m50['precision']:.4f} R:{m50['recall']:.4f} TP:{m50['TP']} FP:{m50['FP']} FN:{m50['FN']}")
        print(f"AP_R40 BEV (0.7)  Easy:{ap['bev']['easy']:.2f}  Mod:{ap['bev']['moderate']:.2f}  Hard:{ap['bev']['hard']:.2f}")
        print(f"AP_R40 3D  (0.7)  Easy:{ap['3d']['easy']:.2f}  Mod:{ap['3d']['moderate']:.2f}  Hard:{ap['3d']['hard']:.2f}")

        # Remove temporary predictions
        tmp_dir_obj.cleanup()

        # Console
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} Reg={avg_reg_loss:.4f} Ang={avg_ang_loss:.4f} | P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
