#!/usr/bin/env python
# train.py
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import datetime
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 프로젝트 루트 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# util 
from lib.helpers.utils_helper import create_logger, set_random_seed

# 우리의 dataset/model
from dataset import CarKeypointDataset
from model import CarKeypointAutoencoder

def compute_length_sym_loss(pred_norm, length_pairs, eps):
    B = pred_norm.size(0)
    pts = pred_norm.view(B, 12, 2)
    losses = []
    for (a, b), (c, d) in length_pairs:
        pa, pb = pts[:, a-1], pts[:, b-1]
        pc, pd = pts[:, c-1], pts[:, d-1]
        dab = torch.norm(pa - pb, dim=1)
        dcd = torch.norm(pc - pd, dim=1)
        long = torch.max(dab, dcd)
        short = torch.min(dab, dcd)
        ratio = short / (long + eps)
        losses.append((1 - ratio) ** 2)
    return torch.mean(torch.stack(losses))

def compute_angle_sym_loss(pred_norm, angle_pairs, eps):
    B = pred_norm.size(0)
    pts = pred_norm.view(B, 12, 2)
    losses = []
    for (a, b, c), (d, e, f) in angle_pairs:
        v1 = pts[:, b-1] - pts[:, a-1]
        v2 = pts[:, c-1] - pts[:, b-1]
        cos1 = (v1 * v2).sum(1) / (torch.norm(v1, 1) * torch.norm(v2, 1) + eps)
        u1 = pts[:, e-1] - pts[:, d-1]
        u2 = pts[:, f-1] - pts[:, e-1]
        cos2 = (u1 * u2).sum(1) / (torch.norm(u1, 1) * torch.norm(u2, 1) + eps)
        losses.append(((cos1 - cos2) / 2) ** 2)
    return torch.mean(torch.stack(losses))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/autoencoder.yaml')
    parser.add_argument('-e', '--evaluate_only', action='store_true')
    args = parser.parse_args()

    assert os.path.exists(args.config), f"{args.config} not found"
    cfg = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    set_random_seed(cfg.get('random_seed', 444))

    # output & logger
    model_name = cfg.get('model_name', 'autoencoder')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg['output']['save_path'], model_name + "_" + timestamp)  # ⬅️ 구분된 폴더 생성
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "train.log")
    logger = create_logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # dataloader
    train_ds = CarKeypointDataset(cfg['dataset']['root_dir'], split='train')
    val_ds = CarKeypointDataset(cfg['dataset']['root_dir'], split=cfg['dataset']['val_split'])
    train_loader = DataLoader(train_ds,
        batch_size=cfg['dataset']['batch_size'], shuffle=True,
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    val_loader = DataLoader(val_ds,
        batch_size=cfg['dataset']['batch_size'], shuffle=False,
        num_workers=cfg['dataset']['num_workers'],
        pin_memory=(device.type == 'cuda')
    )

    # model / optim / loss
    model = CarKeypointAutoencoder(
        input_dim=cfg['model']['input_dim'],
        h1=cfg['model']['h1'],
        h2=cfg['model']['h2']
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg['training']['lr']),
        weight_decay=float(cfg['training']['weight_decay'])
    )
    criterion = torch.nn.MSELoss()
    λ_sym = cfg['loss']['lambda_sym']
    eps = float(cfg['loss']['epsilon'])
    length_pairs = cfg['symmetry']['length_pairs']
    angle_pairs = cfg['symmetry']['angle_triplets']

    best_ckpt = os.path.join(out_dir, 'best.pth')
    if args.evaluate_only:
        assert os.path.isfile(best_ckpt), "No checkpoint for evaluation!"
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        logger.info("##### Testing Only #####")
        logger.info(f"Batch Size: {cfg['dataset']['batch_size']}")
        total_sq, n_pts = 0.0, 0
        model.eval()
        for kps, iso in tqdm(val_loader, desc="Eval"):
            kps = kps.to(device).float()
            iso = iso.to(device).float().unsqueeze(1)
            with torch.no_grad():
                pred = model(kps)
            diff = (pred - kps[:, :24]).view(-1, 12, 2)
            total_sq += (diff**2).sum().item()
            n_pts += diff.numel()
        rmse = (total_sq / n_pts) ** 0.5
        logger.info(f"Test RMSE: {rmse:.2f} px")
        return

    # training loop
    logger.info("##### Training #####")
    logger.info(f"Batch Size: {cfg['dataset']['batch_size']}  LR: {cfg['training']['lr']}")
    best_rmse, best_ep = float('inf'), 0
    for ep in range(1, cfg['training']['epochs'] + 1):
        logger.info(f"Epoch {ep}/{cfg['training']['epochs']}")
        model.train()
        for kps, iso in tqdm(train_loader, desc="Train"):
            kps = kps.to(device).float()
            iso = iso.to(device).float()

            pred = model(kps)
            L_rec = criterion(pred, kps[:, :24])
            L_len = compute_length_sym_loss(pred, length_pairs, eps)
            L_ang = compute_angle_sym_loss(pred, angle_pairs, eps)
            loss = L_rec + λ_sym * (L_len + L_ang)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation & save
        model.eval()
        total_sq, n_pts = 0.0, 0
        for kps, iso in tqdm(val_loader, desc="Val"):
            kps = kps.to(device).float()
            iso = iso.to(device).float().unsqueeze(1)
            with torch.no_grad():
                pred = model(kps)
            diff = (pred - kps[:, :24]).view(-1, 12, 2)
            total_sq += (diff**2).sum().item()
            n_pts += diff.numel()
        val_rmse = (total_sq / n_pts) ** 0.5
        if val_rmse < best_rmse:
            best_rmse, best_ep = val_rmse, ep
            torch.save(model.state_dict(), best_ckpt)
        logger.info(f"Best Result: {best_rmse:.4f}, epoch: {best_ep}")

    # final test
    logger.info("##### Testing #####")
    logger.info(f"Batch Size: {cfg['dataset']['batch_size']}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    total_sq, n_pts = 0.0, 0
    for kps, iso in tqdm(val_loader, desc="Test"):
        kps = kps.to(device).float()
        iso = iso.to(device).float().unsqueeze(1)
        with torch.no_grad():
            pred = model(kps)
        diff = (pred - kps[:, :24]).view(-1, 12, 2)
        total_sq += (diff**2).sum().item()
        n_pts += diff.numel()
    test_rmse = (total_sq / n_pts) ** 0.5
    logger.info(f"Test RMSE: {test_rmse:.2f} px")

if __name__ == '__main__':
    main()
