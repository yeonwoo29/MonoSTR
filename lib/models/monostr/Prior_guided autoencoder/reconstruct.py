#!/usr/bin/env python3
# reconstruct.py
# - AE로 재구성한 키포인트와 입력 키포인트를 비교
# - bbox 스케일로 정규화한 이동량이 임계값 이상인 점만 AE 결과로 대체
# - 나머지는 입력값 유지

import os
import json
import torch
import numpy as np
from typing import List, Tuple, Optional
from model import CarKeypointAutoencoder

# =========================
# 설정 (필요시 여기만 바꾸세요)
# =========================
THRESHOLD_RATIO = 0.15  # '짧은 변의 2%' 이상 움직인 키포인트만 AE 결과로 대체
USE_BBOX_SHORT = True   # True: min(w,h), False: bbox 대각선 사용

# =========================
# 유틸
# =========================
def _bbox_scale(bbox: Optional[List[float]], fallback_scale: float) -> float:
    """
    bbox: [x1, y1, x2, y2] (픽셀)
    반환: 스케일 s (정규화에 사용). 없으면 fallback_scale 사용.
    """
    if bbox is None:
        return max(fallback_scale, 1e-6)
    x1, y1, x2, y2 = map(float, bbox)
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    if USE_BBOX_SHORT:
        return max(min(w, h), 1e-6)
    else:
        return float(np.sqrt(w * w + h * h))

def _pairwise_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    a, b: (12,2) ndarray
    return: (12,) L2 거리
    """
    d = a - b
    return np.sqrt(np.maximum((d ** 2).sum(axis=-1), eps))

def _selective_mix(input_kp: np.ndarray,
                   recon_kp: np.ndarray,
                   scale_s: float,
                   thr_ratio: float) -> np.ndarray:
    """
    input_kp, recon_kp: (12,2) float (px)
    scale_s: 스케일(예: bbox 짧은 변)
    thr_ratio: 비율 임계값 (예: 0.02)
    규칙: ||recon-input|| / s >= thr_ratio 인 점만 AE 결과로 대체
    """
    residual_px = _pairwise_l2(recon_kp, input_kp)      # (12,)
    residual_ratio = residual_px / max(scale_s, 1e-6)   # (12,)
    mask = (residual_ratio >= thr_ratio)                # (12,) bool
    mixed = input_kp.copy()
    mixed[mask] = recon_kp[mask]
    return mixed

# =========================
# AE 로딩/추론
# =========================
def load_model(checkpoint_path: str, device: torch.device):
    model = CarKeypointAutoencoder()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def _ae_forward(kp: np.ndarray, theta: float, model: CarKeypointAutoencoder, device: torch.device) -> np.ndarray:
    """
    기존 코드의 AE 입력 정규화 방식을 유지:
      - 중심 이동 → 가장 긴 변 + margin(40)으로 isotropic 스케일 → [-?~?] 정규화
      - theta는 pi로 나눠서 입력
    반환: AE가 예측한 (12,2) float (px 절대좌표로 역정규화까지 수행)
    """
    # 1) 중심 이동
    centroid = kp.mean(axis=0)                          # (2,)
    kp_centered = kp - centroid

    # 2) isotropic scale (max dimension + margin)
    min_xy = kp.min(axis=0)
    max_xy = kp.max(axis=0)
    scale_iso = float(np.max(max_xy - min_xy)) + 40.0

    # 3) 정규화 & 입력 벡터
    kp_norm = kp_centered / (scale_iso + 1e-6)
    theta_norm = theta / np.pi
    inp = np.concatenate([kp_norm.reshape(-1), [theta_norm]], axis=0)
    inp_tensor = torch.from_numpy(inp).float().unsqueeze(0).to(device)

    # 4) 예측
    with torch.no_grad():
        pred = model(inp_tensor)                       # (1,24)
    pred = pred.cpu().numpy().reshape(12, 2)           # (12,2)

    # 5) 역정규화 (px)
    pred_abs = (pred * scale_iso) + centroid
    return pred_abs, scale_iso, centroid

def reconstruct_keypoints(item: dict, model: CarKeypointAutoencoder, device: torch.device):
    """
    입력: item에 최소한 다음 키가 있다고 가정
      - "keypoints": [[x,y]*12]
      - "theta": float
      - "crop_bbox": [x1,y1,x2,y2]  (선택: 없을 수도 있음)
    출력:
      - (12,2) int 리스트 (선택적 대체 적용 후)
    """
    # 원본 keypoints, theta
    kp_in = np.array(item["keypoints"], dtype=np.float32)  # (12,2)
    theta = float(item["theta"])                           # scalar

    # AE 재구성 (px)
    recon_px, scale_iso, centroid = _ae_forward(kp_in, theta, model, device)

    # 스케일 s 결정: bbox 있으면 bbox, 없으면 AE에서 쓰던 isotropic scale 사용
    crop_bbox = item.get("crop_bbox", None)
    s = _bbox_scale(crop_bbox, fallback_scale=scale_iso)

    # 선택적 대체: 스케일 정규화된 이동량이 thr 이상인 점만 대체
    mixed = _selective_mix(kp_in, recon_px, scale_s=s, thr_ratio=THRESHOLD_RATIO)

    # 정수 좌표로 저장
    mixed_int = np.round(mixed).astype(int)
    return mixed_int.tolist()

# =========================
# 메인
# =========================
def main():
    # — 경로 설정 —
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = r"C:\Users\vimlab\Desktop\Prior_guided autoencoder\checkpoints\autoencoder\sym1, length_pairs 8_20250730_151239\best.pth"
    json_path       = r"C:\Users\vimlab\Desktop\Prior_guided autoencoder\val data\센터넷으로 뽑은 키포인트 원본.json"
    out_path        = os.path.splitext(json_path)[0] + "_reconstructed_sym1_5.json"

    model = load_model(checkpoint_path, device)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_data = []
    for item in data:
        out_data.append({
            "image_id":      item["image_id"],
            "object_index":  item["object_index"],
            "crop_bbox":     item.get("crop_bbox", None),
            "keypoints":     reconstruct_keypoints(item, model, device),
            "theta":         item["theta"]
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Reconstructed (selective) keypoints saved to:\n   {out_path}")
    print(f"   rule: replace if ||Δ|| / s ≥ {THRESHOLD_RATIO} (s = {'min(w,h)' if USE_BBOX_SHORT else 'diag'})")

if __name__ == "__main__":
    main()
