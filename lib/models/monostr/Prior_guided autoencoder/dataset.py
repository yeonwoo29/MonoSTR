# dataset.py
"""
CarKeypointDataset
- 12개의 2D keypoint와 coarse orientation θ(−π~π → −1~1)를 결합해 반환
- train split: (kp_theta, iso_bbox)
- val/test split : (img, kp_theta, centroid, iso_bbox)
"""

import os, json, random, math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

_KEY_NUM = 12

class CarKeypointDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train") -> None:
        self.include_img = False
        self.to_tensor = transforms.Compose([
            transforms.Resize((1920, 1080)),  # 원하는 사이즈로 조정
            transforms.ToTensor()
        ])
        
        samples = []
        for fld in os.listdir(root_dir):
            ann_path = os.path.join(root_dir, fld, "annotation_theta.json")
            img_dir  = os.path.join(root_dir, fld, "images_jpg")
            if not os.path.isfile(ann_path): continue
            with open(ann_path, "r") as f:
                anns = json.load(f)
            for ann in anns:
                coords = [(pt[0], pt[1]) for pt in ann["keypoints"]]
                theta  = float(ann["theta"])           # ➞ JSON 에서 읽어온 raw θ (radian)
                img_path = os.path.join(img_dir, ann["file_name"])
                if os.path.isfile(img_path):
                    samples.append({
                        "img": img_path,
                        "kp": coords,
                        "theta": theta          # ➞ 샘플에 θ 저장
                    })

        random.shuffle(samples)
        cut = int(0.8 * len(samples))
        self.samples = samples[:cut] if split=="train" else samples[cut:]
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _compute_theta(kp_px: np.ndarray) -> float:
        # front lights (idx 4,5), rear lights (idx 6,7)
        front = kp_px[[4,5]].mean(axis=0)
        rear  = kp_px[[6,7]].mean(axis=0)
        vec   = front - rear
        return math.atan2(vec[1], vec[0])

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        kp  = np.array(rec["kp"], dtype=np.float32)        # (12,2)

        # centroid shift
        centroid    = kp.mean(axis=0, keepdims=True)      # (1,2)
        kp_centered = kp - centroid

        # isotropic scale
        min_xy = kp.min(axis=0, keepdims=True)
        max_xy = kp.max(axis=0, keepdims=True)
        bbox_size = max_xy - min_xy
        scale = float(np.max(bbox_size)) + 40.0            # margin=20*2
        iso_bbox = np.array([scale, scale], dtype=np.float32)

        # normalize keypoints
        kp_norm = kp_centered / (scale + 1e-6)             # roughly (−0.5,0.5)

        # orientation θ normalized (−π~π → −1~1) — JSON의 theta 사용
        theta    = rec["theta"] / math.pi
        kp_theta = np.concatenate([kp_norm.reshape(-1), [theta]], axis=0)

        return (
            torch.from_numpy(kp_theta),
            torch.from_numpy(iso_bbox)
        )
