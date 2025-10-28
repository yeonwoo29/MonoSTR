# model.py
"""
Prior-guided Autoencoder
- 입력: 25-D (24 keypoint + θ)
- 출력: 24-D (keypoint만 재구성)
- 구조: FC → BN → LeakyReLU ×2  encoder/decoder
"""

import torch.nn as nn

class CarKeypointAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 25,
        output_dim: int = 24,
        h1: int = 16,
        h2: int = 10,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(),
            nn.Linear(h1, output_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
