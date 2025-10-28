import math
import torch
import torch.nn as nn

class KPToSpatialPosEmbed(nn.Module):
    """
    kp_tensor: (B, Q, 24)  # 24 = 12 keypoints * (x,y) in raw pixels
    returns : (B, 256, H, W) with H=24, W=80 by default
    """
    def __init__(
        self,
        img_w: int,
        img_h: int,
        H: int = 24,
        W: int = 80,
        sigma: float = 1.5,         # 가우시안 표준편차(그리드 단위)
        reduce: str = "mean",       # 'mean' | 'max'
        add_sinusoidal_pe: bool = True
    ):
        super().__init__()
        assert reduce in ("mean", "max")
        self.img_w, self.img_h = img_w, img_h
        self.H, self.W = H, W
        self.sigma = sigma
        self.reduce = reduce
        self.add_sinusoidal_pe = add_sinusoidal_pe

        # 12채널 -> 256채널
        self.proj = nn.Sequential(
            nn.Conv2d(12, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 2D 사인/코사인 positional encoding (선택)
        if add_sinusoidal_pe:
            pe = self._build_2d_sincos_pe(H, W, dim=256)
            self.register_buffer("pe_2d", pe, persistent=False)

        # 그리드 좌표 (H,W,2)
        ys = torch.linspace(0, H - 1, H)
        xs = torch.linspace(0, W - 1, W)
        grid_y, grid_x = torch.meshgrid(ys, xs)
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2)
        self.register_buffer("grid_hw", grid, persistent=False)

    @staticmethod
    def _build_2d_sincos_pe(H, W, dim=256):
        """ (1, dim, H, W) sinusoidal 2D PE """
        assert dim % 4 == 0
        dim_h = dim_w = dim // 2
        dim_h_half = dim_h // 2
        dim_w_half = dim_w // 2

        y = torch.arange(H).unsqueeze(1)  # (H,1)
        x = torch.arange(W).unsqueeze(1)  # (W,1)

        div_h = torch.exp(torch.arange(0, dim_h_half, 1) * (-math.log(10000.0) / dim_h_half))
        div_w = torch.exp(torch.arange(0, dim_w_half, 1) * (-math.log(10000.0) / dim_w_half))

        pe_y = torch.cat([torch.sin(y * div_h), torch.cos(y * div_h)], dim=1)  # (H,dim_h)
        pe_x = torch.cat([torch.sin(x * div_w), torch.cos(x * div_w)], dim=1)  # (W,dim_w)

        pe_y = pe_y[:, None, :]         # (H,1,dim_h)
        pe_x = pe_x[None, :, :]         # (1,W,dim_w)
        pe = torch.cat([
            pe_y.expand(H, W, dim_h),   # (H,W,dim_h)
            pe_x.expand(H, W, dim_w)    # (H,W,dim_w)
        ], dim=-1)                      # (H,W,dim)
        pe = pe.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,dim,H,W)
        return pe

    def forward(self, kp_tensor: torch.Tensor) -> torch.Tensor:
        """
        kp_tensor: (B, Q, 24)
        """
        B, Q, S = kp_tensor.shape
        assert S == 24, "Expected 24 = 12 * (x,y)."
        device = kp_tensor.device

        # (1) raw -> grid 좌표로 정규화
        #    x: [0,img_w] -> [0,W-1],  y: [0,img_h] -> [0,H-1]
        idx = torch.arange(S, device=device)
        is_x = (idx % 2 == 0).float().view(1, 1, S)   # (1,1,24)
        is_y = 1.0 - is_x
        scale = is_x * (self.W - 1) / max(self.img_w, 1e-6) + is_y * (self.H - 1) / max(self.img_h, 1e-6)
        kp_grid = kp_tensor * scale                    # (B,Q,24)
        kp_grid = kp_grid.clamp_min(0.0)               # 음수 방지

        # (2) 12개 키포인트 분리
        kp_grid = kp_grid.view(B, Q, 12, 2)  # (B,Q,12,2) -> (x,y) in grid

        # (3) 히트맵 생성: 각 keypoint를 (H,W)에 가우시안으로 splat
        # grid_hw: (H,W,2) -> (1,1,1,H,W,2)
        grid = self.grid_hw.to(device).view(1, 1, 1, self.H, self.W, 2)
        # kp: (B,Q,12,1,1,2)
        kp = kp_grid.view(B, Q, 12, 1, 1, 2)

        # L2 거리 제곱
        diff = grid - kp  # (B,Q,12,H,W,2)
        dist2 = (diff ** 2).sum(dim=-1)  # (B,Q,12,H,W)

        heat = torch.exp(-0.5 * dist2 / (self.sigma ** 2) + 1e-8)  # (B,Q,12,H,W)

        # (4) Q 축 집계
        if self.reduce == "mean":
            heat_agg = heat.mean(dim=1)           # (B,12,H,W)
        else:
            heat_agg = heat.max(dim=1).values     # (B,12,H,W)

        # (5) 12채널 -> 256채널
        feat = self.proj(heat_agg)                # (B,256,H,W)

        # (6) 원하면 2D 사인/코사인 PE 더하기
        if self.add_sinusoidal_pe:
            feat = feat + self.pe_2d.to(device)   # (B,256,H,W)

        return feat  # (B,256,24,80)





def sine_position_encode_xy(xy, num_pos_feats=128, temperature=10000.0, W=1280, H=384):
    """
    xy: [..., 2]  (x, y) raw pixel coords (float)
    Returns: [..., 2*num_pos_feats]  (Y-part first, then X-part)
    """
    assert H is not None and W is not None, "H,W (image size) must be provided"
    # normalize to [0, 2π]
    # (+eps 방어는 호출부에서 H,W>=2 가정. 필요시 max(H-1,1)로 방어)
    x = xy[..., 0] / max(W - 1, 1) * (2.0 * math.pi)
    y = xy[..., 1] / max(H - 1, 1) * (2.0 * math.pi)

    # [num_pos_feats]
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=xy.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)

    # broadcasting to [..., num_pos_feats]
    pos_x = x[..., None] / dim_t
    pos_y = y[..., None] / dim_t

    # interleave sin/cos on even/odd dims and then flatten back to num_pos_feats
    # (짝수 idx → sin, 홀수 idx → cos)
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

    # concat as [ ... , 2*num_pos_feats ] with Y first (DETR convention)
    pos = torch.cat([pos_y, pos_x], dim=-1)
    return pos


def build_keypoint_position_tensor(keypoints, W=1280, H=384, num_pos_feats=128, temperature=10000.0,
                                   reduce='mean', invalid_xy_mask=None):
    """
    keypoints: [B, 550, 24] (12 keypoints * 2), raw pixel coords (int/float, unnormalized)
    H, W: image size used to normalize
    reduce: 'mean' | 'sum' | 'weighted' (if invalid_xy_mask or weights provided)
    invalid_xy_mask: [B, 550, 12] boolean, True=invalid(무시), False=valid (optional)

    Returns:
      keypoint_position_tensor: [B, 550, 256]
    """
    B, Q, D = keypoints.shape
    assert D == 24, "D must be 24 (12 keypoints * 2)"
    K = 12

    # [B, Q, 12, 2]
    xy = keypoints.view(B, Q, K, 2).to(torch.float32)

    # (선택) invalid 키포인트 마스크가 없으면, (0,0) 같은 dummy를 invalid로 간주할 수도 있음
    if invalid_xy_mask is None:
        # (0,0)을 invalid로 취급하고 싶다면 다음 주석을 해제하세요.
        invalid_xy_mask = (xy[..., 0] == 0) & (xy[..., 1] == 0)  # [B, Q, K], True means invalid
    else:
        invalid_xy_mask = invalid_xy_mask.to(torch.bool)

    # 각 키포인트별 2D 사인-코사인 위치 임베딩: [B, Q, K, 256]
    pos_per_kp = sine_position_encode_xy(
        xy, num_pos_feats=num_pos_feats, temperature=temperature, W=W, H=H
    )

    # invalid 키포인트 무시하고 reduce
    if reduce == 'mean':
        # mask: True=invalid → 가중치 0, False=valid → 가중치 1
        weights = (~invalid_xy_mask).to(pos_per_kp.dtype)[..., None]  # [B, Q, K, 1]
        pos_sum = (pos_per_kp * weights).sum(dim=2)                   # [B, Q, 256]
        denom = weights.sum(dim=2).clamp(min=1e-6)                    # [B, Q, 1]
        pos_out = pos_sum / denom
    elif reduce == 'sum':
        weights = (~invalid_xy_mask).to(pos_per_kp.dtype)[..., None]
        pos_out = (pos_per_kp * weights).sum(dim=2)                   # [B, Q, 256]
    elif reduce == 'weighted':
        # 예: keypoint confidence가 있다면 weights로 사용
        # weights: [B, Q, K, 1] 형태로 전달했다고 가정 (여기선 placeholder)
        raise NotImplementedError("Provide external keypoint weights for 'weighted' reduce.")
    else:
        raise ValueError("reduce must be one of {'mean','sum','weighted'}")

    # pos_out: [B, 550, 256]
    return pos_out