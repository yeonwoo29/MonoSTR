import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer
from .ffc_block import *

class DepthPredictor(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg["num_depth_bins"])
        depth_min = float(model_cfg["depth_min"])
        depth_max = float(model_cfg["depth_max"])
        self.depth_max = depth_max

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = model_cfg["hidden_dim"]
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        depth_encoder_layer = TransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=256, dropout=0.1)
        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)
        
        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, d_model)

        # For AAAI 2026
        inplanes = 256
        self.inplanes = inplanes
        ratio = 0.5
        self.groups = 1
        self.base_width = 64
        self.use_se = False
        self.lfu = True
        self.dilation = 1
        n_blocks = 8
        norm_layer = None 
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer     
        self.ffc_depth = self._make_layer(
            BasicBlock, inplanes * 1, n_blocks, stride=1, ratio_gin=0, ratio_gout=ratio)         
    # For AAAI 2026    
    def _make_layer(self, block, planes, blocks, stride=1, ratio_gin=0.5, ratio_gout=0.5):
            norm_layer = self._norm_layer
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion or ratio_gin == 0:
                downsample = FFC_BN_ACT(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                                        ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=self.lfu)

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                                self.dilation, ratio_gin, ratio_gout, lfu=self.lfu, use_se=self.use_se))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                                    ratio_gin=ratio_gout, ratio_gout=ratio_gout, lfu=self.lfu, use_se=self.use_se))

            return nn.Sequential(*layers)

    def forward(self, feature, mask, pos):

        # foreground depth map
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:], mode='bilinear'))
        src_8 = self.downsample(feature[0])

        src = (src_8 + src_16 + src_32) / 3 #[8, 256, 24, 80]
        scr = self.ffc_depth(src)
        src = self.depth_head(src)
        depth_logits = self.depth_classifier(src)
        
        # Calculate the median value along the depth_num_bins dimension
        #median_values = torch.median(depth_logits, dim=1, keepdim=True).values
        # Apply the threshold: set values below the median to zero
        #thresholded_logits = torch.where(depth_logits >= median_values, depth_logits,  torch.tensor(-float('inf')).to(depth_logits.device))
             
        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)
        
        # depth embeddings with depth positional encodings
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)
        
        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed #+ depth_pos_embed_ip
        
        return depth_logits, depth_embed, weighted_depth


    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta







class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, ratio_gin=0.5, ratio_gout=0.5, lfu=True, use_se=False, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = FFC_BN_ACT(inplanes, width, kernel_size=3, padding=1, stride=stride,
                                ratio_gin=ratio_gin, ratio_gout=ratio_gout, norm_layer=norm_layer, activation_layer=nn.ReLU, enable_lfu=lfu)
        self.conv2 = FFC_BN_ACT(width, planes * self.expansion, kernel_size=3, padding=1,
                                ratio_gin=ratio_gout, ratio_gout=ratio_gout, norm_layer=norm_layer, enable_lfu=lfu)
        self.se_block = FFCSE_block(
            planes * self.expansion, ratio_gout) if use_se else nn.Identity()
        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.downsample is None else self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x_l, x_g = self.se_block(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g