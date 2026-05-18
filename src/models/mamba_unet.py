"""src/models/mamba_unet.py — Hybrid CNN-Mamba U-Net.

Falls back to a pure CNN bottleneck if mamba-ssm is not installed.
In competition mode, use the full Mamba version only after verifying
CUDA kernel compatibility in your Docker target environment.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# Attempt to import Mamba — graceful fallback to plain MLP
try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class _MambaBlock(nn.Module):
    """Single Mamba SSM block with layer norm and residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        if MAMBA_AVAILABLE:
            self.norm = nn.LayerNorm(d_model)
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv)
        else:
            # Fallback: simple MLP bottleneck
            self.norm = nn.LayerNorm(d_model)
            self.mamba = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class _ConvBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HybridCNNMambaUNet(nn.Module):
    """Hybrid CNN encoder + Mamba bottleneck + CNN decoder.

    Architecture:
        CNN Encoder  →  [E1, E2, E3, E4]  (local features)
        Mamba blocks →  sequence modelling over flattened E4  (global context)
        CNN Decoder  ←  skip connections from encoder
    """

    BASE_FILTERS = 32

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_filters: int = 32,
        mamba_depth: int = 2,
        meta_dim: int = 2,
    ) -> None:
        super().__init__()
        f = base_filters
        self.out_channels = out_channels

        # Encoder
        self.enc1 = _ConvBlock3D(in_channels, f)
        self.enc2 = _ConvBlock3D(f, f * 2, stride=2)
        self.enc3 = _ConvBlock3D(f * 2, f * 4, stride=2)
        self.enc4 = _ConvBlock3D(f * 4, f * 8, stride=2)

        # Bottleneck Mamba
        bottleneck_dim = f * 8
        self.mamba_blocks = nn.ModuleList(
            [_MambaBlock(bottleneck_dim) for _ in range(mamba_depth)]
        )

        # Metadata injection (concatenated then projected)
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, bottleneck_dim),
            nn.SiLU(),
        )

        # Decoder
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, 2, stride=2)
        self.dec3 = _ConvBlock3D(f * 8, f * 4)
        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2)
        self.dec2 = _ConvBlock3D(f * 4, f * 2)
        self.up1 = nn.ConvTranspose3d(f * 2, f, 2, stride=2)
        self.dec1 = _ConvBlock3D(f * 2, f)

        self.head = nn.Conv3d(f, out_channels, 1)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(image)          # [B, f,   D,   H,   W  ]
        e2 = self.enc2(e1)             # [B, 2f,  D/2, H/2, W/2]
        e3 = self.enc3(e2)             # [B, 4f,  D/4, H/4, W/4]
        e4 = self.enc4(e3)             # [B, 8f,  D/8, H/8, W/8]

        # Flatten spatial dims → sequence for Mamba
        B, C, d, h, w = e4.shape
        x = e4.flatten(2).permute(0, 2, 1)   # [B, N, C]

        # Inject metadata via additive bias
        meta_feat = self.meta_proj(metadata).unsqueeze(1)   # [B, 1, C]
        x = x + meta_feat

        # Mamba blocks
        for block in self.mamba_blocks:
            x = block(x)

        # Reshape back to spatial
        x = x.permute(0, 2, 1).view(B, C, d, h, w)

        # Decode with skip connections
        x = self.up3(x)
        x = self.dec3(torch.cat([x, e3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, e1], dim=1))

        return self.head(x)
