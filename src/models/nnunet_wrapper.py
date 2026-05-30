"""src/models/nnunet_wrapper.py — DynUNet (nnU-Net style) with FiLM + deep supervision."""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

from src.models.film import FiLMLayer

# Strides and kernel sizes for a 128³ input → 5 downsampling steps → 4³ bottleneck
_STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
_KERNEL_SIZE = [[3, 3, 3]] * 6
_UPSAMPLE_KERNEL_SIZE = [[2, 2, 2]] * 5


class NNUNetWrapper(nn.Module):
    """DynUNet (nnU-Net-style 3-D residual U-Net) with FiLM bottleneck conditioning.

    Supports deep supervision during training: forward() returns
    [B, N, C, D, H, W] when self.training is True and deep_supervision=True,
    and [B, C, D, H, W] otherwise.  StrokeLoss handles both shapes.

    Architecture (128³ input, default channels):
        Encoder  : 128→64→32→16→8→4  (filters [32, 64, 128, 256, 320, 320])
        Bottleneck: [B, 320, 4, 4, 4]  ← FiLM conditioned here
        Decoder  : symmetric with skip connections
        DS heads : 3 additional outputs at ×2, ×4, ×8 lower resolution
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        encoder_channels: tuple[int, ...] = (32, 64, 128, 256, 320, 320),
        deep_supervision: bool = True,
        meta_dim: int = 2,
        film_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self._deep_supervision = deep_supervision
        deep_supr_num = 3 if deep_supervision else 0

        self.backbone = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_KERNEL_SIZE,
            strides=_STRIDES,
            upsample_kernel_size=_UPSAMPLE_KERNEL_SIZE,
            filters=list(encoder_channels),
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
            res_block=True,
        )

        bottleneck_channels = encoder_channels[-1]
        self.film = FiLMLayer(
            meta_dim=meta_dim,
            channels=bottleneck_channels,
            hidden_dim=film_hidden_dim,
        )

        self._current_metadata: torch.Tensor | None = None
        if not hasattr(self.backbone, "bottleneck"):
            raise AttributeError(
                "DynUNet has no 'bottleneck' attribute — "
                "upgrade MONAI to >= 1.1 or check the installed version."
            )
        self._hook_handle = self.backbone.bottleneck.register_forward_hook(
            self._film_hook
        )

    def _film_hook(self, module, input, output) -> torch.Tensor:
        if self._current_metadata is None:
            return output
        return self.film(output, self._current_metadata)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Training + deep_supervision : [B, N, out_channels, D, H, W]
            Otherwise                   : [B, out_channels, D, H, W]
        """
        self._current_metadata = metadata
        try:
            output = self.backbone(image)
        finally:
            self._current_metadata = None

        # DynUNet stacks deep supervision outputs: [B, N, C, D, H, W]
        # Return only full-resolution during eval / inference
        if not self.training and isinstance(output, torch.Tensor) and output.dim() == 6:
            return output[:, 0]
        return output

    def __del__(self) -> None:
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
