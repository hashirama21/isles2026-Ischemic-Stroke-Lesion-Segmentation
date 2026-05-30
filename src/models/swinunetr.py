"""src/models/swinunetr.py — MONAI SwinUNETR with FiLM metadata conditioning."""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from src.models.film import FiLMLayer


class SwinUNETRWrapper(nn.Module):
    """SwinUNETR with FiLM conditioning injected at the bottleneck.

    A forward hook on `encoder10` intercepts and replaces the bottleneck output
    with FiLM-conditioned features **before** the decoder path runs.

    The bottleneck feature map has shape [B, 8 * feature_size, D/32, H/32, W/32].
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.1,
        use_checkpoint: bool = True,
        meta_dim: int = 2,
        film_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

        bottleneck_channels = 8 * feature_size  # 384 for feature_size=48
        self.film = FiLMLayer(
            meta_dim=meta_dim,
            channels=bottleneck_channels,
            hidden_dim=film_hidden_dim,
        )

        self._current_metadata: torch.Tensor | None = None
        self._hook_handle = self.backbone.encoder10.register_forward_hook(
            self._film_hook
        )

    def _film_hook(self, module, input, output) -> torch.Tensor:
        """Return FiLM-conditioned bottleneck features so the decoder uses them."""
        if self._current_metadata is None:
            return output
        return self.film(output, self._current_metadata)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:    [B, 1, D, H, W]
            metadata: [B, meta_dim]

        Returns:
            logits [B, out_channels, D, H, W]
        """
        self._current_metadata = metadata
        try:
            logits = self.backbone(image)
        finally:
            self._current_metadata = None
        return logits

    def __del__(self) -> None:
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
