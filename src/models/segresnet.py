"""src/models/segresnet.py — MONAI SegResNet wrapped with FiLM metadata conditioning."""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

from src.models.film import FiLMLayer


class SegResNetFiLM(nn.Module):
    """SegResNet encoder-decoder with FiLM conditioning at the bottleneck.

    A forward hook on the last encoder block intercepts and replaces its output
    with FiLM-conditioned features **before** the decoder path runs, so the
    metadata signal propagates through the full decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        init_filters: int = 32,
        blocks_down: tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: tuple[int, ...] = (1, 1, 1),
        dropout_prob: float = 0.2,
        meta_dim: int = 2,
        film_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.backbone = SegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=list(blocks_down),
            blocks_up=list(blocks_up),
            dropout_prob=dropout_prob,
        )

        bottleneck_channels = init_filters * (2 ** (len(blocks_down) - 1))
        self.film = FiLMLayer(
            meta_dim=meta_dim,
            channels=bottleneck_channels,
            hidden_dim=film_hidden_dim,
        )

        self._current_metadata: torch.Tensor | None = None
        self._hook_handle = list(self.backbone.down_layers)[-1].register_forward_hook(
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
            logits: [B, out_channels, D, H, W]
        """
        self._current_metadata = metadata
        logits = self.backbone(image)
        self._current_metadata = None
        return logits

    def __del__(self) -> None:
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()
