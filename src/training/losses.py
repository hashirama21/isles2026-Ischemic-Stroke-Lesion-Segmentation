"""src/training/losses.py — Combined segmentation loss with optional deep supervision."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss


class StrokeLoss(nn.Module):
    """Weighted sum of Dice + CrossEntropy + optional Focal loss.

    Supports deep supervision: when logits has shape [B, N, C, D, H, W],
    the loss is computed at each resolution and combined with
    weights [1.0, 0.5, 0.25, 0.125, ...].

    Weights must sum to 1.0.
    """

    _DS_WEIGHTS = (1.0, 0.5, 0.25, 0.125)

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        focal_weight: float = 0.0,
        softmax: bool = True,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        total = dice_weight + ce_weight + focal_weight
        if abs(total - 1.0) >= 1e-3:
            raise ValueError(
                f"Loss weights must sum to 1.0; got "
                f"dice={dice_weight} + ce={ce_weight} + focal={focal_weight} = {total:.6f}"
            )
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.deep_supervision = deep_supervision

        self.dice_loss = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=softmax,
        )
        self.focal: FocalLoss | None = (
            FocalLoss(include_background=False, to_onehot_y=True, use_softmax=softmax)
            if focal_weight > 0.0
            else None
        )

    def forward(
        self,
        logits: torch.Tensor,   # [B, C, D, H, W]  or  [B, N, C, D, H, W] (deep sup.)
        targets: torch.Tensor,  # [B, 1, D, H, W]  long
    ) -> torch.Tensor:
        if self.deep_supervision and logits.dim() == 6:
            return self._deep_supervision_loss(logits, targets)
        return self._single_level_loss(logits, targets)


    def _single_level_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.dice_weight * self.dice_loss(logits, targets)
        if self.ce_weight > 0.0:
            loss = loss + self.ce_weight * F.cross_entropy(
                logits, targets.squeeze(1).long()
            )
        if self.focal is not None:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        return loss

    def _deep_supervision_loss(
        self,
        logits: torch.Tensor,   # [B, N, C, D, H, W]
        targets: torch.Tensor,  # [B, 1, D, H, W]
    ) -> torch.Tensor:
        n_levels = logits.shape[1]
        total_weight = sum(self._DS_WEIGHTS[:n_levels])
        loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        for i in range(n_levels):
            level_logits = logits[:, i]  # [B, C, d, h, w]
            if level_logits.shape[2:] != targets.shape[2:]:
                t = F.interpolate(
                    targets.float(),
                    size=level_logits.shape[2:],
                    mode="nearest",
                ).long()
            else:
                t = targets
            w = self._DS_WEIGHTS[min(i, len(self._DS_WEIGHTS) - 1)]
            loss = loss + w * self._single_level_loss(level_logits, t)

        return loss / total_weight