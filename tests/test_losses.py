"""tests/test_losses.py — Smoke tests for loss functions."""
import pytest
import torch

from src.training.losses import StrokeLoss


class TestStrokeLoss:
    def _batch(self, B=2, C=2, D=16, H=16, W=16):
        logits = torch.randn(B, C, D, H, W)
        targets = torch.randint(0, C, (B, 1, D, H, W)).long()
        return logits, targets

    def test_forward_returns_scalar(self):
        loss_fn = StrokeLoss()
        logits, targets = self._batch()
        assert loss_fn(logits, targets).shape == ()

    def test_forward_is_finite(self):
        loss_fn = StrokeLoss()
        logits, targets = self._batch()
        assert torch.isfinite(loss_fn(logits, targets))

    def test_weights_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            StrokeLoss(dice_weight=0.4, ce_weight=0.4, focal_weight=0.0)

    def test_with_focal(self):
        loss_fn = StrokeLoss(dice_weight=0.4, ce_weight=0.4, focal_weight=0.2)
        logits, targets = self._batch()
        assert torch.isfinite(loss_fn(logits, targets))

    def test_dice_only(self):
        loss_fn = StrokeLoss(dice_weight=1.0, ce_weight=0.0, focal_weight=0.0)
        logits, targets = self._batch()
        assert torch.isfinite(loss_fn(logits, targets))
