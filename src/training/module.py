"""src/training/module.py — PyTorch Lightning LightningModule for ISLES'26."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import MAX_DAYS
from src.training.losses import StrokeLoss
from src.evaluation.metrics import compute_lesion_metrics


class ISLES26Module(pl.LightningModule):
    """Encapsulates model, loss, optimiser, metrics, and logging."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        self.model = instantiate(cfg.model)

        loss_cfg = cfg.training.loss
        self.criterion = StrokeLoss(
            dice_weight=loss_cfg.dice_weight,
            ce_weight=loss_cfg.ce_weight,
            focal_weight=loss_cfg.focal_weight,
            deep_supervision=loss_cfg.get("deep_supervision", False),
        )

        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=True,
        )

        self._num_classes: int = cfg.model.out_channels
        self._sw_roi = list(cfg.training.sw_roi_size)
        self._sw_batch = cfg.training.sw_batch_size
        self._sw_overlap = cfg.training.sw_overlap
        self._sw_mode: str = cfg.training.sw_mode

        self._test_outputs: list[dict[str, Any]] = []

    # Helpers

    def _full_res(self, output: torch.Tensor) -> torch.Tensor:
        """Return full-resolution logits from a potentially deep-supervised output."""
        if isinstance(output, torch.Tensor) and output.dim() == 6:
            return output[:, 0]
        return output

    def _sliding_predictor(self, meta: torch.Tensor):
        """Build a sliding-window predictor that returns full-res logits.

        Sliding-window inference processes one subject at a time, so meta must
        have batch dimension 1; the single subject's metadata is expanded to
        cover all sw_batch_size patch crops passed to the predictor.
        """
        if meta.shape[0] != 1:
            raise ValueError(
                "Sliding-window inference requires batch_size=1 per subject; "
                f"got metadata batch of {meta.shape[0]}. "
                "Set val_batch_size=1 in the data config."
            )
        def _predict(x: torch.Tensor) -> torch.Tensor:
            # x: [sw_batch_size, C, d, h, w] — multiple patches from the same subject
            return self._full_res(self.model(x, meta.expand(x.shape[0], -1)))
        return _predict

    @staticmethod
    def _voxel_spacing_from_batch(batch: dict) -> tuple[float, float, float]:
        """Extract voxel spacing (mm) from image MetaTensor affine, or default 1 mm."""
        image = batch["image"]
        if hasattr(image, "meta") and "affine" in image.meta:
            try:
                affine = image.meta["affine"][0].cpu().numpy()
                spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
                return tuple(float(s) for s in spacing)
            except Exception as exc:
                warnings.warn(
                    f"Failed to extract voxel spacing from affine ({exc}); "
                    "falling back to (1.0, 1.0, 1.0) — surface distances and volumes "
                    "will be in voxel units.",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Image has no affine metadata; falling back to voxel spacing (1.0, 1.0, 1.0). "
                "Surface distances and volumes will be in voxel units.",
                stacklevel=2,
            )
        return (1.0, 1.0, 1.0)

    # Forward

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.model(image, metadata)

    # Training

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"], batch["metadata"])
        loss = self.criterion(logits, batch["label"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # Validation

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        image, label, meta = batch["image"], batch["label"], batch["metadata"]

        logits = sliding_window_inference(
            inputs=image,
            roi_size=self._sw_roi,
            sw_batch_size=self._sw_batch,
            predictor=self._sliding_predictor(meta),
            overlap=self._sw_overlap,
            mode=self._sw_mode,
        )

        loss = self.criterion(logits, label)

        preds = logits.argmax(dim=1, keepdim=True)
        self.dice_metric(
            y_pred=one_hot(preds, num_classes=self._num_classes),
            y=one_hot(label, num_classes=self._num_classes),
        )

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        dice, not_nans = self.dice_metric.aggregate()
        self.dice_metric.reset()
        self.log("val/dice", dice.item(), prog_bar=True, sync_dist=True)

    # Test

    def test_step(self, batch: dict, batch_idx: int) -> None:
        image, label, meta = batch["image"], batch["label"], batch["metadata"]
        spacing_mm = self._voxel_spacing_from_batch(batch)

        logits = sliding_window_inference(
            inputs=image,
            roi_size=self._sw_roi,
            sw_batch_size=self._sw_batch,
            predictor=self._sliding_predictor(meta),
            overlap=self._sw_overlap,
            mode=self._sw_mode,
        )
        preds = logits.argmax(dim=1, keepdim=True)

        vox_vol_ml = float(np.prod(spacing_mm)) * 0.001
        metrics = compute_lesion_metrics(
            pred=preds.squeeze(0).squeeze(0).cpu().numpy(),
            gt=label.squeeze(0).squeeze(0).cpu().numpy(),
            voxel_volume_ml=vox_vol_ml,
            voxel_spacing_mm=spacing_mm,
        )

        meta_vals = meta[0].cpu().numpy()
        days = float(meta_vals[0]) * MAX_DAYS
        metrics["days_post_stroke"] = days
        metrics["chronicity"] = float(meta_vals[1])
        metrics["lesion_volume_ml"] = (
            float(label.squeeze(0).squeeze(0).cpu().numpy().sum()) * vox_vol_ml
        )
        metrics["phase"] = (
            "acute" if days <= 7 else "subacute" if days <= 180 else "chronic"
        )
        metrics["subject_id"] = batch["subject_id"][0]
        metrics["center"] = batch["center"][0]
        self._test_outputs.append(metrics)

    def on_test_epoch_end(self) -> None:
        df = pd.DataFrame(self._test_outputs)
        self._test_outputs.clear()

        mean = df.select_dtypes("number").mean()
        self.log_dict({f"test/{k}": float(v) for k, v in mean.items()})

        out_path = Path(self.cfg.paths.predictions) / "test_metrics.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    # Optimiser / scheduler

    def configure_optimizers(self):
        opt = instantiate(self.cfg.training.optimizer, params=self.parameters())
        sched = instantiate(self.cfg.training.scheduler, optimizer=opt)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val/dice"},
        }
