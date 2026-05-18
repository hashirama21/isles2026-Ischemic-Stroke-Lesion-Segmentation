"""src/data/datamodule.py — PyTorch Lightning DataModule for ISLES'26."""
from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.dataset import ISLES26Dataset
from src.data.sampler import build_lesion_sampler
from src.data.transforms import get_train_transforms, get_val_transforms


class ISLES26DataModule(pl.LightningDataModule):
    """Handles all data loading for ISLES'26.

    Training uses a weighted sampler that over-represents cases containing
    small lesions (< small_lesion_threshold_ml mL), ensuring these clinically
    critical but rare cases are seen more often during training.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._train_ds: ISLES26Dataset | None = None
        self._val_ds: ISLES26Dataset | None = None
        self._test_ds: ISLES26Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        splits_file = Path(self.cfg.splits_file)
        fold = self.cfg.fold

        if stage in ("fit", None):
            self._train_ds = ISLES26Dataset.from_split_file(
                splits_file, fold, "train",
                transforms=get_train_transforms(self.cfg),
            )
            self._val_ds = ISLES26Dataset.from_split_file(
                splits_file, fold, "val",
                transforms=get_val_transforms(self.cfg),
            )

        if stage in ("test", None):
            self._test_ds = ISLES26Dataset.from_split_file(
                splits_file, fold, "test",
                transforms=get_val_transforms(self.cfg),
            )

    def train_dataloader(self) -> DataLoader:
        num_workers = self.cfg.num_workers
        threshold_ml = self.cfg.get("small_lesion_threshold_ml", 1.0)
        sampler = build_lesion_sampler(self._train_ds.samples, threshold_ml)
        return DataLoader(
            self._train_ds,
            batch_size=self.cfg.train_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=num_workers > 0 and self.cfg.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        num_workers = self.cfg.num_workers
        return DataLoader(
            self._val_ds,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=num_workers > 0 and self.cfg.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def __repr__(self) -> str:
        n_train = len(self._train_ds) if self._train_ds else 0
        n_val = len(self._val_ds) if self._val_ds else 0
        return f"ISLES26DataModule(fold={self.cfg.fold}, train={n_train}, val={n_val})"