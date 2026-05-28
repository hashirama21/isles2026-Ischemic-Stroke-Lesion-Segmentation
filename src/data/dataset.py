"""src/data/dataset.py — ISLES'26 dataset with metadata support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from torch.utils.data import Dataset

MAX_DAYS: float = 365.0  # normalization ceiling for days_post_stroke


class ISLES26Dataset(Dataset):
    """PyTorch Dataset for ISLES'26.

    Each sample is a dict with keys:
        image       : MetaTensor  [1, D, H, W]  (affine embedded)
        label       : MetaTensor  [1, D, H, W]  (affine embedded)
        metadata    : FloatTensor [2]  (days_post_stroke_norm, chronicity)
        subject_id  : str
        center      : str
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        transforms=None,
    ) -> None:
        self.samples = samples
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        info = self.samples[idx]

        img_nib = nib.load(info["image_path"])
        lbl_nib = nib.load(info["label_path"])
        affine = torch.from_numpy(img_nib.affine.copy()).float()

        image = MetaTensor(
            torch.from_numpy(np.asarray(img_nib.dataobj, dtype=np.float32)).unsqueeze(0),
            meta={"affine": affine},
        )
        label = MetaTensor(
            torch.from_numpy(np.asarray(lbl_nib.dataobj, dtype=np.int64)).unsqueeze(0),
            meta={"affine": affine},
        )

        days = float(info.get("days_post_stroke", 0.0))
        days_norm = float(np.clip(days, 0.0, MAX_DAYS) / MAX_DAYS)
        chronicity = float(info.get("chronicity", 0.0))
        metadata = torch.tensor([days_norm, chronicity], dtype=torch.float32)

        sample = {
            "image": image,
            "label": label,
            "metadata": metadata,
            "subject_id": info["subject_id"],
            "center": info.get("center", "unknown"),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @classmethod
    def from_split_file(
        cls,
        split_file: Path,
        fold: int,
        split: str,   # "train" | "val" | "test"
        transforms=None,
    ) -> "ISLES26Dataset":
        """Load dataset from a pre-computed JSON split file."""
        with open(split_file) as f:
            splits = json.load(f)

        fold_key = f"fold_{fold}"
        if fold_key not in splits:
            raise KeyError(f"Fold {fold} not found in {split_file}")

        return cls(splits[fold_key][split], transforms=transforms)