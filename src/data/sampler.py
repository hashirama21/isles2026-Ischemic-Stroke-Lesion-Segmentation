"""src/data/sampler.py — Weighted sampler that over-represents small-lesion cases."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import WeightedRandomSampler


def build_lesion_sampler(
    samples: list[dict],
    small_lesion_threshold_ml: float = 1.0,
) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler biased toward small-lesion cases.

    Weights:
        3.0 — at least one lesion component < threshold_ml
        1.0 — lesions present but all ≥ threshold_ml
        0.5 — no lesion at all (background-only case)

    Label files are read once at call time (a one-off cost during setup).

    Args:
        samples:                   List of sample dicts (as stored in the split JSON).
        small_lesion_threshold_ml: Volume threshold in mL below which a lesion
                                   component is considered "small".
    """
    from tqdm import tqdm

    weights = _compute_weights(samples, small_lesion_threshold_ml, verbose=True)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def _compute_weights(
    samples: list[dict],
    threshold_ml: float,
    verbose: bool = False,
) -> torch.Tensor:
    from tqdm import tqdm

    weights: list[float] = []
    iterator = tqdm(samples, desc="Computing sampler weights", leave=False) if verbose else samples
    for s in iterator:
        label_path = s.get("label_path", "")
        if label_path and Path(label_path).exists():
            w = _weight_for_label(Path(label_path), threshold_ml)
        else:
            w = 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def _weight_for_label(label_path: Path, threshold_ml: float) -> float:
    nib_img = nib.load(label_path)
    zooms = nib_img.header.get_zooms()[:3]
    vox_vol_ml = float(np.prod(zooms)) * 0.001
    data = np.asarray(nib_img.dataobj) > 0

    if not data.any():
        return 0.5

    labeled, n = ndimage.label(data)
    for i in range(1, n + 1):
        if (labeled == i).sum() * vox_vol_ml < threshold_ml:
            return 3.0
    return 1.0
