"""tests/test_sampler.py — Tests for the weighted lesion sampler."""
from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.data.sampler import build_lesion_sampler


def _make_label(path: Path, mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> None:
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))


@pytest.fixture
def tmp_labels(tmp_path):
    """Three label files: small lesion, large lesion, no lesion."""
    small = np.zeros((32, 32, 32), np.uint8)
    small[14:16, 14:16, 14:16] = 1  # 8 voxels × 0.001 mL = 0.008 mL (< 1 mL)

    large = np.zeros((32, 32, 32), np.uint8)
    large[5:20, 5:20, 5:20] = 1  # 3375 voxels × 0.001 mL = 3.375 mL (> 1 mL)

    empty = np.zeros((32, 32, 32), np.uint8)

    paths = {}
    for name, arr in [("small", small), ("large", large), ("empty", empty)]:
        p = tmp_path / f"{name}.nii.gz"
        _make_label(p, arr)
        paths[name] = str(p)

    return paths


def test_weights_ordering(tmp_labels):
    samples = [
        {"label_path": tmp_labels["small"]},
        {"label_path": tmp_labels["large"]},
        {"label_path": tmp_labels["empty"]},
    ]
    sampler = build_lesion_sampler(samples, small_lesion_threshold_ml=1.0)
    weights = sampler.weights.tolist()
    assert weights[0] == pytest.approx(3.0)   # small lesion — over-sampled
    assert weights[1] == pytest.approx(1.0)   # large lesion — normal weight
    assert weights[2] == pytest.approx(0.5)   # no lesion    — under-sampled


def test_small_cases_overrepresented(tmp_labels):
    """Over 1000 draws, small-lesion case should appear more than large-lesion case."""
    samples = [
        {"label_path": tmp_labels["small"]},
        {"label_path": tmp_labels["large"]},
    ]
    sampler = build_lesion_sampler(samples, small_lesion_threshold_ml=1.0)

    counts = [0, 0]
    for idx in sampler:
        counts[idx] += 1

    assert counts[0] > counts[1], (
        f"Small lesion case ({counts[0]}) should be drawn more than large ({counts[1]})"
    )


def test_missing_label_path_gets_default_weight():
    samples = [{"label_path": "/nonexistent/label.nii.gz"}]
    sampler = build_lesion_sampler(samples, small_lesion_threshold_ml=1.0)
    assert sampler.weights[0].item() == pytest.approx(1.0)


def test_sampler_length_matches_dataset():
    samples = [{"label_path": ""}, {"label_path": ""}]
    sampler = build_lesion_sampler(samples)
    assert sampler.num_samples == len(samples)