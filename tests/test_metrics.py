"""tests/test_metrics.py — Unit tests for evaluation metrics."""
import numpy as np
import pytest

from src.evaluation.metrics import compute_lesion_metrics


def _sphere(shape, center, radius):
    """Create a binary sphere mask."""
    mask = np.zeros(shape, dtype=bool)
    idx = np.indices(shape)
    dist = np.sqrt(sum((idx[i] - center[i]) ** 2 for i in range(3)))
    mask[dist <= radius] = True
    return mask


# Dice

class TestDice:
    def test_perfect(self):
        gt = _sphere((64, 64, 64), (32, 32, 32), 10)
        m = compute_lesion_metrics(gt.copy(), gt.copy())
        assert abs(m["dice"] - 1.0) < 1e-4

    def test_both_empty_returns_perfect(self):
        shape = (64, 64, 64)
        m = compute_lesion_metrics(np.zeros(shape, bool), np.zeros(shape, bool))
        assert m["dice"] == pytest.approx(1.0)
        assert m["lesion_f1"] == pytest.approx(1.0)
        assert m["hd95_mm"] == pytest.approx(0.0)

    def test_empty_pred(self):
        gt = _sphere((64, 64, 64), (32, 32, 32), 10)
        m = compute_lesion_metrics(np.zeros_like(gt), gt)
        assert m["dice"] == pytest.approx(0.0, abs=1e-4)

    def test_no_overlap(self):
        gt = _sphere((64, 64, 64), (16, 16, 16), 5)
        pred = _sphere((64, 64, 64), (48, 48, 48), 5)
        m = compute_lesion_metrics(pred, gt)
        assert m["dice"] == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap(self):
        gt = _sphere((64, 64, 64), (32, 32, 32), 10)
        pred = _sphere((64, 64, 64), (38, 32, 32), 10)
        m = compute_lesion_metrics(pred, gt)
        assert 0.0 < m["dice"] < 1.0


# Lesion-wise F1

class TestLesionF1:
    def test_single_lesion_detected(self):
        gt = _sphere((64, 64, 64), (32, 32, 32), 8)
        pred = _sphere((64, 64, 64), (32, 32, 32), 8)
        m = compute_lesion_metrics(pred, gt)
        assert m["lesion_f1"] == pytest.approx(1.0, abs=0.01)

    def test_single_lesion_missed(self):
        gt = _sphere((64, 64, 64), (32, 32, 32), 8)
        m = compute_lesion_metrics(np.zeros_like(gt), gt)
        assert m["lesion_recall"] == pytest.approx(0.0, abs=0.01)

    def test_false_positive_lesion(self):
        gt = _sphere((64, 64, 64), (20, 20, 20), 6)
        pred = _sphere((64, 64, 64), (20, 20, 20), 6) | _sphere((64, 64, 64), (45, 45, 45), 6)
        m = compute_lesion_metrics(pred, gt)
        assert m["lesion_precision"] < 1.0
        assert m["lesion_recall"] == pytest.approx(1.0, abs=0.01)


# Volume

class TestVolume:
    def test_equal_volumes(self):
        gt = _sphere((64, 64, 64), (32, 32, 32), 8)
        m = compute_lesion_metrics(gt.copy(), gt.copy(), voxel_volume_ml=0.001)
        assert m["avd_ml"] == pytest.approx(0.0, abs=1e-4)

    def test_volume_difference(self):
        gt = _sphere((64, 64, 64), (32, 32, 32), 10)
        pred = _sphere((64, 64, 64), (32, 32, 32), 5)
        m = compute_lesion_metrics(pred, gt, voxel_volume_ml=0.001)
        assert m["avd_ml"] > 0


# Surface distances

class TestSurfaceDistances:
    def test_perfect_overlap_hd95_zero(self):
        gt = _sphere((64, 64, 64), (32, 32, 32), 8)
        m = compute_lesion_metrics(gt.copy(), gt.copy(), voxel_spacing_mm=(1.0, 1.0, 1.0))
        assert m["hd95_mm"] == pytest.approx(0.0, abs=1e-4)
        assert m["assd_mm"] == pytest.approx(0.0, abs=1e-4)

    def test_anisotropic_spacing_scales_distances(self):
        gt = _sphere((64, 64, 64), (32, 32, 32), 8)
        pred = _sphere((64, 64, 64), (38, 32, 32), 8)
        m1 = compute_lesion_metrics(pred, gt, voxel_spacing_mm=(1.0, 1.0, 1.0))
        m2 = compute_lesion_metrics(pred, gt, voxel_spacing_mm=(2.0, 1.0, 1.0))
        assert m2["hd95_mm"] > m1["hd95_mm"]
