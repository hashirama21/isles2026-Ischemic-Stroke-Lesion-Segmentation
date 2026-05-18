"""tests/test_postprocess.py — Unit tests for the post-processing pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.inference.postprocess import (
    adaptive_threshold,
    fill_holes,
    postprocess_prediction,
    remove_small_components,
)


class TestRemoveSmallComponents:
    def test_removes_small(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        pred[1:3, 1:3, 1:3] = 1   # 8 voxels — should be removed
        pred[8:14, 8:14, 8:14] = 1  # 216 voxels — should stay
        result = remove_small_components(pred, min_volume_vox=100)
        assert result[1, 1, 1] == 0
        assert result[10, 10, 10] == 1

    def test_keeps_all_above_threshold(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        pred[2:8, 2:8, 2:8] = 1
        result = remove_small_components(pred, min_volume_vox=1)
        assert result.sum() == pred.sum()

    def test_empty_input(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        assert remove_small_components(pred, min_volume_vox=10).sum() == 0


class TestFillHoles:
    def test_fills_hole(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        pred[3:13, 3:13, 3:13] = 1
        pred[6:9, 6:9, 6:9] = 0   # punch a hole
        result = fill_holes(pred)
        assert result[7, 7, 7] == 1

    def test_solid_object_unchanged(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        pred[4:12, 4:12, 4:12] = 1
        result = fill_holes(pred)
        assert np.array_equal(result, pred)


class TestAdaptiveThreshold:
    def test_acute_uses_higher_threshold(self):
        prob = np.full((8, 8, 8), 0.40)
        result_acute = adaptive_threshold(prob, days_post_stroke=3, threshold_acute=0.45)
        result_chronic = adaptive_threshold(prob, days_post_stroke=3, threshold_chronic=0.35)
        assert result_acute.sum() == 0      # 0.40 < 0.45
        assert result_chronic.sum() == 0    # not chronic path

    def test_chronic_uses_lower_threshold(self):
        prob = np.full((8, 8, 8), 0.40)
        result = adaptive_threshold(prob, days_post_stroke=400, threshold_chronic=0.35)
        assert result.sum() == prob.size    # 0.40 > 0.35

    def test_boundary_day_180(self):
        prob = np.full((4, 4, 4), 0.42)
        below = adaptive_threshold(prob, days_post_stroke=179, threshold_acute=0.45, threshold_chronic=0.35)
        above = adaptive_threshold(prob, days_post_stroke=181, threshold_acute=0.45, threshold_chronic=0.35)
        assert below.sum() == 0          # 0.42 < 0.45
        assert above.sum() == prob.size  # 0.42 > 0.35


class TestPostprocessPrediction:
    class _Cfg:
        class postprocess:
            min_lesion_volume_ml = 0.05
            threshold_acute = 0.45
            threshold_chronic = 0.35

    def test_removes_noise_and_returns_uint8(self):
        prob_map = np.zeros((2, 16, 16, 16), np.float32)
        prob_map[0] = 0.9
        prob_map[1, 2:4, 2:4, 2:4] = 0.5   # tiny spurious region
        prob_map[1, 8:14, 8:14, 8:14] = 0.5  # larger region

        pred = prob_map.argmax(axis=0).astype(np.uint8)
        meta = np.array([30.0 / 365.0, 1.0])  # 30 days, subacute

        result = postprocess_prediction(
            pred, prob_map, meta, (1.0, 1.0, 1.0), self._Cfg()
        )
        assert result.dtype == np.uint8
        assert result.ndim == 3