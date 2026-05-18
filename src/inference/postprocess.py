"""src/inference/postprocess.py — Post-processing pipeline for stroke lesion masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def remove_small_components(
    pred: np.ndarray,
    min_volume_vox: int,
) -> np.ndarray:
    """Remove connected components smaller than min_volume_vox voxels (26-connectivity)."""
    struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labeled, n = ndimage.label(pred.astype(bool), structure=struct)
    cleaned = np.zeros_like(pred)
    for i in range(1, n + 1):
        mask = labeled == i
        if mask.sum() >= min_volume_vox:
            cleaned[mask] = 1
    return cleaned.astype(pred.dtype)


def fill_holes(pred: np.ndarray) -> np.ndarray:
    """Fill internal holes: per-axial-slice then a full 3-D pass."""
    filled = pred.astype(bool).copy()
    for i in range(filled.shape[0]):
        filled[i] = ndimage.binary_fill_holes(filled[i])
    return ndimage.binary_fill_holes(filled).astype(pred.dtype)


def adaptive_threshold(
    prob_map: np.ndarray,
    days_post_stroke: float,
    threshold_acute: float = 0.45,
    threshold_chronic: float = 0.35,
) -> np.ndarray:
    """Threshold the lesion probability map with a phase-dependent cutoff.

    Chronic lesions (> 180 days) are hypointense on T1w and harder to detect,
    so a lower threshold reduces false negatives at the cost of a few more FPs.

    Args:
        prob_map:          2-D or 3-D array of lesion class probabilities.
        days_post_stroke:  Absolute (de-normalised) days since stroke onset.
        threshold_acute:   Threshold for acute/sub-acute phase (≤ 180 days).
        threshold_chronic: Threshold for chronic phase (> 180 days).

    Returns:
        Binary uint8 mask.
    """
    threshold = threshold_chronic if days_post_stroke > 180 else threshold_acute
    return (prob_map > threshold).astype(np.uint8)


def postprocess_prediction(
    pred: np.ndarray,
    prob_map: np.ndarray,
    metadata: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float],
    cfg,
) -> np.ndarray:
    """Full post-processing chain: adaptive threshold → remove small components → fill holes.

    Args:
        pred:              Raw binary mask [D, H, W] (not used directly; prob_map is
                           re-thresholded for the adaptive step).
        prob_map:          Soft probability map [C, D, H, W] (C=2, channel 1 = lesion).
        metadata:          Numpy array [2] = [days_norm, chronicity].
        voxel_spacing_mm:  Voxel size in mm (D, H, W).
        cfg:               Object / OmegaConf node with postprocess sub-keys:
                           min_lesion_volume_ml, threshold_acute, threshold_chronic.

    Returns:
        Post-processed binary uint8 mask [D, H, W].
    """
    days = float(metadata[0]) * 365.0  # de-normalise

    lesion_probs = prob_map[1] if prob_map.ndim == 4 else prob_map
    result = adaptive_threshold(
        lesion_probs,
        days,
        threshold_acute=cfg.postprocess.threshold_acute,
        threshold_chronic=cfg.postprocess.threshold_chronic,
    )

    vox_vol_ml = float(np.prod(voxel_spacing_mm)) * 0.001
    min_volume_vox = max(1, int(cfg.postprocess.min_lesion_volume_ml / vox_vol_ml))
    result = remove_small_components(result, min_volume_vox)
    result = fill_holes(result)

    return result.astype(np.uint8)
