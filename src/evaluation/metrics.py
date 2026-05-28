"""src/evaluation/metrics.py — Full evaluation suite for ISLES'26.

Computes:
  - Dice score (voxel-wise)
  - Lesion-wise precision, recall, F1 (individual lesion detection)
  - Absolute Volume Difference (AVD) in mL
  - Hausdorff Distance 95th percentile (HD95) in mm
  - Average Symmetric Surface Distance (ASSD) in mm
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


def compute_lesion_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_volume_ml: float = 0.001,
    voxel_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    iou_threshold: float = 0.1,
) -> dict[str, float]:
    """Compute the full ISLES'26 metric suite.

    Args:
        pred:              Binary prediction array [D, H, W].
        gt:                Binary ground truth array [D, H, W].
        voxel_volume_ml:   Volume of one voxel in mL (1 mm³ = 0.001 mL).
        voxel_spacing_mm:  Voxel size in mm per axis, used for surface distances.
        iou_threshold:     Minimum IoU to count a GT lesion as detected.

    Returns:
        Dict with dice, lesion_precision, lesion_recall, lesion_f1,
        avd_ml, hd95_mm, assd_mm.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not pred.any() and not gt.any():
        return {
            "dice": 1.0,
            "avd_ml": 0.0,
            "lesion_precision": 1.0,
            "lesion_recall": 1.0,
            "lesion_f1": 1.0,
            "n_pred_lesions": 0.0,
            "n_gt_lesions": 0.0,
            "hd95_mm": 0.0,
            "assd_mm": 0.0,
        }

    metrics: dict[str, float] = {}

    # 1. Voxel-wise Dice
    intersection = (pred & gt).sum()
    metrics["dice"] = 2.0 * intersection / (pred.sum() + gt.sum())

    # 2. Absolute Volume Difference (mL)
    metrics["avd_ml"] = abs(pred.sum() - gt.sum()) * voxel_volume_ml

    # 3. Lesion-wise detection (F1)
    pred_labels, n_pred = ndimage.label(pred)
    gt_labels, n_gt = ndimage.label(gt)

    # Precision and recall need independent TP counts:
    #   recall    — how many GT lesions were hit by at least one prediction
    #   precision — how many predicted lesions hit at least one GT lesion
    tp_recall = _count_detected(gt_labels, pred_labels, n_gt, iou_threshold)
    tp_precision = _count_detected(pred_labels, gt_labels, n_pred, iou_threshold)

    precision = tp_precision / n_pred if n_pred > 0 else 0.0
    recall = tp_recall / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics["lesion_precision"] = precision
    metrics["lesion_recall"] = recall
    metrics["lesion_f1"] = f1
    metrics["n_pred_lesions"] = float(n_pred)
    metrics["n_gt_lesions"] = float(n_gt)

    # 4. Surface distances (HD95 + ASSD) in mm
    spacing = np.array(voxel_spacing_mm, dtype=np.float64)

    if pred.any() and gt.any():
        pred_surface = _get_surface_points(pred) * spacing
        gt_surface = _get_surface_points(gt) * spacing

        d_pred_gt = _surface_distances(pred_surface, gt_surface)
        d_gt_pred = _surface_distances(gt_surface, pred_surface)

        metrics["hd95_mm"] = float(
            max(np.percentile(d_pred_gt, 95), np.percentile(d_gt_pred, 95))
        )
        metrics["assd_mm"] = float((d_pred_gt.mean() + d_gt_pred.mean()) / 2.0)
    else:
        metrics["hd95_mm"] = float(
            np.sqrt(np.sum((np.array(pred.shape) * spacing) ** 2))
        )
        metrics["assd_mm"] = metrics["hd95_mm"]

    return metrics


def stratified_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subgroup mean metrics for error analysis.

    Subgroups:
        phase       — acute (≤ 7 d) / subacute (8–180 d) / chronic (> 180 d)
        lesion_size — small (< 1 mL) / medium (1–10 mL) / large (> 10 mL)
        center      — top-10 sites individually, rest grouped as 'other'

    Args:
        results_df: DataFrame produced by on_test_epoch_end, expected columns:
                    dice, lesion_f1, avd_ml, hd95_mm, assd_mm,
                    days_post_stroke, lesion_volume_ml, center.

    Returns:
        Long-format DataFrame with columns [group_type, group_value, <metrics>].
    """
    df = results_df.copy()
    num_cols = [c for c in ("dice", "lesion_f1", "avd_ml", "hd95_mm", "assd_mm") if c in df.columns]

    if "days_post_stroke" in df.columns:
        df["phase"] = pd.cut(
            df["days_post_stroke"],
            bins=[-1, 7, 180, float("inf")],
            labels=["acute", "subacute", "chronic"],
        ).astype(str)

    if "lesion_volume_ml" in df.columns:
        df["lesion_size"] = pd.cut(
            df["lesion_volume_ml"],
            bins=[-0.001, 1.0, 10.0, float("inf")],
            labels=["small", "medium", "large"],
        ).astype(str)

    if "center" in df.columns:
        top_centers = df["center"].value_counts().head(10).index
        df["center_group"] = df["center"].where(df["center"].isin(top_centers), "other")

    group_cols = [c for c in ("phase", "lesion_size", "center_group") if c in df.columns]
    frames: list[pd.DataFrame] = []
    for col in group_cols:
        g = df.groupby(col)[num_cols].mean().reset_index()
        g["group_type"] = col
        g = g.rename(columns={col: "group_value"})
        frames.append(g)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Internal helpers

def _count_detected(
    query_labels: np.ndarray,
    ref_labels: np.ndarray,
    n_query: int,
    iou_threshold: float,
) -> int:
    """Count query lesions that have IoU >= iou_threshold with at least one ref lesion."""
    count = 0
    for qid in range(1, n_query + 1):
        q_mask = query_labels == qid
        candidates = np.unique(ref_labels[q_mask])
        candidates = candidates[candidates > 0]
        for rid in candidates:
            r_mask = ref_labels == rid
            iou = (q_mask & r_mask).sum() / (q_mask | r_mask).sum()
            if iou >= iou_threshold:
                count += 1
                break
    return count


def _get_surface_points(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask)
    surface = mask ^ eroded
    return np.column_stack(np.where(surface)).astype(np.float64)


def _surface_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(target)
    dists, _ = tree.query(source)
    return dists