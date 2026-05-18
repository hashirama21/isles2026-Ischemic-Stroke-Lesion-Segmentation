"""src/inference/predict.py — Sliding window inference, TTA, ensembling, and post-processing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from src.utils.misc import load_checkpoint

_TTA_FLIPS = [
    [],
    [2],
    [3],
    [4],
    [2, 3],
    [2, 4],
    [3, 4],
    [2, 3, 4],
]


def predict_with_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    metadata: torch.Tensor,
    roi_size: list[int],
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Run sliding-window inference with 8-fold flip TTA.

    Returns soft probability map [B, C, D, H, W] averaged over all flips.
    """
    if device is None:
        device = next(model.parameters()).device

    image = image.to(device)
    metadata = metadata.to(device)
    accumulated: torch.Tensor | None = None

    for flip_dims in _TTA_FLIPS:
        img_flip = torch.flip(image, flip_dims) if flip_dims else image

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=img_flip,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=lambda x: model(x, metadata[:x.shape[0]]),
                overlap=overlap,
                mode="gaussian",
            )

        probs = F.softmax(logits, dim=1)
        if flip_dims:
            probs = torch.flip(probs, flip_dims)

        accumulated = probs if accumulated is None else accumulated.add_(probs)

    return accumulated / len(_TTA_FLIPS)  # type: ignore[operator]


def ensemble_predict(
    checkpoint_paths: list[Path],
    model_cfgs: list,
    image: torch.Tensor,
    metadata: torch.Tensor,
    roi_size: list[int],
    use_tta: bool = True,
    device: torch.device | None = None,
    postprocess_cfg=None,
    voxel_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Load multiple checkpoints and return a binary segmentation mask.

    Args:
        checkpoint_paths:  .ckpt files (one per fold or architecture).
        model_cfgs:        Hydra model configs matching each checkpoint.
        image:             [1, 1, D, H, W] float tensor.
        metadata:          [1, meta_dim] float tensor.
        roi_size:          Sliding window patch size.
        use_tta:           Apply 8-fold flip TTA.
        postprocess_cfg:   If provided, run post-processing after argmax.
        voxel_spacing_mm:  Voxel size in mm (D, H, W) — used for postprocessing.

    Returns:
        Binary segmentation mask [D, H, W] as np.ndarray uint8.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_probs: list[torch.Tensor] = []

    for ckpt_path, model_cfg in zip(checkpoint_paths, model_cfgs):
        model = load_checkpoint(ckpt_path, model_cfg, device=device)
        model.eval()

        if use_tta:
            probs = predict_with_tta(
                model, image, metadata, roi_size, device=device
            )
        else:
            with torch.no_grad():
                logits = sliding_window_inference(
                    inputs=image.to(device),
                    roi_size=roi_size,
                    sw_batch_size=4,
                    predictor=lambda x: model(x, metadata.to(device)[:x.shape[0]]),
                    overlap=0.5,
                    mode="gaussian",
                )
            probs = F.softmax(logits, dim=1)

        all_probs.append(probs.cpu())

    ensemble_probs = torch.stack(all_probs).mean(dim=0)  # [1, C, D, H, W]

    if postprocess_cfg is not None:
        from src.inference.postprocess import postprocess_prediction
        prob_np = ensemble_probs.squeeze(0).numpy()       # [C, D, H, W]
        pred_np = prob_np.argmax(axis=0).astype(np.uint8)
        meta_np = metadata.squeeze(0).numpy()
        return postprocess_prediction(
            pred_np, prob_np, meta_np, voxel_spacing_mm, postprocess_cfg
        )

    return ensemble_probs.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)