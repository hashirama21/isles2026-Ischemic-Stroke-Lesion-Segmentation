#!/usr/bin/env python
"""scripts/ensemble.py — Ensemble all fold checkpoints with TTA.

Runs on a directory of preprocessed test subjects and writes NIfTI predictions
ready for Grand Challenge Docker packaging.

Usage:
    python scripts/ensemble.py \
        checkpoints_dir=outputs/checkpoints \
        data.processed_dir=data/processed \
        ensemble.use_tta=true
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import torch
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress

from src.data.dataset import MAX_DAYS
from src.data.transforms import get_val_transforms
from src.inference.predict import ensemble_predict
from src.utils.misc import seed_everything

console = Console()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    ckpt_root = Path(cfg.paths.checkpoints).parent   # parent of fold dirs
    processed_dir = Path(cfg.paths.data_proc)
    pred_dir = Path(cfg.paths.predictions) / "ensemble"
    pred_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Device: [bold]{device}[/bold]")

    # Collect all fold checkpoints
    ckpt_paths = sorted(ckpt_root.rglob("*.ckpt"))
    console.print(f"Found [bold]{len(ckpt_paths)}[/bold] checkpoints")

    # Load test subjects
    test_metas = sorted(processed_dir.rglob("metadata.json"))
    val_transform = get_val_transforms(cfg.data)
    roi_size = list(cfg.training.sw_roi_size)
    use_tta = cfg.get("ensemble", {}).get("use_tta", True)

    with Progress() as progress:
        task = progress.add_task("Predicting...", total=len(test_metas))

        for meta_path in test_metas:
            with open(meta_path) as f:
                meta = json.load(f)

            subj_id = meta["subject_id"]
            img_path = meta["image_path"]

            # Build sample dict
            img_nib = nib.load(img_path)
            zooms = img_nib.header.get_zooms()[:3]
            voxel_spacing_mm = tuple(float(z) for z in zooms)

            image_tensor = torch.from_numpy(
                np.asarray(img_nib.dataobj, dtype=np.float32)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

            meta_tensor = torch.tensor(
                [
                    min(float(meta.get("days_post_stroke", 0)) / MAX_DAYS, 1.0),
                    float(meta.get("chronicity", 0)),
                ],
                dtype=torch.float32,
            ).unsqueeze(0)  # [1, 2]

            pred_array = ensemble_predict(
                checkpoint_paths=ckpt_paths,
                image=image_tensor,
                metadata=meta_tensor,
                roi_size=roi_size,
                use_tta=use_tta,
                device=device,
                postprocess_cfg=cfg,
                voxel_spacing_mm=voxel_spacing_mm,
            )

            out_nib = nib.Nifti1Image(
                pred_array.astype(np.uint8), img_nib.affine, img_nib.header
            )
            nib.save(out_nib, pred_dir / f"{subj_id}_pred.nii.gz")

            progress.advance(task)

    console.print(f"[green]✓ Predictions saved → {pred_dir}[/green]")


if __name__ == "__main__":
    main()
