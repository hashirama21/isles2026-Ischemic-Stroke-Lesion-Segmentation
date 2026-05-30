#!/usr/bin/env python
"""docker/process.py — Grand Challenge inference wrapper.

Grand Challenge calls this script for each test case.
Expected environment:
    /input/images/t1-brain-mri/  — contains one .mha or .nii.gz
    /input/                      — may contain metadata.json
    /output/images/stroke-lesion-segmentation/  — write prediction here

Ensemble checkpoints are baked into the Docker image under CKPT_DIR.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from omegaconf import OmegaConf

from src.data.dataset import MAX_DAYS

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
CKPT_DIR = Path("/opt/ml/checkpoints")

IMAGE_INPUT = INPUT_DIR / "images" / "t1-brain-mri"
META_INPUT = INPUT_DIR / "metadata.json"
SEG_OUTPUT = OUTPUT_DIR / "images" / "stroke-lesion-segmentation"


def _sitk_to_ras_affine(img_sitk: sitk.Image) -> np.ndarray:
    """Build a RAS-convention NIfTI affine from a SimpleITK image.

    SimpleITK uses LPS orientation with (x, y, z) physical/index convention.
    GetArrayFromImage returns a numpy array in (z, y, x) order.
    NIfTI/MONAI use RAS orientation.

    Steps:
        1. Build LPS affine for (x, y, z) voxel indexing.
        2. Negate x and y rows to convert LPS → RAS.
        3. Permute affine columns so that numpy (z, y, x) indices
           map to the correct RAS physical coordinates.
    """
    spacing = np.array(img_sitk.GetSpacing())                         # (sx, sy, sz)
    origin = np.array(img_sitk.GetOrigin())                           # (ox, oy, oz) LPS
    direction = np.array(img_sitk.GetDirection()).reshape(3, 3)       # LPS (x,y,z)

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin

    # LPS → RAS: negate first two physical axis rows (L→R, P→A)
    affine_ras = affine_lps.copy()
    affine_ras[[0, 1], :] *= -1

    # Permute columns: numpy array is (z, y, x) but affine was built for (x, y, z)
    # The permutation matrix maps numpy index (nz, ny, nx) → physical (x, y, z) order
    perm = np.array([[0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]], dtype=np.float64)
    return (affine_ras @ perm).astype(np.float32)


def load_input_image() -> tuple[np.ndarray, np.ndarray]:
    """Load the input T1w volume. Returns (data [D,H,W], affine [4,4])."""
    files = sorted(IMAGE_INPUT.glob("*.mha")) + sorted(IMAGE_INPUT.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No image found in {IMAGE_INPUT}")

    img_sitk = sitk.ReadImage(str(files[0]))
    data = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # [D, H, W] = (z,y,x)
    affine = _sitk_to_ras_affine(img_sitk)

    return data, affine


def load_metadata() -> dict:
    if META_INPUT.exists():
        with open(META_INPUT) as f:
            return json.load(f)
    return {}


def preprocess(data: np.ndarray) -> torch.Tensor:
    """Percentile clip + Z-score normalise within brain mask, then tensorise."""
    mask = data > 0
    if mask.any():
        p1, p99 = np.percentile(data[mask], [1, 99])
        data = np.clip(data, p1, p99)
        mean, std = data[mask].mean(), data[mask].std() + 1e-8
        data[mask] = (data[mask] - mean) / std
    return torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]


def run() -> None:
    SEG_OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[process.py] device={device}")

    data, affine = load_input_image()
    image_tensor = preprocess(data)

    raw_meta = load_metadata()
    days = float(raw_meta.get("DAYS_POST_STROKE", 0))
    chronicity = float(raw_meta.get("CHRONICITY", 0))
    meta_tensor = torch.tensor(
        [min(days / MAX_DAYS, 1.0), chronicity], dtype=torch.float32
    ).unsqueeze(0)  # [1, 2]

    ckpt_paths = sorted(CKPT_DIR.glob("*.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {CKPT_DIR}")
    print(f"[process.py] {len(ckpt_paths)} checkpoints found")

    from src.inference.predict import ensemble_predict

    postprocess_cfg = OmegaConf.create({
        "postprocess": {
            "min_lesion_volume_ml": 0.05,
            "threshold_acute": 0.45,
            "threshold_chronic": 0.35,
        }
    })

    pred = ensemble_predict(
        checkpoint_paths=ckpt_paths,
        image=image_tensor,
        metadata=meta_tensor,
        roi_size=[128, 128, 128],
        use_tta=True,
        device=device,
        postprocess_cfg=postprocess_cfg,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    out_nib = nib.Nifti1Image(pred.astype(np.uint8), affine)
    out_path = SEG_OUTPUT / "output.nii.gz"
    nib.save(out_nib, out_path)
    print(f"[process.py] Saved prediction → {out_path}")


if __name__ == "__main__":
    run()
