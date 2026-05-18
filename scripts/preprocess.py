#!/usr/bin/env python
"""scripts/preprocess.py — Skull-strip and normalise raw ISLES'26 T1w volumes.

Usage:
    python scripts/preprocess.py \
        data.raw_dir=data/raw \
        data.out_dir=data/processed \
        preprocess.n_jobs=4

Pipeline per subject:
    1. Load T1w NIfTI + metadata JSON
    2. Skull-strip with HD-BET (fast mode) or SynthStrip fallback
    3. Percentile clipping [1%, 99%] on brain mask
    4. Z-score normalisation within brain mask
    5. Save processed NIfTI + copy metadata JSON
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import Progress

console = Console()


def _skull_strip(img_path: Path, out_path: Path) -> None:
    """Apply skull stripping, writing result to out_path."""
    try:
        from hd_bet.run import run_hd_bet

        run_hd_bet(str(img_path), str(out_path.with_suffix("").with_suffix("")))
        # HD-BET writes <stem>_bet.nii.gz next to the output path
        candidate = out_path.parent / (out_path.stem.replace(".nii", "") + "_bet.nii.gz")
        if candidate.exists():
            shutil.move(str(candidate), str(out_path))
        elif not out_path.exists():
            raise RuntimeError(
                f"HD-BET ran but produced no output at {candidate} or {out_path}"
            )
    except ImportError:
        console.print("[yellow]HD-BET not available — falling back to intensity threshold.[/yellow]")
        img = nib.load(img_path)
        data = img.get_fdata(dtype=np.float32)
        mask = data > np.percentile(data, 10)
        data[~mask] = 0.0
        nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)


def _normalise(img_path: Path, out_path: Path) -> None:
    """Clip [1, 99] percentile then Z-score normalise within brain mask."""
    img = nib.load(img_path)
    data = img.get_fdata(dtype=np.float32)
    mask = data > 0

    p1, p99 = np.percentile(data[mask], [1, 99])
    data = np.clip(data, p1, p99)

    mean = data[mask].mean()
    std = data[mask].std() + 1e-8
    data[mask] = (data[mask] - mean) / std

    nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)


def process_subject(
    subj_dir: Path,
    out_root: Path,
    strip: bool = True,
) -> str:
    """Process a single subject directory. Returns subject ID."""
    subj_id = subj_dir.name
    out_dir = out_root / subj_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expect: <subj>/sub-*_T1w.nii.gz  and  <subj>/sub-*_mask.nii.gz
    t1w_files = sorted(subj_dir.glob("*_T1w.nii.gz"))
    mask_files = sorted(subj_dir.glob("*_mask.nii.gz"))
    json_files = sorted(subj_dir.glob("*.json"))

    if not t1w_files:
        return f"SKIP {subj_id}: no T1w found"

    t1w = t1w_files[0]
    stripped_path = out_dir / "image_stripped.nii.gz"
    final_path = out_dir / "image.nii.gz"

    # 1. Skull strip
    if strip:
        _skull_strip(t1w, stripped_path)
    else:
        shutil.copy(t1w, stripped_path)

    # 2. Normalise
    _normalise(stripped_path, final_path)
    stripped_path.unlink(missing_ok=True)

    # 3. Copy label
    if mask_files:
        shutil.copy(mask_files[0], out_dir / "label.nii.gz")

    # 4. Copy / merge metadata JSON
    meta = {}
    if json_files:
        with open(json_files[0]) as f:
            raw = json.load(f)
        meta["days_post_stroke"] = raw.get("DAYS_POST_STROKE", 0)
        meta["chronicity"] = raw.get("CHRONICITY", 0)
        meta["center"] = raw.get("CENTER", "unknown")
    meta["subject_id"] = subj_id
    meta["image_path"] = str(final_path)
    meta["label_path"] = str(out_dir / "label.nii.gz")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return subj_id


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.paths.data_raw)
    out_dir = Path(cfg.paths.data_proc)
    out_dir.mkdir(parents=True, exist_ok=True)

    subj_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    console.print(f"[bold]Preprocessing {len(subj_dirs)} subjects[/bold] → {out_dir}")

    n_jobs = cfg.get("preprocess", {}).get("n_jobs", 4)

    with Progress() as progress:
        task = progress.add_task("Processing...", total=len(subj_dirs))
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(process_subject, d, out_dir): d for d in subj_dirs
            }
            for fut in as_completed(futures):
                result = fut.result()
                progress.advance(task)

    console.print("[green]✓ Preprocessing complete[/green]")


if __name__ == "__main__":
    main()
