#!/usr/bin/env python
"""scripts/make_splits.py — Create center-stratified K-fold splits.

Splits are stratified by CENTER code to ensure the validation set
contains unseen acquisition protocols — mimicking the hidden test set.

Usage:
    python scripts/make_splits.py \
        data.processed_dir=data/processed \
        splits.n_folds=5 \
        splits.test_ratio=0.1 \
        seed=42
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from rich.console import Console
from sklearn.model_selection import GroupKFold

console = Console()


def load_subjects(processed_dir: Path) -> list[dict]:
    """Load all processed subjects with their metadata."""
    subjects = []
    for meta_file in sorted(processed_dir.rglob("metadata.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        subjects.append(meta)
    return subjects


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    processed_dir = Path(cfg.paths.data_proc)
    splits_dir = Path(cfg.paths.splits)
    splits_dir.mkdir(parents=True, exist_ok=True)

    n_folds = cfg.get("splits", {}).get("n_folds", 5)
    test_ratio = cfg.get("splits", {}).get("test_ratio", 0.1)
    seed = cfg.seed

    random.seed(seed)
    np.random.seed(seed)

    subjects = load_subjects(processed_dir)
    console.print(f"Loaded [bold]{len(subjects)}[/bold] subjects")

    # Hold out a global test set (stratified by center)
    centers = [s.get("center", "unknown") for s in subjects]
    center_to_subjects = defaultdict(list)
    for s, c in zip(subjects, centers):
        center_to_subjects[c].append(s)

    test_set = []
    trainval_set = []
    for center, subs in center_to_subjects.items():
        random.shuffle(subs)
        n_test = max(1, int(len(subs) * test_ratio))
        test_set.extend(subs[:n_test])
        trainval_set.extend(subs[n_test:])

    console.print(
        f"Test: [red]{len(test_set)}[/red]  |  Train+Val: [green]{len(trainval_set)}[/green]"
    )

    # Group K-Fold on trainval — groups = center codes
    X = np.arange(len(trainval_set))
    groups = np.array([s.get("center", "unknown") for s in trainval_set])

    gkf = GroupKFold(n_splits=n_folds)
    splits: dict[str, dict[str, list]] = {}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
        splits[f"fold_{fold_idx}"] = {
            "train": [trainval_set[i] for i in train_idx],
            "val": [trainval_set[i] for i in val_idx],
            "test": test_set,
        }
        console.print(
            f"  Fold {fold_idx}: "
            f"train={len(train_idx)}  val={len(val_idx)}  "
            f"val_centers={len(set(groups[val_idx]))}"
        )

    out_file = splits_dir / "splits_5fold.json"
    with open(out_file, "w") as f:
        json.dump(splits, f, indent=2)

    console.print(f"[green]✓ Splits saved → {out_file}[/green]")


if __name__ == "__main__":
    main()
