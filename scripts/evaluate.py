#!/usr/bin/env python
"""scripts/evaluate.py — Evaluate a trained checkpoint on the test split.

Usage:
    python scripts/evaluate.py \
        checkpoint=outputs/checkpoints/fold0/best.ckpt \
        fold=0
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from src.data.datamodule import ISLES26DataModule
from src.evaluation.metrics import stratified_metrics
from src.training.module import ISLES26Module
from src.utils.misc import seed_everything

console = Console()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    ckpt = cfg.get("checkpoint")
    if ckpt is None:
        raise ValueError("Pass checkpoint=<path> on the command line.")

    datamodule = ISLES26DataModule(cfg.data)
    module = ISLES26Module.load_from_checkpoint(ckpt, cfg=cfg)

    trainer = pl.Trainer(
        accelerator="auto",
        precision=cfg.training.precision,
        logger=False,
    )

    trainer.test(module, datamodule=datamodule)

    csv_path = Path(cfg.paths.predictions) / "test_metrics.csv"
    if not csv_path.exists():
        console.print("[yellow]test_metrics.csv not found — skipping summary[/yellow]")
        return

    df = pd.read_csv(csv_path)

    mean_row = df.select_dtypes("number").mean()
    table = Table(title="Overall Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    for k, v in mean_row.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    strat = stratified_metrics(df)
    if not strat.empty:
        out_strat = Path(cfg.paths.predictions) / "stratified_metrics.csv"
        strat.to_csv(out_strat, index=False)
        console.print(f"[green]✓ Stratified metrics → {out_strat}[/green]")

        num_cols = [c for c in ("dice", "lesion_f1", "avd_ml", "hd95_mm") if c in strat.columns]
        strat_table = Table(title="Stratified Results")
        strat_table.add_column("Group type", style="magenta")
        strat_table.add_column("Group", style="cyan")
        for col in num_cols:
            strat_table.add_column(col, style="green")
        for _, row in strat.iterrows():
            vals = [f"{row[c]:.4f}" for c in num_cols]
            strat_table.add_row(row["group_type"], str(row["group_value"]), *vals)
        console.print(strat_table)


if __name__ == "__main__":
    main()