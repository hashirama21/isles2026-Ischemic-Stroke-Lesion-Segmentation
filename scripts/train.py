#!/usr/bin/env python
"""scripts/train.py — Main training entry point.

Usage:
    python scripts/train.py experiment=segresnet_baseline fold=0
    python scripts/train.py experiment=segresnet_baseline fold=0 debug=true
    python scripts/train.py model=swinunetr training.max_epochs=300 fold=1
"""
from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from rich.console import Console

from src.data.datamodule import ISLES26DataModule
from src.training.module import ISLES26Module
from src.utils.misc import seed_everything

console = Console()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    console.print("[bold blue]ISLES'26 Training[/bold blue]")
    console.print(OmegaConf.to_yaml(cfg))

    datamodule = ISLES26DataModule(cfg.data)

    module = ISLES26Module(cfg)
    console.print(f"Model parameters: {sum(p.numel() for p in module.parameters()):,}")

    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch{epoch:03d}-dice{val/dice:.4f}",
            monitor="val/dice",
            mode="max",
            save_top_k=cfg.training.save_top_k,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/dice",
            patience=cfg.training.patience,
            mode="max",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    logger = None
    if not cfg.debug:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=list(cfg.wandb.tags),
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs if not cfg.debug else 2,
        val_check_interval=cfg.training.val_check_interval,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        fast_dev_run=cfg.debug,
    )

    trainer.fit(module, datamodule=datamodule)
    console.print(f"[green]✓ Training complete. Best ckpt: {callbacks[0].best_model_path}[/green]")

    if not cfg.debug and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
