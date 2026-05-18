"""src/utils/misc.py — General utility helpers."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(
    ckpt_path: Path,
    model_cfg,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Instantiate a model from config and load weights from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate(model_cfg)
    state = torch.load(ckpt_path, map_location=device)

    # Lightning checkpoints nest weights under 'state_dict'
    if "state_dict" in state:
        # Strip 'model.' prefix added by LightningModule
        weights = {
            k.replace("model.", "", 1): v
            for k, v in state["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(weights, strict=True)
    else:
        model.load_state_dict(state, strict=True)

    model.to(device)
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
