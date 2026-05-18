"""src/data/transforms.py — MONAI + TorchIO augmentation pipeline."""
from __future__ import annotations

import torch
import torchio as tio
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    Orientationd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    Spacingd,
    ToTensord,
)
from omegaconf import DictConfig


def get_train_transforms(cfg: DictConfig) -> "_CombinedTransform":
    """Return full training augmentation pipeline."""
    aug = cfg.augmentation

    monai_t = Compose([
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=list(cfg.spacing),
            mode=("bilinear", "nearest"),
        ),
        ResizeWithPadOrCropd(
            keys=["image", "label"],
            spatial_size=list(cfg.spatial_size),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandFlipd(
            keys=["image", "label"],
            prob=aug.random_flip_prob,
            spatial_axis=[0, 1, 2],
        ),
        RandAffined(
            keys=["image", "label"],
            prob=aug.random_affine_prob,
            rotate_range=[0.26, 0.26, 0.26],  # ~15 deg
            shear_range=[0.05, 0.05],
            scale_range=[0.1, 0.1, 0.1],
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            prob=aug.random_elastic_prob,
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandGaussianNoised(keys=["image"], prob=aug.random_noise_prob, std=0.05),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=tuple(aug.gamma_range)),
        ToTensord(keys=["image", "label"]),
    ])

    return _CombinedTransform(_build_tio_transforms(aug), monai_t)


def get_val_transforms(cfg: DictConfig) -> Compose:
    """Return deterministic validation pipeline (no augmentation)."""
    return Compose([
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=list(cfg.spacing),
            mode=("bilinear", "nearest"),
        ),
        ResizeWithPadOrCropd(
            keys=["image", "label"],
            spatial_size=list(cfg.spatial_size),
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])


def _build_tio_transforms(aug: DictConfig) -> tio.Compose:
    return tio.Compose([
        tio.RandomBiasField(p=aug.random_bias_field_prob),
        tio.RandomMotion(p=aug.random_motion_prob),
        tio.RandomGhosting(p=aug.random_ghosting_prob),
        tio.RandomSpike(p=aug.random_spike_prob),
    ])


class _CombinedTransform:
    """Apply TorchIO MRI artefact augmentations then the MONAI spatial/intensity pipeline.

    Affine metadata is extracted from the input MetaTensors before passing through
    TorchIO (which returns plain tensors) and re-attached so MONAI spatial transforms
    (Spacingd, Orientationd) receive the correct spatial information.
    """

    def __init__(self, tio_t: tio.Compose, monai_t: Compose) -> None:
        self._tio = tio_t
        self._monai = monai_t

    def __call__(self, sample: dict) -> dict:
        img, lbl = sample["image"], sample["label"]

        img_tensor = img.as_tensor() if isinstance(img, MetaTensor) else img
        lbl_tensor = lbl.as_tensor() if isinstance(lbl, MetaTensor) else lbl
        meta = (
            {"affine": img.meta["affine"]}
            if isinstance(img, MetaTensor) and "affine" in img.meta
            else {}
        )

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor),
            label=tio.LabelMap(tensor=lbl_tensor.float()),
        )
        subject = self._tio(subject)

        aug_img = subject["image"].data
        aug_lbl = subject["label"].data.long()

        if meta:
            sample["image"] = MetaTensor(aug_img, meta=meta)
            sample["label"] = MetaTensor(aug_lbl, meta=meta)
        else:
            sample["image"] = aug_img
            sample["label"] = aug_lbl

        return self._monai(sample)