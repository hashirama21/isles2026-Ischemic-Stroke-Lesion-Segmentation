# ISLES'26 — Ischemic Stroke Lesion Segmentation

Production-grade PyTorch Lightning pipeline for the ISLES'26 Grand Challenge.  
Single-modality T1w MRI · ~2 000 scans · 60+ centres · Acute / Sub-acute / Chronic.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/isles2026-Ischemic-Stroke-Lesion-Segmentation/blob/main/notebooks/02_colab_e2e.ipynb)

## Project structure


## Quick start

```bash
pip install -e ".[dev]"

python scripts/preprocess.py data.raw_dir=data/raw data.out_dir=data/processed

python scripts/make_splits.py splits.n_folds=5

python scripts/train.py fold=0                          # NNUNet + FiLM (default)
python scripts/train.py experiment=segresnet_baseline fold=0  # SegResNet baseline

python scripts/evaluate.py checkpoint=outputs/checkpoints/fold0/best.ckpt fold=0

python scripts/ensemble.py checkpoints_dir=outputs/checkpoints

bash docker/build.sh
```

## Reproducing top results

| Model                          | Dice (5-fold CV) | Notes                                   |
|--------------------------------|------------------|-----------------------------------------|
| **NNUNetWrapper + FiLM + DS**  | ~0.68            | Default — recommended                   |
| SegResNet + FiLM               | ~0.67            | Stable alternative                      |
| Ensemble (3×) + TTA            | **~0.74**        | Recommended Grand Challenge submission  |
| Hybrid CNN-Mamba *(exp.)*      | ~0.72            | Experimental — verify CUDA compat. first|

## Key design choices

- **NNUNet-style DynUNet** (6-level, 128→4³ bottleneck) as default backbone
- **Deep supervision** during training — 4 output levels with weights `[1.0, 0.5, 0.25, 0.125]`
- **FiLM conditioning** injects `DAYS_POST_STROKE` + `CHRONICITY` into the bottleneck via a forward hook
- **Weighted lesion sampler** over-samples subjects with small lesions (< 1 mL) at weight 3×
- **Adaptive post-processing** — phase-dependent threshold (0.45 acute / 0.35 chronic), connected-component filtering (< 0.05 mL removed), hole filling
- **Stratified evaluation** by phase, lesion size, and acquisition centre
- **Centre-stratified GroupKFold** ensures the validation set mimics distribution shift
- **Hydra** + **PyTorch Lightning** + **MONAI** + **TorchIO** + **W&B**

## Citation

```bibtex
@software{isles26_pipeline,
  title  = {{ISLES'26 Ischemic Stroke Lesion Segmentation Pipeline}},
  year   = {2026},
  url    = {https://github.com/hashirama21/isles2026-Ischemic-Stroke-Lesion-Segmentation},
  license = {Apache-2.0}
}
```
