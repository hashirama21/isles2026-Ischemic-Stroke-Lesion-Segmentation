# ISLES'26 — Flux complet des données, du patient au masque de sortie

## Vue d'ensemble

```
IRM T1w brute (.nii.gz)          Métadonnées cliniques (.json)
         │                                  │
         ▼                                  │
  ┌─────────────────────────────────────────┤
  │         PRÉTRAITEMENT (CPU, 1 fois)     │
  │  skull-strip → clip [p1,p99] → Z-score  │
  └─────────────────────────────────────────┘
         │
         ▼
  image.nii.gz [D × H × W]     days_post_stroke, chronicity
         │                                  │
         ▼                                  ▼
  ┌─────────────────────────────────────────────┐
  │            PIPELINE DE DONNÉES             │
  │  MetaTensor [1,D,H,W] + FloatTensor [2]    │
  │  TorchIO (artefacts MRI) → MONAI (spatial) │
  │  Sortie : [1,128,128,128] + [2]            │
  └─────────────────────────────────────────────┘
         │                                  │
         ▼                                  ▼
  ┌─────────────────────────────────────────────┐
  │               MODÈLE (GPU)                 │
  │  image  [B, 1, 128, 128, 128]              │
  │  metadata [B, 2]  ──► FiLM bottleneck      │
  │                                            │
  │  Sortie : logits [B, 2, 128, 128, 128]     │
  └─────────────────────────────────────────────┘
         │
         ▼
  argmax(dim=1) → masque binaire [B, 1, 128, 128, 128]
         │
         ▼
  Sortie finale Grand Challenge : NIfTI uint8 [D, H, W]
  (0 = fond, 1 = lésion ischémique)
```

---

## 1. Entrées du modèle

### 1.1 Image IRM

| Étape | Forme | Type | Description |
|---|---|---|---|
| NIfTI brut | `[D, H, W]` | float32 | T1w, dimensions variables |
| Après skull-strip | `[D, H, W]` | float32 | Cerveau seulement, fond = 0 |
| Après normalisation | `[D, H, W]` | float32 | Z-score dans le masque cérébral |
| Chargement Dataset | `[1, D, H, W]` | MetaTensor | + affine NIfTI embarquée |
| Après transforms | `[1, 128, 128, 128]` | float32 tensor | Isotropique 1 mm, RAS, Z-scoré |
| En batch | `[B, 1, 128, 128, 128]` | float32 | B = 2 train, 1 val/test |

### 1.2 Vecteur de métadonnées cliniques

| Composante | Source JSON | Encodage | Plage |
|---|---|---|---|
| `days_post_stroke_norm` | `DAYS_POST_STROKE` | `clip(jours, 0, 365) / 365` | [0.0, 1.0] |
| `chronicity` | `CHRONICITY` | valeur brute (0=aigu, 1=subaigu, 2=chronique) | {0, 1, 2} |

```
metadata : FloatTensor [B, 2]
```

---

## 2. Pipeline de prétraitement (scripts/preprocess.py)

Exécuté **une seule fois** sur les données brutes, résultat sauvegardé sur disque.

```
T1w NIfTI brut
    │
    ▼ HD-BET (ou seuillage intensité si non disponible)
Skull-strip : fond → 0, cerveau conservé
    │
    ▼ Percentile clipping sur le masque cérébral
data = clip(data, percentile_1%, percentile_99%)
    │
    ▼ Z-score dans le masque
data[mask] = (data[mask] - mean) / std
    │
    ▼
image.nii.gz  +  label.nii.gz  +  metadata.json
```

---

## 3. Pipeline de transforms (src/data/transforms.py)

### Ordre d'application — entraînement uniquement

```
sample dict {image: MetaTensor[1,D,H,W], label: MetaTensor[1,D,H,W]}
    │
    ▼ ─── TorchIO (artefacts MRI simulés) ───────────────────
    │   RandomBiasField   p=0.4   (inhomogénéité champ B1)
    │   RandomMotion      p=0.2   (flou de mouvement)
    │   RandomGhosting    p=0.2   (artefact k-space)
    │   RandomSpike       p=0.15  (pic RF)
    │   ⚠ Affine NIfTI re-attachée au MetaTensor après TorchIO
    │
    ▼ ─── MONAI (spatial + intensité) ───────────────────────
    │   Orientationd      → réorientation RAS
    │   Spacingd          → rééchantillonnage isotropique 1 mm
    │   ResizeWithPadOrCropd → [128, 128, 128]
    │   NormalizeIntensityd  → Z-score non-zero channel-wise
    │   RandFlipd            p=0.5  (axe aléatoire)
    │   RandAffined          p=0.3  (rotation ±15°, cisaillement, échelle)
    │   Rand3DElasticd       p=0.2  (déformation élastique)
    │   RandGaussianNoised   p=0.4  std=0.05
    │   RandScaleIntensityd  p=0.5  factor±0.3
    │   RandShiftIntensityd  p=0.5  offset±0.1
    │   RandAdjustContrastd  p=0.3  gamma ∈ [0.7, 1.5]
    │
    ▼
{image: [1,128,128,128], label: [1,128,128,128], metadata: [2]}
```

**Validation / Test** : uniquement Orientation → Spacing → Resize → Normalize (déterministe).

---

## 4. Architecture des modèles

Trois architectures disponibles, toutes avec le même contrat d'interface :

```
forward(image: [B, 1, 128, 128, 128], metadata: [B, 2])
    → logits: [B, 2, 128, 128, 128]
```

---

### 4.1 SegResNetFiLM (architecture par défaut)

```
image [B, 1, 128, 128, 128]
    │
    ▼ Encodeur CNN résiduel (down_layers)
    │
    ├─► E1  [B, 32,  128, 128, 128]
    ├─► E2  [B, 64,   64,  64,  64]
    ├─► E3  [B, 128,  32,  32,  32]
    ▼
    E4 (bottleneck) [B, 256, 16, 16, 16]
    │
    │   ◄── FiLM hook intercepts ici ──────────────────────────┐
    │   metadata [B, 2]                                         │
    │       │                                                   │
    │       ▼ MLP  Linear(2→64) → SiLU → Linear(64→512)       │
    │   params [B, 512]  →  γ [B, 256, 1,1,1]  β [B, 256, 1,1,1]
    │       │                                                   │
    │   out = (1 + γ) × E4 + β    [B, 256, 16, 16, 16]       │
    │   ↑ le décodeur reçoit ce tenseur conditionné ───────────┘
    │
    ▼ Décodeur CNN résiduel (up_layers) avec skip connections
    │
    ▼
logits [B, 2, 128, 128, 128]
```

**Paramètres clés** : `init_filters=32`, `blocks_down=[1,2,2,4]` → bottleneck = 32 × 2³ = 256 canaux.

---

### 4.2 SwinUNETRWrapper

```
image [B, 1, 128, 128, 128]
    │
    ▼ Swin Transformer — partition en fenêtres 3D
    │   feature_size = 48, 4 niveaux de résolution
    │
    ▼ encoder10 (bottleneck) [B, 384, 4, 4, 4]
    │   ◄── FiLM hook (même mécanique que SegResNet)
    │       metadata [B, 2] → γ, β [B, 384, 1,1,1]
    │       out = (1 + γ) × feats + β
    │
    ▼ Décodeur CNN avec skip connections
    │
    ▼
logits [B, 2, 128, 128, 128]
```

---

### 4.3 HybridCNNMambaUNet

```
image [B, 1, 128, 128, 128]
    │
    ▼ Encodeur CNN
    E1 [B, 32,  128,128,128]
    E2 [B, 64,   64, 64, 64]   stride=2
    E3 [B, 128,  32, 32, 32]   stride=2
    E4 [B, 256,  16, 16, 16]   stride=2
    │
    ▼ Aplatissement spatial → séquence
    x [B, N=4096, 256]   (N = 16×16×16)
    │
    │   metadata [B, 2]
    │       ▼ Linear(2→256) → SiLU
    │   meta_feat [B, 1, 256]
    │   x = x + meta_feat   (biais additif sur toute la séquence)
    │
    ▼ Mamba SSM blocks (×2)   (modélisation de contexte global)
    x [B, N, 256]
    │
    ▼ Reshape → [B, 256, 16, 16, 16]
    │
    ▼ Décodeur CNN + skip connections
    up3 → cat(E3) → dec3
    up2 → cat(E2) → dec2
    up1 → cat(E1) → dec1
    │
    ▼ Conv 1×1×1
logits [B, 2, 128, 128, 128]
```

---

## 5. Mécanisme FiLM (Feature-wise Linear Modulation)

**Objectif** : conditionner le réseau sur l'âge de la lésion pour adapter la prédiction
aux différences d'apparence IRM entre lésions aiguës et chroniques.

```
metadata [B, 2]
    │
    ▼ MLP
    Linear(2  → 64)   → SiLU
    Linear(64 → 2C)            C = nb canaux bottleneck

params [B, 2C]
    │
    ├─► γ = params[:, :C]  → reshape [B, C, 1, 1, 1]
    └─► β = params[:, C:]  → reshape [B, C, 1, 1, 1]

features [B, C, D, H, W]
    │
    ▼
out = (1 + γ) × features + β
```

**Initialisation** : poids et biais de la dernière couche = 0 → γ=0, β=0 au départ
→ identité garantie en début d'entraînement, convergence stable.

---

## 6. Fonction de perte (src/training/losses.py)

```
logits  [B, 2, D, H, W]     (sorties brutes du modèle)
targets [B, 1, D, H, W]     (masque binaire ground truth, long)

L = w_dice × DiceLoss(softmax(logits), onehot(targets))
  + w_ce   × CrossEntropy(logits, targets)
  + w_focal× FocalLoss(softmax(logits), onehot(targets))   [si w_focal > 0]

Poids par défaut : w_dice=0.5, w_ce=0.5, w_focal=0.0
```

- **DiceLoss** : gère le déséquilibre de classes (lésion << fond), `include_background=False`
- **CrossEntropy** : signal de gradient dense voxel par voxel
- **FocalLoss** : optionnel, amplifie les petites lésions difficiles

---

## 7. Inférence (src/inference/predict.py)

### 7.1 Sliding Window Inference

Les volumes 128³ en entrée peuvent être plus grands en test. L'inférence découpe le volume en fenêtres qui se chevauchent.

```
Volume test [1, 1, D, H, W]   (D,H,W peuvent être > 128)
    │
    ▼ Sliding Window (roi=128³, overlap=50%, mode=gaussian)
    │   ┌──────────────────┐  ┌──────────────────┐
    │   │  patch 128³ #1   │  │  patch 128³ #2   │  ...
    │   └──────────────────┘  └──────────────────┘
    │         │ modèle(patch, metadata)
    │         ▼ logits [1, 2, 128, 128, 128]
    │
    ▼ Fusion gaussienne des prédictions chevauchantes
logits reconstruits [1, 2, D, H, W]
```

### 7.2 Test-Time Augmentation (TTA) — 8 passes

```
Pour chaque transformation de flip :
    [], [D], [H], [W], [D,H], [D,W], [H,W], [D,H,W]

    ┌─► flip(image, dims)
    │       ▼ sliding_window_inference(...)
    │   logits flippés
    │       ▼ softmax → probs
    │       ▼ flip inverse
    └── probs alignées [1, 2, D, H, W]

Moyenne des 8 probabilités
    ▼
probs_tta [1, 2, D, H, W]
```

### 7.3 Ensemble multi-fold

```
checkpoint_fold0  checkpoint_fold1  checkpoint_fold2  ...
       │                │                │
       ▼                ▼                ▼
  probs_tta0       probs_tta1       probs_tta2
       │                │                │
       └────────────────┴────────────────┘
                        │
                    mean(axis=0)
                        │
                        ▼
         ensemble_probs [1, 2, D, H, W]
```

---

## 8. Sortie finale

```
ensemble_probs [1, 2, D, H, W]   float32, probabilités softmax
    │
    ▼ argmax(dim=1)
mask [1, D, H, W]                long  {0, 1}
    │
    ▼ squeeze + numpy
mask [D, H, W]                   uint8  {0, 1}
    │
    ▼ nib.Nifti1Image(mask, affine_originale)
output.nii.gz                    NIfTI uint8
    │
    0 = tissu sain / fond
    1 = lésion ischémique
```

**L'affine originale de l'image d'entrée est préservée** → le masque de sortie est dans
le même espace voxel que l'image soumise, directement superposable.

---

## 9. Métriques d'évaluation (src/evaluation/metrics.py)

| Métrique | Calcul | Unité | Idéal |
|---|---|---|---|
| **Dice** | `2×|P∩G| / (|P|+|G|)` | — | 1.0 |
| **Lesion F1** | détection IoU > 0.1 par lésion | — | 1.0 |
| **AVD** | `|voxels_pred − voxels_gt| × 0.001` | mL | 0.0 |
| **HD95** | 95e percentile distances surfaces | mm | 0.0 |
| **ASSD** | moyenne symétrique distances surfaces | mm | 0.0 |

Convention : prédiction **et** ground truth vides → toutes les métriques = score parfait.

---

## 10. Résumé des dimensions clés

```
Entrée modèle  :  image    [B,  1, 128, 128, 128]  float32
                  metadata [B,  2]                  float32

Bottleneck     :  SegResNet   [B, 256,  16,  16,  16]
                  SwinUNETR   [B, 384,   4,   4,   4]
                  MambaUNet   [B, 256,  16,  16,  16] (après sequence → spatial)

Logits         :  [B,  2, 128, 128, 128]  float32  (canal 0=fond, canal 1=lésion)

Probs softmax  :  [B,  2, 128, 128, 128]  float32  ∈ [0, 1], somme = 1 par voxel

Sortie finale  :  [D,  H,  W]             uint8    {0 = fond, 1 = lésion}
```
