"""src/models/film.py — Feature-wise Linear Modulation for metadata injection."""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Injects scalar metadata into a feature map via affine conditioning.

    Given metadata vector z ∈ R^{meta_dim}, produces (γ, β) ∈ R^{channels}
    and applies:   out = γ * features + β

    Args:
        meta_dim:    Dimension of the metadata input vector.
        channels:    Number of feature map channels to condition.
        hidden_dim:  Width of the MLP that maps metadata → (γ, β).
    """

    def __init__(self, meta_dim: int, channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels * 2),  # γ and β concatenated
        )
        # Initialise to identity (γ=1, β=0) for training stability
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, D, H, W]  (or [B, C, H, W] for 2-D)
            metadata: [B, meta_dim]

        Returns:
            Conditioned feature map, same shape as `features`.
        """
        params = self.mlp(metadata)           # [B, 2C]
        channels = features.shape[1]
        gamma, beta = params[:, :channels], params[:, channels:]

        # Reshape for broadcasting over spatial dims
        n_spatial = features.dim() - 2
        view_shape = (-1, channels) + (1,) * n_spatial
        gamma = gamma.view(view_shape)
        beta = beta.view(view_shape)

        return (1.0 + gamma) * features + beta  # residual init: γ near 0 → identity
