"""tests/test_models.py — Smoke tests for model forward passes."""
import pytest
import torch


def _make_batch(B=1, D=32, H=32, W=32):
    image = torch.randn(B, 1, D, H, W)
    metadata = torch.rand(B, 2)
    return image, metadata


class TestSegResNetFiLM:
    def test_forward_shape(self):
        from src.models.segresnet import SegResNetFiLM
        model = SegResNetFiLM(in_channels=1, out_channels=2, init_filters=8)
        model.eval()
        image, meta = _make_batch()
        with torch.no_grad():
            out = model(image, meta)
        assert out.shape == (1, 2, 32, 32, 32)

    def test_output_is_finite(self):
        from src.models.segresnet import SegResNetFiLM
        model = SegResNetFiLM(in_channels=1, out_channels=2, init_filters=8)
        model.eval()
        image, meta = _make_batch()
        with torch.no_grad():
            out = model(image, meta)
        assert torch.isfinite(out).all()


class TestHybridMamba:
    def test_forward_shape(self):
        from src.models.mamba_unet import HybridCNNMambaUNet
        model = HybridCNNMambaUNet(
            in_channels=1, out_channels=2, base_filters=8, mamba_depth=1
        )
        model.eval()
        image, meta = _make_batch()
        with torch.no_grad():
            out = model(image, meta)
        assert out.shape == (1, 2, 32, 32, 32)


class TestFiLMLayer:
    def test_identity_init(self):
        """FiLM initialised to zero weight → output ≈ input."""
        from src.models.film import FiLMLayer
        film = FiLMLayer(meta_dim=2, channels=16, hidden_dim=32)
        features = torch.randn(2, 16, 4, 4, 4)
        metadata = torch.rand(2, 2)
        out = film(features, metadata)
        # γ=0, β=0 → (1+0)*features + 0 = features
        assert torch.allclose(out, features, atol=1e-5)
