"""
Validates that Mamba2 preserves the sequence shape contract under forward pass.
"""

import pytest
import torch

from mamba_ssm import Mamba2


@pytest.mark.parametrize("B,Z,D_MODEL", [
    (2, 16, 64),     # small prototype shape
    (1, 32, 128),    # different sequence length
    (4, 8, 32),      # larger batch, shorter sequence
])
def test_mamba2_forward_preserves_shape(B: int, Z: int, D_MODEL: int) -> None:
    """Mamba2 forward must return (B, Z, D_MODEL) unchanged."""
    device = torch.device("cuda")
    model = Mamba2(d_model=D_MODEL).to(device).eval()

    seq = torch.randn(B, Z, D_MODEL, device=device, dtype=torch.float32).contiguous()

    with torch.no_grad():
        out = model(seq)

    assert out.shape == (B, Z, D_MODEL), (
        f"Expected {(B, Z, D_MODEL)}, got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "Mamba2 output contains NaN or inf."


def test_mamba2_forward_batch_independence() -> None:
    """Processing rows together must equal processing them individually."""
    device = torch.device("cuda")
    B, Z, D_MODEL = 2, 16, 64

    model = Mamba2(d_model=D_MODEL).to(device).eval()
    seq = torch.randn(B, Z, D_MODEL, device=device).contiguous()

    with torch.no_grad():
        batched = model(seq)
        individual = torch.cat([model(seq[i : i + 1]) for i in range(B)], dim=0)

    assert torch.allclose(batched, individual, rtol=1e-4, atol=1e-5), (
        "Mamba2 batched output differs from individual row outputs."
    )
