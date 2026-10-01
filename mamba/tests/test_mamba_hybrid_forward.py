"""
CUDA-only tests for MambaHybrid with the real Mamba2 z-context path.

These tests require CUDA and mamba_ssm. They must be run on the server using
~/mamba-env, not ~/denv and not the local CPU environment.

Run on the server:
    ~/mamba-env/bin/python -m pytest mamba/tests/test_mamba_hybrid_forward.py -v
"""

import pytest
import torch

pytest.importorskip("mamba_ssm")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MambaHybrid use_z_context=True requires CUDA.",
)

from idssp.sonk.model.mamba_hybrid import MambaHybrid


# Non-cubic shapes to prevent cubic tensors from hiding axis bugs.
# X and Y must be divisible by 2^num_downs.
_TEST_CASES = [
    # (B, C, X, Y, Z, base_channels, num_downs)
    (2, 1, 32, 48, 7, 16, 4),
    (1, 1, 16, 32, 5, 16, 4),
    (3, 1, 16, 16, 4, 8, 3),
]


def _build_cuda_model(
    base_channels: int,
    num_downs: int,
) -> MambaHybrid:
    """Build a deterministic MambaHybrid model on CUDA."""
    torch.manual_seed(0)
    return MambaHybrid(
        in_channels=1,
        num_classes=3,
        base_channels=base_channels,
        num_downs=num_downs,
        use_z_context=True,
        norm_type="group",
        norm_num_groups=8,
    ).to("cuda").eval()


@pytest.mark.parametrize("b,c,x,y,z,base_channels,num_downs", _TEST_CASES)
def test_forward_shape_use_z_context_true(b, c, x, y, z, base_channels, num_downs):
    """The z-context path must preserve the external MONAI tensor contract."""
    model = _build_cuda_model(base_channels, num_downs)

    torch.manual_seed(42)
    volume = torch.randn(b, c, x, y, z, device="cuda")

    with torch.no_grad():
        output = model(volume)

    assert output.shape == (b, 3, x, y, z), (
        f"Expected output shape {(b, 3, x, y, z)}, got {tuple(output.shape)}."
    )
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("base_channels,num_downs", [(16, 4), (8, 3)])
def test_z_context_module_contract(base_channels, num_downs):
    """Stage 5 and Stage 8 must have the settled channel contract."""
    c_bot = base_channels * (2 ** num_downs)

    model = _build_cuda_model(base_channels, num_downs)

    assert hasattr(model, "z_sequence")
    assert hasattr(model, "fusion_conv")

    assert model.fusion_conv.in_channels == 2 * c_bot
    assert model.fusion_conv.out_channels == c_bot
    assert model.fusion_conv.kernel_size == (1, 1)


@pytest.mark.parametrize("b,c,x,y,z,base_channels,num_downs", _TEST_CASES)
def test_volume_permutation_equivariance_use_z_context_true(
    b, c, x, y, z, base_channels, num_downs
):
    """
    Permuting whole volumes in the batch must permute the output identically.

    This is a volume-level row-order check. It preserves the z sequence within
    each volume, which is the correct equivariance property for a causal
    sequence model along z.
    """
    if b < 2:
        pytest.skip("Batch permutation test requires at least two volumes.")

    model = _build_cuda_model(base_channels, num_downs)

    torch.manual_seed(42)
    volume = torch.randn(b, c, x, y, z, device="cuda")

    perm = torch.arange(b - 1, -1, -1, device="cuda")

    with torch.no_grad():
        output_orig = model(volume)
        output_perm = model(volume[perm])

    assert torch.allclose(output_perm, output_orig[perm], atol=1e-4), (
        "MambaHybrid z-context path is not equivariant to volume permutation. "
        "Batch or z-context row order may be corrupted."
    )
