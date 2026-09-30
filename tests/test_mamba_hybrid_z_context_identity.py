"""
Tests for the MambaHybrid identity z-context scaffold.

These tests cover the `use_z_context=True` scaffold path where Stage 5 is an
`nn.Identity()` placeholder. They are CPU-only and do not require `mamba_ssm`.

The z-flip test is identity-specific. Once the real causal Mamba2 block is
introduced, that test will need to be revised because a causal sequence model
is not generally equivariant to reversing the z-axis.

Run:
    ~/envs/dev-thesis/bin/python -m pytest tests/test_mamba_hybrid_z_context_identity.py -v
"""

import torch
import pytest

from idssp.sonk.model.mamba_hybrid import MambaHybrid


# Non-cubic shapes to prevent cubic tensors from hiding axis bugs.
# X and Y must be divisible by 2^num_downs.
_TEST_CASES = [
    # (B, C, X, Y, Z, base_channels, num_downs)
    (2, 1, 32, 48, 7, 16, 4),   # non-cubic, non-power-of-2 Z
    (1, 1, 16, 32, 5, 16, 4),   # single volume, asymmetric X/Y
    (3, 1, 16, 16, 4, 8, 3),    # different base_channels, fewer downs
]


def _build_model(
    base_channels: int,
    num_downs: int,
    use_z_context: bool,
) -> MambaHybrid:
    """Build a deterministic MambaHybrid model in eval mode."""
    torch.manual_seed(0)
    return MambaHybrid(
        in_channels=1,
        num_classes=3,
        base_channels=base_channels,
        num_downs=num_downs,
        use_z_context=use_z_context,
        norm_type="group",
        norm_num_groups=8,
    ).eval()


@pytest.mark.parametrize("b,c,x,y,z,base_channels,num_downs", _TEST_CASES)
def test_forward_shape_z_context_true(b, c, x, y, z, base_channels, num_downs):
    """The scaffold path must preserve the external MONAI tensor contract."""
    model = _build_model(base_channels, num_downs, use_z_context=True)

    torch.manual_seed(42)
    volume = torch.randn(b, c, x, y, z)

    with torch.no_grad():
        output = model(volume)

    assert output.shape == (b, 3, x, y, z), (
        f"Expected output shape {(b, 3, x, y, z)}, got {tuple(output.shape)}."
    )
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("b,c,x,y,z,base_channels,num_downs", _TEST_CASES)
def test_forward_shape_z_context_false(b, c, x, y, z, base_channels, num_downs):
    """The ablation path must also preserve the external MONAI tensor contract."""
    model = _build_model(base_channels, num_downs, use_z_context=False)

    torch.manual_seed(42)
    volume = torch.randn(b, c, x, y, z)

    with torch.no_grad():
        output = model(volume)

    assert output.shape == (b, 3, x, y, z), (
        f"Expected output shape {(b, 3, x, y, z)}, got {tuple(output.shape)}."
    )
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("base_channels,num_downs", [(16, 4), (8, 3)])
def test_fusion_conv_contract(base_channels, num_downs):
    """Stage 8 fusion conv must map 2*C_bot channels back to C_bot channels."""
    c_bot = base_channels * (2 ** num_downs)

    model_true = _build_model(base_channels, num_downs, use_z_context=True)
    assert isinstance(model_true.z_context, torch.nn.Identity)
    assert model_true.fusion_conv.in_channels == 2 * c_bot
    assert model_true.fusion_conv.out_channels == c_bot
    assert model_true.fusion_conv.kernel_size == (1, 1)

    model_false = _build_model(base_channels, num_downs, use_z_context=False)
    assert not hasattr(model_false, "z_context")
    assert not hasattr(model_false, "fusion_conv")


@pytest.mark.parametrize("b,c,x,y,z,base_channels,num_downs", _TEST_CASES)
def test_batch_permutation_equivariance_z_context_true(
    b, c, x, y, z, base_channels, num_downs
):
    """
    Permuting whole volumes in the batch must permute the output identically.

    This is a volume-level row-order check. It is appropriate for the z-context
    scaffold because the sequence axis belongs to each volume.
    """
    if b < 2:
        pytest.skip("Batch permutation test requires at least two volumes.")

    model = _build_model(base_channels, num_downs, use_z_context=True)

    torch.manual_seed(42)
    volume = torch.randn(b, c, x, y, z)

    perm = torch.arange(b - 1, -1, -1)

    with torch.no_grad():
        output_orig = model(volume)
        output_perm = model(volume[perm])

    assert torch.allclose(output_perm, output_orig[perm], atol=1e-5), (
        "MambaHybrid z-context scaffold is not equivariant to volume permutation. "
        "Batch or z-context row order may be corrupted."
    )


@pytest.mark.parametrize("b,c,x,y,z,base_channels,num_downs", _TEST_CASES)
def test_identity_z_flip_equivariance(b, c, x, y, z, base_channels, num_downs):
    """
    Identity-specific z-axis check.

    With an identity Stage 5, reversing the z-axis of the input must reverse
    the z-axis of the output. This validates the split/merge row order together
    with the identity z-context path.

    This test must be revised once real causal Mamba2 replaces nn.Identity().
    """
    model = _build_model(base_channels, num_downs, use_z_context=True)

    torch.manual_seed(42)
    volume = torch.randn(b, c, x, y, z)

    with torch.no_grad():
        output_orig = model(volume)
        output_z_flip = model(torch.flip(volume, dims=[4]))

    expected = torch.flip(output_orig, dims=[4])

    assert torch.allclose(output_z_flip, expected, atol=1e-5), (
        "Identity z-context scaffold is not equivariant to z-axis reversal. "
        "Stage 1, Stage 11, or the z-context reshape logic may be corrupted."
    )
