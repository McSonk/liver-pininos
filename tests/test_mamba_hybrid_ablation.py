"""
CPU-only tests for the MambaHybrid ablation path.

These tests do not require mamba_ssm and must remain runnable in the baseline
CPU environment.

The z-context path (`use_z_context=True`) is tested separately under
`mamba/tests/`, because it requires CUDA and mamba_ssm.

Run:
    ~/envs/thesis/bin/python -m pytest tests/test_mamba_hybrid_ablation.py -v
"""

import pytest
import torch

from idssp.sonk.model.mamba_hybrid import MambaHybrid


# Non-cubic shapes to prevent cubic tensors from hiding axis bugs.
# X and Y must be divisible by 2^num_downs.
_TEST_CASES = [
    # (B, C, X, Y, Z, base_channels, num_downs)
    (2, 1, 32, 48, 7, 16, 4),
    (1, 1, 16, 32, 5, 16, 4),
    (3, 1, 16, 16, 4, 8, 3),
]


def _build_ablation_model(
    base_channels: int,
    num_downs: int,
) -> MambaHybrid:
    """Build a deterministic CPU-safe MambaHybrid ablation model."""
    torch.manual_seed(0)
    return MambaHybrid(
        in_channels=1,
        num_classes=3,
        base_channels=base_channels,
        num_downs=num_downs,
        use_z_context=False,
        norm_type="group",
        norm_num_groups=8,
    ).eval()


@pytest.mark.parametrize("b,c,x,y,z,base_channels,num_downs", _TEST_CASES)
def test_forward_shape_use_z_context_false(b, c, x, y, z, base_channels, num_downs):
    """The ablation path must preserve the external MONAI tensor contract."""
    model = _build_ablation_model(base_channels, num_downs)

    torch.manual_seed(42)
    volume = torch.randn(b, c, x, y, z)

    with torch.no_grad():
        output = model(volume)

    assert output.shape == (b, 3, x, y, z), (
        f"Expected output shape {(b, 3, x, y, z)}, got {tuple(output.shape)}."
    )
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("base_channels,num_downs", [(16, 4), (8, 3)])
def test_ablation_path_has_no_z_context_modules(base_channels, num_downs):
    """The ablation path must not construct Mamba or fusion modules."""
    model = _build_ablation_model(base_channels, num_downs)

    assert not hasattr(model, "z_sequence")
    assert not hasattr(model, "fusion_conv")


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="This contract test is intended for CPU environments without CUDA.",
)
def test_use_z_context_true_fails_without_required_environment():
    """
    Requesting the Mamba path in an unsupported environment must fail
    immediately. There is no fallback.
    """
    with pytest.raises((ImportError, RuntimeError)):
        MambaHybrid(
            in_channels=1,
            num_classes=3,
            base_channels=16,
            num_downs=4,
            use_z_context=True,
        )
