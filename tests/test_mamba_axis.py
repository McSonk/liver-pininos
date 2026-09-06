"""
CPU-only unit tests for Mamba-hybrid axis handling.

These tests are deliberately:

    fast
    synthetic
    non-cubic
    independent of LiTS data
    independent of real preprocessing
    independent of mamba_ssm

They verify the external spatial contract:

    input:  (B, C, X, Y, Z)
    output: (B, K, X, Y, Z)

where Z is the last spatial axis and is used as the slice/sequence axis.
"""

import pytest
import torch

from idssp.sonk.model.mamba_axis import (AxialSliceMeta, merge_axial_slices,
                                         split_into_axial_slices)


def test_split_shape_non_cubic() -> None:
    """
    The split operation should treat the last spatial axis as the slice axis.
    """
    volume = torch.randn(2, 1, 16, 24, 32)

    slices, meta = split_into_axial_slices(volume)

    assert slices.shape == (64, 1, 16, 24)
    assert meta.batch_size == 2
    assert meta.x == 16
    assert meta.y == 24
    assert meta.z == 32


def test_split_maps_last_spatial_axis_to_slice_dimension() -> None:
    """
    Verify actual axis mapping, not only shape.

    For input shape (1, 1, X, Y, Z), slice z should equal volume[..., z].
    """
    x = 2
    y = 3
    z = 4

    volume = torch.arange(1 * 1 * x * y * z, dtype=torch.float32).reshape(
        1, 1, x, y, z
    )

    slices, meta = split_into_axial_slices(volume)

    assert slices.shape == (z, 1, x, y)
    assert meta.z == z

    for slice_index in range(z):
        assert torch.equal(slices[slice_index, 0], volume[0, 0, :, :, slice_index])


def test_merge_restores_external_contract_and_axis_order() -> None:
    """
    Splitting and merging should restore the exact original tensor when the
    number of channels is unchanged.
    """
    volume = torch.arange(2 * 3 * 2 * 3 * 4, dtype=torch.float32).reshape(
        2, 3, 2, 3, 4
    )

    slices, meta = split_into_axial_slices(volume)
    merged = merge_axial_slices(slices, meta)

    assert merged.shape == (2, 3, 2, 3, 4)
    assert torch.equal(merged, volume)


def test_merge_accepts_different_channel_count() -> None:
    """
    Merge should allow a different channel dimension, for example:

        input channels:  1
        output channels: NUM_CLASSES = 3

    This is required because the model output has NUM_CLASSES channels.
    """
    volume = torch.randn(2, 1, 4, 5, 6)

    _, meta = split_into_axial_slices(volume)

    num_classes = 3
    batch_size = meta.batch_size
    z = meta.z

    logits = torch.randn(batch_size * z, num_classes, meta.x, meta.y)

    merged = merge_axial_slices(logits, meta)

    assert merged.shape == (batch_size, num_classes, meta.x, meta.y, z)


def test_split_rejects_non_5d_input() -> None:
    """
    The helper should fail loudly if the input is not a batched 3D volume.
    """
    bad_volume = torch.randn(1, 1, 16, 24)

    with pytest.raises(ValueError):
        split_into_axial_slices(bad_volume)


def test_merge_rejects_wrong_slice_count() -> None:
    """
    Merge should fail if B * Z does not match the first dimension.
    """
    meta = AxialSliceMeta(batch_size=2, x=4, y=5, z=6)

    # Expected slice count is 2 * 6 = 12, but 11 is provided.
    bad_slices = torch.randn(11, 3, 4, 5)

    with pytest.raises(ValueError):
        merge_axial_slices(bad_slices, meta)


def test_merge_rejects_wrong_spatial_shape() -> None:
    """
    Merge should fail if X/Y do not match the stored metadata.
    """
    meta = AxialSliceMeta(batch_size=1, x=4, y=5, z=6)

    # Y is wrong: expected 5, got 9.
    bad_slices = torch.randn(6, 3, 4, 9)

    with pytest.raises(ValueError):
        merge_axial_slices(bad_slices, meta)


@pytest.mark.skip(
    reason=(
        "Mamba-hybrid model not implemented yet. Enable this once the model "
        "factory can instantiate the Mamba model."
    )
)
def test_future_mamba_model_external_contract() -> None:
    """
    Future model-level test.

    Once the Mamba-hybrid model exists, this should be enabled and adapted to
    the real model factory.

    Expected contract:

        input:  (B, C, X, Y, Z)
        output: (B, NUM_CLASSES, X, Y, Z)

    Example:

        model = get_mamba_model()
        model.eval()

        x = torch.randn(1, 1, 16, 24, 32)

        with torch.inference_mode():
            y = model(x)

        assert y.shape == (1, 3, 16, 24, 32)
    """
    # TODO: implement this when we have mamba model
    pass
