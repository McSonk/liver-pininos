"""
Axis-handling helpers for the 2.5D Mamba-hybrid architecture.

External tensor contract:

    input volume:     (B, C, X, Y, Z)
    axial slices:     (B * Z, C, X, Y)
    restored volume:  (B, C, X, Y, Z)

The z-axis is the last spatial axis, matching the deterministic preprocessing
convention after Orientationd(axcodes="LAS").

These helpers are deliberately pure tensor operations. They do not depend on
MONAI, config, LiTS data, or mamba_ssm, so they can be unit-tested on CPU.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AxialSliceMeta:
    """
    Metadata needed to restore a batched 3D volume from per-slice tensors.

    Attributes
    ----------
    batch_size:
        Number of 3D volumes/patches.
    x:
        Left-right spatial size.
    y:
        Posterior-anterior spatial size.
    z:
        Inferior-superior spatial size. This is the Mamba sequence axis.
    """

    batch_size: int
    x: int
    y: int
    z: int


def split_into_axial_slices(
    volume: torch.Tensor,
) -> tuple[torch.Tensor, AxialSliceMeta]:
    """
    Split a batched 3D volume into independent axial slices.

    Parameters
    ----------
    volume:
        Tensor with shape (B, C, X, Y, Z).

    Returns
    -------
    slices:
        Tensor with shape (B * Z, C, X, Y).
    meta:
        Metadata required to restore the original 3D layout.
    """
    if volume.ndim != 5:
        raise ValueError(
            "Expected a 5D tensor with shape (B, C, X, Y, Z), "
            f"got shape {tuple(volume.shape)}."
        )

    batch_size, channels, x, y, z = volume.shape
    meta = AxialSliceMeta(batch_size=batch_size, x=x, y=y, z=z)

    # (B, C, X, Y, Z) -> (B, Z, C, X, Y) -> (B * Z, C, X, Y)
    slices = volume.permute(0, 4, 1, 2, 3).reshape(batch_size * z, channels, x, y)

    return slices, meta


def merge_axial_slices(
    slices: torch.Tensor,
    meta: AxialSliceMeta,
) -> torch.Tensor:
    """
    Restore a batched 3D volume from independent axial slices.

    Parameters
    ----------
    slices:
        Tensor with shape (B * Z, K, X, Y), where K may be the number of
        feature channels or NUM_CLASSES. It does not need to match the input
        channel count.
    meta:
        Metadata produced by split_into_axial_slices.

    Returns
    -------
    volume:
        Tensor with shape (B, K, X, Y, Z).
    """
    if slices.ndim != 4:
        raise ValueError(
            "Expected a 4D tensor with shape (B * Z, K, X, Y), "
            f"got shape {tuple(slices.shape)}."
        )

    expected_slice_count = meta.batch_size * meta.z

    if slices.shape[0] != expected_slice_count:
        raise ValueError(
            "Slice count does not match AxialSliceMeta. "
            f"Expected B * Z = {expected_slice_count}, "
            f"got {slices.shape[0]}."
        )

    if slices.shape[2] != meta.x or slices.shape[3] != meta.y:
        raise ValueError(
            "Spatial slice shape does not match AxialSliceMeta. "
            f"Expected (X, Y) = ({meta.x}, {meta.y}), "
            f"got ({slices.shape[2]}, {slices.shape[3]})."
        )

    channels = slices.shape[1]

    # (B * Z, K, X, Y) -> (B, Z, K, X, Y) -> (B, K, X, Y, Z)
    volume = slices.reshape(
        meta.batch_size,
        meta.z,
        channels,
        meta.x,
        meta.y,
    ).permute(0, 2, 3, 4, 1)

    return volume
