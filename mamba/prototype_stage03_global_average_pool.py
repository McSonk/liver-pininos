"""
prototype_stage03_global_average_pool.py

Stage 3 prototype for the 2.5D Mamba-hybrid architecture.

This validates only the Stage 3 shape contract:

    Stage 2:
        Dummy bottleneck:
        (B * Z, BOTTLENECK_CHANNELS, BOTTLENECK_X, BOTTLENECK_Y)

    Stage 3:
        Global average pool over spatial dimensions:
        (B * Z, BOTTLENECK_CHANNELS)

In the full architecture, this produces one vector per axial slice.
That vector will later become one Mamba sequence token, and its channel
dimension becomes Mamba's d_model.

This deliberately does not include:
    - Mamba
    - Stage 4 sequence reshape
    - fusion
    - the decoder
    - Stage 7/8
    - Stage 11 logits logic

Run:
    python prototype_stage03_global_average_pool.py
"""

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small non-cubic shapes.
#
# These match the Stage 2 prototype so the scripts remain comparable.
# X and Y are divisible by 16 because Stage 2 performs four stride-2
# downsampling steps.
# -----------------------------------------------------------------------------
B = 2
C = 1
X = 32
Y = 48
Z = 16

NUM_DOWNS = 4
BASE_CHANNELS = 4

DOWNSAMPLE_FACTOR = 2 ** NUM_DOWNS

BOTTLENECK_CHANNELS = BASE_CHANNELS * DOWNSAMPLE_FACTOR
BOTTLENECK_X = X // DOWNSAMPLE_FACTOR
BOTTLENECK_Y = Y // DOWNSAMPLE_FACTOR

# In the real architecture, Mamba's d_model will be equal to the bottleneck
# channel count after Stage 2 / Stage 3.
D_MODEL = BOTTLENECK_CHANNELS

PERMUTATION_ORDER = (0, 4, 1, 2, 3)  # (B, C, X, Y, Z) -> (B, Z, C, X, Y)


def split_volume_to_axial_slices(volume: torch.Tensor) -> torch.Tensor:
    """
    Stage 1:
    (B, C, X, Y, Z) -> (B * Z, C, X, Y)

    Row order:
        row = b * Z + z
    """
    b, c, x, y, z = volume.shape
    return volume.permute(*PERMUTATION_ORDER).reshape(b * z, c, x, y)


def dummy_down_path(features: torch.Tensor, base_channels: int, num_downs: int) -> torch.Tensor:
    """
    Placeholder Stage 2 down path.

    This is not a real encoder. It only mimics the expected shape schedule:

        - spatial size is halved `num_downs` times
        - channel count is doubled `num_downs` times

    Input:
        (N, C, H, W)

    Output:
        (N, base_channels * 2**num_downs, H / 2**num_downs, W / 2**num_downs)
    """
    rows, channels, height, width = features.shape

    # Expand input channels to the base channel width if needed.
    if channels != base_channels:
        if base_channels % channels != 0:
            raise ValueError(
                f"Cannot expand channel count from {channels} to {base_channels} "
                "using integer repetition."
            )

        features = features.repeat(1, base_channels // channels, 1, 1)

    for _ in range(num_downs):
        if features.shape[-2] % 2 != 0 or features.shape[-1] % 2 != 0:
            raise ValueError(
                "Spatial dimensions must be divisible by 2 at every downsampling "
                f"step. Got shape {tuple(features.shape)}."
            )

        # Halve spatial size.
        features = F.avg_pool2d(features, kernel_size=2, stride=2)

        # Double channel count.
        features = features.repeat(1, 2, 1, 1)

    return features


def global_average_pool(features: torch.Tensor) -> torch.Tensor:
    """
    Stage 3:
    (N, C, H, W) -> (N, C)

    Global average pool over the spatial dimensions H and W.

    The batch dimension N is preserved. In this architecture, N = B * Z,
    so each pooled vector still corresponds to one axial slice row.
    """
    if features.ndim != 4:
        raise ValueError(
            "global_average_pool expects a 4D tensor with shape (N, C, H, W), "
            f"got shape {tuple(features.shape)}."
        )

    return features.mean(dim=(2, 3))


def main() -> None:
    print("=" * 80)
    print("Stage 3 prototype: global average pool")
    print("=" * 80)

    print("Intended shapes:")
    print(f"    Batch size (B):                  {B}")
    print(f"    Channels (C):                    {C}")
    print(f"    Spatial dimensions (X, Y, Z):    ({X}, {Y}, {Z})")
    print(f"    Number of downsamples:           {NUM_DOWNS}")
    print(f"    Base channels:                   {BASE_CHANNELS}")
    print(f"    Bottleneck channels / d_model:   {D_MODEL}")
    print(f"    Bottleneck spatial size:         ({BOTTLENECK_X}, {BOTTLENECK_Y})")

    # -------------------------------------------------------------------------
    # Sanity checks on prototype constants.
    # -------------------------------------------------------------------------
    assert X % DOWNSAMPLE_FACTOR == 0, (
        f"X={X} is not divisible by {DOWNSAMPLE_FACTOR}. "
        "Choose X divisible by 16 for four downsamples."
    )
    assert Y % DOWNSAMPLE_FACTOR == 0, (
        f"Y={Y} is not divisible by {DOWNSAMPLE_FACTOR}. "
        "Choose Y divisible by 16 for four downsamples."
    )

    # -------------------------------------------------------------------------
    # Stage 0: input volume.
    # -------------------------------------------------------------------------
    volume = torch.arange(
        B * C * X * Y * Z,
        dtype=torch.float32,
    ).reshape(B, C, X, Y, Z)

    print("Stage 0")
    print(f"    Input volume: {tuple(volume.shape)}")
    assert volume.shape == (B, C, X, Y, Z)

    # -------------------------------------------------------------------------
    # Stage 1: split into axial slices.
    # -------------------------------------------------------------------------
    slices = split_volume_to_axial_slices(volume)

    print("Stage 1")
    print(f"    Axial slices: {tuple(slices.shape)}")
    assert slices.shape == (B * Z, C, X, Y)

    # -------------------------------------------------------------------------
    # Verify Stage 1 row order.
    #
    # Row order:
    #     row = b * Z + z
    # -------------------------------------------------------------------------
    print("    verifying Stage 1 row order...")
    for b in range(B):
        for z in range(Z):
            row = b * Z + z
            expected_slice = volume[b, :, :, :, z]

            assert slices[row].shape == expected_slice.shape
            assert torch.equal(slices[row], expected_slice), (
                f"Stage 1 row-order mismatch: b={b}, z={z}, row={row}"
            )

    print("    Stage 1 row order verified: row = b * Z + z")

    # -------------------------------------------------------------------------
    # Stage 2: dummy 2D encoder bottleneck.
    # -------------------------------------------------------------------------
    bottleneck = dummy_down_path(
        slices,
        base_channels=BASE_CHANNELS,
        num_downs=NUM_DOWNS,
    )

    print("Stage 2")
    print(f"    Dummy bottleneck: {tuple(bottleneck.shape)}")

    assert bottleneck.shape == (
        B * Z,
        BOTTLENECK_CHANNELS,
        BOTTLENECK_X,
        BOTTLENECK_Y,
    ), (
        f"Stage 2 shape mismatch. Expected "
        f"{(B * Z, BOTTLENECK_CHANNELS, BOTTLENECK_X, BOTTLENECK_Y)}, "
        f"got {tuple(bottleneck.shape)}."
    )

    # -------------------------------------------------------------------------
    # Stage 3: global average pool over spatial dimensions.
    #
    # Input:
    #     (B * Z, BOTTLENECK_CHANNELS, BOTTLENECK_X, BOTTLENECK_Y)
    #
    # Output:
    #     (B * Z, BOTTLENECK_CHANNELS)
    #
    # Each axial slice row becomes one pooled vector.
    # -------------------------------------------------------------------------
    pooled = global_average_pool(bottleneck)

    print("Stage 3")
    print(f"    Pooled slice vectors: {tuple(pooled.shape)}")

    assert pooled.shape == (B * Z, D_MODEL), (
        f"Stage 3 shape mismatch. Expected {(B * Z, D_MODEL)}, "
        f"got {tuple(pooled.shape)}."
    )

    assert pooled.ndim == 2, (
        f"Stage 3 output must be 2D: (B * Z, D_MODEL). Got {pooled.ndim}D."
    )

    assert pooled.shape[0] == bottleneck.shape[0], (
        "Stage 3 changed the merged slice dimension. "
        f"Stage 2 had {bottleneck.shape[0]} rows, Stage 3 has {pooled.shape[0]} rows."
    )

    assert pooled.shape[1] == BOTTLENECK_CHANNELS, (
        "Stage 3 channel dimension must equal the bottleneck channel count. "
        f"Expected {BOTTLENECK_CHANNELS}, got {pooled.shape[1]}."
    )

    # -------------------------------------------------------------------------
    # Stage 3 row-order canary.
    #
    # Global average pooling must reduce spatial dimensions without mixing
    # batch rows. Each output row must still correspond to the same input row.
    #
    # Create a bottleneck where every voxel in row i has value i.
    # After global average pooling, every element of pooled row i should
    # still be i.
    # -------------------------------------------------------------------------
    print("    verifying Stage 3 row-order canary...")

    row_ids = torch.arange(B * Z, dtype=torch.float32)

    row_bottleneck = (
        row_ids.view(B * Z, 1, 1, 1)
        .expand(B * Z, BOTTLENECK_CHANNELS, BOTTLENECK_X, BOTTLENECK_Y)
        .contiguous()
    )

    row_pooled = global_average_pool(row_bottleneck)

    assert row_pooled.shape == (B * Z, D_MODEL), (
        f"Stage 3 canary shape mismatch. Expected {(B * Z, D_MODEL)}, "
        f"got {tuple(row_pooled.shape)}."
    )

    expected_row_pooled = row_ids.unsqueeze(1).expand(B * Z, D_MODEL)

    assert torch.allclose(row_pooled, expected_row_pooled, rtol=0.0, atol=1e-5), (
        "Stage 3 row-order canary failed: pooled vectors no longer correspond "
        "to the original merged slice rows."
    )

    # -------------------------------------------------------------------------
    # Spot-check the row mapping explicitly.
    # -------------------------------------------------------------------------
    for b in (0, B - 1):
        for z in (0, Z - 1):
            row = b * Z + z

            assert row_pooled[row, 0].item() == float(row), (
                f"Stage 3 row-order mismatch at first channel: "
                f"b={b}, z={z}, row={row}."
            )

            assert row_pooled[row, -1].item() == float(row), (
                f"Stage 3 row-order mismatch at last channel: "
                f"b={b}, z={z}, row={row}."
            )

    print("    Stage 3 row-order canary verified.")
    print("    Row mapping verified: row = b * Z + z")

    print("=" * 80)
    print("Stage 3 prototype passed.")
    print("Next step: prototype Stage 4:")
    print("    reshape pooled vectors into a Mamba sequence:")
    print("    (B * Z, D_MODEL) -> (B, Z, D_MODEL)")
    print("=" * 80)


if __name__ == "__main__":
    main()
