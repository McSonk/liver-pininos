"""
prototype_stage02_dummy_bottleneck.py

Stage 2 prototype for the 2.5D Mamba-hybrid architecture.

This validates only the Stage 2 shape contract:

    Stage 1:
        Axial slices: (B * Z, C, X, Y)

    Stage 2:
        Dummy 2D encoder bottleneck:
        (B * Z, BOTTLENECK_CHANNELS, BOTTLENECK_X, BOTTLENECK_Y)

This deliberately does not include:
    - the real 2D encoder
    - skip connection caching
    - Mamba
    - fusion
    - the decoder
    - Stage 7/8
    - Stage 11 logits logic

The dummy down path used here is only a placeholder. It uses average pooling
and channel repetition to mimic the expected spatial/channel schedule. It is
not a real encoder and must not be reused as one.

Run:
    python prototype_stage02_dummy_bottleneck.py
"""

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small non-cubic shapes.
#
# X and Y are chosen to be divisible by 16 because Stage 2 performs four
# stride-2 downsampling steps:
#
#   X -> X/2 -> X/4 -> X/8 -> X/16
#   Y -> Y/2 -> Y/4 -> Y/8 -> Y/16
#
# Z does not need to be divisible by 16 because the 2D encoder does not
# downsample along Z. Z is only the sequence/slice dimension.
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
    Params
    -----
    features: torch.Tensor
        Input features of shape (N, C, H, W)
    base_channels: int
        The base channel count. The output channel count will be
        base_channels * 2**num_downs.
    num_downs: int
        The number of downsampling steps. Each step halves the spatial size
        and doubles the channel count.
    Returns
    -------
    torch.Tensor
        Output features of shape (N, base_channels * 2**num_downs, H / 2**num_downs, W / 2**num_downs)
    Raises
    ------
    ValueError
        If the input spatial dimensions are not divisible by 2 at every downsampling step.
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


def main() -> None:
    print("=" * 80)
    print("Stage 2 prototype: dummy 2D encoder bottleneck")
    print("=" * 80)

    print("Intended shapes:")
    print(f"    Batch size (B):                  {B}")
    print(f"    Channels (C):                    {C}")
    print(f"    Spatial dimensions (X, Y, Z):    ({X}, {Y}, {Z})")
    print(f"    Number of downsamples:           {NUM_DOWNS}")
    print(f"    Base channels:                   {BASE_CHANNELS}")
    print(f"    Bottleneck channels:             {BOTTLENECK_CHANNELS}")
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
    #
    # This is repeated here so the Stage 2 prototype remains standalone.
    # -------------------------------------------------------------------------
    slices = split_volume_to_axial_slices(volume)

    print("Stage 1")
    print(f"    Axial slices: {tuple(slices.shape)}")
    assert slices.shape == (B * Z, C, X, Y)

    # -------------------------------------------------------------------------
    # Verify Stage 1 row order again.
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
    #
    # The real encoder will process each row independently:
    #
    #     (B * Z, C, X, Y)
    #
    # and produce:
    #
    #     (B * Z, BOTTLENECK_CHANNELS, BOTTLENECK_X, BOTTLENECK_Y)
    #
    # The first dimension must remain B * Z, and row i must still correspond
    # to the same axial slice.
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

    assert bottleneck.shape[0] == slices.shape[0], (
        "Stage 2 changed the merged slice dimension. "
        f"Stage 1 had {slices.shape[0]} rows, Stage 2 has {bottleneck.shape[0]} rows."
    )

    # -------------------------------------------------------------------------
    # Row-order canary for the placeholder down path.
    #
    # This proves that the placeholder Stage 2 operation does not mix or
    # reorder the merged slice rows.
    #
    # Each input row is filled with a unique constant equal to its row index.
    # After the placeholder down path, each output row should still contain
    # that same constant value.
    # -------------------------------------------------------------------------
    print("    verifying Stage 2 row-order canary...")

    row_ids = torch.arange(B * Z, dtype=torch.float32)

    canary = (
        row_ids.view(B * Z, 1, 1, 1)
        .expand(B * Z, BASE_CHANNELS, X, Y)
        .contiguous()
    )

    canary_bottleneck = dummy_down_path(
        canary,
        base_channels=BASE_CHANNELS,
        num_downs=NUM_DOWNS,
    )

    assert canary_bottleneck.shape == bottleneck.shape, (
        f"Canary bottleneck shape mismatch. Expected {tuple(bottleneck.shape)}, "
        f"got {tuple(canary_bottleneck.shape)}."
    )

    row_means = canary_bottleneck.mean(dim=(1, 2, 3))

    assert torch.allclose(row_means, row_ids, rtol=0.0, atol=1e-5), (
        "Stage 2 row-order canary failed: output rows no longer correspond "
        "to the original merged slice rows."
    )

    print("    Stage 2 row-order canary verified.")

    # -------------------------------------------------------------------------
    # Spot-check the row mapping explicitly.
    # -------------------------------------------------------------------------
    for b in (0, B - 1):
        for z in (0, Z - 1):
            row = b * Z + z
            assert row_means[row].item() == float(row), (
                f"Row-order mismatch at b={b}, z={z}, row={row}."
            )

    print("    Row mapping verified: row = b * Z + z")

    print("=" * 80)
    print("Stage 2 prototype passed.")
    print("Next step: prototype Stage 3 and Stage 4:")
    print("    global average pool -> sequence reshape for Mamba")
    print("=" * 80)


if __name__ == "__main__":
    main()
