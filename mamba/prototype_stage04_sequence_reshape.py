"""
prototype_stage04_sequence_reshape.py

Stage 4 prototype for the 2.5D Mamba-hybrid architecture.

This validates only the Stage 4 sequence reshape:

    Stage 3:
        Pooled slice vectors:
        (B * Z, D_MODEL)

    Stage 4:
        Mamba sequence layout:
        (B, Z, D_MODEL)

This is the tensor layout expected by Mamba:

    (batch, sequence_length, d_model)

where:
    batch           = B
    sequence_length = Z
    d_model         = D_MODEL

This deliberately does not include:
    - Mamba
    - fusion
    - the decoder
    - Stage 7/8
    - Stage 11 logits logic

Run:
    python prototype_stage04_sequence_reshape.py
"""

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small non-cubic shapes.
#
# These match the Stage 2 and Stage 3 prototypes so the scripts remain
# comparable.
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
    """
    if features.ndim != 4:
        raise ValueError(
            "global_average_pool expects a 4D tensor with shape (N, C, H, W), "
            f"got shape {tuple(features.shape)}."
        )

    return features.mean(dim=(2, 3))


def reshape_pooled_to_sequence(pooled: torch.Tensor, b: int, z: int) -> torch.Tensor:
    """
    Stage 4:
    (B * Z, D_MODEL) -> (B, Z, D_MODEL)

    This reshape is valid only because Stage 1 established the row order:

        row = b * Z + z

    Therefore, after reshaping:

        seq[b, z] == pooled[b * Z + z]
    """
    rows, d_model = pooled.shape

    assert rows == b * z, (
        f"Stage 4 input row count mismatch. Expected B * Z = {b * z}, "
        f"got {rows}."
    )

    return pooled.reshape(b, z, d_model)


def reshape_sequence_to_pooled(seq: torch.Tensor) -> torch.Tensor:
    """
    Inverse of Stage 4:
    (B, Z, D_MODEL) -> (B * Z, D_MODEL)

    This will later correspond to Stage 6, where the Mamba output is re-merged
    so it can be fused back onto the spatial bottleneck.
    """
    b, z, d_model = seq.shape
    return seq.reshape(b * z, d_model)


def main() -> None:
    print("=" * 80)
    print("Stage 4 prototype: sequence reshape for Mamba")
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
    # -------------------------------------------------------------------------
    pooled = global_average_pool(bottleneck)

    print("Stage 3")
    print(f"    Pooled slice vectors: {tuple(pooled.shape)}")

    assert pooled.shape == (B * Z, D_MODEL), (
        f"Stage 3 shape mismatch. Expected {(B * Z, D_MODEL)}, "
        f"got {tuple(pooled.shape)}."
    )

    # -------------------------------------------------------------------------
    # Stage 4: reshape pooled vectors into Mamba sequence layout.
    #
    # Input:
    #     (B * Z, D_MODEL)
    #
    # Output:
    #     (B, Z, D_MODEL)
    #
    # Mamba expects:
    #     (batch, sequence_length, d_model)
    # -------------------------------------------------------------------------
    seq = reshape_pooled_to_sequence(pooled, B, Z)

    print("Stage 4")
    print(f"    Mamba sequence: {tuple(seq.shape)}")

    assert seq.shape == (B, Z, D_MODEL), (
        f"Stage 4 shape mismatch. Expected {(B, Z, D_MODEL)}, "
        f"got {tuple(seq.shape)}."
    )

    assert seq.ndim == 3, (
        f"Stage 4 output must be 3D: (B, Z, D_MODEL). Got {seq.ndim}D."
    )

    assert seq.shape[0] == B, (
        f"Stage 4 batch dimension mismatch. Expected B={B}, got {seq.shape[0]}."
    )

    assert seq.shape[1] == Z, (
        f"Stage 4 sequence length must equal Z. Expected Z={Z}, got {seq.shape[1]}."
    )

    assert seq.shape[2] == D_MODEL, (
        f"Stage 4 d_model dimension mismatch. Expected D_MODEL={D_MODEL}, "
        f"got {seq.shape[2]}."
    )

    # -------------------------------------------------------------------------
    # Verify that Stage 4 preserves the Stage 1 row order.
    #
    # For every b and z:
    #     seq[b, z] == pooled[b * Z + z]
    # -------------------------------------------------------------------------
    print("    verifying Stage 4 row order...")
    for b in range(B):
        for z in range(Z):
            row = b * Z + z

            assert torch.equal(seq[b, z], pooled[row]), (
                f"Stage 4 row-order mismatch: b={b}, z={z}, row={row}"
            )

    print("    Stage 4 row order verified: seq[b, z] == pooled[b * Z + z]")

    # -------------------------------------------------------------------------
    # Stage 4 round-trip canary.
    #
    # Use an identifiable pooled tensor to prove:
    #
    #     (B * Z, D_MODEL)
    #         -> (B, Z, D_MODEL)
    #         -> (B * Z, D_MODEL)
    #
    # preserves every element and every row.
    # -------------------------------------------------------------------------
    print("    verifying Stage 4 round-trip canary...")

    pooled_ids = torch.arange(
        B * Z * D_MODEL,
        dtype=torch.float32,
    ).reshape(B * Z, D_MODEL)

    seq_ids = reshape_pooled_to_sequence(pooled_ids, B, Z)

    assert seq_ids.shape == (B, Z, D_MODEL), (
        f"Stage 4 canary sequence shape mismatch. Expected {(B, Z, D_MODEL)}, "
        f"got {tuple(seq_ids.shape)}."
    )

    for b in range(B):
        for z in range(Z):
            row = b * Z + z

            assert torch.equal(seq_ids[b, z], pooled_ids[row]), (
                f"Stage 4 canary row-order mismatch: b={b}, z={z}, row={row}"
            )

    flat_ids = reshape_sequence_to_pooled(seq_ids)

    assert flat_ids.shape == pooled_ids.shape, (
        f"Stage 4 canary round-trip shape mismatch. "
        f"Expected {tuple(pooled_ids.shape)}, got {tuple(flat_ids.shape)}."
    )

    assert torch.equal(pooled_ids, flat_ids), (
        "Stage 4 round-trip failed: reshaping back to (B * Z, D_MODEL) "
        "did not recover the original pooled tensor element-wise."
    )

    print("    Stage 4 round-trip canary verified.")

    print("=" * 80)
    print("Stage 4 prototype passed.")
    print("Next step: prototype Stage 5:")
    print("    real Mamba call on CUDA in ~/mamba-env")
    print("    (B, Z, D_MODEL) -> Mamba -> (B, Z, D_MODEL)")
    print("=" * 80)


if __name__ == "__main__":
    main()
