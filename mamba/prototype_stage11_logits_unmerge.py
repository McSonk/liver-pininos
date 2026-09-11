"""
prototype_stage11_logits_unmerge.py

Stage 11 prototype for the 2.5D Mamba-hybrid architecture.

This validates the logits un-merge round-trip:

    Stage 10 output (slice logits):
        (B * Z, NUM_CLASSES, X, Y)

    Stage 11 un-merge:
        (B * Z, NUM_CLASSES, X, Y)
        -> reshape to (B, Z, NUM_CLASSES, X, Y)
        -> permute to (B, NUM_CLASSES, X, Y, Z)

    Inverse (for round-trip validation):
        (B, NUM_CLASSES, X, Y, Z)
        -> permute to (B, Z, NUM_CLASSES, X, Y)
        -> reshape to (B * Z, NUM_CLASSES, X, Y)

This is the final reshape that converts per-slice 2D logits back into a 3D
volume matching the MONAI convention (B, C, X, Y, Z).

This deliberately does not include:
    - Mamba
    - the encoder
    - the decoder
    - the classification head (Stage 10)
    - softmax or any activations

This script runs on CPU. No CUDA or mamba_ssm required.

Run:
    python prototype_stage11_logits_unmerge.py
"""

import torch


# -----------------------------------------------------------------------------
# Small non-cubic shapes.
#
# These are deliberately non-cubic to expose axis-ordering bugs.
# X, Y here represent the FULL spatial resolution at which the decoder
# produces logits (not the bottleneck resolution).
# -----------------------------------------------------------------------------
B = 2
NUM_CLASSES = 3
X = 16
Y = 24
Z = 32

# Permutation orders matching Stage 1 conventions
# Forward: (B, Z, C, X, Y) -> (B, C, X, Y, Z)
UNMERGE_PERMUTE = (0, 2, 3, 4, 1)

# Inverse: (B, C, X, Y, Z) -> (B, Z, C, X, Y)
MERGE_PERMUTE = (0, 4, 1, 2, 3)


def unmerge_slice_logits_to_volume(slice_logits: torch.Tensor, b: int, z: int) -> torch.Tensor:
    """
    Stage 11:
    (B * Z, NUM_CLASSES, X, Y) -> (B, NUM_CLASSES, X, Y, Z)

    Row order:
        slice_logits[b * Z + z] corresponds to volume[:, :, :, :, z]

    The reshape splits the merged row dimension back into (B, Z), then the
    permute moves Z to the last spatial axis.
    """
    rows, num_classes, x, y = slice_logits.shape

    assert rows == b * z, (
        f"Stage 11 input row count mismatch. Expected B * Z = {b * z}, got {rows}."
    )

    # (B * Z, NUM_CLASSES, X, Y) -> (B, Z, NUM_CLASSES, X, Y)
    reshaped = slice_logits.reshape(b, z, num_classes, x, y)

    # (B, Z, NUM_CLASSES, X, Y) -> (B, NUM_CLASSES, X, Y, Z)
    return reshaped.permute(*UNMERGE_PERMUTE)


def split_volume_to_slice_logits(volume_logits: torch.Tensor) -> torch.Tensor:
    """
    Inverse of Stage 11:
    (B, NUM_CLASSES, X, Y, Z) -> (B * Z, NUM_CLASSES, X, Y)

    This is the same operation as Stage 1's split, applied to logits.
    """
    b, num_classes, x, y, z = volume_logits.shape

    # (B, NUM_CLASSES, X, Y, Z) -> (B, Z, NUM_CLASSES, X, Y)
    permuted = volume_logits.permute(*MERGE_PERMUTE)

    # (B, Z, NUM_CLASSES, X, Y) -> (B * Z, NUM_CLASSES, X, Y)
    return permuted.reshape(b * z, num_classes, x, y)


def main() -> None:
    print("=" * 80)
    print("Stage 11 prototype: logits un-merge round-trip")
    print("=" * 80)

    print("Intended shapes:")
    print(f"    Batch size (B):                  {B}")
    print(f"    Classes (NUM_CLASSES):           {NUM_CLASSES}")
    print(f"    Spatial dimensions (X, Y, Z):    ({X}, {Y}, {Z})")
    print(f"    Merged rows (B * Z):             {B * Z}")
    print(f"    Slice logits shape:              {(B * Z, NUM_CLASSES, X, Y)}")
    print(f"    Volume logits shape:             {(B, NUM_CLASSES, X, Y, Z)}")
    print(f"    Un-merge permute:                {UNMERGE_PERMUTE}")
    print(f"    Merge permute (inverse):         {MERGE_PERMUTE}")

    # =========================================================================
    # Test 1: Shape contract
    # =========================================================================
    print("-" * 80)
    print("Test 1: Shape contract")
    print("-" * 80)

    slice_logits = torch.randn(B * Z, NUM_CLASSES, X, Y)

    print(f"    Input (slice logits):    {tuple(slice_logits.shape)}")
    assert slice_logits.shape == (B * Z, NUM_CLASSES, X, Y)

    volume_logits = unmerge_slice_logits_to_volume(slice_logits, B, Z)

    print(f"    Output (volume logits):  {tuple(volume_logits.shape)}")
    assert volume_logits.shape == (B, NUM_CLASSES, X, Y, Z), (
        f"Stage 11 shape mismatch. Expected {(B, NUM_CLASSES, X, Y, Z)}, "
        f"got {tuple(volume_logits.shape)}."
    )

    # Round-trip back
    slice_logits_back = split_volume_to_slice_logits(volume_logits)

    print(f"    Round-trip (slices):     {tuple(slice_logits_back.shape)}")
    assert slice_logits_back.shape == (B * Z, NUM_CLASSES, X, Y), (
        f"Stage 11 round-trip shape mismatch. Expected {(B * Z, NUM_CLASSES, X, Y)}, "
        f"got {tuple(slice_logits_back.shape)}."
    )

    print("    Shape contract verified.")

    # =========================================================================
    # Test 2: Element-wise round-trip with identifiable values
    # =========================================================================
    print("-" * 80)
    print("Test 2: Element-wise round-trip (arange)")
    print("-" * 80)

    identifiable_logits = torch.arange(
        B * Z * NUM_CLASSES * X * Y,
        dtype=torch.float32,
    ).reshape(B * Z, NUM_CLASSES, X, Y)

    print(f"    Input shape:     {tuple(identifiable_logits.shape)}")
    print(f"    First 5 values:  {identifiable_logits.flatten()[:5].tolist()}")
    print(f"    Last 5 values:   {identifiable_logits.flatten()[-5:].tolist()}")

    # Un-merge to volume
    volume = unmerge_slice_logits_to_volume(identifiable_logits, B, Z)
    print(f"    Volume shape:    {tuple(volume.shape)}")

    # Round-trip back to slices
    round_trip = split_volume_to_slice_logits(volume)
    print(f"    Round-trip:      {tuple(round_trip.shape)}")

    assert torch.equal(identifiable_logits, round_trip), (
        "Stage 11 round-trip failed: element-wise values do not match after "
        "un-merge and re-split."
    )

    print("    Element-wise round-trip verified: all values preserved.")

    # =========================================================================
    # Test 3: Row-order verification
    # =========================================================================
    print("-" * 80)
    print("Test 3: Row-order verification")
    print("-" * 80)

    # For every batch b and slice z, verify that:
    #   slice_logits[b * Z + z] == volume[b, :, :, :, z]
    # where volume = unmerge_slice_logits_to_volume(slice_logits)

    print(f"    Checking {B * Z} rows...")

    for b in range(B):
        for z in range(Z):
            row = b * Z + z

            # The slice at row `row` should end up at volume[b, :, :, :, z]
            expected_slice = identifiable_logits[row]
            actual_slice = volume[b, :, :, :, z]

            assert expected_slice.shape == actual_slice.shape, (
                f"Shape mismatch at b={b}, z={z}, row={row}: "
                f"expected {tuple(expected_slice.shape)}, got {tuple(actual_slice.shape)}"
            )

            assert torch.equal(expected_slice, actual_slice), (
                f"Stage 11 row-order mismatch: b={b}, z={z}, row={row}. "
                f"Volume slice at [b, :, :, :, z] does not match slice_logits[row]."
            )

    print(f"    Row order verified: row = b * Z + z for all {B * Z} rows.")

    # =========================================================================
    # Test 4: Row-order canary with row-index encoding
    # =========================================================================
    print("-" * 80)
    print("Test 4: Row-order canary (row-index encoding)")
    print("-" * 80)

    row_ids = torch.arange(B * Z, dtype=torch.float32)

    # Create logits where every voxel in row i has value i.
    # Shape: (B * Z, NUM_CLASSES, X, Y)
    canary_logits = (
        row_ids
        .view(B * Z, 1, 1, 1)
        .expand(B * Z, NUM_CLASSES, X, Y)
        .contiguous()
    )

    canary_volume = unmerge_slice_logits_to_volume(canary_logits, B, Z)

    assert canary_volume.shape == (B, NUM_CLASSES, X, Y, Z), (
        f"Canary volume shape mismatch. Expected {(B, NUM_CLASSES, X, Y, Z)}, "
        f"got {tuple(canary_volume.shape)}."
    )

    # Verify: for every b and z, all voxels at volume[b, :, :, :, z] should
    # equal the row index b * Z + z.
    for b in range(B):
        for z in range(Z):
            row = b * Z + z
            expected_value = float(row)

            actual = canary_volume[b, :, :, :, z]
            assert torch.allclose(actual, torch.full_like(actual, expected_value), rtol=0.0, atol=1e-6), (
                f"Canary row-order mismatch at b={b}, z={z}, row={row}. "
                f"Expected all values to be {expected_value}."
            )

    print(f"    Row-order canary verified for all {B * Z} rows.")

    # =========================================================================
    # Test 5: Inverse round-trip (volume -> slices -> volume)
    # =========================================================================
    print("-" * 80)
    print("Test 5: Inverse round-trip (volume -> slices -> volume)")
    print("-" * 80)

    # Start from a volume
    identifiable_volume = torch.arange(
        B * NUM_CLASSES * X * Y * Z,
        dtype=torch.float32,
    ).reshape(B, NUM_CLASSES, X, Y, Z)

    print(f"    Volume shape:    {tuple(identifiable_volume.shape)}")

    # Split to slices
    slices = split_volume_to_slice_logits(identifiable_volume)
    print(f"    Slices shape:    {tuple(slices.shape)}")

    # Un-merge back to volume
    volume_back = unmerge_slice_logits_to_volume(slices, B, Z)
    print(f"    Volume back:     {tuple(volume_back.shape)}")

    assert volume_back.shape == identifiable_volume.shape, (
        f"Inverse round-trip shape mismatch. Expected {tuple(identifiable_volume.shape)}, "
        f"got {tuple(volume_back.shape)}."
    )

    assert torch.equal(identifiable_volume, volume_back), (
        "Stage 11 inverse round-trip failed: volume -> slices -> volume did not "
        "preserve element-wise values."
    )

    print("    Inverse round-trip verified: volume -> slices -> volume.")

    # =========================================================================
    # Test 6: MONAI convention compliance
    # =========================================================================
    print("-" * 80)
    print("Test 6: MONAI convention compliance")
    print("-" * 80)

    # The final volume shape must be (B, C, X, Y, Z) which matches MONAI's
    # expected (batch, channels, spatial_0, spatial_1, spatial_2)
    #
    # In this project:
    #   spatial_0 = X (left-right)
    #   spatial_1 = Y (anterior-posterior)
    #   spatial_2 = Z (superior-inferior, the slice axis)
    #
    # This is consistent with Orientationd(axcodes="LAS") in transforms.py

    assert volume_logits.ndim == 5, (
        f"Volume logits must be 5D, got {volume_logits.ndim}D."
    )
    assert volume_logits.shape[0] == B, "Batch dimension mismatch."
    assert volume_logits.shape[1] == NUM_CLASSES, "Channel (class) dimension mismatch."
    assert volume_logits.shape[2] == X, "X spatial dimension mismatch."
    assert volume_logits.shape[3] == Y, "Y spatial dimension mismatch."
    assert volume_logits.shape[4] == Z, "Z (slice) dimension mismatch."

    print("    MONAI convention verified: (B, NUM_CLASSES, X, Y, Z)")
    print("    Z is the last spatial axis (consistent with Orientationd LAS).")

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("Stage 11 prototype passed.")
    print("")
    print("Validated:")
    print("    1. Shape contract: (B*Z, K, X, Y) -> (B, K, X, Y, Z)")
    print("    2. Element-wise round-trip with arange")
    print("    3. Row-order preservation: row = b * Z + z")
    print("    4. Row-order canary with row-index encoding")
    print("    5. Inverse round-trip: volume -> slices -> volume")
    print("    6. MONAI convention: (B, C, X, Y, Z) with Z last")
    print("")
    print("This completes the reshape prototype sequence:")
    print("    Stage 0/1  - Spatial split/merge")
    print("    Stage 2    - Dummy bottleneck")
    print("    Stage 3    - Global average pool")
    print("    Stage 4    - Sequence reshape")
    print("    Stage 5    - Mamba2 forward (fp32)")
    print("    Stage 5b   - Mamba2 AMP forward/backward")
    print("    Stage 6    - Re-merge for fusion")
    print("    Stage 7a   - Broadcast z-context")
    print("    Stage 7b   - Concatenate with bottleneck")
    print("    Stage 8    - 1x1 fusion conv")
    print("    Stage 9    - Placeholder decoder canary")
    print("    Stage 11   - Logits un-merge (this script)")
    print("")
    print("Next step: integrate the validated reshape logic into the real model")
    print("in idssp/sonk/model/models.py.")
    print("=" * 80)


if __name__ == "__main__":
    main()
