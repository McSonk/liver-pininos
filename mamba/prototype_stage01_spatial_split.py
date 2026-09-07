"""
prototype_stage01_spatial_split.py

First minimum prototype for the 2.5D Mamba-hybrid architecture.

This validates only:

    Stage 0:
        Input volume: (B, C, X, Y, Z)

    Stage 1:
        Permute + merge B and Z:
        (B, C, X, Y, Z) -> (B * Z, C, X, Y)

    Inverse:
        (B * Z, C, X, Y) -> (B, C, X, Y, Z)

This deliberately does not include:
    - Mamba
    - the real encoder
    - the real decoder
    - fusion
    - stage 7/8
    - stage 11 logits logic

Run:
    python prototype_stage01_spatial_split.py
"""

import torch


# -----------------------------------------------------------------------------
# Small non-cubic shapes.
#
# Non-cubic dimensions are intentional. They make axis-ordering bugs visible.
# For example, if X and Y are accidentally swapped, the shapes or element-wise
# round-trip assertions will fail.
# -----------------------------------------------------------------------------
B = 2
C = 1
X = 16
Y = 24
Z = 32

PERMUTATION_ORDER = (0, 4, 1, 2, 3)  # (B, C, X, Y, Z) -> (B, Z, C, X, Y)
PERMUTATION_ORDER_INVERSE = (0, 2, 3, 4, 1)  # (B, Z, C, X, Y) -> (B, C, X, Y, Z)


def main() -> None:
    print("=" * 80)
    print("Stage 0/1 prototype: spatial split and merge")
    print("=" * 80)

    print("Intended volume shape:")
    print(f"    Batch size (B): {B}")
    print(f"    Channels (C): {C}")
    print(f"    Spatial dimensions (X, Y, Z): ({X}, {Y}, {Z})")

    # -------------------------------------------------------------------------
    # Stage 0: input patch.
    #
    # MONAI convention used in this project:
    #     (B, C, X, Y, Z)
    #
    # The z-axis is the last spatial axis, i.e. tensor dimension 4.
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
    # We need the 2D encoder to process each axial slice independently.
    #
    # Starting shape:
    #     (B, C, X, Y, Z)
    #
    # Permute to:
    #     (B, Z, C, X, Y)
    #
    # Then merge B and Z:
    #     (B * Z, C, X, Y)
    #
    # Row order:
    #     row = b * Z + z
    # -------------------------------------------------------------------------
    print("Stage 1")
    print(f"    permuted shape (B, Z, C, X, Y): {tuple(volume.permute(*PERMUTATION_ORDER).shape)}")
    slices = volume.permute(*PERMUTATION_ORDER).reshape(B * Z, C, X, Y)

    print(f"    axial slices (B * Z, C, X, Y): {tuple(slices.shape)}")
    assert slices.shape == (B * Z, C, X, Y)

    # -------------------------------------------------------------------------
    # Verify the row order explicitly.
    # The merged tensor has 64 rows.
    #
    # Each row corresponds to one axial slice from one volume.
    # The row order is:
    #    row = b * Z + z
    #
    # For each batch b and slice z, the merged row b * Z + z must contain
    # exactly the slice volume[b, :, :, :, z].
    # -------------------------------------------------------------------------
    print("    verifying row order...")
    for b in range(B):
        for z in range(Z):
            row = b * Z + z
            expected_slice = volume[b, :, :, :, z]

            assert slices[row].shape == expected_slice.shape
            assert torch.equal(slices[row], expected_slice), (
                f"Stage 1 row-order mismatch: b={b}, z={z}, row={row}"
            )

    print("    row order verified: row = b * Z + z")

    # -------------------------------------------------------------------------
    # Inverse of Stage 1: merge axial slices back into a volume.
    #
    # Starting shape:
    #     (B * Z, C, X, Y)
    #
    # Reshape to:
    #     (B, Z, C, X, Y)
    #
    # Permute back to:
    #     (B, C, X, Y, Z)
    # -------------------------------------------------------------------------
    print("Stage 1 inverse")
    merged = slices.reshape(B, Z, C, X, Y).permute(*PERMUTATION_ORDER_INVERSE)

    print(f"    Merged volume:        {tuple(merged.shape)}")
    assert merged.shape == volume.shape

    # -------------------------------------------------------------------------
    # Element-wise round-trip check.
    #
    # This is the important part. Shape equality alone is not enough, because
    # a wrong permutation can still produce the correct shape.
    # -------------------------------------------------------------------------
    print("    verifying element-wise equality...")
    assert torch.equal(volume, merged), (
        "Spatial split/merge round-trip failed: merged volume does not match "
        "the original volume element-wise."
    )
    print("    element-wise equality verified: merged volume matches original volume")

    print("=" * 80)
    print("Stage 0/1 prototype passed.")
    print("Next step: prototype Stage 2 with a dummy bottleneck shape only.")
    print("=" * 80)


if __name__ == "__main__":
    main()
