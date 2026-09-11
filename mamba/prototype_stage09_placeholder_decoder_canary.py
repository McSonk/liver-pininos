"""
prototype_stage09_placeholder_decoder_canary.py

Stage 9 placeholder decoder canary for the 2.5D Mamba-hybrid architecture.

This script validates only the generic property that a convolutional
upsampling path does not reorder the merged slice rows.

It uses a throwaway decoder consisting of one ConvTranspose2d layer.

This is NOT the real decoder. It has:
    - no skip connections
    - no real channel schedule
    - no Mamba branch
    - no encoder
    - no final segmentation head

Its only purpose is to prove that a conv-based upsampling operation preserves
the row order:

    row = b * Z + z

This placeholder canary does NOT retire the obligation to re-run the same
row-order canary against the real Stage 9 decoder once it is implemented.

Run:
    python prototype_stage09_placeholder_decoder_canary.py
"""

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Prototype constants.
#
# These match the row count and bottleneck spatial size used in the previous
# prototypes, but the decoder channel count here is arbitrary.
# -----------------------------------------------------------------------------
B = 2
Z = 16

ROWS = B * Z

D_MODEL = 64
BOTTLENECK_X = 2
BOTTLENECK_Y = 3

# Arbitrary placeholder decoder output channels.
# This is not the real decoder channel schedule.
PLACEHOLDER_OUT_CHANNELS = 8


def make_stage8_like_input(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a Stage-8-shaped tensor where every row has a unique constant value.

    Returns
    -------
    stage8_like:
        Tensor of shape:
            (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

        Every voxel in row i has value i.

    row_ids:
        Tensor of shape:
            (B * Z,)

        The expected row identity for each merged slice row.
    """
    row_ids = torch.arange(ROWS, dtype=torch.float32, device=device)

    stage8_like = (
        row_ids
        .view(ROWS, 1, 1, 1)
        .expand(ROWS, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
        .contiguous()
    )

    return stage8_like, row_ids


def build_placeholder_decoder(device: torch.device) -> nn.ConvTranspose2d:
    """
    Build a throwaway decoder layer.

    One ConvTranspose2d layer is enough to demonstrate the row-order property.
    The real decoder will be deeper and will consume skip connections, but this
    placeholder is deliberately minimal.
    """
    placeholder_decoder = nn.ConvTranspose2d(
        in_channels=D_MODEL,
        out_channels=PLACEHOLDER_OUT_CHANNELS,
        kernel_size=2,
        stride=2,
        bias=False,
    ).to(device)

    placeholder_decoder.eval()

    # Known weights make the canary exact.
    #
    # If every input channel in a row has value `row_id`, and every weight is
    # 1 / D_MODEL, then each output channel also becomes `row_id`.
    with torch.no_grad():
        placeholder_decoder.weight.fill_(1.0 / D_MODEL)

    return placeholder_decoder


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Stage 9 prototype: placeholder decoder row-order canary")
    print("=" * 80)

    print("Environment:")
    print(f"    Device:                          {device}")

    print("Intended shapes:")
    print(f"    Batch size (B):                  {B}")
    print(f"    Slice count (Z):                 {Z}")
    print(f"    Merged rows (B * Z):             {ROWS}")
    print(f"    Input channels (D_MODEL):        {D_MODEL}")
    print(f"    Input spatial size:              ({BOTTLENECK_X}, {BOTTLENECK_Y})")
    print(f"    Placeholder output channels:     {PLACEHOLDER_OUT_CHANNELS}")
    print(f"    Placeholder output spatial size: ({BOTTLENECK_X * 2}, {BOTTLENECK_Y * 2})")

    # -------------------------------------------------------------------------
    # Create a Stage-8-shaped canary input.
    #
    # Each merged row represents one axial slice. Row i is filled with the
    # constant value i.
    # -------------------------------------------------------------------------
    stage8_like, row_ids = make_stage8_like_input(device)

    print("Stage 8-like canary input")
    print(f"    Shape: {tuple(stage8_like.shape)}")

    assert stage8_like.shape == (
        ROWS,
        D_MODEL,
        BOTTLENECK_X,
        BOTTLENECK_Y,
    ), (
        f"Stage 8-like input shape mismatch. Expected "
        f"{(ROWS, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)}, "
        f"got {tuple(stage8_like.shape)}."
    )

    # -------------------------------------------------------------------------
    # Build the placeholder decoder.
    # -------------------------------------------------------------------------
    placeholder_decoder = build_placeholder_decoder(device)

    print("Stage 9 placeholder decoder")
    print(
        f"    ConvTranspose2d("
        f"{D_MODEL}, "
        f"{PLACEHOLDER_OUT_CHANNELS}, "
        f"kernel_size=2, stride=2)"
    )

    # -------------------------------------------------------------------------
    # Run the placeholder decoder.
    # -------------------------------------------------------------------------
    with torch.no_grad():
        decoded = placeholder_decoder(stage8_like)

    expected_shape = (
        ROWS,
        PLACEHOLDER_OUT_CHANNELS,
        BOTTLENECK_X * 2,
        BOTTLENECK_Y * 2,
    )

    print(f"    Decoded shape: {tuple(decoded.shape)}")

    assert decoded.shape == expected_shape, (
        f"Stage 9 placeholder output shape mismatch. Expected {expected_shape}, "
        f"got {tuple(decoded.shape)}."
    )

    assert decoded.shape[0] == ROWS, (
        f"Stage 9 placeholder changed the merged row count. "
        f"Expected {ROWS}, got {decoded.shape[0]}."
    )

    assert torch.isfinite(decoded).all(), (
        "Stage 9 placeholder decoder produced non-finite values (NaN or inf)."
    )

    # -------------------------------------------------------------------------
    # Verify row identities are preserved.
    #
    # Because the input row i is a constant value i, and the placeholder
    # decoder has known weights, every output voxel in row i should still be i.
    # -------------------------------------------------------------------------
    print("    verifying placeholder decoder row identities...")

    decoded_means = decoded.mean(dim=(1, 2, 3))

    assert torch.allclose(decoded_means, row_ids, rtol=0.0, atol=1e-5), (
        "Stage 9 placeholder decoder canary failed: decoded row means do not "
        "match the original row identities."
    )

    # Explicit row-order mapping:
    #     row = b * Z + z
    for b in range(B):
        for z in range(Z):
            row = b * Z + z
            expected_value = float(row)

            assert abs(decoded[row, 0, 0, 0].item() - expected_value) <= 1e-5, (
                f"Stage 9 placeholder row mismatch at first output voxel: "
                f"b={b}, z={z}, row={row}."
            )

            assert abs(decoded[row, -1, -1, -1].item() - expected_value) <= 1e-5, (
                f"Stage 9 placeholder row mismatch at last output voxel: "
                f"b={b}, z={z}, row={row}."
            )

    print("    Placeholder decoder row identities verified: row = b * Z + z")

    # -------------------------------------------------------------------------
    # Verify batch independence.
    #
    # Processing all rows together must produce the same result as processing
    # each row independently in the same order.
    # -------------------------------------------------------------------------
    print("    verifying placeholder decoder batch independence...")

    with torch.no_grad():
        individual_outputs = [
            placeholder_decoder(stage8_like[i : i + 1])
            for i in range(ROWS)
        ]
        expected_decoded = torch.cat(individual_outputs, dim=0)

    assert torch.allclose(decoded, expected_decoded, rtol=0.0, atol=1e-5), (
        "Stage 9 placeholder decoder batch-independence check failed: "
        "batched output differs from row-wise output."
    )

    print("    Placeholder decoder batch independence verified.")

    print("=" * 80)
    print("Stage 9 placeholder decoder canary passed.")
    print("")
    print("Reminder:")
    print("    This placeholder does NOT validate the real decoder.")
    print("    The same row-order canary must be re-run against the real")
    print("    Stage 9 decoder once skip connections and the real channel")
    print("    schedule are implemented.")
    print("")
    print("Next step: prototype Stage 11 logits un-merge round-trip:")
    print("    (B * Z, NUM_CLASSES, X, Y)")
    print("    -> (B, NUM_CLASSES, X, Y, Z)")
    print("    -> (B * Z, NUM_CLASSES, X, Y)")
    print("=" * 80)


if __name__ == "__main__":
    main()
