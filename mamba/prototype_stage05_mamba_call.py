"""
prototype_stage05_mamba_call.py

Stage 5 prototype for the 2.5D Mamba-hybrid architecture.

This validates the Mamba input/output contract:

    Stage 4 output:
        (B, Z, D_MODEL)

    Stage 5 Mamba output:
        (B, Z, D_MODEL)

Mamba is expected to process the sequence dimension Z and preserve the
tensor shape.

This deliberately does not include:
    - bidirectional Mamba
    - fusion
    - the decoder
    - Stage 7/8
    - Stage 11 logits logic

Run on the server:
    source ~/mamba-env/bin/activate
    python prototype_stage05_mamba_call.py

Run locally:
    python prototype_stage05_mamba_call.py

Notes
-----
- mamba_ssm requires CUDA.
- If CUDA or mamba_ssm is unavailable, this script bypasses Mamba with an
  identity path so the surrounding reshape logic can still be smoke-tested.
"""

import torch
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


# -----------------------------------------------------------------------------
# Small non-cubic shapes.
#
# These match the previous prototypes so the scripts remain comparable.
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Stage 5 prototype: Mamba call")
    print("=" * 80)

    print("Environment:")
    print(f"    Device:                          {device}")
    print(f"    mamba_ssm available:             {Mamba is not None}")

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
        device=device,
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
    # -------------------------------------------------------------------------
    seq = reshape_pooled_to_sequence(pooled, B, Z)

    print("Stage 4")
    print(f"    Mamba sequence: {tuple(seq.shape)}")

    assert seq.shape == (B, Z, D_MODEL), (
        f"Stage 4 shape mismatch. Expected {(B, Z, D_MODEL)}, "
        f"got {tuple(seq.shape)}."
    )

    # -------------------------------------------------------------------------
    # Verify Stage 4 row order.
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
    # Stage 5: Mamba call.
    #
    # Mamba expects:
    #     (batch, sequence_length, d_model)
    #
    # In this architecture:
    #     batch           = B
    #     sequence_length = Z
    #     d_model         = D_MODEL
    #
    # The expected output shape is unchanged:
    #     (B, Z, D_MODEL)
    # -------------------------------------------------------------------------
    use_mamba = device.type == "cuda" and Mamba is not None

    print("Stage 5")

    if use_mamba:
        print("    Running real Mamba forward pass...")

        mamba_block = Mamba(
            d_model=D_MODEL,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device)

        mamba_block.eval()

        # Mamba's CUDA kernels are commonly used under mixed precision.
        # Try fp32 first for simplicity, then fall back to fp16 if the
        # installed mamba_ssm build rejects fp32 inputs.
        with torch.no_grad():
            try:
                z_context = mamba_block(seq)
            except (RuntimeError, NotImplementedError) as exc:
                print(f"    [WARNING] Mamba fp32 forward failed: {exc}")
                print("    [WARNING] Retrying Mamba forward pass in fp16.")

                mamba_block = mamba_block.half()
                z_context = mamba_block(seq.half())

        # Normalise back to fp32 for downstream prototype assertions.
        z_context = z_context.float()
    else:
        if device.type != "cuda":
            reason = "CUDA is not available"
        else:
            reason = "mamba_ssm is not installed"

        print(f"    [WARNING] Bypassing Mamba branch: {reason}.")
        print("    [WARNING] Using identity z-context for local smoke testing.")
        print("    [WARNING] Run this script on the server inside ~/mamba-env")
        print("              to validate the real Mamba call.")

        z_context = seq

    print(f"    Mamba output sequence: {tuple(z_context.shape)}")

    assert z_context.shape == (B, Z, D_MODEL), (
        f"Stage 5 shape mismatch. Expected {(B, Z, D_MODEL)}, "
        f"got {tuple(z_context.shape)}."
    )

    assert z_context.shape == seq.shape, (
        "Stage 5 must preserve the Mamba sequence shape. "
        f"Input shape: {tuple(seq.shape)}, output shape: {tuple(z_context.shape)}."
    )

    assert torch.isfinite(z_context).all(), (
        "Stage 5 produced non-finite values (NaN or inf)."
    )

    print("    Stage 5 shape contract verified: (B, Z, D_MODEL) -> (B, Z, D_MODEL)")

    print("=" * 80)
    print("Stage 5 prototype passed.")
    print("Next step: prototype Stage 6:")
    print("    re-merge Mamba output for fusion:")
    print("    (B, Z, D_MODEL) -> (B * Z, D_MODEL)")
    print("=" * 80)


if __name__ == "__main__":
    main()
