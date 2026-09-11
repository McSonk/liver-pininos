"""
prototype_stage08_fusion_conv.py

Stage 8 prototype for the 2.5D Mamba-hybrid architecture.

This validates the 1x1 fusion convolution:

    Stage 7b fused tensor:
        (B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

    Stage 8 fusion conv:
        Conv2d(2 * D_MODEL, D_MODEL, kernel_size=1)

    Stage 8 output:
        (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

This deliberately does not include:
    - the real decoder
    - skip connections
    - Stage 11 logits logic

Run on the server:
    source ~/mamba-env/bin/activate
    python prototype_stage08_fusion_conv.py

Run locally:
    python prototype_stage08_fusion_conv.py

Notes
-----
- mamba_ssm requires CUDA.
- If CUDA or mamba_ssm is unavailable, this script bypasses Mamba2 with an
  identity path so the Stage 8 fusion logic can still be smoke-tested.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None


# -----------------------------------------------------------------------------
# Small non-cubic shapes.
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

D_MODEL = BOTTLENECK_CHANNELS

PERMUTATION_ORDER = (0, 4, 1, 2, 3)


def split_volume_to_axial_slices(volume: torch.Tensor) -> torch.Tensor:
    b, c, x, y, z = volume.shape
    return volume.permute(*PERMUTATION_ORDER).reshape(b * z, c, x, y)


def dummy_down_path(features: torch.Tensor, base_channels: int, num_downs: int) -> torch.Tensor:
    rows, channels, height, width = features.shape

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
        features = F.avg_pool2d(features, kernel_size=2, stride=2)
        features = features.repeat(1, 2, 1, 1)

    return features


def global_average_pool(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError(
            "global_average_pool expects a 4D tensor with shape (N, C, H, W), "
            f"got shape {tuple(features.shape)}."
        )
    return features.mean(dim=(2, 3))


def reshape_pooled_to_sequence(pooled: torch.Tensor, b: int, z: int) -> torch.Tensor:
    rows, d_model = pooled.shape
    assert rows == b * z, (
        f"Stage 4 input row count mismatch. Expected B * Z = {b * z}, got {rows}."
    )
    return pooled.reshape(b, z, d_model)


def reshape_sequence_to_pooled(seq: torch.Tensor) -> torch.Tensor:
    b, z, d_model = seq.shape
    return seq.reshape(b * z, d_model)


def broadcast_z_context(z_flat: torch.Tensor, bottleneck_x: int, bottleneck_y: int) -> torch.Tensor:
    if z_flat.ndim != 2:
        raise ValueError(
            "broadcast_z_context expects a 2D tensor with shape (B * Z, D_MODEL), "
            f"got shape {tuple(z_flat.shape)}."
        )
    rows, d_model = z_flat.shape
    return (
        z_flat
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(rows, d_model, bottleneck_x, bottleneck_y)
    )


def concat_bottleneck_and_z_context(
    bottleneck: torch.Tensor,
    z_spatial: torch.Tensor,
) -> torch.Tensor:
    """
    Stage 7b:
    Concatenate spatial bottleneck and broadcast z-context along dim=1.
    """
    if bottleneck.shape != z_spatial.shape:
        raise ValueError(
            "Stage 7b requires bottleneck and z_spatial to have identical shapes. "
            f"Got bottleneck {tuple(bottleneck.shape)} and z_spatial {tuple(z_spatial.shape)}."
        )

    return torch.cat((bottleneck, z_spatial), dim=1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(123)

    print("=" * 80)
    print("Stage 8 prototype: 1x1 fusion convolution")
    print("=" * 80)

    print("Environment:")
    print(f"    Device:                          {device}")
    print(f"    mamba_ssm Mamba2 available:      {Mamba2 is not None}")

    print("Intended shapes:")
    print(f"    Batch size (B):                  {B}")
    print(f"    Channels (C):                    {C}")
    print(f"    Spatial dimensions (X, Y, Z):    ({X}, {Y}, {Z})")
    print(f"    Number of downsamples:           {NUM_DOWNS}")
    print(f"    Base channels:                   {BASE_CHANNELS}")
    print(f"    Bottleneck channels / d_model:   {D_MODEL}")
    print(f"    Bottleneck spatial size:         ({BOTTLENECK_X}, {BOTTLENECK_Y})")

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    assert X % DOWNSAMPLE_FACTOR == 0
    assert Y % DOWNSAMPLE_FACTOR == 0

    # -------------------------------------------------------------------------
    # Stage 0: input volume
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
    # Stage 1: split into axial slices
    # -------------------------------------------------------------------------
    slices = split_volume_to_axial_slices(volume)

    print("Stage 1")
    print(f"    Axial slices: {tuple(slices.shape)}")
    assert slices.shape == (B * Z, C, X, Y)

    print("    verifying Stage 1 row order...")
    for b in range(B):
        for z in range(Z):
            row = b * Z + z
            assert torch.equal(slices[row], volume[b, :, :, :, z]), (
                f"Stage 1 row-order mismatch: b={b}, z={z}, row={row}"
            )
    print("    Stage 1 row order verified.")

    # -------------------------------------------------------------------------
    # Stage 2: dummy 2D encoder bottleneck
    # -------------------------------------------------------------------------
    bottleneck = dummy_down_path(slices, BASE_CHANNELS, NUM_DOWNS)

    print("Stage 2")
    print(f"    Dummy bottleneck: {tuple(bottleneck.shape)}")
    assert bottleneck.shape == (B * Z, BOTTLENECK_CHANNELS, BOTTLENECK_X, BOTTLENECK_Y)

    # -------------------------------------------------------------------------
    # Stage 3: global average pool
    # -------------------------------------------------------------------------
    pooled = global_average_pool(bottleneck)

    print("Stage 3")
    print(f"    Pooled slice vectors: {tuple(pooled.shape)}")
    assert pooled.shape == (B * Z, D_MODEL)

    # -------------------------------------------------------------------------
    # Stage 4: reshape to Mamba sequence
    # -------------------------------------------------------------------------
    seq = reshape_pooled_to_sequence(pooled, B, Z)

    print("Stage 4")
    print(f"    Mamba sequence: {tuple(seq.shape)}")
    assert seq.shape == (B, Z, D_MODEL)

    # -------------------------------------------------------------------------
    # Stage 5: Mamba2 call
    # -------------------------------------------------------------------------
    use_mamba = device.type == "cuda" and Mamba2 is not None

    print("Stage 5")

    if use_mamba:
        print("    Running real Mamba2 forward pass...")
        mamba_block = Mamba2(d_model=D_MODEL).to(device)
        mamba_block.eval()
        seq = seq.contiguous()

        with torch.no_grad():
            try:
                z_context = mamba_block(seq)
            except (RuntimeError, NotImplementedError, AssertionError) as exc:
                print(f"    [WARNING] Mamba2 fp32 forward failed: {exc}")
                print("    [WARNING] Retrying Mamba2 forward pass in fp16.")
                mamba_block = mamba_block.half()
                z_context = mamba_block(seq.half())

        z_context = z_context.float()
    else:
        reason = (
            "CUDA is not available"
            if device.type != "cuda"
            else "mamba_ssm.Mamba2 is not installed"
        )
        print(f"    [WARNING] Bypassing Mamba2 branch: {reason}.")
        z_context = seq

    print(f"    Mamba2 output sequence: {tuple(z_context.shape)}")
    assert z_context.shape == (B, Z, D_MODEL)
    assert torch.isfinite(z_context).all()

    # -------------------------------------------------------------------------
    # Stage 6: re-merge
    # -------------------------------------------------------------------------
    z_flat = reshape_sequence_to_pooled(z_context).contiguous()

    print("Stage 6")
    print(f"    Re-merged z-context: {tuple(z_flat.shape)}")
    assert z_flat.shape == (B * Z, D_MODEL)

    # -------------------------------------------------------------------------
    # Stage 7a: broadcast z-context spatially
    # -------------------------------------------------------------------------
    z_spatial = broadcast_z_context(z_flat, BOTTLENECK_X, BOTTLENECK_Y)

    print("Stage 7a")
    print(f"    Broadcast z-context: {tuple(z_spatial.shape)}")
    assert z_spatial.shape == (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

    # -------------------------------------------------------------------------
    # Stage 7b: concatenate bottleneck and z-context
    # -------------------------------------------------------------------------
    fused = concat_bottleneck_and_z_context(bottleneck, z_spatial)

    print("Stage 7b")
    print(f"    Fused tensor: {tuple(fused.shape)}")
    assert fused.shape == (B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

    # -------------------------------------------------------------------------
    # Stage 8: 1x1 fusion convolution
    #
    # Input:
    #     (B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    #
    # Output:
    #     (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    #
    # This restores the channel count to D_MODEL, which is the intended
    # decoder input width. Keeping this width constant makes the Mamba branch
    # easier to ablate later.
    # -------------------------------------------------------------------------
    fusion_conv = nn.Conv2d(
        in_channels=2 * D_MODEL,
        out_channels=D_MODEL,
        kernel_size=1,
        bias=True,
    ).to(device)

    fusion_conv.eval()

    print("Stage 8")
    print(f"    Fusion conv: Conv2d({2 * D_MODEL}, {D_MODEL}, kernel_size=1)")

    with torch.no_grad():
        fused_out = fusion_conv(fused)

    print(f"    Fusion output: {tuple(fused_out.shape)}")

    assert fused_out.shape == (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y), (
        f"Stage 8 shape mismatch. Expected "
        f"{(B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)}, "
        f"got {tuple(fused_out.shape)}."
    )

    assert fused_out.shape[0] == B * Z, (
        f"Stage 8 row count mismatch. Expected {B * Z}, got {fused_out.shape[0]}."
    )

    assert fused_out.shape[1] == D_MODEL, (
        f"Stage 8 output channel mismatch. Expected D_MODEL={D_MODEL}, "
        f"got {fused_out.shape[1]}."
    )

    assert fused_out.shape[1] == bottleneck.shape[1], (
        "Stage 8 output channel count must match the original bottleneck "
        "channel count so that the decoder input width remains constant "
        "with or without the Mamba branch."
    )

    assert fused_out.shape[2:] == (BOTTLENECK_X, BOTTLENECK_Y), (
        f"Stage 8 spatial size mismatch. Expected ({BOTTLENECK_X}, {BOTTLENECK_Y}), "
        f"got {tuple(fused_out.shape[2:])}."
    )

    assert torch.isfinite(fused_out).all(), (
        "Stage 8 produced non-finite values (NaN or inf)."
    )

    print("    Stage 8 shape contract verified.")

    # -------------------------------------------------------------------------
    # Stage 8 row-order canary.
    #
    # A 1×1 Conv2d operates independently at each spatial location and does not
    # mix the batch dimension.. It must not mix or reorder merged slice rows.
    #
    # This canary uses a separate Conv2d with known weights. Each input row is
    # filled with a unique constant equal to its row index. With all weights
    # set to 1 / in_channels, every output channel should recover the same
    # row index.
    # -------------------------------------------------------------------------
    print("    verifying Stage 8 row-order canary...")

    row_ids = torch.arange(B * Z, dtype=torch.float32, device=device)

    canary_conv = nn.Conv2d(
        in_channels=2 * D_MODEL,
        out_channels=D_MODEL,
        kernel_size=1,
        bias=False,
    ).to(device)

    canary_conv.eval()

    with torch.no_grad():
        canary_conv.weight.fill_(1.0 / (2 * D_MODEL))

    canary_fused = (
        row_ids
        .view(B * Z, 1, 1, 1)
        .expand(B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
        .contiguous()
    )

    with torch.no_grad():
        canary_out = canary_conv(canary_fused)

    assert canary_out.shape == (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y), (
        f"Stage 8 canary shape mismatch. Expected "
        f"{(B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)}, "
        f"got {tuple(canary_out.shape)}."
    )

    canary_means = canary_out.mean(dim=(1, 2, 3))

    assert torch.allclose(canary_means, row_ids, rtol=0.0, atol=1e-5), (
        "Stage 8 row-order canary failed: output rows do not preserve the "
        "input row identities."
    )

    # Spot-check explicit row mapping.
    for b in (0, B - 1):
        for z in (0, Z - 1):
            row = b * Z + z
            expected_value = float(row)

            assert abs(canary_out[row, 0, 0, 0].item() - expected_value) <= 1e-5, (
                f"Stage 8 canary row mismatch at first spatial location: "
                f"b={b}, z={z}, row={row}."
            )

            assert abs(canary_out[row, -1, -1, -1].item() - expected_value) <= 1e-5, (
                f"Stage 8 canary row mismatch at last spatial location: "
                f"b={b}, z={z}, row={row}."
            )

    print("    Stage 8 row-order canary verified.")

    print("=" * 80)
    print("Stage 8 prototype passed.")
    print("Next step: prototype Stage 9 placeholder decoder canary:")
    print("    verify that a conv-based upsampling path does not reorder rows.")
    print("=" * 80)


if __name__ == "__main__":
    main()
