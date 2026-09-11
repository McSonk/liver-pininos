"""
prototype_stage07b_concat.py

Stage 7b prototype for the 2.5D Mamba-hybrid architecture.

This validates the concatenation of the broadcast z-context with the
Stage 2 bottleneck:

    Stage 2 bottleneck:
        (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

    Stage 7a broadcast z-context:
        (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

    Stage 7b concat (dim=1):
        (B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

This deliberately does not include:
    - Stage 8 fusion conv
    - the decoder
    - Stage 11 logits logic

Run on the server:
    source ~/mamba-env/bin/activate
    python prototype_stage07b_concat.py

Run locally:
    python prototype_stage07b_concat.py

Notes
-----
- mamba_ssm requires CUDA.
- If CUDA or mamba_ssm is unavailable, this script bypasses Mamba2 with an
  identity path so the Stage 7b concat logic can still be smoke-tested.
"""

import torch
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Stage 7b prototype: concatenate z-context with bottleneck")
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
        B * C * X * Y * Z, dtype=torch.float32, device=device,
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
        reason = "CUDA is not available" if device.type != "cuda" else "mamba_ssm.Mamba2 is not installed"
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
    # Stage 7b: concatenate bottleneck and broadcast z-context along dim=1
    #
    # Input:
    #     bottleneck:  (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    #     z_spatial:   (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    #
    # Output:
    #     fused:       (B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    #
    # The first D_MODEL channels come from the spatial bottleneck.
    # The second D_MODEL channels come from the z-context.
    # -------------------------------------------------------------------------
    fused = torch.cat((bottleneck, z_spatial), dim=1)

    print("Stage 7b")
    print(f"    Fused tensor: {tuple(fused.shape)}")

    expected_channels = 2 * D_MODEL

    assert fused.shape == (B * Z, expected_channels, BOTTLENECK_X, BOTTLENECK_Y), (
        f"Stage 7b shape mismatch. Expected "
        f"{(B * Z, expected_channels, BOTTLENECK_X, BOTTLENECK_Y)}, "
        f"got {tuple(fused.shape)}."
    )

    assert fused.shape[1] == expected_channels, (
        f"Stage 7b channel count mismatch. Expected {expected_channels}, "
        f"got {fused.shape[1]}."
    )

    assert fused.shape[0] == B * Z, (
        f"Stage 7b row count mismatch. Expected {B * Z}, got {fused.shape[0]}."
    )

    assert fused.shape[2:] == (BOTTLENECK_X, BOTTLENECK_Y), (
        f"Stage 7b spatial size mismatch. Expected ({BOTTLENECK_X}, {BOTTLENECK_Y}), "
        f"got {tuple(fused.shape[2:])}."
    )

    # -------------------------------------------------------------------------
    # Verify that the first half of channels matches the bottleneck
    # and the second half matches the broadcast z-context.
    # -------------------------------------------------------------------------
    print("    verifying Stage 7b channel composition...")

    bottleneck_half = fused[:, :D_MODEL, :, :]
    z_context_half = fused[:, D_MODEL:, :, :]

    assert torch.equal(bottleneck_half, bottleneck), (
        "Stage 7b first channel half does not match the original bottleneck."
    )

    # z_spatial is a broadcast view; the concatenated copy should equal it
    # element-wise after materialisation.
    assert torch.allclose(z_context_half, z_spatial, rtol=0.0, atol=1e-6), (
        "Stage 7b second channel half does not match the broadcast z-context."
    )

    print("    Channel composition verified.")

    # -------------------------------------------------------------------------
    # Verify row order is preserved through concatenation.
    # -------------------------------------------------------------------------
    print("    verifying Stage 7b row order...")
    for b in range(B):
        for z in range(Z):
            row = b * Z + z

            assert torch.equal(fused[row, :D_MODEL, 0, 0], bottleneck[row, :, 0, 0]), (
                f"Stage 7b row-order mismatch in bottleneck half: b={b}, z={z}, row={row}"
            )

            assert torch.equal(fused[row, D_MODEL:, 0, 0], z_flat[row]), (
                f"Stage 7b row-order mismatch in z-context half: b={b}, z={z}, row={row}"
            )

    print("    Stage 7b row order verified: row = b * Z + z")

    # -------------------------------------------------------------------------
    # Stage 7b concat canary.
    #
    # Create identifiable bottleneck and z-context tensors where each row has
    # a unique constant. After concatenation, the first D_MODEL channels of
    # row i should equal the bottleneck row id, and the second D_MODEL channels
    # should equal the z-context row id.
    # -------------------------------------------------------------------------
    print("    verifying Stage 7b concat canary...")

    row_ids = torch.arange(B * Z, dtype=torch.float32, device=device)

    canary_bottleneck = (
        row_ids.view(B * Z, 1, 1, 1)
        .expand(B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
        .contiguous()
    )

    canary_z_flat = row_ids.unsqueeze(1).expand(B * Z, D_MODEL)
    canary_z_spatial = broadcast_z_context(canary_z_flat, BOTTLENECK_X, BOTTLENECK_Y)

    canary_fused = torch.cat((canary_bottleneck, canary_z_spatial), dim=1)

    assert canary_fused.shape == (B * Z, expected_channels, BOTTLENECK_X, BOTTLENECK_Y), (
        f"Stage 7b canary shape mismatch. Expected "
        f"{(B * Z, expected_channels, BOTTLENECK_X, BOTTLENECK_Y)}, "
        f"got {tuple(canary_fused.shape)}."
    )

    canary_bn_mean = canary_fused[:, :D_MODEL, :, :].mean(dim=(1, 2, 3))
    canary_zc_mean = canary_fused[:, D_MODEL:, :, :].mean(dim=(1, 2, 3))

    assert torch.allclose(canary_bn_mean, row_ids, rtol=0.0, atol=1e-5), (
        "Stage 7b canary failed: bottleneck half does not preserve row identities."
    )

    assert torch.allclose(canary_zc_mean, row_ids, rtol=0.0, atol=1e-5), (
        "Stage 7b canary failed: z-context half does not preserve row identities."
    )

    print("    Stage 7b concat canary verified.")

    print("=" * 80)
    print("Stage 7b prototype passed.")
    print("Next step: prototype Stage 8:")
    print("    1x1 fusion conv to reduce channels back to D_MODEL:")
    print("    (B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)")
    print("    -> Conv2d(2*D_MODEL, D_MODEL, 1)")
    print("    -> (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)")
    print("=" * 80)


if __name__ == "__main__":
    main()
