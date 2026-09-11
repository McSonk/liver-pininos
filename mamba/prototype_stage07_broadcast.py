"""
prototype_stage07_broadcast.py

Stage 7a prototype for the 2.5D Mamba-hybrid architecture.

This validates the spatial broadcast of the re-merged z-context:

    Stage 6 output:
        (B * Z, D_MODEL)

    Stage 7a broadcast:
        (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

This prepares the z-context for concatenation with the Stage 2 bottleneck in
Stage 7b.

This deliberately does not include:
    - Stage 7b concat
    - Stage 8 fusion conv
    - the decoder
    - Stage 11 logits logic

Run on the server:
    source ~/mamba-env/bin/activate
    python prototype_stage07_broadcast.py

Run locally:
    python prototype_stage07_broadcast.py

Notes
-----
- mamba_ssm requires CUDA.
- If CUDA or mamba_ssm is unavailable, this script bypasses Mamba2 with an
  identity path so the Stage 7 broadcast logic can still be smoke-tested.
"""

import torch
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None


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
    """
    rows, d_model = pooled.shape

    assert rows == b * z, (
        f"Stage 4 input row count mismatch. Expected B * Z = {b * z}, "
        f"got {rows}."
    )

    return pooled.reshape(b, z, d_model)


def reshape_sequence_to_pooled(seq: torch.Tensor) -> torch.Tensor:
    """
    Stage 6:
    (B, Z, D_MODEL) -> (B * Z, D_MODEL)
    """
    b, z, d_model = seq.shape
    return seq.reshape(b * z, d_model)


def broadcast_z_context(z_flat: torch.Tensor, bottleneck_x: int, bottleneck_y: int) -> torch.Tensor:
    """
    Stage 7a:
    (B * Z, D_MODEL) -> (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)

    Each slice's z-context vector is broadcast over the bottleneck spatial
    dimensions using `unsqueeze` and `expand`.

    This does not allocate a full spatial copy by itself. The expanded tensor
    is a broadcast view. Materialisation happens later when the tensor is
    concatenated with the bottleneck in Stage 7b.
    """
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
    print("Stage 7a prototype: broadcast z-context spatially")
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
    # Stage 5: Mamba2 call.
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
        if device.type != "cuda":
            reason = "CUDA is not available"
        else:
            reason = "mamba_ssm.Mamba2 is not installed"

        print(f"    [WARNING] Bypassing Mamba2 branch: {reason}.")
        print("    [WARNING] Using identity z-context for local smoke testing.")

        z_context = seq

    print(f"    Mamba2 output sequence: {tuple(z_context.shape)}")

    assert z_context.shape == (B, Z, D_MODEL), (
        f"Stage 5 shape mismatch. Expected {(B, Z, D_MODEL)}, "
        f"got {tuple(z_context.shape)}."
    )

    assert torch.isfinite(z_context).all(), (
        "Stage 5 produced non-finite values (NaN or inf)."
    )

    # -------------------------------------------------------------------------
    # Stage 6: re-merge Mamba2 output for fusion.
    # -------------------------------------------------------------------------
    z_flat = reshape_sequence_to_pooled(z_context)

    print("Stage 6")
    print(f"    Re-merged z-context: {tuple(z_flat.shape)}")

    assert z_flat.shape == (B * Z, D_MODEL), (
        f"Stage 6 shape mismatch. Expected {(B * Z, D_MODEL)}, "
        f"got {tuple(z_flat.shape)}."
    )

    # Make the Stage 7 broadcast deterministic with respect to storage layout.
    z_flat = z_flat.contiguous()

    # -------------------------------------------------------------------------
    # Stage 7a: broadcast z-context spatially.
    #
    # Input:
    #     (B * Z, D_MODEL)
    #
    # Output:
    #     (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    #
    # Each channel vector from Mamba is broadcast over the bottleneck spatial
    # grid so it can later be concatenated with the spatial bottleneck tensor.
    # -------------------------------------------------------------------------
    z_spatial = broadcast_z_context(z_flat, BOTTLENECK_X, BOTTLENECK_Y)

    print("Stage 7a")
    print(f"    Broadcast z-context: {tuple(z_spatial.shape)}")
    print(f"    Broadcast strides:   {z_spatial.stride()}")

    assert z_spatial.shape == (
        B * Z,
        D_MODEL,
        BOTTLENECK_X,
        BOTTLENECK_Y,
    ), (
        f"Stage 7a shape mismatch. Expected "
        f"{(B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)}, "
        f"got {tuple(z_spatial.shape)}."
    )

    assert z_spatial.ndim == 4, (
        f"Stage 7a output must be 4D. Got {z_spatial.ndim}D."
    )

    assert z_spatial.shape[0] == z_flat.shape[0], (
        "Stage 7a changed the merged slice dimension. "
        f"Expected {z_flat.shape[0]} rows, got {z_spatial.shape[0]}."
    )

    assert z_spatial.shape[1] == D_MODEL, (
        f"Stage 7a channel dimension mismatch. Expected D_MODEL={D_MODEL}, "
        f"got {z_spatial.shape[1]}."
    )

    # -------------------------------------------------------------------------
    # Verify that this is a broadcast view.
    #
    # Expanded spatial dimensions should have zero stride. This indicates that
    # the same underlying memory is being reused rather than a full spatial
    # copy being allocated.
    # -------------------------------------------------------------------------
    if BOTTLENECK_X > 1:
        assert z_spatial.stride(2) == 0, (
            "Stage 7a X broadcast dimension does not have zero stride. "
            f"Got stride {z_spatial.stride(2)}."
        )

    if BOTTLENECK_Y > 1:
        assert z_spatial.stride(3) == 0, (
            "Stage 7a Y broadcast dimension does not have zero stride. "
            f"Got stride {z_spatial.stride(3)}."
        )

    print("    Broadcast view verified: expanded spatial strides are zero.")

    # -------------------------------------------------------------------------
    # Verify broadcast values.
    #
    # Since every spatial location contains the same channel vector, the
    # spatial mean should equal the original flat vector.
    # -------------------------------------------------------------------------
    spatial_mean = z_spatial.mean(dim=(2, 3))

    assert spatial_mean.shape == z_flat.shape, (
        f"Stage 7a spatial mean shape mismatch. Expected {tuple(z_flat.shape)}, "
        f"got {tuple(spatial_mean.shape)}."
    )

    assert torch.allclose(spatial_mean, z_flat, rtol=0.0, atol=1e-6), (
        "Stage 7a value check failed: spatial mean of broadcast z-context "
        "does not match the original z-context vector."
    )

    # -------------------------------------------------------------------------
    # Verify row order explicitly.
    # -------------------------------------------------------------------------
    print("    verifying Stage 7a row order...")
    for b in range(B):
        for z in range(Z):
            row = b * Z + z

            assert torch.equal(z_spatial[row, :, 0, 0], z_flat[row]), (
                f"Stage 7a row-order mismatch at first spatial location: "
                f"b={b}, z={z}, row={row}"
            )

            assert torch.equal(z_spatial[row, :, -1, -1], z_flat[row]), (
                f"Stage 7a row-order mismatch at last spatial location: "
                f"b={b}, z={z}, row={row}"
            )

    print("    Stage 7a row order verified: row = b * Z + z")

    # -------------------------------------------------------------------------
    # Stage 7a broadcast canary.
    #
    # Create an identifiable flat z-context where every element in row i is i.
    # After broadcasting, every spatial location in row i should still be i.
    # -------------------------------------------------------------------------
    print("    verifying Stage 7a broadcast canary...")

    row_ids = torch.arange(B * Z, dtype=torch.float32, device=device)
    print(f"    Canary row IDs: {row_ids.tolist()}")
    print(f"    Canary row IDs shape: {tuple(row_ids.shape)}")

    canary_flat = row_ids.unsqueeze(1).expand(B * Z, D_MODEL)
    canary_spatial = broadcast_z_context(canary_flat, BOTTLENECK_X, BOTTLENECK_Y)

    assert canary_spatial.shape == (
        B * Z,
        D_MODEL,
        BOTTLENECK_X,
        BOTTLENECK_Y,
    ), (
        f"Stage 7a canary shape mismatch. Expected "
        f"{(B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)}, "
        f"got {tuple(canary_spatial.shape)}."
    )

    expected_canary = (
        row_ids
        .view(B * Z, 1, 1, 1)
        .expand(B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    )

    assert torch.equal(canary_spatial, expected_canary), (
        "Stage 7a broadcast canary failed: broadcast values do not match "
        "the expected row identities."
    )

    canary_mean = canary_spatial.mean(dim=(2, 3))

    print(f"    Canary spatial mean shape: {tuple(canary_mean.shape)}")

    assert torch.allclose(
        canary_mean,
        canary_flat,
        rtol=0.0,
        atol=1e-6,
    ), (
        "Stage 7a broadcast canary failed: spatial mean does not preserve "
        "row identities."
    )

    # Optional additional per-row check.
    assert torch.allclose(canary_mean.mean(dim=1), row_ids, rtol=0.0, atol=1e-6), (
        "Stage 7a broadcast canary failed: per-row spatial mean does not preserve "
        "row identities."
    )

    print("    Stage 7a broadcast canary verified.")

    print("=" * 80)
    print("Stage 7a prototype passed.")
    print("Next step: prototype Stage 7b:")
    print("    concatenate broadcast z-context with the Stage 2 bottleneck:")
    print("    (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)")
    print("    +")
    print("    (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)")
    print("    ->")
    print("    (B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)")
    print("=" * 80)


if __name__ == "__main__":
    main()
