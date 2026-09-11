"""
prototype_stages00_to_08_unified.py
Unified prototype for the 2.5D Mamba-hybrid architecture (Stages 0 to 8).

This script consolidates the core reshape story (Stages 0-8) into a single
execution flow, eliminating the redundancy of maintaining separate scripts
that re-implement the same dummy encoder and Mamba bypass logic.

It includes all recommended assertions:
- Spatial split/merge round-trip (Stage 1)
- Dummy bottleneck shape and row-order canary (Stage 2)
- Global average pool shape (Stage 3)
- Sequence reshape round-trip (Stage 4/6)
- Mamba2 forward pass with bypass (Stage 5)
- Spatial broadcast and stride verification (Stage 7a)
- Concatenation shape, channel composition, and row-order canary (Stage 7b)
- 1x1 fusion convolution shape and row-order canary (Stage 8)

Run on the server:
    source ~/mamba-env/bin/activate
    python prototype_stages_all.py
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
# X and Y are divisible by 16 to support four stride-2 downsampling steps.
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

PERMUTE_FORWARD = (0, 4, 1, 2, 3)   # (B, C, X, Y, Z) -> (B, Z, C, X, Y)
PERMUTE_INVERSE = (0, 2, 3, 4, 1)   # (B, Z, C, X, Y) -> (B, C, X, Y, Z)


def split_volume_to_axial_slices(volume: torch.Tensor) -> torch.Tensor:
    b, c, x, y, z = volume.shape
    return volume.permute(*PERMUTE_FORWARD).reshape(b * z, c, x, y)


def merge_axial_slices_to_volume(slices: torch.Tensor, b: int, c: int, x: int, y: int, z: int) -> torch.Tensor:
    return slices.reshape(b, z, c, x, y).permute(*PERMUTE_INVERSE)


def dummy_down_path(features: torch.Tensor, base_channels: int, num_downs: int) -> torch.Tensor:
    rows, channels, height, width = features.shape
    if channels != base_channels:
        if base_channels % channels != 0:
            raise ValueError(f"Cannot expand channel count from {channels} to {base_channels}.")
        features = features.repeat(1, base_channels // channels, 1, 1)
    
    for _ in range(num_downs):
        if features.shape[-2] % 2 != 0 or features.shape[-1] % 2 != 0:
            raise ValueError(f"Spatial dimensions must be divisible by 2. Got {tuple(features.shape)}.")
        features = F.avg_pool2d(features, kernel_size=2, stride=2)
        features = features.repeat(1, 2, 1, 1)
    return features


def global_average_pool(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {tuple(features.shape)}.")
    return features.mean(dim=(2, 3))


def reshape_pooled_to_sequence(pooled: torch.Tensor, b: int, z: int) -> torch.Tensor:
    rows, d_model = pooled.shape
    assert rows == b * z, f"Row count mismatch. Expected {b * z}, got {rows}."
    return pooled.reshape(b, z, d_model)


def reshape_sequence_to_pooled(seq: torch.Tensor) -> torch.Tensor:
    b, z, d_model = seq.shape
    return seq.reshape(b * z, d_model)


def broadcast_z_context(z_flat: torch.Tensor, bottleneck_x: int, bottleneck_y: int) -> torch.Tensor:
    rows, d_model = z_flat.shape
    return z_flat.unsqueeze(-1).unsqueeze(-1).expand(rows, d_model, bottleneck_x, bottleneck_y)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(123)

    print("=" * 80)
    print("Unified Prototype: Stages 0 to 8")
    print("=" * 80)
    print(f"Device: {device} | Mamba2 available: {Mamba2 is not None}")
    print(f"Shapes: B={B}, C={C}, X={X}, Y={Y}, Z={Z}")
    print(f"Bottleneck: {BOTTLENECK_CHANNELS} channels, ({BOTTLENECK_X}, {BOTTLENECK_Y}) spatial")

    # =========================================================================
    # Stage 0 & 1: Spatial split/merge round-trip
    # =========================================================================
    print("\n--- Stage 0 & 1: Spatial split/merge round-trip ---")
    volume = torch.arange(B * C * X * Y * Z, dtype=torch.float32, device=device).reshape(B, C, X, Y, Z)
    slices = split_volume_to_axial_slices(volume)
    assert slices.shape == (B * Z, C, X, Y)
    
    # Row order check
    for b in range(B):
        for z in range(Z):
            assert torch.equal(slices[b * Z + z], volume[b, :, :, :, z])
            
    merged = merge_axial_slices_to_volume(slices, B, C, X, Y, Z)
    assert torch.equal(volume, merged), "Spatial round-trip failed."
    print("    Spatial split/merge round-trip verified.")

    # =========================================================================
    # Stage 2: Dummy bottleneck
    # =========================================================================
    print("\n--- Stage 2: Dummy bottleneck ---")
    bottleneck = dummy_down_path(slices, BASE_CHANNELS, NUM_DOWNS)
    assert bottleneck.shape == (B * Z, BOTTLENECK_CHANNELS, BOTTLENECK_X, BOTTLENECK_Y)
    
    # Row-order canary
    row_ids = torch.arange(B * Z, dtype=torch.float32, device=device)
    canary_in = row_ids.view(B * Z, 1, 1, 1).expand(B * Z, BASE_CHANNELS, X, Y).contiguous()
    canary_out = dummy_down_path(canary_in, BASE_CHANNELS, NUM_DOWNS)
    assert torch.allclose(canary_out.mean(dim=(1, 2, 3)), row_ids, atol=1e-5)
    print("    Dummy bottleneck shape and row-order canary verified.")

    # =========================================================================
    # Stage 3: Global average pool
    # =========================================================================
    print("\n--- Stage 3: Global average pool ---")
    pooled = global_average_pool(bottleneck)
    assert pooled.shape == (B * Z, D_MODEL)
    print("    Global average pool shape verified.")

    # =========================================================================
    # Stage 4 & 6: Sequence reshape round-trip
    # =========================================================================
    print("\n--- Stage 4 & 6: Sequence reshape round-trip ---")

    # Use a pure arange tensor to isolate the reshape logic from upstream operations.
    # This strictly follows the checklist requirement for an "arange-based pooled tensor".
    identifiable_pooled = torch.arange(
        B * Z * D_MODEL, dtype=torch.float32, device=device
    ).reshape(B * Z, D_MODEL)

    seq = reshape_pooled_to_sequence(identifiable_pooled, B, Z)
    assert seq.shape == (B, Z, D_MODEL)

    # Explicit row-order check
    for b in range(B):
        for z in range(Z):
            row = b * Z + z
            assert torch.equal(seq[b, z], identifiable_pooled[row]), \
                f"Stage 4 row-order mismatch at b={b}, z={z}, row={row}"

    flat_back = reshape_sequence_to_pooled(seq)
    assert torch.equal(identifiable_pooled, flat_back), "Stage 4/6 sequence round-trip failed."

    print("    Sequence reshape round-trip (arange-based) verified.")

    # =========================================================================
    # Stage 5: Mamba2 call
    # =========================================================================
    print("\n--- Stage 5: Mamba2 call ---")
    use_mamba = device.type == "cuda" and Mamba2 is not None
    if use_mamba:
        mamba_block = Mamba2(d_model=D_MODEL).to(device).eval()
        seq_c = seq.contiguous()
        with torch.no_grad():
            try:
                z_context = mamba_block(seq_c)
            except Exception:
                print("    [WARNING] Mamba2 fp32 failed, retrying in fp16.")
                z_context = mamba_block.half()(seq_c.half()).float()
        print("    Real Mamba2 forward pass executed.")
    else:
        z_context = seq  # Identity bypass for local smoke testing
        print("    [WARNING] Mamba2 bypassed (identity). Run on server for real Mamba2.")

    assert z_context.shape == (B, Z, D_MODEL)
    print("    Mamba2 forward pass (or bypass) shape verified.")

    # =========================================================================
    # Stage 7a: Spatial broadcast
    # =========================================================================
    print("\n--- Stage 7a: Spatial broadcast ---")
    z_flat = reshape_sequence_to_pooled(z_context).contiguous()
    z_spatial = broadcast_z_context(z_flat, BOTTLENECK_X, BOTTLENECK_Y)
    assert z_spatial.shape == (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    if BOTTLENECK_X > 1: assert z_spatial.stride(2) == 0
    if BOTTLENECK_Y > 1: assert z_spatial.stride(3) == 0
    print("    Spatial broadcast shape and zero-stride view verified.")

    # =========================================================================
    # Stage 7b: Concatenation
    # =========================================================================
    print("\n--- Stage 7b: Concatenation ---")
    fused = torch.cat((bottleneck, z_spatial), dim=1)
    assert fused.shape == (B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    assert torch.equal(fused[:, :D_MODEL], bottleneck)
    
    # Concat canary
    canary_bn = row_ids.view(B * Z, 1, 1, 1).expand(B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y).contiguous()
    canary_zc = broadcast_z_context(row_ids.unsqueeze(1).expand(B * Z, D_MODEL), BOTTLENECK_X, BOTTLENECK_Y)
    canary_fused = torch.cat((canary_bn, canary_zc), dim=1)
    assert torch.allclose(canary_fused[:, :D_MODEL].mean(dim=(1, 2, 3)), row_ids, atol=1e-5)
    assert torch.allclose(canary_fused[:, D_MODEL:].mean(dim=(1, 2, 3)), row_ids, atol=1e-5)
    print("    Concatenation shape, composition, and row-order canary verified.")

    # =========================================================================
    # Stage 8: 1x1 Fusion Convolution
    # =========================================================================
    print("\n--- Stage 8: 1x1 Fusion Convolution ---")
    fusion_conv = nn.Conv2d(2 * D_MODEL, D_MODEL, kernel_size=1, bias=True).to(device).eval()
    with torch.no_grad():
        fused_out = fusion_conv(fused)
    assert fused_out.shape == (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)
    
    # Fusion canary
    canary_conv = nn.Conv2d(2 * D_MODEL, D_MODEL, kernel_size=1, bias=False).to(device).eval()
    with torch.no_grad():
        canary_conv.weight.fill_(1.0 / (2 * D_MODEL))
        canary_fused_in = row_ids.view(B * Z, 1, 1, 1).expand(B * Z, 2 * D_MODEL, BOTTLENECK_X, BOTTLENECK_Y).contiguous()
        canary_out = canary_conv(canary_fused_in)
    assert torch.allclose(canary_out.mean(dim=(1, 2, 3)), row_ids, atol=1e-5)
    print("    1x1 fusion convolution shape and row-order canary verified.")

    print("\n" + "=" * 80)
    print("Unified prototype passed all assertions.")
    print("=" * 80)


if __name__ == "__main__":
    main()
