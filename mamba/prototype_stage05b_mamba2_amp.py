"""
prototype_stage05b_mamba2_amp.py

Stage 5b prototype for the 2.5D Mamba-hybrid architecture.

This validates that Mamba2 works under the same kind of CUDA AMP training
context used by the real training loop.

It tests:

    Input:
        (B, Z, D_MODEL)

    Mamba2:
        (B, Z, D_MODEL) -> (B, Z, D_MODEL)

    Under:
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        torch.amp.GradScaler("cuda")
        backward()
        gradient clipping
        optimizer step

This script deliberately does not test the full segmentation model.
It isolates the Mamba2 block and verifies that it can participate in an
AMP training step.

Run on the server:
    source ~/mamba-env/bin/activate
    python prototype_stage05b_mamba2_amp.py

Notes
-----
- This script requires CUDA.
- This script requires mamba_ssm.Mamba2.
- There is no CPU bypass, because the purpose of this test is specifically
  CUDA AMP compatibility.
"""

import torch
from torch import nn

try:
    from mamba_ssm import Mamba2
except ImportError as exc:
    raise ImportError(
        "Could not import Mamba2 from mamba_ssm. "
        "Run this inside ~/mamba-env on the server."
    ) from exc


# -----------------------------------------------------------------------------
# Prototype sequence shape.
#
# This matches the previous small prototypes:
#
#   Stage 4 / Stage 5 input:
#       (B, Z, D_MODEL) = (2, 16, 64)
#
# Full A100 equivalent later:
#       (B, Z, D_MODEL) = (8, 128, 256)
# -----------------------------------------------------------------------------
B = 2
Z = 16
D_MODEL = 64

NUM_STEPS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_GRAD_NORM = 1.0


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or inf values."


def main() -> None:
    print("=" * 80)
    print("Stage 5b prototype: Mamba2 AMP forward/backward validation")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Stage 5b must be run on the server "
            "inside ~/mamba-env."
        )

    device = torch.device("cuda")

    print("Environment:")
    print(f"    Device:                    {device}")
    print(f"    CUDA device name:           {torch.cuda.get_device_name(0)}")
    print(f"    torch version:              {torch.__version__}")
    print("    Mamba2 available:           True")

    print("Mamba2 AMP test shape:")
    print(f"    B:                          {B}")
    print(f"    Z / sequence length:         {Z}")
    print(f"    D_MODEL:                    {D_MODEL}")
    print(f"    Input shape:                {(B, Z, D_MODEL)}")
    print("    AMP dtype:                  torch.float16")
    print(f"    Number of optimizer steps:  {NUM_STEPS}")

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    # -------------------------------------------------------------------------
    # Build Mamba2 block.
    #
    # Keep parameters in fp32. Do NOT manually call .half().
    # The real training loop also keeps the model normally and lets autocast
    # decide which operations run in reduced precision.
    # -------------------------------------------------------------------------
    mamba_block = Mamba2(d_model=D_MODEL).to(device)
    mamba_block.train()

    num_params = count_parameters(mamba_block)
    print("Mamba2")
    print(f"    Constructor:                Mamba2(d_model={D_MODEL})")
    print(f"    Parameters:                 {num_params}")

    optimizer = torch.optim.AdamW(
        mamba_block.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Mirrors the GradScaler usage pattern in training.py.
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    print("AMP validation")
    print(f"    Initial GradScaler scale:    {scaler.get_scale()}")

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)

        # New input each step. requires_grad=True verifies that gradients can
        # flow backward through Mamba2 to the upstream encoder in the real model.
        seq = torch.randn(
            B,
            Z,
            D_MODEL,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        ).contiguous()

        assert seq.shape == (B, Z, D_MODEL)
        assert seq.dtype == torch.float32

        # ---------------------------------------------------------------------
        # AMP forward.
        #
        # This is the key part: it mirrors the real training loop's autocast
        # context around self.model(images).
        # ---------------------------------------------------------------------
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            z_context = mamba_block(seq)

            assert z_context.shape == (B, Z, D_MODEL), (
                f"Mamba2 AMP output shape mismatch. Expected {(B, Z, D_MODEL)}, "
                f"got {tuple(z_context.shape)}."
            )

            # Use a simple finite scalar loss.
            #
            # Cast to fp32 for the loss reduction to avoid unnecessary numerical
            # fragility in the prototype. This is analogous to many training
            # paths where losses are accumulated in fp32.
            loss = z_context.float().pow(2).mean()

        print(f"    Step {step + 1}/{NUM_STEPS}")
        print(f"        Input dtype:             {seq.dtype}")
        print(f"        Output dtype:            {z_context.dtype}")
        print(f"        Loss dtype:              {loss.dtype}")
        print(f"        Loss value:              {loss.item():.6f}")

        assert_finite_tensor("Mamba2 AMP output", z_context)
        assert_finite_tensor("Mamba2 AMP loss", loss)

        # ---------------------------------------------------------------------
        # AMP backward with GradScaler.
        # ---------------------------------------------------------------------
        scaler.scale(loss).backward()

        # Verify gradient flows back to the input sequence.
        assert seq.grad is not None, "Input sequence gradient is None."
        assert_finite_tensor("Input sequence gradient", seq.grad)

        # Verify at least some Mamba2 parameters received gradients.
        param_grads = [
            p.grad for p in mamba_block.parameters()
            if p.grad is not None
        ]

        assert len(param_grads) > 0, (
            "No Mamba2 parameter gradients were produced."
        )

        for idx, grad in enumerate(param_grads):
            assert_finite_tensor(f"Mamba2 parameter gradient {idx}", grad)

        # Unscale before gradient clipping, matching the usual AMP pattern.
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            mamba_block.parameters(),
            max_norm=MAX_GRAD_NORM,
        )

        assert torch.isfinite(grad_norm), (
            f"Gradient norm is not finite: {grad_norm}"
        )

        print(f"        Input grad finite:       True")
        print(f"        Param grads finite:      True")
        print(f"        Grad norm before clip:   {float(grad_norm):.6f}")

        # Optimizer step through GradScaler.
        previous_scale = scaler.get_scale()

        scaler.step(optimizer)
        scaler.update()

        for name, param in mamba_block.named_parameters():
            assert torch.isfinite(param).all(), (
                f"Mamba2 parameter '{name}' became non-finite "
                f"after step {step + 1}."
            )

        new_scale = scaler.get_scale()

        print(f"        GradScaler scale:        {previous_scale} -> {new_scale}")

        torch.cuda.synchronize()

    print("=" * 80)
    print("Stage 5b prototype passed.")
    print("Mamba2 survived fp16 autocast forward, scaled backward, gradient clipping,")
    print("and optimizer stepping.")
    print("=" * 80)
    print("Next step: prototype Stage 7:")
    print("    broadcast re-merged z-context spatially:")
    print("    (B * Z, D_MODEL) -> (B * Z, D_MODEL, BOTTLENECK_X, BOTTLENECK_Y)")
    print("=" * 80)


if __name__ == "__main__":
    main()
