"""
Validates that Mamba2 participates correctly in AMP training:
autocast forward, GradScaler backward, gradient clipping, optimizer step.
"""

import pytest
import torch
from torch.amp import GradScaler, autocast

from mamba_ssm import Mamba2


def test_mamba2_amp_forward_backward_step() -> None:
    """Full AMP training step must complete without error and update weights."""
    device = torch.device("cuda")
    B, Z, D_MODEL = 2, 16, 64
    LR = 1e-4

    model = Mamba2(d_model=D_MODEL).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scaler = GradScaler("cuda", enabled=True)

    params_before = [p.detach().clone() for p in model.parameters()]

    optimizer.zero_grad(set_to_none=True)

    seq = torch.randn(
        B, Z, D_MODEL, device=device, dtype=torch.float32, requires_grad=True
    ).contiguous()

    with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        out = model(seq)
        loss = out.float().pow(2).mean()

    assert out.shape == (B, Z, D_MODEL)
    assert torch.isfinite(out).all()
    assert torch.isfinite(loss).all()

    scaler.scale(loss).backward()

    assert seq.grad is not None, "Input gradient is None."
    assert torch.isfinite(seq.grad).all()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    # Verify weights actually changed (GradScaler did not skip the step)
    params_changed = any(
        not torch.equal(before, p.detach())
        for before, p in zip(params_before, model.parameters())
    )
    assert params_changed, (
        "No parameter changed after optimizer step. "
        "GradScaler may have skipped the update."
    )
