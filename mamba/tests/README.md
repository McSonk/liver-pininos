# Mamba-specific tests

These tests require CUDA and `mamba_ssm`. They must be run on the server
inside `~/mamba-env`. They are NOT part of the CPU-only suite in `tests/`.

## Running

    source ~/mamba-env/bin/activate
    python -m pytest mamba/tests/ -v

## Requirements

- NVIDIA GPU with CUDA support
- `mamba_ssm` installed (available only in `~/mamba-env`)
- PyTorch with CUDA backend

## What is tested here vs `tests/`

| Directory | Environment | Requires CUDA? | Requires mamba_ssm? |
|---|---|---|---|
| `tests/` | `~/envs/dev-thesis` | No | No |
| `mamba/tests/` | `~/mamba-env` (server only) | Yes | Yes |
