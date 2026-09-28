"""
Shared fixtures for Mamba-specific tests.

All tests in this directory require CUDA and mamba_ssm.
If either is unavailable, the entire directory is skipped gracefully.
"""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Skip all tests in this directory if CUDA or mamba_ssm is unavailable."""
    try:
        from mamba_ssm import Mamba2  # noqa: F401
    except ImportError:
        skip_mamba = pytest.mark.skip(
            reason="mamba_ssm is not installed. Run inside ~/mamba-env."
        )
        for item in items:
            item.add_marker(skip_mamba)
        return

    if not torch.cuda.is_available():
        skip_cuda = pytest.mark.skip(
            reason="CUDA is not available. Mamba tests require a GPU."
        )
        for item in items:
            item.add_marker(skip_cuda)
