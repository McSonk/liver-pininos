"""
Shared fixtures for Mamba-specific tests.

All tests in this directory require CUDA and mamba_ssm.
If either is unavailable, only tests under this conftest directory are skipped.
"""

from pathlib import Path

import pytest
import torch

_CONFTEST_DIR = Path(__file__).resolve().parent


def _is_local_item(item: pytest.Item) -> bool:
    """Return True if the collected item belongs to this conftest directory."""
    path = getattr(item, "path", None)
    return path is not None and path.resolve().is_relative_to(_CONFTEST_DIR)


def pytest_collection_modifyitems(items):
    """Skip Mamba tests in this directory if CUDA or mamba_ssm is unavailable."""
    local_items = [item for item in items if _is_local_item(item)]

    if not local_items:
        return

    try:
        from mamba_ssm import Mamba2  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(
            reason="mamba_ssm is not installed. Run inside ~/mamba-env."
        )
        for item in local_items:
            item.add_marker(skip)
        return

    if not torch.cuda.is_available():
        skip = pytest.mark.skip(
            reason="CUDA is not available. Mamba tests require a GPU."
        )
        for item in local_items:
            item.add_marker(skip)
