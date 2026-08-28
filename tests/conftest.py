"""Shared fixtures for the unit-test suite.

This module provides fixtures that are reused across test files for components
that depend on a real, frozen ``Config`` instance or on a clean module-level
config singleton.
"""

from pathlib import Path

import pytest

from idssp.sonk import config as config_module
from idssp.sonk.config import AvailableModels, Config, Mode


@pytest.fixture(autouse=True)
def reset_config_singleton(monkeypatch):
    """Reset the module-level config singleton before and after each test.

    ``idssp.sonk.config._config`` is forced back to ``None`` before the test so
    a previously initialised config cannot leak into the current test. The
    original value is restored automatically via ``monkeypatch``.
    """
    original = config_module._config
    monkeypatch.setattr(config_module, "_config", None)
    yield
    monkeypatch.setattr(config_module, "_config", original)


@pytest.fixture
def minimal_config(tmp_path) -> Config:
    """Build a minimal, valid frozen :class:`Config` directly.

    The instance is constructed explicitly (without calling ``config.init()``)
    so tests that need a real ``Config`` for ``dataclasses.replace()`` can do
    so without depending on environment variables or hardware detection.
    All paths point under the per-test temporary directory.
    """
    output_dir = tmp_path / "output"
    return Config(
        RUN_ID="dummy-run",
        ENV="local",
        DEVICE="cpu",
        HC_GPU=False,
        LEARNING_RATE=1e-4,
        SLIDING_WINDOW_BATCH_SIZE=4,
        COSINE_ETA_MIN=1e-6,
        WARMUP_EPOCHS=5,
        NUM_EPOCHS=10,
        NUM_CLASSES=3,
        TUMOUR_CLASS_INDEX=2,
        DICE_CE_WEIGHTS=[0.5, 1.0, 3.0],
        ISO_SPACING=(1.0, 1.0, 1.0),
        TRAIN_PATCH_SIZE=(64, 64, 64),
        VAL_PATCH_SIZE=(64, 64, 64),
        MODEL=AvailableModels.SEG_RES_NET,
        MODE=Mode.TRAIN,
        OUTPUT_DIR=output_dir,
        RUN_DIR=output_dir / "run",
        CHECKPOINT_DIR=output_dir / "checkpoints",
        LOG_DIR=output_dir / "logs",
        TENSORBOARD_DIR=output_dir / "tensorboard",
        STATS_DIR=output_dir / "stats",
        SPLIT_JSON=output_dir / "split.json",
        TRAIN_STATS_DIR=output_dir / "train_stats",
        CT_ROOT=tmp_path / "ct_root",
        CT_TEST=tmp_path / "ct_test",
        PERSISTENT_DATASET_DIR=None,
        PRE_TRAINED_MODEL_PATH=None,
        LOG_LEVEL_CONSOLE="INFO",
        LOG_LEVEL_FILE="DEBUG",
        ENABLE_EMAIL_NOTIFICATIONS=False,
        SMTP_HOST="",
        SMTP_PORT=-1,
        EMAIL_SENDER="",
        EMAIL_PASSWORD="",
        EMAIL_RECIPIENT="",
        ENABLE_TELEGRAM_NOTIFICATIONS=False,
        TELEGRAM_BOT_TOKEN="",
        TELEGRAM_CHAT_ID="",
    )
