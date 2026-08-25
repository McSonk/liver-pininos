"""Unit tests for idssp.sonk.model.training logic-only helpers.

Covers AugmentedDataset, ModelBuilder._should_log_overlay,
ModelBuilder._should_notify, ModelBuilder._validate_checkpoint,
and EarlyStopper.__call__. All tests use mocks and synthetic data;
no GPU, network, real LiTS data, or real .env is used.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from idssp.sonk.model.training import (
    AugmentedDataset,
    EarlyStopper,
    ModelBuilder,
)


# ---------------------------------------------------------------------------
# AugmentedDataset
# ---------------------------------------------------------------------------

class FakeBaseDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_augmented_dataset_length_matches_base():
    base = FakeBaseDataset([1, 2, 3])
    aug = AugmentedDataset.__new__(AugmentedDataset)
    aug.base_ds = base
    aug.aug = lambda x: x  # identity transform
    assert len(aug) == 3


def test_augmented_dataset_getitem_applies_random_transform():
    base = FakeBaseDataset([10, 20])
    captured = {}

    def record_transform(x):
        captured["input"] = x
        return x * 2

    aug = AugmentedDataset.__new__(AugmentedDataset)
    aug.base_ds = base
    aug.aug = record_transform

    result = aug[1]
    assert captured["input"] == 20
    assert result == 40


# ---------------------------------------------------------------------------
# ModelBuilder._should_log_overlay
# ---------------------------------------------------------------------------

def test_should_log_overlay_epoch_le_10_returns_true():
    mb = ModelBuilder.__new__(ModelBuilder)
    for epoch in range(0, 11):  # 0-indexed: epochs 0-10 correspond to first 11 epochs
        assert mb._should_log_overlay(epoch) is True


@pytest.mark.parametrize("epoch,expected", [
    (10, True),    # epoch 10 (11th epoch) <= 10
    (11, False),   # 11 % 5 = 1
    (12, False),   # 12 % 5 = 2
    (13, False),   # 13 % 5 = 3
    (14, False),   # 14 % 5 = 4
    (15, True),    # 15 % 5 = 0
    (16, False),
    (20, True),
    (25, True),
    (30, True),
])
def test_should_log_overlay_epoch_11_to_30_every_5(epoch, expected):
    mb = ModelBuilder.__new__(ModelBuilder)
    assert mb._should_log_overlay(epoch) == expected


@pytest.mark.parametrize("epoch,expected", [
    (30, True),    # 30 is in <= 30 range, handled above
    (31, False),   # 31 % 10 = 1
    (40, True),    # 40 % 10 = 0
    (41, False),
    (50, True),
    (60, True),
])
def test_should_log_overlay_epoch_ge_31_every_10(epoch, expected):
    mb = ModelBuilder.__new__(ModelBuilder)
    assert mb._should_log_overlay(epoch) == expected


# ---------------------------------------------------------------------------
# ModelBuilder._should_notify
# ---------------------------------------------------------------------------

def test_should_notify_epoch_le_50_every_5():
    mb = ModelBuilder.__new__(ModelBuilder)
    for epoch in range(0, 51):
        if epoch % 5 == 0:
            assert mb._should_notify(epoch) is True
        else:
            assert mb._should_notify(epoch) is False


@pytest.mark.parametrize("epoch,expected", [
    (50, True),    # 50 % 5 = 0
    (51, False),   # 51 % 10 = 1
    (60, True),    # 60 % 10 = 0
    (61, False),
    (100, True),
    (101, False),  # 101 % 20 = 1
    (120, True),   # 120 % 20 = 0
    (121, False),
    (140, True),
])
def test_should_notify_boundary_epochs(epoch, expected):
    mb = ModelBuilder.__new__(ModelBuilder)
    assert mb._should_notify(epoch) == expected


# ---------------------------------------------------------------------------
# ModelBuilder._validate_checkpoint
# ---------------------------------------------------------------------------

class MockConfig:
    def __init__(self):
        self.MODEL = MagicMock()
        self.MODEL.value = "SEG_RES_NET"
        self.NUM_CLASSES = 3
        self.NUM_EPOCHS = 200
        self.ISO_SPACING = 1.0
        self.HU_WINDOW_MIN = -175
        self.HU_WINDOW_MAX = 250


def _make_builder_with_config():
    mb = ModelBuilder.__new__(ModelBuilder)
    mb.config = MockConfig()
    return mb


def test_validate_checkpoint_missing_model_state_dict_raises():
    mb = _make_builder_with_config()
    checkpoint = {"epoch": 5}
    with pytest.raises(ValueError, match="Required key 'model_state_dict' not found"):
        mb._validate_checkpoint(checkpoint, Path("fake.pth"))


def test_validate_checkpoint_model_mismatch_raises():
    mb = _make_builder_with_config()
    checkpoint = {
        "model_state_dict": {},
        "config_snapshot": {"MODEL": "U_NET", "NUM_CLASSES": 3},
    }
    with pytest.raises(ValueError, match="MODEL mismatch"):
        mb._validate_checkpoint(checkpoint, Path("fake.pth"))


def test_validate_checkpoint_num_classes_mismatch_raises():
    mb = _make_builder_with_config()
    checkpoint = {
        "model_state_dict": {},
        "config_snapshot": {"MODEL": "SEG_RES_NET", "NUM_CLASSES": 2},
    }
    with pytest.raises(ValueError, match="NUM_CLASSES mismatch"):
        mb._validate_checkpoint(checkpoint, Path("fake.pth"))


def test_validate_checkpoint_preprocessing_mismatch_warns(caplog):
    mb = _make_builder_with_config()
    checkpoint = {
        "model_state_dict": {},
        "config_snapshot": {
            "MODEL": "SEG_RES_NET",
            "NUM_CLASSES": 3,
            "ISO_SPACING": 2.0,
            "HU_WINDOW_MIN": -100,
            "HU_WINDOW_MAX": 200,
        },
    }
    caplog.set_level("WARNING")
    mb._validate_checkpoint(checkpoint, Path("fake.pth"))
    assert "Preprocessing mismatch" in caplog.text
    assert "ISO_SPACING" in caplog.text
    assert "HU_WINDOW_MIN" in caplog.text
    assert "HU_WINDOW_MAX" in caplog.text


def test_validate_checkpoint_missing_config_snapshot_warns(caplog):
    mb = _make_builder_with_config()
    checkpoint = {"model_state_dict": {}}
    caplog.set_level("WARNING")
    mb._validate_checkpoint(checkpoint, Path("fake.pth"))
    assert "does not contain 'config_snapshot'" in caplog.text


def test_validate_checkpoint_epoch_ge_num_epochs_raises():
    mb = _make_builder_with_config()
    checkpoint = {
        "model_state_dict": {},
        "config_snapshot": {"MODEL": "SEG_RES_NET", "NUM_CLASSES": 3},
        "epoch": 200,  # equal to NUM_EPOCHS
    }
    with pytest.raises(RuntimeError, match="Checkpoint epoch.*>=.*current NUM_EPOCHS"):
        mb._validate_checkpoint(checkpoint, Path("fake.pth"))

    checkpoint["epoch"] = 201
    with pytest.raises(RuntimeError, match="Checkpoint epoch.*>=.*current NUM_EPOCHS"):
        mb._validate_checkpoint(checkpoint, Path("fake.pth"))


def test_validate_checkpoint_valid_passes():
    mb = _make_builder_with_config()
    checkpoint = {
        "model_state_dict": {},
        "config_snapshot": {"MODEL": "SEG_RES_NET", "NUM_CLASSES": 3},
        "epoch": 50,
    }
    mb._validate_checkpoint(checkpoint, Path("fake.pth"))  # should not raise


# ---------------------------------------------------------------------------
# EarlyStopper
# ---------------------------------------------------------------------------

class MockConfigForEarlyStopper:
    """Mock config with all attributes accessed by EarlyStopper and save_checkpoint."""
    VERSION = "test"
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.001
    CHECKPOINT_DIR = Path("/tmp")
    MODEL = MagicMock()
    MODEL.value = "SEG_RES_NET"
    NUM_CLASSES = 3
    DICE_CE_WEIGHTS = [1.0, 1.0, 1.0]
    LEARNING_RATE = 1e-4
    WARMUP_EPOCHS = 5
    COSINE_ETA_MIN = 1e-6
    DEVICE = "cpu"

    def to_dict(self):
        return {
            "VERSION": self.VERSION,
            "MODEL": self.MODEL.value,
            "NUM_CLASSES": self.NUM_CLASSES,
            "ISO_SPACING": 1.0,
            "HU_WINDOW_MIN": -175,
            "HU_WINDOW_MAX": 250,
            "EARLY_STOPPING_PATIENCE": self.EARLY_STOPPING_PATIENCE,
            "EARLY_STOPPING_MIN_DELTA": self.EARLY_STOPPING_MIN_DELTA,
        }


def _make_fake_builder(tmp_path):
    """Create a fake builder with all attributes needed by EarlyStopper.save_checkpoint."""
    builder = MagicMock()
    builder.writer = MagicMock()
    builder.config = MockConfigForEarlyStopper()
    builder.config.CHECKPOINT_DIR = tmp_path

    # Mock model with state_dict
    mock_model = MagicMock()
    mock_model.state_dict.return_value = {}
    builder.model = mock_model

    # Mock optimizer with state_dict
    mock_optimizer = MagicMock()
    mock_optimizer.state_dict.return_value = {}
    mock_optimizer.param_groups = [{"lr": 1e-4}]
    builder.optimizer = mock_optimizer

    # Mock scaler
    builder.scaler = None

    # Mock scheduler
    builder.scheduler = None

    # Mock history
    builder.history = {"train_loss": [], "val_loss": [], "val_dice": []}

    return builder


def test_early_stopper_improvement_resets_counter_and_returns_false(tmp_path, monkeypatch):
    # Patch config.get to return our mock config
    monkeypatch.setattr("idssp.sonk.model.training.config.get", lambda: MockConfigForEarlyStopper())

    builder = _make_fake_builder(tmp_path)
    es = EarlyStopper(builder)

    # First call: improvement from -1.0 to 0.5
    result = es(0, 0.5, 0.6, 0.5)
    assert result is False
    assert es.best_tumour_dice == 0.5
    assert es.epochs_no_improve == 0
    builder.writer.add_scalar.assert_called()
    builder.save_checkpoint.assert_called_once()


def test_early_stopper_no_improvement_increments_counter(tmp_path, monkeypatch):
    monkeypatch.setattr("idssp.sonk.model.training.config.get", lambda: MockConfigForEarlyStopper())

    builder = _make_fake_builder(tmp_path)
    es = EarlyStopper(builder)

    # First call establishes baseline
    es(0, 0.5, 0.6, 0.5)
    # Second call: no improvement (0.5005 is within min_delta)
    result = es(1, 0.5005, 0.6, 0.5005)
    assert result is False
    assert es.epochs_no_improve == 1
    # Third call: no improvement
    result = es(2, 0.5005, 0.6, 0.5005)
    assert result is False
    assert es.epochs_no_improve == 2


def test_early_stopper_improvement_requires_exceeds_min_delta(tmp_path, monkeypatch):
    monkeypatch.setattr("idssp.sonk.model.training.config.get", lambda: MockConfigForEarlyStopper())

    builder = _make_fake_builder(tmp_path)
    es = EarlyStopper(builder)

    es(0, 0.5, 0.6, 0.5)
    # Exactly at min_delta should NOT be improvement (strict >)
    result = es(1, 0.501, 0.6, 0.501)  # 0.501 == 0.5 + 0.001
    assert result is False
    assert es.epochs_no_improve == 1  # no improvement
    # Just above min_delta SHOULD be improvement
    result = es(2, 0.5011, 0.6, 0.5011)
    assert result is False
    assert es.epochs_no_improve == 0
    assert es.best_tumour_dice == 0.5011


def test_early_stopper_returns_true_after_patience_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr("idssp.sonk.model.training.config.get", lambda: MockConfigForEarlyStopper())

    builder = _make_fake_builder(tmp_path)
    es = EarlyStopper(builder)

    # Establish baseline
    es(0, 0.5, 0.6, 0.5)
    # 3 epochs with no improvement (patience = 3)
    es(1, 0.5, 0.6, 0.5)  # epochs_no_improve = 1
    es(2, 0.5, 0.6, 0.5)  # epochs_no_improve = 2
    result = es(3, 0.5, 0.6, 0.5)  # epochs_no_improve = 3 == patience -> should return True
    assert result is True


def test_early_stopper_monitors_tumour_dice_only(tmp_path, monkeypatch):
    monkeypatch.setattr("idssp.sonk.model.training.config.get", lambda: MockConfigForEarlyStopper())

    builder = _make_fake_builder(tmp_path)
    es = EarlyStopper(builder)

    # Establish baseline with good tumour dice
    es(0, 0.5, 0.6, 0.5)
    # Mean dice improves but tumour dice does not - should NOT trigger improvement
    result = es(1, 0.7, 0.8, 0.5)
    assert result is False
    assert es.epochs_no_improve == 1
    # Liver dice improves but tumour dice does not - should NOT trigger improvement
    result = es(2, 0.5, 0.8, 0.5)
    assert result is False
    assert es.epochs_no_improve == 2


def test_early_stopper_writer_and_checkpoint_only_on_improvement(tmp_path, monkeypatch):
    monkeypatch.setattr("idssp.sonk.model.training.config.get", lambda: MockConfigForEarlyStopper())

    builder = _make_fake_builder(tmp_path)
    es = EarlyStopper(builder)

    # Improvement call
    es(0, 0.5, 0.6, 0.5)
    assert builder.writer.add_scalar.call_count == 3  # Mean, Liver, Tumour
    assert builder.save_checkpoint.call_count == 1

    # Reset mocks
    builder.writer.reset_mock()
    builder.save_checkpoint.reset_mock()

    # No improvement call
    es(1, 0.5, 0.6, 0.5)
    # Should NOT call writer or save_checkpoint
    assert builder.writer.add_scalar.call_count == 0
    assert builder.save_checkpoint.call_count == 0


# ---------------------------------------------------------------------------
# EarlyStopper exact patience sequence
# ---------------------------------------------------------------------------

def test_early_stopper_exact_patience_sequence(tmp_path, monkeypatch):
    """Verify the exact sequence: after improvement, P consecutive non-improving
    calls return False; the (P+1)th non-improving call returns True."""
    monkeypatch.setattr("idssp.sonk.model.training.config.get", lambda: MockConfigForEarlyStopper())

    builder = _make_fake_builder(tmp_path)
    es = EarlyStopper(builder)
    patience = MockConfigForEarlyStopper.EARLY_STOPPING_PATIENCE  # 3

    # Initial improvement
    es(0, 0.5, 0.6, 0.5)
    assert es.epochs_no_improve == 0

    # P consecutive non-improving epochs should return False
    for i in range(patience):
        result = es(i + 1, 0.5, 0.6, 0.5)
        assert result is False, f"Epoch {i + 1} should return False"
        assert es.epochs_no_improve == i + 1

    # The next non-improving epoch should return True
    result = es(patience + 1, 0.5, 0.6, 0.5)
    assert result is True