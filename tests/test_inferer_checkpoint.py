"""Unit tests for checkpoint loading and the limited-environment guard.

Targets ``idssp/sonk/model/inferer``:

- ``InferenceEngine.load_checkpoint``
- the limited-environment guard inside ``InferenceEngine.run_inference``

No real checkpoint files, models, or full inference are exercised here; the
heavy dependencies (``torch.load``, ``get_model``, ``SlidingWindowInferer``,
``get_validation_transforms``) are patched so only the alignment logic and the
guard are exercised.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from idssp.sonk.config import AvailableModels
from idssp.sonk.model import inferer


def _make_fake_checkpoint() -> dict:
    """Return a fake checkpoint payload with a training config snapshot."""
    return {
        "config_snapshot": {
            "NUM_CLASSES": 3,
            "ISO_SPACING": [1.0, 1.0, 1.0],
            "HU_WINDOW_MIN": -175,
            "HU_WINDOW_MAX": 250,
            "TRAIN_PATCH_SIZE": [64, 64, 64],
            "MODEL": "seg-res-net",
            "TUMOUR_CLASS_INDEX": 2,
            "SLIDING_WINDOW_BATCH_SIZE": 99,
            "RAND_CROP_NUM_SAMPLES": 99,
        },
        "model_state_dict": {},
        "best_dice": 0.8,
    }


def _write_dummy_checkpoint(tmp_path) -> Path:
    """Create a dummy checkpoint file under ``tmp_path``."""
    path = tmp_path / "best_model.pth"
    path.write_bytes(b"dummy checkpoint bytes")
    return path


def _patch_inferer_dependencies(monkeypatch, minimal_config, checkpoint_payload, checkpoint_path):
    """Patch the external dependencies used by ``load_checkpoint``.

    Returns the patched mock model so tests can assert on its usage.
    """
    monkeypatch.setattr(inferer.config, "get", lambda: minimal_config)
    monkeypatch.setattr(inferer.torch, "load", MagicMock(return_value=checkpoint_payload))
    mock_model = MagicMock()
    monkeypatch.setattr(inferer, "get_model", MagicMock(return_value=mock_model))
    monkeypatch.setattr(inferer, "SlidingWindowInferer", MagicMock())
    return mock_model


# --------------------------------------------------------------------------- #
# load_checkpoint
# --------------------------------------------------------------------------- #

def test_load_checkpoint_missing_path_raises_file_not_found(monkeypatch, minimal_config):
    monkeypatch.setattr(inferer.config, "get", lambda: minimal_config)
    missing = Path("/nonexistent/checkpoint.pth")

    engine = inferer.InferenceEngine(missing)

    with pytest.raises(FileNotFoundError):
        engine.load_checkpoint()


def test_load_checkpoint_converts_list_spacing_to_tuple(monkeypatch, tmp_path, minimal_config):
    payload = _make_fake_checkpoint()
    ckpt_path = _write_dummy_checkpoint(tmp_path)
    _patch_inferer_dependencies(monkeypatch, minimal_config, payload, ckpt_path)

    engine = inferer.InferenceEngine(ckpt_path)
    engine.load_checkpoint()

    assert isinstance(engine.config.ISO_SPACING, tuple)
    assert isinstance(engine.config.TRAIN_PATCH_SIZE, tuple)
    assert engine.config.ISO_SPACING == (1.0, 1.0, 1.0)
    assert engine.config.TRAIN_PATCH_SIZE == (64, 64, 64)


def test_load_checkpoint_aligns_strict_keys(monkeypatch, tmp_path, minimal_config):
    payload = _make_fake_checkpoint()
    ckpt_path = _write_dummy_checkpoint(tmp_path)
    _patch_inferer_dependencies(monkeypatch, minimal_config, payload, ckpt_path)

    engine = inferer.InferenceEngine(ckpt_path)
    engine.load_checkpoint()

    assert engine.config.NUM_CLASSES == 3
    assert engine.config.HU_WINDOW_MIN == -175
    assert engine.config.HU_WINDOW_MAX == 250
    assert engine.config.TUMOUR_CLASS_INDEX == 2


def test_load_checkpoint_converts_model_string_to_enum(monkeypatch, tmp_path, minimal_config):
    payload = _make_fake_checkpoint()
    ckpt_path = _write_dummy_checkpoint(tmp_path)
    _patch_inferer_dependencies(monkeypatch, minimal_config, payload, ckpt_path)

    engine = inferer.InferenceEngine(ckpt_path)
    engine.load_checkpoint()

    assert engine.config.MODEL == AvailableModels.SEG_RES_NET
    assert isinstance(engine.config.MODEL, AvailableModels)


def test_load_checkpoint_warns_on_noncritical_mismatch(monkeypatch, tmp_path, minimal_config, caplog):
    payload = _make_fake_checkpoint()
    ckpt_path = _write_dummy_checkpoint(tmp_path)
    _patch_inferer_dependencies(monkeypatch, minimal_config, payload, ckpt_path)

    with caplog.at_level("WARNING"):
        engine = inferer.InferenceEngine(ckpt_path)
        engine.load_checkpoint()

    assert any("SLIDING_WINDOW_BATCH_SIZE" in r.message for r in caplog.records)
    assert any("RAND_CROP_NUM_SAMPLES" in r.message for r in caplog.records)


def test_load_checkpoint_builds_model_from_aligned_config(monkeypatch, tmp_path, minimal_config):
    payload = _make_fake_checkpoint()
    ckpt_path = _write_dummy_checkpoint(tmp_path)
    _patch_inferer_dependencies(monkeypatch, minimal_config, payload, ckpt_path)

    engine = inferer.InferenceEngine(ckpt_path)
    engine.load_checkpoint()

    _, kwargs = inferer.get_model.call_args
    assert kwargs == {}
    aligned_config = inferer.get_model.call_args.args[0]
    assert aligned_config.NUM_CLASSES == 3
    assert aligned_config.MODEL == AvailableModels.SEG_RES_NET


# --------------------------------------------------------------------------- #
# run_inference limited-environment guard
# --------------------------------------------------------------------------- #

def test_run_inference_limited_env_raises_runtime_error(monkeypatch, tmp_path, minimal_config):
    ckpt_path = _write_dummy_checkpoint(tmp_path)
    _patch_inferer_dependencies(monkeypatch, minimal_config, _make_fake_checkpoint(), ckpt_path)
    monkeypatch.setattr(inferer, "get_validation_transforms", MagicMock())
    monkeypatch.setattr(inferer.config, "is_limited_env", lambda config=None, include_vram=True: True)

    engine = inferer.InferenceEngine(ckpt_path)
    engine.load_checkpoint()

    with pytest.raises(RuntimeError):
        engine.run_inference(test_files=[], save_path=tmp_path / "out")
