"""Unit tests for the model factory and pretrained-weight loading.

Targets ``idssp/sonk/model/models``:

- ``_load_monai_pretrained_weights``
- ``get_model``
- ``get_swin_unetr`` (patch-size validation only)
- ``get_swin_unetr_pretrain`` (error path)

No real MONAI networks (UNet, SegResNet, SwinUNETR) are constructed here; the
heavy model classes are patched so only the factory logic is exercised.
"""

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from idssp.sonk.config import AvailableModels
from idssp.sonk.model import models


class _TinyModule(nn.Module):
    """Minimal real module used as a stand-in for a segmentation head."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


def _make_torch_load(monkeypatch, payload):
    """Force ``torch.load`` inside the models module to return ``payload``."""
    monkeypatch.setattr(models.torch, "load", MagicMock(return_value=payload))


def _patch_load_state(model, monkeypatch):
    """Replace a real model's ``load_state_dict`` with a recording mock.
    
    PyTorch's load_state_dict returns a named tuple of (missing_keys, unexpected_keys).
    Because the production code unpacks this into two variables, the mock must 
    be configured to return a 2-tuple.
    """
    spy = MagicMock(return_value=([], []))
    monkeypatch.setattr(model, "load_state_dict", spy)
    return spy


# --------------------------------------------------------------------------- #
# get_model dispatch
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "model_enum, builder_name",
    [
        (AvailableModels.U_NET, "get_unet"),
        (AvailableModels.SEG_RES_NET, "get_seg_res_net"),
        (AvailableModels.SWIN_UNETR, "get_swin_unetr"),
        (AvailableModels.SWIN_UNETR_PRETRAIN, "get_swin_unetr_pretrain"),
    ],
)
def test_get_model_dispatches_to_builder(monkeypatch, minimal_config, model_enum, builder_name):
    sentinel = object()
    builder = MagicMock(return_value=sentinel)
    monkeypatch.setattr(models, builder_name, builder)
    cfg = dataclasses.replace(minimal_config, MODEL=model_enum)

    result = models.get_model(cfg)

    assert result is sentinel
    builder.assert_called_once_with(cfg)


def test_get_model_unsupported_model_raises_value_error(monkeypatch, minimal_config):
    for name in ("get_unet", "get_seg_res_net", "get_swin_unetr", "get_swin_unetr_pretrain"):
        monkeypatch.setattr(models, name, MagicMock())
    cfg = dataclasses.replace(minimal_config, MODEL="unsupported-model")

    with pytest.raises(ValueError):
        models.get_model(cfg)


# --------------------------------------------------------------------------- #
# get_swin_unetr patch-size validation
# --------------------------------------------------------------------------- #

def test_get_swin_unetr_wrong_spatial_dims_raises_value_error(monkeypatch, minimal_config):
    monkeypatch.setattr(models, "SwinUNETR", MagicMock())
    cfg = dataclasses.replace(minimal_config, TRAIN_PATCH_SIZE=(64, 64))

    with pytest.raises(ValueError):
        models.get_swin_unetr(cfg)


def test_get_swin_unetr_dimension_not_divisible_raises_value_error(monkeypatch, minimal_config):
    monkeypatch.setattr(models, "SwinUNETR", MagicMock())
    cfg = dataclasses.replace(minimal_config, TRAIN_PATCH_SIZE=(64, 64, 33))

    with pytest.raises(ValueError):
        models.get_swin_unetr(cfg)


def test_get_swin_unetr_valid_size_uses_patched_class(monkeypatch, minimal_config):
    sentinel = object()
    patched = MagicMock(return_value=sentinel)
    monkeypatch.setattr(models, "SwinUNETR", patched)

    result = models.get_swin_unetr(minimal_config)

    assert result is sentinel
    patched.assert_called_once_with(
        spatial_dims=3,
        in_channels=1,
        out_channels=minimal_config.NUM_CLASSES,
        feature_size=48,
        use_checkpoint=True,
        norm_name="instance",
    )


# --------------------------------------------------------------------------- #
# _load_monai_pretrained_weights
# --------------------------------------------------------------------------- #

def test_load_accepts_direct_state_dict(monkeypatch):
    model = _TinyModule()
    state_dict = dict(model.state_dict())
    _make_torch_load(monkeypatch, state_dict)
    spy = _patch_load_state(model, monkeypatch)

    models._load_monai_pretrained_weights(model, Path("pretrained.pth"))

    spy.assert_called_once()
    call_args, call_kwargs = spy.call_args
    assert set(call_args[0].keys()) == {"linear.weight", "linear.bias"}
    assert call_kwargs["strict"] is False


def test_load_requests_weights_only(monkeypatch):
    model = _TinyModule()
    _make_torch_load(monkeypatch, dict(model.state_dict()))
    _patch_load_state(model, monkeypatch)

    models._load_monai_pretrained_weights(model, Path("pretrained.pth"))

    load_call = models.torch.load.call_args
    assert load_call.kwargs["weights_only"] is True


@pytest.mark.parametrize("wrapper_key", ["state_dict", "model_state_dict", "model"])
def test_load_accepts_wrapped_state_dict(monkeypatch, wrapper_key):
    model = _TinyModule()
    inner = dict(model.state_dict())
    _make_torch_load(monkeypatch, {wrapper_key: inner})
    spy = _patch_load_state(model, monkeypatch)

    models._load_monai_pretrained_weights(model, Path("pretrained.pth"))

    call_args, _ = spy.call_args
    assert set(call_args[0].keys()) == {"linear.weight", "linear.bias"}


def test_load_full_module_checkpoint_raises_runtime_error(tmp_path):
    """A full-module checkpoint is rejected under weights_only=True."""
    model = _TinyModule()
    path = tmp_path / "full_module.pth"
    torch.save(model, path)
    with pytest.raises(RuntimeError):
        models._load_monai_pretrained_weights(model, path)


def test_load_strips_module_prefix(monkeypatch):
    model = _TinyModule()
    prefixed = {f"module.{k}": v for k, v in model.state_dict().items()}
    _make_torch_load(monkeypatch, prefixed)
    spy = _patch_load_state(model, monkeypatch)

    models._load_monai_pretrained_weights(model, Path("pretrained.pth"))

    call_args, _ = spy.call_args
    assert set(call_args[0].keys()) == {"linear.weight", "linear.bias"}


def test_load_torch_failure_raises_runtime_error(monkeypatch):
    model = _TinyModule()

    def failing_load(*args, **kwargs):
        raise OSError("simulated io failure")

    monkeypatch.setattr(models.torch, "load", failing_load)
    _patch_load_state(model, monkeypatch)

    with pytest.raises(RuntimeError):
        models._load_monai_pretrained_weights(model, Path("pretrained.pth"))


def test_load_unsupported_format_raises_type_error(monkeypatch):
    model = _TinyModule()
    _make_torch_load(monkeypatch, 12345)
    _patch_load_state(model, monkeypatch)

    with pytest.raises(TypeError):
        models._load_monai_pretrained_weights(model, Path("pretrained.pth"))


# --------------------------------------------------------------------------- #
# get_swin_unetr_pretrain error path
# --------------------------------------------------------------------------- #

def test_get_swin_unetr_pretrain_wraps_loading_failure(monkeypatch, minimal_config):
    monkeypatch.setattr(models, "SwinUNETR", MagicMock())

    def failing_loader(*args, **kwargs):
        raise ValueError("pretrained load exploded")

    monkeypatch.setattr(models, "_load_monai_pretrained_weights", failing_loader)
    cfg = dataclasses.replace(
        minimal_config,
        PRE_TRAINED_MODEL_PATH=Path("pretrained.pth"),
    )

    with pytest.raises(RuntimeError):
        models.get_swin_unetr_pretrain(cfg)
