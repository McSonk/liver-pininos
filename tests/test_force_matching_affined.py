"""Unit tests for idssp.sonk.model.transforms.ForceMatchingAffined.

Covers affine placeholder detection, affine normalisation, case-name
validation, and the __call__ correction behaviour.

All tests use synthetic affines and lightweight fake metadata objects (plus a
real MONAI MetaTensor case) created in memory. No real LiTS data, network, or
real environment settings are used.

The tests deliberately preserve the production invariants:

- _ALLOWED_LITS_VOLUMES is {48, 49, 50, 51, 52} (not widened).
- _IDENTITY_AFFINE_THRESHOLD is 1e-3 (not changed).
"""

import numpy as np
import pytest
import torch

from idssp.sonk.model.transforms import (
    _ALLOWED_LITS_VOLUMES,
    _IDENTITY_AFFINE_THRESHOLD,
    ForceMatchingAffined,
)


# ---------------------------------------------------------------------------
# Constants and identity affines
# ---------------------------------------------------------------------------

def test_allowed_lits_volumes_unchanged():
    assert _ALLOWED_LITS_VOLUMES == frozenset({48, 49, 50, 51, 52})


def test_identity_affine_threshold_unchanged():
    assert _IDENTITY_AFFINE_THRESHOLD == 1e-3


def _identity_affine():
    return torch.eye(4)


def _non_identity_affine(spacing=2.0):
    aff = torch.eye(4)
    aff[0, 0] = spacing
    aff[1, 1] = spacing
    aff[2, 2] = spacing
    return aff


# ---------------------------------------------------------------------------
# _is_placeholder_affine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "affine",
    [
        _identity_affine(),
        _identity_affine().numpy(),
        torch.eye(4)[None],  # batched (1, 4, 4)
    ],
)
def test_is_placeholder_affine_detects_identity(affine):
    assert ForceMatchingAffined()._is_placeholder_affine(affine)


def test_is_placeholder_affine_detects_identity_like_with_translation():
    aff = torch.eye(4)
    aff[0, 3] = 12.0
    aff[1, 3] = -3.5
    aff[2, 3] = 7.0
    assert ForceMatchingAffined()._is_placeholder_affine(aff)


def test_is_placeholder_affine_rejects_non_identity_spacing():
    assert not ForceMatchingAffined()._is_placeholder_affine(_non_identity_affine())


def test_is_placeholder_affine_rejects_none():
    assert not ForceMatchingAffined()._is_placeholder_affine(None)


def test_is_placeholder_affine_rejects_near_identity_above_threshold():
    # Scale just above the 1e-3 threshold must not be considered a placeholder.
    aff = torch.eye(4)
    aff[0, 0] = 1.0 + 2 * _IDENTITY_AFFINE_THRESHOLD
    assert not ForceMatchingAffined()._is_placeholder_affine(aff)


# ---------------------------------------------------------------------------
# _normalise_affine
# ---------------------------------------------------------------------------

def test_normalise_affine_none_returns_none():
    assert ForceMatchingAffined._normalise_affine(None) is None


def test_normalise_affine_accepts_torch_tensor():
    aff = ForceMatchingAffined._normalise_affine(torch.eye(4))
    assert isinstance(aff, torch.Tensor)
    assert aff.shape == (4, 4)


def test_normalise_affine_accepts_numpy_array():
    aff = ForceMatchingAffined._normalise_affine(np.eye(4))
    assert isinstance(aff, torch.Tensor)
    assert aff.shape == (4, 4)


def test_normalise_affine_accepts_batched_shape():
    aff = ForceMatchingAffined._normalise_affine(torch.eye(4).unsqueeze(0))
    assert aff.shape == (4, 4)


def test_normalise_affine_returns_cpu_float32():
    aff = ForceMatchingAffined._normalise_affine(torch.eye(4, dtype=torch.float64))
    assert aff.dtype == torch.float32
    assert aff.device.type == "cpu"


def test_normalise_affine_rejects_invalid_2d_shape():
    with pytest.raises(ValueError, match="shape \\(4, 4\\)"):
        ForceMatchingAffined._normalise_affine(torch.ones(3, 3))


def test_normalise_affine_rejects_wrong_batch_size():
    with pytest.raises(ValueError, match="batch size 1"):
        ForceMatchingAffined._normalise_affine(torch.ones(2, 4, 4))


# ---------------------------------------------------------------------------
# _validate_case_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("volume_id", [48, 49, 50, 51, 52])
def test_validate_case_name_accepts_allowed_ids(volume_id):
    name = f"segmentation-{volume_id}.nii.gz"
    assert ForceMatchingAffined._validate_case_name(name) == volume_id


@pytest.mark.parametrize("volume_id", [47, 53])
def test_validate_case_name_rejects_disallowed_ids(volume_id):
    name = f"segmentation-{volume_id}.nii.gz"
    with pytest.raises(ValueError, match="only validated for LiTS volumes"):
        ForceMatchingAffined._validate_case_name(name)


@pytest.mark.parametrize(
    "malformed",
    ["volume-52.nii.gz", "segmentation-52.nii", "segmentation-52", "52.nii.gz", ""],
)
def test_validate_case_name_rejects_malformed_names(malformed):
    with pytest.raises(ValueError, match="unexpected filename format"):
        ForceMatchingAffined._validate_case_name(malformed)


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------

class _FakeMeta:
    """Minimal stand-in for a MONAI object exposing `.meta` and `.affine`."""
    def __init__(self, meta, affine=None):
        self.meta = meta
        self.affine = affine


class _ReadOnlyAffineMeta:
    """Stand-in that raises AttributeError when trying to set `.affine`.
    This forces the __call__ method to use the fallback: label.meta["affine"] = ...
    """
    __slots__ = ("meta",)
    
    def __init__(self, meta):
        self.meta = meta


def _make_transform():
    return ForceMatchingAffined(image_key="image", label_key="label")


def _data_for(case_name, img_affine, lbl_affine, lbl_obj=None):
    image = _FakeMeta({"affine": img_affine}, affine=img_affine)
    if lbl_obj is None:
        label = _FakeMeta(
            {"affine": lbl_affine, "filename_or_obj": case_name},
            affine=lbl_affine,
        )
    else:
        label = lbl_obj
    return {"image": image, "label": label}


def test_call_returns_data_when_objects_lack_meta():
    data = {"image": "volume", "label": "seg"}
    out = _make_transform()(data)
    assert out is data


def test_call_returns_data_when_affine_missing():
    image = _FakeMeta({})
    label = _FakeMeta({"filename_or_obj": "segmentation-52.nii.gz"})
    data = {"image": image, "label": label}
    out = _make_transform()(data)
    assert out is data


def test_call_copies_image_affine_to_placeholder_label():
    img_aff = _non_identity_affine(spacing=1.5)
    lbl_aff = _identity_affine()
    data = _data_for("segmentation-52.nii.gz", img_aff, lbl_aff)

    out = _make_transform()(data)

    assert torch.equal(out["label"].affine, img_aff)


def test_call_sets_meta_affine_when_affine_attr_missing():
    img_aff = _non_identity_affine(spacing=1.5)
    lbl_aff = _identity_affine()
    # Use the read-only object to trigger the AttributeError fallback
    label_obj = _ReadOnlyAffineMeta({"affine": lbl_aff, "filename_or_obj": "segmentation-52.nii.gz"})
    data = _data_for("segmentation-52.nii.gz", img_aff, lbl_aff, lbl_obj=label_obj)

    out = _make_transform()(data)

    assert torch.equal(out["label"].meta["affine"], img_aff)


def test_call_raises_for_unapproved_volume_when_correction_would_apply():
    img_aff = _non_identity_affine(spacing=2.0)
    lbl_aff = _identity_affine()
    data = _data_for("segmentation-47.nii.gz", img_aff, lbl_aff)

    with pytest.raises(ValueError, match="only validated for LiTS volumes"):
        _make_transform()(data)


def test_call_does_not_overwrite_valid_label_affine():
    img_aff = _non_identity_affine(spacing=1.5)
    lbl_aff = _non_identity_affine(spacing=3.0)
    data = _data_for("segmentation-52.nii.gz", img_aff, lbl_aff)

    out = _make_transform()(data)

    assert torch.equal(out["label"].affine, lbl_aff)


def test_call_no_correction_when_label_valid_and_image_placeholder():
    # Even for a non-approved volume, no correction happens (and thus no
    # validation is triggered) when the label affine is not a placeholder.
    img_aff = _identity_affine()
    lbl_aff = _non_identity_affine(spacing=2.0)
    data = _data_for("segmentation-47.nii.gz", img_aff, lbl_aff)

    out = _make_transform()(data)

    assert torch.equal(out["label"].affine, lbl_aff)


def test_call_real_metatensor_copies_affine():
    from monai.data import MetaTensor

    img = MetaTensor(torch.zeros(1, 4, 4, 4), affine=_non_identity_affine(2.0))
    label = MetaTensor(torch.zeros(1, 4, 4, 4), affine=_identity_affine())
    label.meta["filename_or_obj"] = "segmentation-52.nii.gz"

    data = {"image": img, "label": label}
    out = _make_transform()(data)

    # MONAI's MetaTensor enforces float64 for affines, so we cast the expected
    # float32 tensor to match the actual dtype before comparing.
    expected = _non_identity_affine(2.0).to(out["label"].affine.dtype)
    assert torch.allclose(out["label"].affine, expected)
