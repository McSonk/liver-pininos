"""Unit tests for idssp.sonk.model.data CSV/data helpers.

Covers the pure helper functions used for per-case CSV export and the
`VolumeWrapper.find_slice_thresholds` logic. All inputs are synthetic,
in-memory pandas objects and small NumPy arrays; temporary files are
created only under pytest's ``tmp_path``.
No config singleton, network access, GPU, real LiTS data, or real environment
settings are used.
"""

import numpy as np
import pandas as pd
import pytest

from idssp.sonk.model.data import (
    DatasetSummary,
    VolumeWrapper,
    _BOOLEAN_COLUMNS,
    _INTEGER_COLUMNS,
    _canonicalise_numeric_columns,
    _format_csv_float,
    _is_scalar_number,
    _to_nullable_int,
)


# ---------------------------------------------------------------------------
# _format_csv_float
# ---------------------------------------------------------------------------

def test_format_csv_float_nan_returns_empty():
    assert _format_csv_float(np.nan) == ""
    assert _format_csv_float(float("nan")) == ""


def test_format_csv_float_roundtrips_via_17g():
    value = 0.00013178507486979167
    assert _format_csv_float(value) == format(float(value), ".17g")


def test_format_csv_float_integer_like_float_no_trailing_zero():
    assert _format_csv_float(70.0) == "70"
    assert _format_csv_float(6.404754638671875) == "6.404754638671875"


def test_format_csv_float_accepts_int_input():
    assert _format_csv_float(3) == "3"


# ---------------------------------------------------------------------------
# _is_scalar_number
# ---------------------------------------------------------------------------

def test_is_scalar_number_accepts_python_numerics():
    assert _is_scalar_number(1)
    assert _is_scalar_number(1.5)


def test_is_scalar_number_accepts_numpy_numerics():
    assert _is_scalar_number(np.int64(3))
    assert _is_scalar_number(np.float32(2.5))


@pytest.mark.parametrize("value", [
    pytest.param(True, id="bool"),
    pytest.param("abc", id="str"),
    pytest.param(None, id="none"),
    pytest.param([1, 2], id="list"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param(np.array([1.0, 2.0]), id="numpy_array"),
])
def test_is_scalar_number_rejects_non_numbers(value):
    assert not _is_scalar_number(value)


# ---------------------------------------------------------------------------
# _to_nullable_int
# ---------------------------------------------------------------------------

def test_to_nullable_int_converts_integer_like_to_int64():
    series = pd.Series([1, 2, 3], name="case_index")
    result = _to_nullable_int(series)
    assert str(result.dtype) == "Int64"
    assert list(result) == [1, 2, 3]


def test_to_nullable_int_preserves_missing():
    series = pd.Series([1, None, 3], name="case_index")
    result = _to_nullable_int(series)
    assert str(result.dtype) == "Int64"
    assert result.isna().tolist() == [False, True, False]


def test_to_nullable_int_raises_for_fractional_values():
    series = pd.Series([1.0, 2.5, 3.0], name="x")
    with pytest.raises(ValueError, match="fractional"):
        _to_nullable_int(series)


def test_to_nullable_int_raises_for_non_numeric_values():
    series = pd.Series([1, "abc", 3], name="x")
    with pytest.raises(ValueError, match="non-numeric"):
        _to_nullable_int(series)


# ---------------------------------------------------------------------------
# _canonicalise_numeric_columns
# ---------------------------------------------------------------------------

def test_canonicalise_converts_integer_columns_to_nullable_int():
    df = pd.DataFrame({
        "liver_voxels": [10, 20, None],
        "has_tumor": [True, False, True],
    })
    result = _canonicalise_numeric_columns(df)
    assert str(result["liver_voxels"].dtype) == "Int64"
    assert str(result["has_tumor"].dtype) == "boolean"


def test_canonicalise_formats_float_columns_deterministically():
    df = pd.DataFrame({"ratio": [70.0, 0.5]})
    result = _canonicalise_numeric_columns(df)
    assert result["ratio"].tolist() == ["70", "0.5"]


def test_canonicalise_handles_object_column_with_none_and_numbers():
    df = pd.DataFrame({"col": [1.0, None, 2.0]})
    result = _canonicalise_numeric_columns(df)
    assert result["col"].tolist() == ["1", "", "2"]


def test_canonicalise_does_not_touch_non_numeric_object_column():
    df = pd.DataFrame({"col": ["a", "b", None]})
    result = _canonicalise_numeric_columns(df)
    assert result["col"].tolist() == ["a", "b", None]


def test_integer_and_boolean_column_sets_exist():
    assert "liver_first" in _INTEGER_COLUMNS
    assert "has_tumor" in _BOOLEAN_COLUMNS


# ---------------------------------------------------------------------------
# DatasetSummary._flatten_dict
# ---------------------------------------------------------------------------

def _make_summary():
    return object.__new__(DatasetSummary)


def test_flatten_dict_flattens_nested_with_underscore():
    summary = _make_summary()
    d = {"liver": {"first": 1, "last": 5}, "tumor": 2}
    result = summary._flatten_dict(d)
    assert result == {"liver_first": 1, "liver_last": 5, "tumor": 2}


def test_flatten_dict_converts_list_to_semicolon_string():
    summary = _make_summary()
    result = summary._flatten_dict({"volumes": [1.0, 2.0, 3.0]})
    assert result["volumes"] == "1.0;2.0;3.0"


def test_flatten_dict_converts_numpy_scalar_to_native():
    summary = _make_summary()
    result = summary._flatten_dict({"vox": np.int64(3), "ratio": np.float64(1.5)})
    assert result["vox"] == 3
    assert type(result["vox"]) is int
    assert result["ratio"] == 1.5
    assert type(result["ratio"]) is float


# ---------------------------------------------------------------------------
# DatasetSummary.export_csv_auto
# ---------------------------------------------------------------------------

def test_export_csv_auto_raises_when_no_rows():
    summary = _make_summary()
    summary.per_case_rows = []
    with pytest.raises(ValueError, match="No data analysed"):
        summary.export_csv_auto("unused.csv")


def test_export_csv_auto_writes_csv_excluding_default_keys(tmp_path):
    summary = _make_summary()
    summary.per_case_rows = [
        {
            "case_name": "volume-1.nii.gz",
            "case_index": 0,
            "image_path": "a.nii.gz",
            "label_path": "b.nii.gz",
            "tumor_voxels": 5,
            "has_tumor": True,
        }
    ]
    out = tmp_path / "out.csv"
    summary.export_csv_auto(out)

    df = pd.read_csv(out)
    assert list(df.columns) == ["case_name", "tumor_voxels", "has_tumor"]
    assert df.iloc[0]["tumor_voxels"] == 5
    assert bool(df.iloc[0]["has_tumor"])


def test_export_csv_auto_excludes_extra_keys(tmp_path):
    summary = _make_summary()
    summary.per_case_rows = [
        {"case_name": "v.nii.gz", "tumor_voxels": 5, "extra_column": 42}
    ]
    out = tmp_path / "out.csv"
    summary.export_csv_auto(out, exclude_keys=["extra_column"])
    df = pd.read_csv(out)
    assert "extra_column" not in df.columns


def test_export_csv_auto_sorts_by_case_name_and_places_first(tmp_path):
    summary = _make_summary()
    summary.per_case_rows = [
        {"case_name": "volume-2.nii.gz", "tumor_voxels": 2},
        {"case_name": "volume-10.nii.gz", "tumor_voxels": 3},
        {"case_name": "volume-1.nii.gz", "tumor_voxels": 1},
    ]
    out = tmp_path / "out.csv"
    summary.export_csv_auto(out)
    df = pd.read_csv(out)
    assert df["case_name"].tolist() == [
        "volume-1.nii.gz",
        "volume-10.nii.gz",
        "volume-2.nii.gz",
    ]
    assert df.columns[0] == "case_name"


# ---------------------------------------------------------------------------
# VolumeWrapper.find_slice_thresholds
# ---------------------------------------------------------------------------

def _make_volume_wrapper(label):
    from idssp.sonk.model.data import VolumeWrapper
    wrapper = VolumeWrapper("img.nii.gz", "lbl.nii.gz")
    wrapper.label_data = label
    wrapper.image_data = np.zeros(label.shape)
    return wrapper


def test_find_slice_thresholds_liver_and_tumor_present():
    # Shape (4, 4, 6): liver on slices 1..4, tumour on slices 2..3.
    label = np.zeros((4, 4, 6), dtype=np.uint8)
    label[:, :, 1:5] = 1
    label[:, :, 2:4] = 2
    wrapper = _make_volume_wrapper(label)
    wrapper.find_slice_thresholds()
    assert wrapper.slice_thresholds["liver"] == {"first": 1, "last": 4}
    assert wrapper.slice_thresholds["tumor"] == {"first": 2, "last": 3}


def test_find_slice_thresholds_uses_slice_axis_2():
    label = np.zeros((4, 4, 6), dtype=np.uint8)
    label[0, 0, 0] = 1
    label[0, 0, 5] = 1
    wrapper = _make_volume_wrapper(label)
    wrapper.find_slice_thresholds()
    assert wrapper.slice_thresholds["liver"] == {"first": 0, "last": 5}


def test_find_slice_thresholds_no_liver_no_tumor():
    label = np.zeros((4, 4, 6), dtype=np.uint8)
    wrapper = _make_volume_wrapper(label)
    wrapper.find_slice_thresholds()
    assert wrapper.slice_thresholds["liver"] == {"first": None, "last": None}
    assert wrapper.slice_thresholds["tumor"] == {"first": None, "last": None}


def test_find_slice_thresholds_liver_without_tumor():
    label = np.zeros((4, 4, 6), dtype=np.uint8)
    label[:, :, 2] = 1
    wrapper = _make_volume_wrapper(label)
    wrapper.find_slice_thresholds()
    assert wrapper.slice_thresholds["liver"] == {"first": 2, "last": 2}
    assert wrapper.slice_thresholds["tumor"] == {"first": None, "last": None}