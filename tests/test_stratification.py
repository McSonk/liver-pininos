"""Unit tests for idssp.sonk.stats.stratification.

Covers the pure binning helpers, statistical safety wrappers, and
save_stratification_metadata. All inputs are synthetic and in-memory;
no config singleton, network, GPU, or real LiTS data is used.

Column-name note: the code deliberately mixes spellings. The DataFrame
column is ``has_tumor`` (US) while the ``bin_tumour_vol`` parameter is
``has_tumour`` (GB). Tests use whichever exact name the function under
test uses and do not normalise.
"""

import json

import numpy as np
import pandas as pd
import pytest

import idssp.sonk.stats.stratification as strat
from idssp.sonk.stats.stratification import (
    _format_median_iqr,
    _mask_no_tumour_volume,
    _safe_chi2_p,
    _safe_kruskal_p,
    _spacing_label,
    bin_liver_hu,
    bin_spacing,
    bin_tumour_vol,
    save_stratification_metadata,
)


# ---------------------------------------------------------------------------
# bin_spacing
# ---------------------------------------------------------------------------

def test_bin_spacing_nan_returns_unknown():
    assert bin_spacing(np.nan) == "unknown"


@pytest.mark.parametrize("value,expected", [
    pytest.param(0.5, "thin", id="well_below_thin"),
    pytest.param(1.0, "thin", id="exact_thin_boundary"),
    pytest.param(1.1, "medium", id="just_above_thin"),
    pytest.param(1.5, "medium", id="exact_medium_boundary"),
    pytest.param(1.6, "thick", id="just_above_medium"),
    pytest.param(5.0, "thick", id="well_above_medium"),
])
def test_bin_spacing_boundaries(value, expected):
    assert bin_spacing(value) == expected


# ---------------------------------------------------------------------------
# bin_liver_hu
# ---------------------------------------------------------------------------

def test_bin_liver_hu_nan_returns_unknown():
    assert bin_liver_hu(np.nan) == "unknown"


@pytest.mark.parametrize("value,expected", [
    pytest.param(-100.0, "low", id="well_below_low"),
    pytest.param(59.9, "low", id="below_low_boundary"),
    pytest.param(60.0, "mid", id="exact_low_boundary"),
    pytest.param(99.9, "mid", id="below_mid_boundary"),
    pytest.param(100.0, "high", id="exact_mid_boundary"),
    pytest.param(150.0, "high", id="well_above_mid"),
])
def test_bin_liver_hu_boundaries(value, expected):
    assert bin_liver_hu(value) == expected


# ---------------------------------------------------------------------------
# bin_tumour_vol
# ---------------------------------------------------------------------------

def test_bin_tumour_vol_no_tumour_returns_none():
    assert bin_tumour_vol(100.0, has_tumour=False) == "none"


def test_bin_tumour_vol_present_nan_returns_unknown():
    assert bin_tumour_vol(np.nan, has_tumour=True) == "unknown"


@pytest.mark.parametrize("value,expected", [
    pytest.param(4.9, "small", id="below_small_boundary"),
    pytest.param(5.0, "medium", id="exact_small_boundary"),
    pytest.param(49.9, "medium", id="below_medium_boundary"),
    pytest.param(50.0, "large", id="exact_medium_boundary"),
    pytest.param(100.0, "large", id="well_above_medium"),
])
def test_bin_tumour_vol_boundaries(value, expected):
    assert bin_tumour_vol(value, has_tumour=True) == expected


# ---------------------------------------------------------------------------
# _spacing_label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    pytest.param(0.5, "Thin (<=1.0 mm)", id="thin"),
    pytest.param(1.2, "Medium (>1.0 to 1.5 mm)", id="medium"),
    pytest.param(2.0, "Thick (>1.5 mm)", id="thick"),
    pytest.param(np.nan, "Unknown", id="unknown"),
])
def test_spacing_label_maps_bin_to_human_label(value, expected):
    assert _spacing_label(value) == expected


# ---------------------------------------------------------------------------
# _format_median_iqr
# ---------------------------------------------------------------------------

def test_format_median_iqr_returns_formatted_string():
    series = pd.Series([1.0, 2.0, 3.0])
    assert _format_median_iqr(series) == "2.00 [1.50-2.50]"


def test_format_median_iqr_empty_series_returns_na():
    assert _format_median_iqr(pd.Series([], dtype=float)) == "N/A"


def test_format_median_iqr_all_nan_returns_na():
    assert _format_median_iqr(pd.Series([np.nan, np.nan])) == "N/A"


def test_format_median_iqr_drops_nan_values():
    series = pd.Series([1.0, np.nan, 2.0, 3.0])
    assert _format_median_iqr(series) == "2.00 [1.50-2.50]"


# ---------------------------------------------------------------------------
# _safe_kruskal_p
# ---------------------------------------------------------------------------

def test_safe_kruskal_p_empty_group_returns_none():
    groups = [pd.Series([1.0, 2.0]), pd.Series([], dtype=float)]
    assert _safe_kruskal_p(groups) is None


def test_safe_kruskal_p_single_value_group_returns_none():
    groups = [pd.Series([1.0]), pd.Series([2.0, 3.0])]
    assert _safe_kruskal_p(groups) is None


def test_safe_kruskal_p_identical_values_returns_none():
    groups = [pd.Series([1.0, 1.0]), pd.Series([1.0, 1.0])]
    assert _safe_kruskal_p(groups) is None


def test_safe_kruskal_p_valid_inputs_return_float():
    groups = [pd.Series([1.0, 2.0, 3.0]), pd.Series([4.0, 5.0, 6.0])]
    p = _safe_kruskal_p(groups)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# _safe_chi2_p
# ---------------------------------------------------------------------------

def test_safe_chi2_p_empty_frame_returns_none():
    assert _safe_chi2_p(pd.DataFrame()) is None


def test_safe_chi2_p_zero_total_returns_none():
    df = pd.DataFrame({"a": [0, 0], "b": [0, 0]})
    assert _safe_chi2_p(df) is None


def test_safe_chi2_p_zero_row_returns_none():
    df = pd.DataFrame({"a": [5, 0], "b": [5, 0]})
    assert _safe_chi2_p(df) is None


def test_safe_chi2_p_zero_column_returns_none():
    df = pd.DataFrame({"a": [5, 5], "b": [0, 0]})
    assert _safe_chi2_p(df) is None


def test_safe_chi2_p_valid_inputs_return_float():
    df = pd.DataFrame({"a": [10, 20], "b": [20, 10]})
    p = _safe_chi2_p(df)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_safe_chi2_p_chi2_contingency_raises_returns_none(monkeypatch):
    def _boom(*args, **kwargs):
        raise ValueError("degenerate contingency table")

    monkeypatch.setattr(strat, "chi2_contingency", _boom)
    df = pd.DataFrame({"a": [10, 20], "b": [20, 10]})
    assert _safe_chi2_p(df) is None


# ---------------------------------------------------------------------------
# _mask_no_tumour_volume
# ---------------------------------------------------------------------------

def test_mask_no_tumour_volume_sets_nan_when_has_tumor_false():
    df = pd.DataFrame({
        "has_tumor": [True, False, True],
        "tumour_volume_ml": [5.0, 0.0, 10.0],
    })
    result = _mask_no_tumour_volume(df)
    assert result.loc[0, "tumour_volume_ml"] == 5.0
    assert pd.isna(result.loc[1, "tumour_volume_ml"])
    assert result.loc[2, "tumour_volume_ml"] == 10.0


def test_mask_no_tumour_volume_does_not_mutate_input():
    df = pd.DataFrame({
        "has_tumor": [True, False],
        "tumour_volume_ml": [5.0, 0.0],
    })
    _mask_no_tumour_volume(df)
    assert df.loc[1, "tumour_volume_ml"] == 0.0


def test_mask_no_tumour_volume_missing_columns_returns_unchanged():
    df = pd.DataFrame({"case_name": ["a", "b"]})
    result = _mask_no_tumour_volume(df)
    assert result.equals(df)


# ---------------------------------------------------------------------------
# save_stratification_metadata
# ---------------------------------------------------------------------------

def _meta_df(cases, strat_keys):
    return pd.DataFrame({"case_name": cases, "strat_key": strat_keys})


def test_save_stratification_metadata_writes_valid_json_and_excludes_test(tmp_path):
    train = _meta_df(["vol-b", "vol-a"], ["bin-z", "bin-a"])
    val = _meta_df(["vol-c"], ["bin-z"])
    test = _meta_df(["vol-d"], ["bin-z"])
    out = tmp_path / "meta.json"

    result = save_stratification_metadata(train, val, test, str(out))

    assert result == out.resolve()
    data = json.loads(out.read_text())

    assert data["train"] == ["vol-a", "vol-b"]
    assert data["val"] == ["vol-c"]
    assert "test" not in data
    assert data["bins"] == {
        "bin-a": ["vol-a"],
        "bin-z": ["vol-b", "vol-c"],
    }


def test_save_stratification_metadata_include_test(tmp_path):
    train = _meta_df(["vol-b", "vol-a"], ["bin-a", "bin-a"])
    val = _meta_df(["vol-c"], ["bin-a"])
    test = _meta_df(["vol-d", "vol-e"], ["bin-a", "bin-a"])
    out = tmp_path / "meta.json"

    save_stratification_metadata(train, val, test, str(out), include_test=True)

    data = json.loads(out.read_text())
    assert data["test"] == ["vol-d", "vol-e"]
    assert data["bins"]["bin-a"] == ["vol-a", "vol-b", "vol-c", "vol-d", "vol-e"]


def test_save_stratification_metadata_creates_parent_dir(tmp_path):
    train = _meta_df(["vol-a"], ["bin-a"])
    val = _meta_df(["vol-b"], ["bin-a"])
    test = _meta_df(["vol-c"], ["bin-a"])
    out = tmp_path / "sub" / "nested" / "meta.json"

    save_stratification_metadata(train, val, test, str(out), include_test=True)

    assert out.exists()
    data = json.loads(out.read_text())
    assert "creation_date" in data
    assert "stratification_method" in data
