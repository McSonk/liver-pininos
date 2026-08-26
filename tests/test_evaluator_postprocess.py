"""Unit tests for idssp.sonk.model.evaluator post-processing and report export.

Covers `_post_process_class_map` (largest-connected-component liver retention,
stray-liver removal, tumour-outside-anatomy removal, fragmented-liver warning)
and `MetricsEvaluator.generate_report` (empty-input handling, empty-DataFrame
skipping, CSV naming, sorting, and output-directory behaviour).

All inputs are small synthetic 3D NumPy arrays and in-memory pandas DataFrames;
temporary files are created only under pytest's ``tmp_path``. No config
singleton, network access, GPU, real LiTS data, or real environment settings
are used.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from idssp.sonk.model.evaluator import MetricsEvaluator, _post_process_class_map

# Class layout used by the whole codebase (see AGENTS.md / Test Plan 1):
#   0 = background, 1 = liver, 2 = tumour.
BG, LIVER, TUMOUR = 0, 1, 2


def _set_connected(vol, coords):
    """Assign a run of voxels, each 6-adjacent to the previous one."""
    prev = None
    for c in coords:
        vol[c] = LIVER
        if prev is None:
            prev = c
            continue
        # Enforce face-adjacency so the whole run forms one connected component.
        if abs(c[0] - prev[0]) + abs(c[1] - prev[1]) + abs(c[2] - prev[2]) != 1:
            raise AssertionError("non-adjacent voxel in run")
        prev = c
    return vol


def _blank():
    return np.zeros((8, 8, 8), dtype=np.uint8)


# ---------------------------------------------------------------------------
# _post_process_class_map: basic invariants
# ---------------------------------------------------------------------------

def test_post_process_returns_same_shape():
    pred = _blank()
    pred[2, 2, 2] = LIVER
    result = _post_process_class_map(pred)
    assert result.shape == pred.shape
    assert result.dtype == pred.dtype


def test_post_process_empty_input_is_unchanged():
    pred = _blank()
    result = _post_process_class_map(pred)
    assert np.array_equal(result, pred)


def test_post_process_no_liver_removes_stray_tumour():
    # No liver, but a stray tumour voxel: it should be stripped because there
    # is no retained anatomical component to anchor it.
    pred = _blank()
    pred[4, 4, 4] = TUMOUR
    result = _post_process_class_map(pred)
    assert (result == TUMOUR).sum() == 0


# ---------------------------------------------------------------------------
# _post_process_class_map: largest-connected-component liver retention
# ---------------------------------------------------------------------------

def test_post_process_keeps_largest_liver_component():
    # A large connected liver blob plus a small stray liver blob.
    pred = _blank()
    _set_connected(pred, [
        (2, 2, 2), (2, 2, 3), (2, 2, 4), (2, 2, 5), (2, 3, 5), (2, 4, 5),
    ])
    # Stray, small liver blob far away.
    pred[6, 6, 6] = LIVER
    result = _post_process_class_map(pred)
    # Only the large blob survives; the stray voxel is dropped.
    assert (result == LIVER).sum() == 6
    assert result[6, 6, 6] == BG


def test_post_process_removes_stray_liver_voxels():
    pred = _blank()
    _set_connected(pred, [
        (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 3, 6), (3, 4, 6),
    ])
    pred[0, 0, 0] = LIVER  # lone stray voxel
    result = _post_process_class_map(pred)
    assert result[0, 0, 0] == BG
    assert (result == LIVER).sum() == 5


# ---------------------------------------------------------------------------
# _post_process_class_map: tumour handling relative to retained anatomy
# ---------------------------------------------------------------------------

def test_post_process_keeps_tumour_inside_retained_liver():
    pred = _blank()
    _set_connected(pred, [
        (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 4, 5),
    ])
    pred[3, 3, 4] = TUMOUR  # tumour embedded inside the retained liver
    result = _post_process_class_map(pred)
    assert result[3, 3, 4] == TUMOUR
    assert (result == TUMOUR).sum() == 1


def test_post_process_removes_tumour_outside_retained_anatomy():
    pred = _blank()
    _set_connected(pred, [
        (3, 3, 3), (3, 3, 4), (3, 3, 5),
    ])
    pred[7, 7, 7] = TUMOUR  # far-away tumour not connected to the liver
    result = _post_process_class_map(pred)
    assert result[7, 7, 7] == BG
    assert (result == TUMOUR).sum() == 0


def test_post_process_keeps_tumour_adjacent_to_retained_liver():
    # Tumour immediately adjacent to the retained liver component belongs to
    # the same anatomical component and must be preserved.
    pred = _blank()
    _set_connected(pred, [
        (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 4, 5),
    ])
    pred[3, 3, 6] = TUMOUR  # faces the last liver voxel
    result = _post_process_class_map(pred)
    assert result[3, 3, 6] == TUMOUR
    assert (result == TUMOUR).sum() == 1


# ---------------------------------------------------------------------------
# _post_process_class_map: fragmented-liver warning behaviour
# ---------------------------------------------------------------------------

def test_post_process_warns_when_most_liver_discarded(caplog):
    # Largest component holds 5 of 11 liver voxels -> ~54% discarded,
    # which crosses the 50% warning threshold.
    pred = _blank()
    _set_connected(pred, [
        (2, 2, 2), (2, 2, 3), (2, 2, 4), (2, 2, 5), (2, 2, 6),  # kept
    ])
    _set_connected(pred, [(5, 5, 5), (5, 5, 6), (5, 5, 7)])       # dropped
    _set_connected(pred, [(0, 0, 0), (0, 0, 1), (0, 0, 2)])       # dropped
    with caplog.at_level(logging.WARNING):
        _post_process_class_map(pred)
    assert any("discarded" in message for message in caplog.messages)


def test_post_process_no_warning_when_liver_mostly_retained(caplog):
    # Stray voxel is a tiny fraction; no warning expected.
    pred = _blank()
    _set_connected(pred, [
        (3, 3, 3), (3, 3, 4), (3, 3, 5), (3, 3, 6), (3, 3, 7),
        (3, 4, 7), (3, 5, 7), (3, 6, 7), (3, 7, 7), (3, 7, 6),
    ])
    pred[0, 0, 0] = LIVER  # one stray voxel among eleven
    with caplog.at_level(logging.WARNING):
        _post_process_class_map(pred)
    assert not any("LCC" in message for message in caplog.messages)


def test_post_process_warns_only_when_liver_present(caplog):
    # No liver at all, so the fraction-based guard should not fire.
    pred = _blank()
    pred[4, 4, 4] = TUMOUR
    with caplog.at_level(logging.WARNING):
        _post_process_class_map(pred)
    assert not any("LCC" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# MetricsEvaluator.generate_report
# ---------------------------------------------------------------------------

def _make_evaluator(default_dir=None):
    """Build a MetricsEvaluator without running its heavy __init__."""
    evaluator = object.__new__(MetricsEvaluator)
    evaluator.config = type(
        "FakeConfig", (), {"RUN_DIR": default_dir if default_dir is not None else None}
    )()
    return evaluator


def test_generate_report_raises_on_empty_dict():
    evaluator = _make_evaluator()
    with pytest.raises(ValueError, match="empty"):
        evaluator.generate_report({})


def test_generate_report_skips_empty_dataframes(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    results = {
        "raw": pd.DataFrame({"case_name": ["a"]}),
        "pp": pd.DataFrame(),
    }
    evaluator.generate_report(results, tmp_path)
    assert (tmp_path / "test_evaluation_results_raw.csv").exists()
    assert not (tmp_path / "test_evaluation_results_pp.csv").exists()


def test_generate_report_writes_named_csvs(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    results = {
        "raw": pd.DataFrame({"case_name": ["a"]}),
        "pp": pd.DataFrame({"case_name": ["b"]}),
    }
    evaluator.generate_report(results, tmp_path)
    assert (tmp_path / "test_evaluation_results_raw.csv").exists()
    assert (tmp_path / "test_evaluation_results_pp.csv").exists()


def test_generate_report_sorts_by_case_name(tmp_path):
    evaluator = _make_evaluator(tmp_path)
    results = {
        "raw": pd.DataFrame({"case_name": ["vol-2", "vol-10", "vol-1"]}),
    }
    evaluator.generate_report(results, tmp_path)
    written = pd.read_csv(tmp_path / "test_evaluation_results_raw.csv")
    assert written["case_name"].tolist() == ["vol-1", "vol-10", "vol-2"]


def test_generate_report_uses_provided_output_dir(tmp_path):
    evaluator = _make_evaluator()
    out = tmp_path / "custom_reports"
    results = {"raw": pd.DataFrame({"case_name": ["a"]})}
    returned = evaluator.generate_report(results, out)
    assert (out / "test_evaluation_results_raw.csv").exists()
    assert returned == str(out)


def test_generate_report_default_dir_uses_config_run_dir(tmp_path):
    evaluator = _make_evaluator(default_dir=tmp_path)
    results = {"raw": pd.DataFrame({"case_name": ["a"]})}
    evaluator.generate_report(results)
    assert (tmp_path / "reports" / "test_evaluation_results_raw.csv").exists()
