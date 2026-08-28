"""Unit tests for idssp.sonk.disk.loader.

Covers CustomDataset discovery/pairing, DataCollector.read_dir,
extract_images_and_labels, _load_split, and get_stratified_split.

All tests use synthetic LiTS-like file trees under tmp_path and a mocked
config singleton. No real LiTS data, network, or real .env is used.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from idssp.sonk.disk import loader as loader_mod
from idssp.sonk.disk.loader import CustomDataset, DataCollector

# ---------------------------------------------------------------------------
# CustomDataset.discover_and_pair / get_lits_paths
# ---------------------------------------------------------------------------

def test_discover_and_pair_raises_when_files_none():
    ds = CustomDataset("LiTS", files=None)
    with pytest.raises(ValueError, match="Files have not been set"):
        ds.discover_and_pair()


def test_discover_and_pair_unsupported_source_raises():
    ds = CustomDataset("UnknownSource", files=[])
    with pytest.raises(ValueError, match="not supported"):
        ds.discover_and_pair()


def test_get_lits_paths_pairs_volume_with_segmentation(tmp_path):
    (tmp_path / "volume-0.nii.gz").touch()
    (tmp_path / "segmentation-0.nii.gz").touch()
    (tmp_path / "volume-1.nii.gz").touch()
    (tmp_path / "segmentation-1.nii.gz").touch()

    files = sorted(str(p) for p in tmp_path.glob("*"))
    ds = CustomDataset("LiTS", files)

    paired = ds.get_lits_paths()
    names = {(Path(p["image"]).name, Path(p["label"]).name) for p in paired}
    assert names == {
        ("volume-0.nii.gz", "segmentation-0.nii.gz"),
        ("volume-1.nii.gz", "segmentation-1.nii.gz"),
    }


def test_get_lits_paths_missing_label_warns_and_not_paired(tmp_path, caplog):
    (tmp_path / "volume-0.nii.gz").touch()

    ds = CustomDataset("LiTS", [str(tmp_path / "volume-0.nii.gz")])

    with caplog.at_level("WARNING", logger="idssp.sonk.disk.loader"):
        paired = ds.get_lits_paths()

    assert paired == []
    assert "Label file not found for image volume-0.nii.gz" in caplog.text


def test_get_lits_paths_non_volume_file_warns_unpaired(tmp_path, caplog):
    (tmp_path / "volume-0.nii.gz").touch()
    (tmp_path / "segmentation-0.nii.gz").touch()
    (tmp_path / "notes.txt").touch()

    files = sorted(str(p) for p in tmp_path.glob("*"))
    ds = CustomDataset("LiTS", files)

    with caplog.at_level("WARNING", logger="idssp.sonk.disk.loader"):
        paired = ds.get_lits_paths()

    assert len(paired) == 1
    assert "notes.txt" in caplog.text
    assert "not identified" in caplog.text


def test_get_lits_paths_label_not_reported_as_unpaired(tmp_path, caplog):
    (tmp_path / "volume-0.nii.gz").touch()
    (tmp_path / "segmentation-0.nii.gz").touch()

    files = sorted(str(p) for p in tmp_path.glob("*"))
    ds = CustomDataset("LiTS", files)

    with caplog.at_level("WARNING", logger="idssp.sonk.disk.loader"):
        paired = ds.get_lits_paths()

    assert len(paired) == 1
    assert "segmentation-0.nii.gz" not in caplog.text


# ---------------------------------------------------------------------------
# DataCollector.read_dir
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config(monkeypatch):
    """Return a MagicMock config and patch loader's config.get to use it."""
    cfg = MagicMock()
    monkeypatch.setattr(loader_mod.config, "get", lambda: cfg)
    return cfg


def test_read_dir_unsupported_source_raises(mock_config, tmp_path):
    dc = DataCollector()
    with pytest.raises(ValueError, match="not supported"):
        dc.read_dir(tmp_path, "UnknownSource")


def test_read_dir_missing_directory_raises(mock_config, tmp_path):
    dc = DataCollector()
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        dc.read_dir(missing, "LiTS")


def test_read_dir_empty_directory_raises(mock_config, tmp_path):
    dc = DataCollector()
    with pytest.raises(ValueError, match="No files found"):
        dc.read_dir(tmp_path, "LiTS")


def test_read_dir_odd_file_count_warns(mock_config, tmp_path, caplog):
    (tmp_path / "volume-0.nii.gz").touch()
    (tmp_path / "segmentation-0.nii.gz").touch()
    (tmp_path / "volume-1.nii.gz").touch()

    dc = DataCollector()
    with caplog.at_level("WARNING", logger="idssp.sonk.disk.loader"):
        dc.read_dir(tmp_path, "LiTS")

    assert "odd number of files (3)" in caplog.text
    assert len(dc.d_sets) == 1


def test_read_dir_even_file_count_no_warning(mock_config, tmp_path, caplog):
    (tmp_path / "volume-0.nii.gz").touch()
    (tmp_path / "segmentation-0.nii.gz").touch()

    dc = DataCollector()
    with caplog.at_level("WARNING", logger="idssp.sonk.disk.loader"):
        dc.read_dir(tmp_path, "LiTS")

    assert "odd number of files" not in caplog.text
    assert len(dc.d_sets) == 1


# ---------------------------------------------------------------------------
# DataCollector.extract_images_and_labels
# ---------------------------------------------------------------------------

def test_extract_raises_when_no_dataset_loaded(mock_config):
    dc = DataCollector()
    with pytest.raises(ValueError, match="No datasets have been loaded"):
        dc.extract_images_and_labels()


def test_extract_images_and_labels_returns_pairs(mock_config, tmp_path):
    (tmp_path / "volume-0.nii.gz").touch()
    (tmp_path / "segmentation-0.nii.gz").touch()

    dc = DataCollector()
    dc.read_dir(tmp_path, "LiTS")
    paired = dc.extract_images_and_labels()

    assert len(paired) == 1
    assert Path(paired[0]["image"]).name == "volume-0.nii.gz"
    assert Path(paired[0]["label"]).name == "segmentation-0.nii.gz"


# ---------------------------------------------------------------------------
# DataCollector._load_split
# ---------------------------------------------------------------------------

def _build_datasources(tmp_path, names):
    ds = []
    for name in names:
        img = tmp_path / f"volume-{name}.nii.gz"
        lab = tmp_path / f"segmentation-{name}.nii.gz"
        img.touch()
        lab.touch()
        ds.append({"image": str(img), "label": str(lab)})
    return ds


def _collector_with_data(tmp_path, names):
    dc = DataCollector.__new__(DataCollector)
    dc.datasources = _build_datasources(tmp_path, names)
    dc.d_sets = []
    return dc


def test_load_split_returns_train_and_val(mock_config, tmp_path):
    dc = _collector_with_data(tmp_path, [0, 1, 2])
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"train": ["volume-0.nii.gz"], "val": ["volume-1.nii.gz"]}),
        encoding="utf-8",
    )

    train, val = dc._load_split(split_file)
    assert [Path(f["image"]).name for f in train] == ["volume-0.nii.gz"]
    assert [Path(f["image"]).name for f in val] == ["volume-1.nii.gz"]


def test_load_split_references_missing_file_raises(mock_config, tmp_path):
    dc = _collector_with_data(tmp_path, [0])
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"train": ["volume-0.nii.gz"], "val": ["volume-9.nii.gz"]}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="not found on disk"):
        dc._load_split(split_file)


def test_load_split_disk_extra_file_warns(mock_config, tmp_path, caplog):
    dc = _collector_with_data(tmp_path, [0, 1])
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"train": ["volume-0.nii.gz"], "val": []}),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING", logger="idssp.sonk.disk.loader"):
        train, val = dc._load_split(split_file)

    assert len(train) == 1
    assert val == []
    assert "not in the split JSON" in caplog.text
    assert "volume-1.nii.gz" in caplog.text


# ---------------------------------------------------------------------------
# DataCollector.get_stratified_split
# ---------------------------------------------------------------------------

def test_get_stratified_split_raises_when_no_data(mock_config):
    dc = DataCollector.__new__(DataCollector)
    dc.datasources = []
    with pytest.raises(ValueError, match="No data loaded"):
        dc.get_stratified_split()


def test_get_stratified_split_missing_json_raises(mock_config, tmp_path):
    cfg = mock_config
    cfg.SPLIT_JSON = tmp_path / "missing.json"
    dc = _collector_with_data(tmp_path, [0])
    dc.config = cfg

    with pytest.raises(FileNotFoundError, match="split JSON file not found"):
        dc.get_stratified_split()


def test_get_stratified_split_returns_train_val(mock_config, tmp_path):
    cfg = mock_config
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"train": ["volume-0.nii.gz"], "val": ["volume-1.nii.gz"]}),
        encoding="utf-8",
    )
    cfg.SPLIT_JSON = split_file

    dc = _collector_with_data(tmp_path, [0, 1])
    dc.config = cfg

    train, val = dc.get_stratified_split()
    assert [Path(f["image"]).name for f in train] == ["volume-0.nii.gz"]
    assert [Path(f["image"]).name for f in val] == ["volume-1.nii.gz"]
