# Test modules

This package contains a set of unit tests to assure the quality of the code
is the same across development.

It has the following modules:

## `tests/test_stratification.py`

### Purpose

This test file verifies the dataset stratification logic used to divide LiTS cases into training, validation, and test subsets. The module under test is `idssp/sonk/stats/stratification.py`.

Stratification is scientifically important because the split files determine the effective sample size, `N`, for every results table in the thesis. A subtle bug in binning, masking, or metadata export could silently bias the splits or corrupt the recorded split metadata.

The tests therefore protect the correctness and reproducibility of the split-generation logic without requiring real LiTS volumes, GPU access, network access, or credentials.

---

### Scope

The file tests pure and mostly pure helper functions involved in:

- binning case-level statistics into stratification categories,
- formatting summary statistics,
- safely computing statistical checks on small or degenerate groups,
- masking tumour-volume values for tumour-negative cases,
- saving stratification metadata to JSON.

All test inputs are synthetic and in-memory. The tests use small pandas DataFrames, NumPy arrays, and temporary files created through pytest's `tmp_path` fixture.

The file does **not** run the full split-generation notebook, load real LiTS data, or modify production source code.

---

### Main behaviours tested

| Area | Functions tested | Behaviour verified |
|---|---|---|
| Spacing binning | `bin_spacing`, `_spacing_label` | NaN handling, exact boundary classification, and human-readable label mapping |
| Liver HU binning | `bin_liver_hu` | NaN handling and low/mid/high boundary classification |
| Tumour volume binning | `bin_tumour_vol` | tumour-absent cases, NaN tumour volume, and small/medium/large boundary classification |
| Summary formatting | `_format_median_iqr` | median/IQR formatting, empty series handling, all-NaN handling, and NaN dropping |
| Statistical safety | `_safe_kruskal_p`, `_safe_chi2_p` | returns `None` for degenerate inputs instead of raising; returns valid numeric p-values for valid inputs |
| Tumour-negative masking | `_mask_no_tumour_volume` | sets `tumour_volume_ml` to `NaN` when the case has no tumour, while preserving the existing DataFrame schema |
| Metadata persistence | `save_stratification_metadata` | writes valid JSON, sorts case names and bins, excludes or includes the test split as requested, and creates parent directories when needed |

---

### Why this test file matters

The stratification module sits upstream of model training and evaluation. If it misclassifies a boundary case, masks tumour volumes incorrectly, or saves inconsistent split metadata, the error can propagate into every downstream experiment.

This test file is important because it catches:

- off-by-one boundary errors in binning functions,
- crashes or undefined behaviour on degenerate statistical inputs,
- incorrect handling of tumour-negative LiTS volumes,
- malformed or non-reproducible stratification metadata,
- accidental changes to the split-saving behaviour.

These are high-value checks because they are cheap to run and protect the experimental foundation of the thesis.


---

### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_stratification.py -v
```

The test suite is fast and should complete in a few seconds on CPU.

---

### Limitations

This file tests the stratification helper functions and metadata export logic. It does not verify the broader scientific suitability of the chosen stratification strategy, nor does it regenerate or validate the official split JSON files under `files/splits/`.

## `tests/test_data_csv.py`

### Purpose

This test file verifies the CSV/data-helper logic in `idssp/sonk/model/data.py`.

The tested code is responsible for turning per-case dataset analysis results into clean, consistent, CSV-friendly tables. These CSV files are important because they support dataset summaries, stratification, thesis tables, exploratory analysis, and quality-control checks.

The tests use only synthetic pandas objects, NumPy arrays, and temporary files. They do not require GPU access, real LiTS volumes, network access, or real environment settings.

---

### Scope

The file tests pure and mostly pure helper functions involved in:

- formatting floating-point values for CSV output,
- identifying scalar numeric values,
- converting integer-like columns to nullable pandas integer columns,
- canonicalising numeric and boolean columns before export,
- flattening nested dictionaries into flat CSV rows,
- exporting per-case dataset summaries to CSV,
- finding liver and tumour slice ranges in a `VolumeWrapper`.

The file does not load real NIfTI volumes or run model training/inference.

---

### Main behaviours tested

| Area | Functions/classes tested | Behaviour verified |
|---|---|---|
| Float formatting | `_format_csv_float` | `NaN` becomes an empty string, integer-like floats avoid unnecessary `.0`, and values are formatted deterministically with `.17g` |
| Numeric detection | `_is_scalar_number` | Python and NumPy scalar numbers are accepted; booleans, strings, `None`, containers, and arrays are rejected |
| Nullable integers | `_to_nullable_int` | integer-like values become nullable `Int64`, missing values are preserved, and fractional/non-numeric values raise `ValueError` |
| Column canonicalisation | `_canonicalise_numeric_columns` | known integer columns become nullable integers, known boolean columns become nullable booleans, and numeric object columns are formatted consistently |
| Schema constants | `_INTEGER_COLUMNS`, `_BOOLEAN_COLUMNS` | key schema constants include expected mixed-spelling fields such as `has_tumor` |
| Dictionary flattening | `DatasetSummary._flatten_dict` | nested dictionaries are flattened with underscore-separated keys, lists become semicolon-separated strings, and NumPy scalars become native Python values |
| CSV export | `DatasetSummary.export_csv_auto` | empty exports raise, CSV files are written, internal/path fields are excluded, extra excluded columns are removed, `case_name` is placed first, and rows are sorted by case name |
| Slice thresholds | `VolumeWrapper.find_slice_thresholds` | first/last liver and tumour slices are detected using axis `2`; absent structures return `None` thresholds |

---

### Why this test file matters

This file protects the data-reporting layer of the project.

The per-case CSV summaries are used to inspect the dataset, support stratification, and produce analysis tables. If numeric formatting, missing-value handling, column typing, or schema conventions are wrong, downstream analysis can become silently misleading.

These tests catch problems such as:

- `NaN` values being exported incorrectly,
- integer columns being converted to floats,
- boolean columns losing their intended type,
- fractional values being silently accepted in integer columns,
- nested dictionaries being flattened incorrectly,
- internal file-path fields leaking into exported CSVs,
- `case_name` ordering changing unexpectedly,
- liver/tumour slice ranges being computed on the wrong axis,
- mixed `tumor` / `tumour` column names being normalised accidentally.

This is especially important because the repository currently uses a mixed spelling schema:

- American spelling in fields such as `has_tumor`, `tumor_voxels`, and `slice_thresholds["tumor"]`;
- British spelling in fields such as `tumour_volume_ml` and `tumour_hu_mean`.

The tests preserve the schema used by the executable code rather than imposing a new naming convention.

---

### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests do not normalise `tumor` / `tumour` spelling.
- The tests use the current executable schema from `data.py`.
- The tests use synthetic data only.
- Temporary CSV files are written only under pytest's `tmp_path`.
- No real LiTS volumes, model training, inference, GPU access, or external services are required.

---

### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_data_csv.py -v
```

---

### Limitations

This file tests CSV/data-helper behaviour and slice-threshold logic. It does not validate the medical correctness of the underlying per-case measurements, nor does it load real medical images.

Full volume loading, NIfTI metadata handling, and heavier integration checks are covered separately in later smoke/integration test plans.

## `tests/test_training_logic.py`

### Purpose

This test file verifies the training control logic in `idssp/sonk/model/training.py`.

The tested code governs how the training pipeline behaves over the course of a long training run: when to apply augmentations, when to log visual overlays, when to send notifications, whether a checkpoint is safe to resume from, and when to stop training and save the best model.

These are high-value tests because a bug in any of these components can silently corrupt a multi-day training run, waste GPU compute, or cause the best model checkpoint to be lost.

The tests use only mocks, synthetic values, and lightweight config objects. They do not require GPU access, real LiTS volumes, real environment settings, network access, or real checkpoint files.

---

### Scope

The file tests logic-only methods in `idssp/sonk/model/training.py`:

- `AugmentedDataset` (length and item retrieval with augmentation),
- `ModelBuilder._should_log_overlay` (epoch-based TensorBoard overlay schedule),
- `ModelBuilder._should_notify` (epoch-based notification schedule),
- `ModelBuilder._validate_checkpoint` (checkpoint compatibility validation),
- `EarlyStopper.__call__` (early stopping and best-model checkpoint logic).

The file does not test model construction, data loading, transform pipelines, or the full training loop.

---

### Main behaviours tested

| Area | Function/class tested | Behaviour verified |
|---|---|---|
| Data augmentation wrapper | `AugmentedDataset` | Length matches base dataset; `__getitem__` applies the random transform to the base item |
| Overlay logging schedule | `ModelBuilder._should_log_overlay` | Every epoch for epochs 0–10; every 5 epochs for 11–30; every 10 epochs for 31+; zero-indexed epoch convention |
| Notification schedule | `ModelBuilder._should_notify` | Every 5 epochs for 0–50; every 10 epochs for 51–100; every 20 epochs for 101+; zero-indexed epoch convention |
| Checkpoint validation | `ModelBuilder._validate_checkpoint` | Hard fail on missing `model_state_dict`; hard fail on `MODEL` mismatch; hard fail on `NUM_CLASSES` mismatch; warning on preprocessing mismatches; warning on missing `config_snapshot`; hard fail when saved epoch >= `NUM_EPOCHS` |
| Early stopping | `EarlyStopper.__call__` | Monitors tumour Dice only; improvement requires strict `>` beyond `min_delta`; resets counter on improvement; increments counter on non-improvement; returns `True` only after `patience + 1` consecutive non-improving calls; writer and checkpoint saving occur only on improvement |

---

### Why this test file matters

This file protects the decision-making layer of the training pipeline.

Training runs on the server can take days. A subtle bug in the early stopping logic, checkpoint validation, or logging schedule can waste significant compute, produce misleading results, or cause the best model to be lost.

These tests catch problems such as:

- data augmentation silently not being applied,
- logging or notification schedules firing at the wrong epochs,
- resuming from an incompatible checkpoint without warning,
- early stopping triggering too early or too late,
- the best model checkpoint not being saved on improvement,
- a non-tumour metric incorrectly triggering a checkpoint save,
- the patience boundary being off by one.

The early stopping tests are especially important because `AGENTS.md` specifies that early stopping must monitor tumour Dice only, with `EARLY_STOPPING_PATIENCE=35` and `EARLY_STOPPING_MIN_DELTA=0.001`. Any deviation from this behaviour would undermine the validity of the trained model.

---

### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests do not call `config.init()` or depend on the real `.env` file.
- The tests do not instantiate real MONAI networks or run real training loops.
- The tests do not write real checkpoint files; `EarlyStopper.save_checkpoint` is patched out.
- The tests use `ModelBuilder.__new__()` to bypass heavy initialisation where only logic methods are exercised.
- The tests respect the convention that early stopping monitors tumour Dice only.
- The tests respect the zero-indexed epoch convention used by the training loop.
- The tests verify the exact patience sequence: with patience `P`, the first `P` non-improving calls return `False`, and only the `(P+1)`th returns `True`.

---

### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_training_logic.py -v
```

---

### Limitations

This file tests training control logic only. It does not verify:

- model construction or forward-pass correctness (covered in later smoke tests),
- data loading or transform pipeline behaviour,
- the full training loop end-to-end,
- actual checkpoint serialisation and deserialisation (the save/load round-trip),
- TensorBoard output format,
- notification delivery.

Full pipeline integration is verified separately via `python main.py --fast-run` as described in `AGENTS.md`.

## `tests/test_loader.py`

### Purpose

This test file verifies the data discovery, pairing, and split-loading logic in `idssp/sonk/disk/loader.py`.

The tested code is responsible for finding LiTS CT volumes, pairing them with their corresponding segmentation masks, and loading the stratified train/validation split JSONs. These operations are critical because silent failures in data loading can lead to corrupted datasets, mismatched image-label pairs, or incorrect sample sizes in downstream results tables.

The tests use only synthetic LiTS-like file trees created under pytest's `tmp_path`. They do not require real LiTS data, GPU access, network access, or real environment settings.

---

### Scope

The file tests the `CustomDataset` and `DataCollector` classes in `idssp/sonk/disk/loader.py`:

- `CustomDataset.discover_and_pair` and `get_lits_paths` (file discovery and ID-based pairing),
- `DataCollector.read_dir` (directory validation and file counting),
- `DataCollector.extract_images_and_labels` (pair extraction),
- `DataCollector._load_split` (JSON split loading and disk validation),
- `DataCollector.get_stratified_split` (end-to-end split retrieval).

The file does not load real NIfTI volumes, parse image headers, or run model training.

---

### Main behaviours tested

| Area | Functions/classes tested | Behaviour verified |
|---|---|---|
| File pairing | `CustomDataset.get_lits_paths` | Correctly pairs `volume-X.nii.gz` with `segmentation-X.nii.gz`; warns and skips when labels are missing; ignores non-volume files without reporting them as unpaired |
| Source validation | `CustomDataset.discover_and_pair` | Raises `ValueError` for unsupported dataset sources or when files have not been set |
| Directory reading | `DataCollector.read_dir` | Raises `FileNotFoundError` for missing directories; raises `ValueError` for empty directories; warns when the file count is odd (indicating unpaired files) |
| Split loading | `DataCollector._load_split` | Returns correct train/val lists; raises `FileNotFoundError` when the JSON references files missing from disk; warns when disk contains files not listed in the JSON |
| Split retrieval | `DataCollector.get_stratified_split` | Raises `ValueError` when no data is loaded; raises `FileNotFoundError` when the configured split JSON is missing; returns correct train/val pairs when valid |

---

### Why this test file matters

This file protects the data ingestion layer of the project.

As noted in `AGENTS.md`, split files produce different sample sizes (N) in every results table. If the loader silently accepts corrupted splits, mismatches image-label pairs, or fails to warn about missing files, downstream metrics and thesis tables will be invalid.

These tests catch problems such as:

- volumes being silently dropped because their segmentation masks are missing,
- image-label pairs being mismatched due to ID parsing errors,
- stratified splits being loaded incorrectly because the JSON references files that no longer exist on disk,
- silent acceptance of directories containing unpaired or non-volume files,
- unsupported dataset sources being accepted without error.

By enforcing strict validation and explicit warnings, these tests ensure that the training pipeline fails loudly when data is misconfigured, rather than silently proceeding with a degraded dataset.

---

### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests do not load real LiTS volumes or parse real NIfTI headers.
- The tests use synthetic file trees created under `tmp_path`.
- The tests mock the `config` singleton to avoid depending on the real `.env` file.
- Warning paths are verified using `caplog` to ensure the loader emits the expected diagnostics for missing labels, unpaired files, and split mismatches.

---

### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_loader.py -v
```

---

### Limitations

This file tests file discovery, pairing, and split loading. It does not verify:

- the actual loading and parsing of NIfTI image data,
- the application of MONAI transforms to the loaded paths,
- the correctness of the stratification algorithm itself (covered in `test_stratification.py`),
- or the full end-to-end training pipeline.
