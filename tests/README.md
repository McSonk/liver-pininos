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

## `tests/test_force_matching_affined.py`

### Purpose

This test file verifies the `ForceMatchingAffined` transform in `idssp/sonk/model/transforms.py`.

This transform is a targeted, safety-gated fix for known broken LiTS volumes (specifically volumes 48–52) where the segmentation mask's NIfTI header contains a placeholder identity affine matrix. The transform copies the correct affine from the image to the label, but only when strict safety conditions are met.

The tests use synthetic affines, lightweight fake metadata objects, and real MONAI `MetaTensor` instances. They do not require real LiTS data, GPU access, or network access.

---

### Scope

The file tests the `ForceMatchingAffined` class and its internal helpers:

- `_is_placeholder_affine` (detecting broken identity-like affines),
- `_normalise_affine` (converting various affine formats to a standard CPU tensor),
- `_validate_case_name` (enforcing the strict allow-list of LiTS volume IDs),
- `__call__` (the end-to-end correction logic and safety guards).

The tests deliberately preserve the production invariants: `_ALLOWED_LITS_VOLUMES` remains `{48, 49, 50, 51, 52}` and `_IDENTITY_AFFINE_THRESHOLD` remains `1e-3`.

---

### Main behaviours tested

| Area | Functions tested | Behaviour verified |
|---|---|---|
| Constants | `_ALLOWED_LITS_VOLUMES`, `_IDENTITY_AFFINE_THRESHOLD` | Values remain exactly as defined in production; not widened or loosened |
| Placeholder detection | `_is_placeholder_affine` | Detects identity matrices, identity matrices with translation, and batched affines; rejects non-identity spacing, `None`, and near-identity values above the threshold |
| Affine normalisation | `_normalise_affine` | Accepts torch tensors, NumPy arrays, and batched `(1, 4, 4)` shapes; returns a CPU `float32` `(4, 4)` tensor; raises `ValueError` for invalid shapes or wrong batch sizes |
| Case name validation | `_validate_case_name` | Accepts allowed IDs (48–52); raises `ValueError` for disallowed IDs (e.g., 47, 53) and malformed filenames |
| Correction logic | `__call__` | Returns data unchanged if objects lack `.meta` or affines; copies image affine to label only when label is placeholder and image is not; writes to `.meta["affine"]` if `.affine` attribute is read-only; raises `ValueError` if correction is attempted on an unapproved volume; does not overwrite a valid non-placeholder label affine |

---

### Why this test file matters

This file protects a critical data-handling safeguard. 

As noted in `AGENTS.md`, LiTS volumes 48–52 have severe affine mismatches. If the correction logic is too broad, it could silently overwrite valid affines on other volumes, corrupting spatial metadata and ruining downstream metrics. If it is too narrow or fails to trigger, the model will train on geometrically misaligned masks for those specific cases.

These tests ensure that the correction is applied **only** to the explicitly approved volumes, **only** when the label affine is demonstrably broken, and **never** when the label already contains valid spatial information.

---

### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests do not widen `_ALLOWED_LITS_VOLUMES` or change `_IDENTITY_AFFINE_THRESHOLD`.
- The tests use synthetic data and lightweight mock objects to avoid loading real NIfTI files.
- The tests verify both the primary execution path (setting `.affine`) and the fallback path (setting `.meta["affine"]` when the attribute is read-only).

---

### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_force_matching_affined.py -v
```

---

### Limitations

This file tests the affine correction logic in isolation. It does not verify the full MONAI transform pipeline, the actual loading of NIfTI files from disk, or the downstream impact of the corrected affines on model training.

## `tests/test_evaluator_postprocess.py`

### Purpose

This test file verifies the post-processing and report-export logic in the evaluation pipeline (specifically `_post_process_class_map` and `MetricsEvaluator.generate_report`).

The tested code is responsible for enforcing anatomical realism on the model's raw 3D segmentation predictions (e.g., removing disconnected liver fragments and stray tumours) and for exporting the final per-case metrics to CSV files for thesis reporting. 

These tests are critical because incorrect post-processing can silently alter final Dice and HD95 scores, and incorrect CSV export can lead to incomplete or misaligned thesis tables. The tests use small synthetic 3D NumPy arrays and in-memory pandas DataFrames, requiring no GPU, real LiTS data, or network access.

---

### Scope

The file tests two main components:

- `_post_process_class_map`: The 3D NumPy array cleanup function that enforces largest-connected-component (LCC) rules for the liver and tumour classes.
- `MetricsEvaluator.generate_report`: The method that aggregates results and exports raw and post-processed metrics to CSV.

The file does not run full-volume inference, load real NIfTI files, or compute actual MONAI metrics.

---

### Main behaviours tested

| Area | Functions/classes tested | Behaviour verified |
|---|---|---|
| Basic invariants | `_post_process_class_map` | Returns an array of the same shape and dtype; handles empty inputs safely; removes stray tumours when no liver is present |
| Liver retention | `_post_process_class_map` | Retains only the largest connected component of the liver; removes small, disconnected stray liver voxels |
| Tumour anchoring | `_post_process_class_map` | Preserves tumour voxels that are inside or immediately adjacent to the retained liver anatomy; strips tumour voxels that are disconnected from the main anatomy |
| Fragmentation warning | `_post_process_class_map` | Emits a warning if more than 50% of predicted liver voxels are discarded during LCC cleanup; remains silent when the liver is mostly retained or absent |
| Report export | `MetricsEvaluator.generate_report` | Raises `ValueError` on empty input dictionaries; skips empty DataFrames without crashing; writes correctly named raw and post-processed CSVs; sorts rows lexicographically by `case_name`; respects custom output directories or falls back to `config.RUN_DIR` |

---

### Why this test file matters

This file protects the integrity of your final reported metrics and thesis tables.

As noted in `AGENTS.md`, post-processing asymmetry is a known factor: LCC cleanup helps spatially coherent models (like SegResNet) but can hurt global-attention models (like SwinUNETR) by stripping true-positive tumours that fall slightly outside the predicted liver boundary. Therefore, the exact behaviour of the cleanup function must be strictly verified.

These tests catch problems such as:

- valid tumours being silently deleted because they were not perfectly enclosed by the predicted liver mask,
- fragmented predictions passing through without triggering the >50% discard warning,
- stray false-positive liver blobs artificially inflating raw Dice scores,
- CSV export logic silently dropping the post-processed results if the DataFrame happens to be empty,
- case names being sorted in a way that misaligns raw and post-processed rows.

By verifying both the anatomical logic and the export mechanics, this file ensures that the numbers you put in your thesis are exactly what the model actually produced.

---

### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests strictly enforce the 3-class layout defined in `AGENTS.md`: `0` (background), `1` (liver), `2` (tumour).
- The tests use synthetic 3D NumPy arrays and explicitly enforce 6-connectivity (face-adjacency) when building connected components, matching `scipy.ndimage.label` defaults.
- The tests verify lexicographical sorting for `case_name` (e.g., `vol-1`, `vol-10`, `vol-2`), matching standard pandas behaviour.
- The tests use `caplog` to verify the exact conditions under which the fragmentation warning is emitted.

---

### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_evaluator_postprocess.py -v
```

---

### Limitations

This file tests the post-processing logic and CSV export in isolation. It does not verify the end-to-end sliding-window inference pipeline, the MONAI metric calculations (Dice/HD95), or the `Invertd` transform used to map predictions back to the original scanner space. Those integration steps are verified via the `--fast-run` entrypoint and server-side validation scripts.
