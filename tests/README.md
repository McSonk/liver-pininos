# Test modules

This package contains the unit-test suite for `idssp.sonk`. The tests assure
that code behaviour remains consistent across development.

All tests are CPU-only and deterministic. They use synthetic data only:
in-memory pandas objects, tiny NumPy/torch tensors, and fake LiTS-like file
trees under pytest's `tmp_path`. No GPU, network access, real LiTS volumes,
real credentials, or the real `.env` file is required.

Shared fixtures (config-singleton reset and a minimal frozen `Config`) live
in `tests/conftest.py`.

| Series | Scope | Files | Tests |
|---|---|---|---|
| Series 1 (Plan 1) | Stratification, CSV/data helpers, training control logic, LiTS loader, affine correction, evaluation post-processing | 6 | 191 |
| Series 2 (Plan 2) | Configuration, model factory, inference checkpoint alignment | 3 | 70 |

Run the whole suite from the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/ -v
```

Notification dispatch (mail/Telegram) is intentionally out of scope. Heavier
smoke/integration tests (real model forward passes, transform-pipeline
execution, `Invertd` inversion) are deferred to a later plan; those paths are
currently verified via the `--fast-run` entrypoint and server-side scripts,
as described in `AGENTS.md`.

## Series 1 — Pure-logic unit tests (Plan 1)

### `tests/test_stratification.py`

#### Purpose

This test file verifies the dataset stratification logic used to divide LiTS cases into training, validation, and test subsets. The module under test is `idssp/sonk/stats/stratification.py`.

Stratification is scientifically important because the split files determine the effective sample size, `N`, for every results table in the thesis. A subtle bug in binning, masking, or metadata export could silently bias the splits or corrupt the recorded split metadata.

The tests therefore protect the correctness and reproducibility of the split-generation logic without requiring real LiTS volumes, GPU access, network access, or credentials.

---

#### Scope

The file tests pure and mostly pure helper functions involved in:

- binning case-level statistics into stratification categories,
- formatting summary statistics,
- safely computing statistical checks on small or degenerate groups,
- masking tumour-volume values for tumour-negative cases,
- saving stratification metadata to JSON.

All test inputs are synthetic and in-memory. The tests use small pandas DataFrames, NumPy arrays, and temporary files created through pytest's `tmp_path` fixture.

The file does **not** run the full split-generation notebook, load real LiTS data, or modify production source code.

---

#### Main behaviours tested

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

#### Why this test file matters

The stratification module sits upstream of model training and evaluation. If it misclassifies a boundary case, masks tumour volumes incorrectly, or saves inconsistent split metadata, the error can propagate into every downstream experiment.

This test file is important because it catches:

- off-by-one boundary errors in binning functions,
- crashes or undefined behaviour on degenerate statistical inputs,
- incorrect handling of tumour-negative LiTS volumes,
- malformed or non-reproducible stratification metadata,
- accidental changes to the split-saving behaviour.

These are high-value checks because they are cheap to run and protect the experimental foundation of the thesis.

---

#### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_stratification.py -v
```

The test suite is fast and should complete in a few seconds on CPU.

---

#### Limitations

This file tests the stratification helper functions and metadata export logic. It does not verify the broader scientific suitability of the chosen stratification strategy, nor does it regenerate or validate the official split JSON files under `files/splits/`.

### `tests/test_data_csv.py`

#### Purpose

This test file verifies the CSV/data-helper logic in `idssp/sonk/model/data.py`.

The tested code is responsible for turning per-case dataset analysis results into clean, consistent, CSV-friendly tables. These CSV files are important because they support dataset summaries, stratification, thesis tables, exploratory analysis, and quality-control checks.

The tests use only synthetic pandas objects, NumPy arrays, and temporary files. They do not require GPU access, real LiTS volumes, network access, or real environment settings.

---

#### Scope

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

#### Main behaviours tested

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

#### Why this test file matters

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

#### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests do not normalise `tumor` / `tumour` spelling.
- The tests use the current executable schema from `data.py`.
- The tests use synthetic data only.
- Temporary CSV files are written only under pytest's `tmp_path`.
- No real LiTS volumes, model training, inference, GPU access, or external services are required.

---

#### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_data_csv.py -v
```

---

#### Limitations

This file tests CSV/data-helper behaviour and slice-threshold logic. It does not validate the medical correctness of the underlying per-case measurements, nor does it load real medical images.

Full volume loading, NIfTI metadata handling, and heavier integration checks are covered separately in later smoke/integration test plans.

### `tests/test_training_logic.py`

#### Purpose

This test file verifies the training control logic in `idssp/sonk/model/training.py`.

The tested code governs how the training pipeline behaves over the course of a long training run: when to apply augmentations, when to log visual overlays, when to send notifications, whether a checkpoint is safe to resume from, and when to stop training and save the best model.

These are high-value tests because a bug in any of these components can silently corrupt a multi-day training run, waste GPU compute, or cause the best model checkpoint to be lost.

The tests use only mocks, synthetic values, and lightweight config objects. They do not require GPU access, real LiTS volumes, real environment settings, network access, or real checkpoint files.

---

#### Scope

The file tests logic-only methods in `idssp/sonk/model/training.py`:

- `AugmentedDataset` (length and item retrieval with augmentation),
- `ModelBuilder._should_log_overlay` (epoch-based TensorBoard overlay schedule),
- `ModelBuilder._should_notify` (epoch-based notification schedule),
- `ModelBuilder._validate_checkpoint` (checkpoint compatibility validation),
- `EarlyStopper.__call__` (early stopping and best-model checkpoint logic).

The file does not test model construction, data loading, transform pipelines, or the full training loop.

---

#### Main behaviours tested

| Area | Function/class tested | Behaviour verified |
|---|---|---|
| Data augmentation wrapper | `AugmentedDataset` | Length matches base dataset; `__getitem__` applies the random transform to the base item |
| Overlay logging schedule | `ModelBuilder._should_log_overlay` | Every epoch for epochs 0–10; every 5 epochs for 11–30; every 10 epochs for 31+; zero-indexed epoch convention |
| Notification schedule | `ModelBuilder._should_notify` | Every 5 epochs for 0–50; every 10 epochs for 51–100; every 20 epochs for 101+; zero-indexed epoch convention |
| Checkpoint validation | `ModelBuilder._validate_checkpoint` | Hard fail on missing `model_state_dict`; hard fail on `MODEL` mismatch; hard fail on `NUM_CLASSES` mismatch; warning on preprocessing mismatches; warning on missing `config_snapshot`; hard fail when saved epoch >= `NUM_EPOCHS` |
| Early stopping | `EarlyStopper.__call__` | Monitors tumour Dice only; improvement requires strict `>` beyond `min_delta`; resets counter on improvement; increments counter on non-improvement; returns `True` only after `patience + 1` consecutive non-improving calls; writer and checkpoint saving occur only on improvement |

---

#### Why this test file matters

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

#### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests do not call `config.init()` or depend on the real `.env` file.
- The tests do not instantiate real MONAI networks or run real training loops.
- The tests do not write real checkpoint files; `EarlyStopper.save_checkpoint` is patched out.
- The tests use `ModelBuilder.__new__()` to bypass heavy initialisation where only logic methods are exercised.
- The tests respect the convention that early stopping monitors tumour Dice only.
- The tests respect the zero-indexed epoch convention used by the training loop.
- The tests verify the exact patience sequence: with patience `P`, the first `P` non-improving calls return `False`, and only the `(P+1)`th returns `True`.

---

#### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_training_logic.py -v
```

---

#### Limitations

This file tests training control logic only. It does not verify:

- model construction or forward-pass correctness (covered in later smoke tests),
- data loading or transform pipeline behaviour,
- the full training loop end-to-end,
- actual checkpoint serialisation and deserialisation (the save/load round-trip),
- TensorBoard output format,
- notification delivery.

Full pipeline integration is verified separately via `python main.py --fast-run` as described in `AGENTS.md`.

### `tests/test_loader.py`

#### Purpose

This test file verifies the data discovery, pairing, and split-loading logic in `idssp/sonk/disk/loader.py`.

The tested code is responsible for finding LiTS CT volumes, pairing them with their corresponding segmentation masks, and loading the stratified train/validation split JSONs. These operations are critical because silent failures in data loading can lead to corrupted datasets, mismatched image-label pairs, or incorrect sample sizes in downstream results tables.

The tests use only synthetic LiTS-like file trees created under pytest's `tmp_path`. They do not require real LiTS data, GPU access, network access, or real environment settings.

---

#### Scope

The file tests the `CustomDataset` and `DataCollector` classes in `idssp/sonk/disk/loader.py`:

- `CustomDataset.discover_and_pair` and `get_lits_paths` (file discovery and ID-based pairing),
- `DataCollector.read_dir` (directory validation and file counting),
- `DataCollector.extract_images_and_labels` (pair extraction),
- `DataCollector._load_split` (JSON split loading and disk validation),
- `DataCollector.get_stratified_split` (end-to-end split retrieval).

The file does not load real NIfTI volumes, parse image headers, or run model training.

---

#### Main behaviours tested

| Area | Functions/classes tested | Behaviour verified |
|---|---|---|
| File pairing | `CustomDataset.get_lits_paths` | Correctly pairs `volume-X.nii.gz` with `segmentation-X.nii.gz`; warns and skips when labels are missing; excludes non-volume files from pairing and logs a warning for each |
| Source validation | `CustomDataset.discover_and_pair` | Raises `ValueError` for unsupported dataset sources or when files have not been set |
| Directory reading | `DataCollector.read_dir` | Raises `FileNotFoundError` for missing directories; raises `ValueError` for empty directories; warns when the file count is odd (indicating unpaired files) |
| Split loading | `DataCollector._load_split` | Returns correct train/val lists; raises `FileNotFoundError` when the JSON references files missing from disk; warns when disk contains files not listed in the JSON |
| Split retrieval | `DataCollector.get_stratified_split` | Raises `ValueError` when no data is loaded; raises `FileNotFoundError` when the configured split JSON is missing; returns correct train/val pairs when valid |

---

#### Why this test file matters

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

#### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests do not load real LiTS volumes or parse real NIfTI headers.
- The tests use synthetic file trees created under `tmp_path`.
- The tests mock the `config` singleton to avoid depending on the real `.env` file.
- Warning paths are verified using `caplog` to ensure the loader emits the expected diagnostics for missing labels, unpaired files, and split mismatches.

---

#### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_loader.py -v
```

---

#### Limitations

This file tests file discovery, pairing, and split loading. It does not verify:

- the actual loading and parsing of NIfTI image data,
- the application of MONAI transforms to the loaded paths,
- the correctness of the stratification algorithm itself (covered in `test_stratification.py`),
- or the full end-to-end training pipeline.

### `tests/test_force_matching_affined.py`

#### Purpose

This test file verifies the `ForceMatchingAffined` transform in `idssp/sonk/model/transforms.py`.

This transform is a targeted, safety-gated fix for known broken LiTS volumes (specifically volumes 48–52) where the segmentation mask's NIfTI header contains a placeholder identity affine matrix. The transform copies the correct affine from the image to the label, but only when strict safety conditions are met.

The tests use synthetic affines, lightweight fake metadata objects, and real MONAI `MetaTensor` instances. They do not require real LiTS data, GPU access, or network access.

---

#### Scope

The file tests the `ForceMatchingAffined` class and its internal helpers:

- `_is_placeholder_affine` (detecting broken identity-like affines),
- `_normalise_affine` (converting various affine formats to a standard CPU tensor),
- `_validate_case_name` (enforcing the strict allow-list of LiTS volume IDs),
- `__call__` (the end-to-end correction logic and safety guards).

The tests deliberately preserve the production invariants: `_ALLOWED_LITS_VOLUMES` remains `{48, 49, 50, 51, 52}` and `_IDENTITY_AFFINE_THRESHOLD` remains `1e-3`.

---

#### Main behaviours tested

| Area | Functions tested | Behaviour verified |
|---|---|---|
| Constants | `_ALLOWED_LITS_VOLUMES`, `_IDENTITY_AFFINE_THRESHOLD` | Values remain exactly as defined in production; not widened or loosened |
| Placeholder detection | `_is_placeholder_affine` | Detects identity matrices, identity matrices with translation, and batched affines; rejects non-identity spacing, `None`, and near-identity values above the threshold |
| Affine normalisation | `_normalise_affine` | Accepts torch tensors, NumPy arrays, and batched `(1, 4, 4)` shapes; returns a CPU `float32` `(4, 4)` tensor; raises `ValueError` for invalid shapes or wrong batch sizes |
| Case name validation | `_validate_case_name` | Accepts allowed IDs (48–52); raises `ValueError` for disallowed IDs (e.g., 47, 53) and malformed filenames |
| Correction logic | `__call__` | Returns data unchanged if objects lack `.meta` or affines; copies image affine to label only when label is placeholder and image is not; writes to `.meta["affine"]` if `.affine` attribute is read-only; raises `ValueError` if correction is attempted on an unapproved volume; does not overwrite a valid non-placeholder label affine |

---

#### Why this test file matters

This file protects a critical data-handling safeguard.

As noted in `AGENTS.md`, LiTS volumes 48–52 have severe affine mismatches. If the correction logic is too broad, it could silently overwrite valid affines on other volumes, corrupting spatial metadata and ruining downstream metrics. If it is too narrow or fails to trigger, the model will train on geometrically misaligned masks for those specific cases.

These tests ensure that the correction is applied **only** to the explicitly approved volumes, **only** when the label affine is demonstrably broken, and **never** when the label already contains valid spatial information.

---

#### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests do not widen `_ALLOWED_LITS_VOLUMES` or change `_IDENTITY_AFFINE_THRESHOLD`.
- The tests use synthetic data and lightweight mock objects to avoid loading real NIfTI files.
- The tests verify both the primary execution path (setting `.affine`) and the fallback path (setting `.meta["affine"]` when the attribute is read-only).

---

#### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_force_matching_affined.py -v
```

---

#### Limitations

This file tests the affine correction logic in isolation. It does not verify the full MONAI transform pipeline, the actual loading of NIfTI files from disk, or the downstream impact of the corrected affines on model training.

### `tests/test_evaluator_postprocess.py`

#### Purpose

This test file verifies the post-processing and report-export logic in the evaluation pipeline (specifically `_post_process_class_map` and `MetricsEvaluator.generate_report`).

The tested code is responsible for enforcing anatomical realism on the model's raw 3D segmentation predictions (e.g., removing disconnected liver fragments and stray tumours) and for exporting the final per-case metrics to CSV files for thesis reporting.

These tests are critical because incorrect post-processing can silently alter final Dice and HD95 scores, and incorrect CSV export can lead to incomplete or misaligned thesis tables. The tests use small synthetic 3D NumPy arrays and in-memory pandas DataFrames, requiring no GPU, real LiTS data, or network access.

---

#### Scope

The file tests two main components:

- `_post_process_class_map`: The 3D NumPy array cleanup function that enforces largest-connected-component (LCC) rules for the liver and tumour classes.
- `MetricsEvaluator.generate_report`: The method that aggregates results and exports raw and post-processed metrics to CSV.

The file does not run full-volume inference, load real NIfTI files, or compute actual MONAI metrics.

---

#### Main behaviours tested

| Area | Functions/classes tested | Behaviour verified |
|---|---|---|
| Basic invariants | `_post_process_class_map` | Returns an array of the same shape and dtype; handles empty inputs safely; removes stray tumours when no liver is present |
| Liver retention | `_post_process_class_map` | Retains only the largest connected component of the liver; removes small, disconnected stray liver voxels |
| Tumour anchoring | `_post_process_class_map` | Preserves tumour voxels that are inside or immediately adjacent to the retained liver anatomy; strips tumour voxels that are disconnected from the main anatomy |
| Fragmentation warning | `_post_process_class_map` | Emits a warning if more than 50% of predicted liver voxels are discarded during LCC cleanup; remains silent when the liver is mostly retained or absent |
| Report export | `MetricsEvaluator.generate_report` | Raises `ValueError` on empty input dictionaries; skips empty DataFrames without crashing; writes correctly named raw and post-processed CSVs; sorts rows lexicographically by `case_name`; respects custom output directories or falls back to a reports directory under `config.RUN_DIR` |

---

#### Why this test file matters

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

#### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests strictly enforce the 3-class layout defined in `AGENTS.md`: `0` (background), `1` (liver), `2` (tumour).
- The tests use synthetic 3D NumPy arrays and explicitly enforce 6-connectivity (face-adjacency) when building connected components, matching `scipy.ndimage.label` defaults.
- The tests verify lexicographical sorting for `case_name` (e.g., `vol-1`, `vol-10`, `vol-2`), matching standard pandas behaviour.
- The tests use `caplog` to verify the exact conditions under which the fragmentation warning is emitted.

---

#### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_evaluator_postprocess.py -v
```

---

#### Limitations

This file tests the post-processing logic and CSV export in isolation. It does not verify the end-to-end sliding-window inference pipeline, the MONAI metric calculations (Dice/HD95), or the `Invertd` transform used to map predictions back to the original scanner space. Those integration steps are verified via the `--fast-run` entrypoint and server-side validation scripts.

## Series 2 — Config, factory, and inference unit tests (Plan 2)

### `tests/test_config.py`

#### Purpose

This test file verifies the configuration, environment detection, and serialisation logic in `idssp/sonk/config.py`.

The tested code is the foundation of the pipeline, responsible for reading environment variables, detecting hardware capabilities (CPU/GPU/RAM), validating required paths, and providing a frozen `Config` singleton to all downstream modules.

These tests are critical because a silent failure in configuration (e.g., misidentifying a limited environment or leaking credentials into a checkpoint) can invalidate training runs or compromise security. The tests use mocked hardware detection, controlled environment variables, and temporary directories, requiring no GPU, network access, or real `.env` file.

---

#### Scope

The file tests the core components of `idssp/sonk/config.py`:

- `AvailableModels` and `Mode` enums.
- The frozen `Config` dataclass and module-level singleton lifecycle.
- `init()` (environment validation, path creation, fallback logic, and notification validation).
- `get()` (singleton access).
- `is_limited_env()` (hardware and environment gating).
- `to_dict()` and `to_param_dict()` (serialisation and secret exclusion).
- `get_cgroup_memory_limit_bytes()` and `get_container_usage()` (container memory detection).

The file does not test the full training loop, model construction, or real network/SMTP connectivity.

---

#### Main behaviours tested

| Area | Functions/classes tested | Behaviour verified |
|---|---|---|
| Enums and Dataclass | `AvailableModels`, `Mode`, `Config` | Correct string values for enums; `Config` is strictly frozen (assignment raises `FrozenInstanceError`) |
| Singleton lifecycle | `get()`, `init()` | `get()` raises `RuntimeError` before initialisation; re-initialising with the same `Mode` returns the same instance; re-initialising with a different `Mode` raises `RuntimeError` |
| Environment validation | `init()` | Raises `EnvironmentError` for missing `PIN_ENV`; raises `ValueError` for unrecognised `PIN_ENV` or missing required variables (`LITS_CT_ROOT`, `SPLIT_JSON`, etc.); raises `FileNotFoundError` for missing CT root directory |
| Fallbacks and paths | `init()` | Invalid log levels fall back to `INFO`/`DEBUG`; cache sources fall back to `disk` when RAM is low; run and log directories are created automatically |
| Notification validation | `init()` | Raises `ValueError` when email/Telegram notifications are enabled but required credentials or fields are missing/invalid |
| Hardware gating | `is_limited_env()` | Returns `True` for local environments, CPU devices, or low-VRAM GPUs (when `include_vram=True`); returns `False` for high-VRAM cloud environments |
| Serialisation | `to_dict()`, `to_param_dict()` | Enums converted to strings; tuples to lists; Paths to strings; sensitive fields (credentials, tokens) strictly excluded; noisy/path keys excluded from parameter dicts |
| Container memory | `get_cgroup_memory_limit_bytes()`, `get_container_usage()` | Correctly parses cgroup v2 numeric limits and `"max"` fallbacks; handles cgroup v1 sentinels; returns `-1` sentinels for missing files or unlimited environments |

---

#### Why this test file matters

This file protects the foundational configuration layer of the project.

Every downstream module (data loading, training, inference) relies on the `Config` singleton to determine execution paths. If `is_limited_env()` incorrectly identifies a machine, the pipeline might attempt GPU-only sliding-window inference on a CPU, or fail to invert spatial transforms correctly, resulting in geometrically invalid NIfTI files. If `to_dict()` fails to exclude secrets, credentials could be written into checkpoint files.

These tests catch problems such as:

- silent acceptance of missing or invalid environment variables,
- incorrect hardware detection leading to out-of-memory crashes or invalid outputs,
- credentials leaking into serialised config snapshots,
- failure to respect Docker/container memory limits (cgroup parsing errors).

---

#### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests never load the real repository `.env` file (`load_dotenv` is patched to a no-op).
- The tests patch hardware detection (`torch.cuda.is_available`, `psutil.virtual_memory`) to ensure deterministic behaviour regardless of the host machine.
- The tests use targeted patches for cgroup file reading to avoid broad `open()` mocks that could affect unrelated operations.
- The `Config` singleton is automatically reset before and after every test via an `autouse` fixture in `conftest.py`.

---

#### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_config.py -v
```

---

#### Limitations

This file tests configuration logic and environment detection in isolation. It does not verify the actual loading of NIfTI data, the execution of the training loop, or the real delivery of email/Telegram notifications.

### `tests/test_models_factory.py`

#### Purpose

This test file verifies the model factory dispatch, architecture validation, and pretrained-weight loading logic in `idssp/sonk/model/models.py`.

The tested code determines which neural network architecture is constructed during training and evaluation, and ensures that pretrained weights are loaded safely and correctly. These tests are critical because they provide the regression net required before modifying `AvailableModels` and `get_model()` to add the 2.5D Mamba-hybrid architecture.

The tests use lightweight mock objects and a tiny `torch.nn.Module` stand-in. No heavy MONAI networks (UNet, SegResNet, SwinUNETR) are actually constructed, and no real checkpoint files are read from disk.

---

#### Scope

The file tests the core components of `idssp/sonk/model/models.py`:

- `get_model()` (factory dispatch).
- `get_swin_unetr()` (patch-size validation).
- `_load_monai_pretrained_weights()` (checkpoint unwrapping and state-dict cleaning).
- `get_swin_unetr_pretrain()` (error wrapping for pretrained weight loading).

The file does not test model forward passes, loss computation, or the full training loop.

---

#### Main behaviours tested

| Area | Functions tested | Behaviour verified |
|---|---|---|
| Factory dispatch | `get_model()` | Routes `U_NET`, `SEG_RES_NET`, `SWIN_UNETR`, and `SWIN_UNETR_PRETRAIN` to their respective builders; raises `ValueError` for unsupported model strings |
| Architecture validation | `get_swin_unetr()` | Raises `ValueError` when `TRAIN_PATCH_SIZE` does not contain exactly 3 spatial dimensions; raises `ValueError` when any dimension is not divisible by 32 |
| Weight loading security | `_load_monai_pretrained_weights()` | Enforces `weights_only=True` when calling `torch.load` |
| Checkpoint unwrapping | `_load_monai_pretrained_weights()` | Accepts direct state dicts; unwraps checkpoints nested under `state_dict`, `model_state_dict`, or `model` keys; accepts full `torch.nn.Module` checkpoints |
| State-dict cleaning | `_load_monai_pretrained_weights()` | Strips `module.` prefixes added by `DataParallel`/`DDP`; calls `load_state_dict` with `strict=False` |
| Error handling | `_load_monai_pretrained_weights()`, `get_swin_unetr_pretrain()` | Wraps `torch.load` failures in `RuntimeError`; raises `TypeError` for unsupported checkpoint formats; wraps pretrained loading failures in `get_swin_unetr_pretrain` |

---

#### Why this test file matters

This file protects the model construction layer of the project.

As noted in `AGENTS.md`, the primary target architecture (2.5D Mamba-hybrid) is not yet in `AvailableModels`. When it is added, `get_model()` and the factory dispatch logic will be modified. These tests ensure that adding a new architecture does not accidentally break the dispatch for the existing baselines (UNet, SegResNet, SwinUNETR).

Additionally, SwinUNETR is highly sensitive to patch sizes. If the validation logic in `get_swin_unetr()` is weakened or removed, the pipeline could attempt to construct a network with invalid dimensions, leading to cryptic MONAI shape errors deep inside the forward pass.

---

#### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests do not construct real MONAI networks; heavy classes like `SwinUNETR` are patched with `MagicMock`.
- The tests use a tiny `torch.nn.Module` (`_TinyModule`) to verify state-dict key cleaning without the overhead of a full segmentation model.
- The tests enforce `weights_only=True` for checkpoint loading, matching the security invariant in `AGENTS.md`.

---

#### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_models_factory.py -v
```

---

#### Limitations

This file tests factory dispatch and weight-loading logic in isolation. It does not verify that the constructed models produce correct output shapes, nor does it test the integration of these models into the `ModelBuilder` training loop.

### `tests/test_inferer_checkpoint.py`

#### Purpose

This test file verifies the checkpoint loading, config alignment, and limited-environment guard logic in `idssp/sonk/model/inferer.py`.

The tested code is responsible for loading a training checkpoint and reconstructing the exact preprocessing environment required for valid full-volume inference. This is critical because inference must use the same spatial parameters (spacing, patch size, HU window) and architecture (model type, number of classes) as training. Misalignment would produce geometrically invalid or semantically incorrect predictions.

The tests use mocked heavy dependencies (`torch.load`, `get_model`, `SlidingWindowInferer`, `get_validation_transforms`) and a real frozen `Config` fixture. No real checkpoint files, MONAI models, or full inference runs are exercised.

---

#### Scope

The file tests two components of `idssp/sonk/model/inferer.py`:

- `InferenceEngine.load_checkpoint()` — checkpoint loading and config alignment.
- `InferenceEngine.run_inference()` — limited-environment guard only.

The file does not test full-volume sliding-window inference, NIfTI export, or the `Invertd` transform pipeline.

---

#### Main behaviours tested

| Area | Functions tested | Behaviour verified |
|---|---|---|
| Path validation | `load_checkpoint()` | Raises `FileNotFoundError` when the checkpoint path does not exist |
| Spatial field conversion | `load_checkpoint()` | JSON lists for `ISO_SPACING` and `TRAIN_PATCH_SIZE` are converted back to Python tuples |
| Strict key alignment | `load_checkpoint()` | `NUM_CLASSES`, `HU_WINDOW_MIN`, `HU_WINDOW_MAX`, and `TUMOUR_CLASS_INDEX` are aligned from the checkpoint snapshot |
| Enum conversion | `load_checkpoint()` | `MODEL` string is converted back to the `AvailableModels` enum |
| Non-critical warnings | `load_checkpoint()` | Mismatches in `SLIDING_WINDOW_BATCH_SIZE` and `RAND_CROP_NUM_SAMPLES` produce warnings but do not raise |
| Model construction | `load_checkpoint()` | The model is built using the aligned config, not the original environment config |
| Limited-environment guard | `run_inference()` | Raises `RuntimeError` in limited environments because random crops cannot be inverted to scanner space |

---

#### Why this test file matters

This file protects the inference portability layer of the project.

Training runs on the server (cloud environment) with specific hyperparameters, while inference and evaluation run locally. The checkpoint's `config_snapshot` is the bridge between these two environments. If the alignment logic fails silently, the inference engine might use the wrong model architecture, wrong number of output classes, or wrong spatial resampling parameters, producing predictions that look plausible but are fundamentally incorrect.

The limited-environment guard is equally important. As noted in `AGENTS.md`, local/CPU runs use random-crop patches that cannot be inverted back to the original scanner space. Without this guard, `Invertd` would silently produce NIfTI files in preprocessed space, corrupting downstream metric calculations.

---

#### Important conventions preserved by the tests

- The tests do not modify production source code.
- The tests use a real frozen `Config` instance (from `conftest.py`) for `dataclasses.replace()` compatibility, not a `MagicMock`.
- The tests do not load real checkpoint files; `torch.load` is patched with a controlled payload.
- The tests do not construct real MONAI models; `get_model` is patched.
- The tests do not run full inference; only the guard logic is exercised.

---

#### How to run

From the repository root:

```bash
~/envs/dev-thesis/bin/python -m pytest tests/test_inferer_checkpoint.py -v
```

---

#### Limitations

This file tests checkpoint loading and config alignment in isolation. It does not verify full-volume sliding-window inference, the `Invertd` spatial inversion pipeline, or NIfTI export correctness. Those integration paths are verified via `scripts/validate.sh` (which runs `do_inference.py`) on the server.