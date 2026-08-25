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
