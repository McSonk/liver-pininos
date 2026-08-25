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

