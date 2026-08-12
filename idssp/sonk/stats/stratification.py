"""
Stratification utilities for analysing a given dataset.

This module provides functions to:
- print statistics for Train/Val/Test splits;
- perform simple dataset checks;
- generate a thesis draft table comparing splits;
- save stratification metadata for the training pipeline.

It also includes binning helpers that can be used when constructing
stratification keys.
"""

import datetime
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.stats import chi2_contingency, kruskal


_SPACING_TABLE_LABELS = {
    "thin": "Thin (<=1.0 mm)",
    "medium": "Medium (1.25-1.5 mm)",
    "thick": "Thick (>=2.0 mm)",
    "other": "Other",
    "unknown": "Unknown",
}


def bin_spacing(z: float) -> str:
    """
    Bin slice spacing into canonical categories.

    Parameters
    ----------
    z : float
        Slice spacing in mm.

    Returns
    -------
    str
        One of ``"thin"``, ``"medium"``, ``"thick"``, ``"other"``,
        or ``"unknown"``.
    """
    if pd.isna(z):
        return "unknown"

    if z <= 1.0:
        return "thin"

    if 1.25 <= z <= 1.5:
        return "medium"

    if z >= 2.0:
        return "thick"

    return "other"


def _spacing_label(z: float) -> str:
    """
    Convert a spacing value into a human-readable table label.
    """
    return _SPACING_TABLE_LABELS[bin_spacing(z)]


def bin_liver_hu(hu: float) -> str:
    """
    Bin mean liver HU into canonical categories.

    Parameters
    ----------
    hu : float
        Mean liver HU.

    Returns
    -------
    str
        One of ``"low"``, ``"mid"``, ``"high"``, or ``"unknown"``.
    """
    if pd.isna(hu):
        return "unknown"

    if hu < 60:
        return "low"

    if hu < 100:
        return "mid"

    return "high"


def bin_tumor_vol(vol: float, has_tumor: bool) -> str:
    """
    Bin tumour volume into canonical categories.

    Parameters
    ----------
    vol : float
        Tumour volume in ml.
    has_tumor : bool
        Whether the case contains a tumour.

    Returns
    -------
    str
        One of ``"none"``, ``"small"``, ``"medium"``, ``"large"``,
        or ``"unknown"``.
    """
    if not has_tumor:
        return "none"

    if pd.isna(vol):
        return "unknown"

    if vol < 5:
        return "small"

    if vol < 50:
        return "medium"

    return "large"


def _format_median_iqr(series: pd.Series) -> str:
    """
    Format a numeric series as median [Q1-Q3].
    """
    series = series.dropna()

    if series.empty:
        return "N/A"

    return (
        f"{series.median():.2f} "
        f"[{series.quantile(0.25):.2f}-{series.quantile(0.75):.2f}]"
    )


def _safe_kruskal_p(groups: list[pd.Series]) -> Optional[float]:
    """
    Run Kruskal-Wallis safely.

    Returns ``None`` when the test cannot be meaningfully computed.
    """
    groups = [group.dropna() for group in groups]

    if any(group.empty for group in groups):
        return None

    if any(len(group) < 2 for group in groups):
        return None

    combined = pd.concat(groups)
    if combined.nunique(dropna=True) < 2:
        return None

    result = kruskal(*groups)
    p_value = float(result.pvalue)

    if pd.isna(p_value):
        return None

    return p_value


def _safe_chi2_p(contingency: pd.DataFrame) -> Optional[float]:
    """
    Run chi-square safely on a contingency table.

    Returns ``None`` when the test cannot be meaningfully computed.
    """
    if contingency.empty:
        return None

    if contingency.values.sum() == 0:
        return None

    if (contingency.sum(axis=0) == 0).any():
        return None

    if (contingency.sum(axis=1) == 0).any():
        return None

    try:
        _, p_val, _, expected = chi2_contingency(contingency)
    except ValueError:
        return None

    if (expected < 5).any():
        print(
            "WARNING: Chi-square assumption may be violated: "
            "some expected counts are below 5."
        )

    if pd.isna(p_val):
        return None

    return float(p_val)


def _mask_no_tumour_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure tumour volume is treated as missing for cases without tumour.

    This prevents no-tumour cases encoded as zero volume from being
    analysed as real tumour volumes.
    """
    df = df.copy()

    if "has_tumor" in df.columns and "tumour_volume_ml" in df.columns:
        df.loc[~df["has_tumor"].astype(bool), "tumour_volume_ml"] = float("nan")

    return df


def print_stats(df: pd.DataFrame, name: str) -> None:
    """
    Print general statistics about a given dataset split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset split to analyse.
    name : str
        Name of the dataset split, for example ``"Training"``,
        ``"Validation"``, or ``"Test"``.
    """
    has_tumor = df["has_tumor"].astype(bool)

    mean_vol = df.loc[has_tumor, "tumour_volume_ml"].mean()
    mean_vol_str = "N/A" if pd.isna(mean_vol) else f"{mean_vol:.2f} ml"

    mean_hu = df["liver_hu_mean"].mean()
    mean_hu_str = "N/A" if pd.isna(mean_hu) else f"{mean_hu:.1f} HU"

    spacing_groups = df["spacing_z"].apply(bin_spacing)

    print(f"--- {name} Set ---")
    print(f"Total cases: {len(df)}")
    print(f"Cases without tumours: {(~has_tumor).sum()}")
    print(f"Cases with tumours: {has_tumor.sum()}")
    print(f"Tumour volume (mean): {mean_vol_str}")
    print(f"Liver HU (mean): {mean_hu_str}")

    print("Spacing distribution:")
    for key, label in _SPACING_TABLE_LABELS.items():
        print(f"  {label}: {(spacing_groups == key).mean():.1%}")

    print("\n")


def print_checks(df: pd.DataFrame, name: str) -> None:
    """
    Print simple checks for a given dataset split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset split to analyse.
    name : str
        Name of the dataset split.
    """
    has_tumor = df["has_tumor"].astype(bool)

    df_small = df.loc[
        has_tumor & (df["tumour_volume_ml"] < 5.0),
        "tumour_volume_ml",
    ]

    if df_small.empty:
        small_volume_range = "N/A"
    else:
        small_volume_range = (
            f"{df_small.min():.2f} - {df_small.max():.2f} ml"
        )

    print(f"--- {name} Set Checks ---")
    print(f"Unique stratification keys: {df['strat_key'].nunique()}")
    print(f"Cases with small tumours (<5 ml): {len(df_small)}")
    print(f"Volume range for small tumours: {small_volume_range}")

    print(f"Thick slices (>=2.0 mm): {(df['spacing_z'] >= 2.0).sum()}")
    print(f"Multi-focal cases (>=5 lesions): {(df['num_lesions'] >= 5).sum()}")
    print(f"Low liver HU cases (<60 HU): {(df['liver_hu_mean'] < 60).sum()}")

    print("\n")


def generate_thesis_table_1(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Print a draft 'Table 1' comparing Train, Val, and Test splits.

    Continuous variables are summarised as median [IQR] and compared
    using Kruskal-Wallis. Categorical variables are summarised as N (%)
    and compared using chi-square.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    val_df : pd.DataFrame
        Validation dataset.
    test_df : pd.DataFrame
        Test dataset.
    """
    # 1. Prepare data and add split identifiers.
    train_df = _mask_no_tumour_volume(train_df)
    val_df = _mask_no_tumour_volume(val_df)
    test_df = _mask_no_tumour_volume(test_df)

    train_label = f"Train (n={len(train_df)})"
    val_label = f"Val (n={len(val_df)})"
    test_label = f"Test (n={len(test_df)})"

    train_df["Split"] = train_label
    val_df["Split"] = val_label
    test_df["Split"] = test_label

    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Bin spacing_z for chi-square.
    df["Spacing Group"] = df["spacing_z"].apply(_spacing_label)

    # 2. Define variables.
    continuous_vars = {
        "Liver HU Mean (HU)": "liver_hu_mean",
        "Liver Volume (ml)": "liver_volume_ml",
        "Tumour Volume (ml)*": "tumour_volume_ml",
        "Number of Lesions": "num_lesions",
        "Liver Texture Variance": "liver_texture_variance",
        "Tumour to Liver Ratio": "tumor_to_liver_ratio",
    }

    categorical_vars = {
        "Has Tumour": "has_tumor",
        "Slice Thickness Group": "Spacing Group",
    }

    # 3. Statistical analysis and formatting.
    print(
        f"{'Variable':<25} | {train_label:<22} | "
        f"{val_label:<22} | {test_label:<22} | {'p-value':<10}"
    )
    print("-" * 120)

    # Continuous variables.
    for label, col in continuous_vars.items():
        train_values = train_df[col].dropna()
        val_values = val_df[col].dropna()
        test_values = test_df[col].dropna()

        train_str = _format_median_iqr(train_values)
        val_str = _format_median_iqr(val_values)
        test_str = _format_median_iqr(test_values)

        p_val = _safe_kruskal_p([train_values, val_values, test_values])

        if p_val is None:
            p_str = "N/A"
        elif p_val >= 0.001:
            p_str = f"{p_val:.3f}"
        else:
            p_str = "<0.001"

        print(
            f"{label:<25} | {train_str:<22} | "
            f"{val_str:<22} | {test_str:<22} | {p_str:<10}"
        )

    print("-" * 120)

    # Categorical variables.
    for label, col in categorical_vars.items():
        contingency = pd.crosstab(df[col], df["Split"])

        # Ensure all splits are present in columns.
        for split_col in [train_label, val_label, test_label]:
            if split_col not in contingency.columns:
                contingency[split_col] = 0

        contingency = contingency[[train_label, val_label, test_label]]

        p_val = _safe_chi2_p(contingency)

        if p_val is None:
            p_str = "N/A"
        elif p_val >= 0.001:
            p_str = f"{p_val:.3f}"
        else:
            p_str = "<0.001"

        print(
            f"{label:<25} | {'':<22} | {'':<22} | {'':<22} | {p_str:<10}"
        )

        for cat in contingency.index:
            train_n = contingency.loc[cat, train_label]
            val_n = contingency.loc[cat, val_label]
            test_n = contingency.loc[cat, test_label]

            train_pct = (train_n / len(train_df) * 100) if len(train_df) else 0.0
            val_pct = (val_n / len(val_df) * 100) if len(val_df) else 0.0
            test_pct = (test_n / len(test_df) * 100) if len(test_df) else 0.0

            cat_str = f"  {str(cat)}"

            train_fmt = f"{train_n} ({train_pct:.1f}%)"
            val_fmt = f"{val_n} ({val_pct:.1f}%)"
            test_fmt = f"{test_n} ({test_pct:.1f}%)"

            print(
                f"{cat_str:<25} | {train_fmt:<22} | "
                f"{val_fmt:<22} | {test_fmt:<22} | {'':<10}"
            )

    print("-" * 120)
    print("* Tumour volume calculated only for cases where has_tumor == True.")
    print(
        "Note: A p-value > 0.05 indicates no statistically detectable difference "
        "between the splits for the tested variable. It does not prove balance, "
        "especially with small sample sizes."
    )


def save_stratification_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str,
    include_test: bool = False,
    stratification_method: str = "iterative_multilabel_with_pre_allocation",
    pre_allocation_rule: Optional[str] = (
        "Cases with liver_hu_mean < 60 were deterministically pre-allocated "
        "(1 to train, 1 to val, 1 to test) prior to iterative stratification "
        "to guarantee rare protocol representation."
    ),
) -> Path:
    """
    Save stratification metadata to a JSON file.

    By default, the test set is excluded because this file is intended
    for training pipeline configuration. Set ``include_test=True`` if you
    explicitly want to include the test split in the same metadata file.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split.
    val_df : pd.DataFrame
        Validation split.
    test_df : pd.DataFrame
        Test split.
    output_path : str
        Output JSON file path.
    include_test : bool, default=False
        Whether to include the test split in the metadata.
    stratification_method : str, default="iterative_multilabel_with_pre_allocation"
        Description of the stratification method.
    pre_allocation_rule : str | None, default=None
        Description of deterministic pre-allocation, if used.

    Returns
    -------
    Path
        The saved JSON file path.
    """
    split_dfs = {
        "train": train_df,
        "val": val_df,
    }

    if include_test:
        split_dfs["test"] = test_df

    bins = {}

    for split_df in split_dfs.values():
        for key, group in split_df.groupby("strat_key"):
            key = str(key)

            if key not in bins:
                bins[key] = []

            bins[key].extend(group["case_name"].astype(str).tolist())

    bins = {
        key: sorted(cases)
        for key, cases in sorted(bins.items())
    }

    metadata = {
        "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stratification_method": stratification_method,
        "stratified_by": "composite_key (slice_group + liver_hu_group + tumor_group)",
        "train": sorted(train_df["case_name"].astype(str).tolist()),
        "val": sorted(val_df["case_name"].astype(str).tolist()),
        "bins": bins,
    }

    if pre_allocation_rule is not None:
        metadata["pre_allocation_rule"] = pre_allocation_rule

    if include_test:
        metadata["test"] = sorted(test_df["case_name"].astype(str).tolist())

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Stratification metadata successfully saved to {out_path.resolve()}")

    return out_path.resolve()
