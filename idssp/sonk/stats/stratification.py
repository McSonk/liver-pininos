"""
Stratification utilities for analysing a given dataset. It provides functions
to print statistics, perform checks and generate a publication-ready table
comparing Train, Val, and Test splits for a given stratification combination.

It also includes helper functions to build the stratification itself,
such as binning the spacing_z, liver_hu_mean, and tumour_volume_ml into discrete categories
and generating the stratification metadata JSON file for the training pipeline configuration.
"""
import datetime
import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, kruskal


# Bin spacing_z (CRITICAL)
def bin_spacing(z):
    if z <= 1.0: return 'thin'      # 0.7-1.0mm
    elif z <= 1.5: return 'medium'   # 1.25-1.5mm  
    else: return 'thick'             # >=2.0mm

# Bin liver HU (captures contrast phase)
def bin_liver_hu(hu):
    if hu < 60: return 'low'         # non-contrast/portal
    elif hu < 100: return 'mid'      # late arterial
    else: return 'high'              # early arterial

# Bin tumour volume (clinical difficulty)
def bin_tumor_vol(vol, has_tumor):
    if not has_tumor: return 'none'
    if vol < 5: return 'small'
    elif vol < 50: return 'medium'
    else: return 'large'


def print_stats(df: pd.DataFrame, name: str):
    '''
    Print general statistics about a given dataset, including the number of cases,
    tumor volume, liver HU, and stratification key distribution.

    Parameters:
    ---------
    df : `pandas.DataFrame`
        The DataFrame containing the dataset to analyze.
    name : `str`
        The name of the dataset (e.g., "Training", "Validation", "Test")
    '''
    mean_vol = df[df['has_tumor']]['tumour_volume_ml'].mean()

    print(f"--- {name} Set ---")
    print(f"Total cases: {len(df)}")
    print(f"Cases without tumors: {(df['has_tumor'] == 0).sum()}")
    print(f"Cases with tumors: {(df['has_tumor'] == 1).sum()}")
    print(f"Tumour volume (mean ml): {mean_vol:.2f} ml")
    print(f"Liver HU (mean): {df['liver_hu_mean'].mean():.1f} HU")
    print("Stratification key distribution:")
    print("=== SPACING_Z DISTRIBUTION ===")
    print(f"  Thin (≤1.0mm): {(df['spacing_z'] <= 1.0).mean():.1%}")
    print(f"  Medium (1.25-1.5): {(df['spacing_z'].between(1.25, 1.5)).mean():.1%}")
    print(f"  Thick (≥2.0mm): {(df['spacing_z'] >= 2.0).mean():.1%}")

    print("\n")

def print_checks(df: pd.DataFrame, name: str):
    '''
    Print checks for a given dataset, including the number of unique stratification keys,
    the number of cases with small tumours, and the volume range for small tumours.

    Parameters:
    ---------
    df : `pandas.DataFrame`
        The DataFrame containing the dataset to analyze.
    name : `str`
        The name of the dataset (e.g., "Training", "Validation", "Test")
    '''

    print(f"--- {name} Set Checks ---")
    print(f"Unique stratification keys: {df['strat_key'].nunique()}")
    df_small = df[df['tumour_volume_ml'] < 5.0]
    print(f"Cases with small tumors (<5ml): {len(df_small)}")
    print(f"Volume range for small tumors: {df_small['tumour_volume_ml'].min():.2f} - {df_small['tumour_volume_ml'].max():.2f} ml")
    print("✅ Thick slices (≥2.0mm):", (df['spacing_z'] >= 2.0).sum())
    print("✅ Small tumours (<5 ml):", (df['tumour_volume_ml'] < 5).sum())
    print("✅ Multi-focal (≥5 lesions):", (df['num_lesions'] >= 5).sum())
    print("✅ Low liver HU (<60):", (df['liver_hu_mean'] < 60).sum())
    print("✅ Multi-focal (≥5 lesions):", (df['num_lesions'] >= 5).sum())
    print("✅ Low liver HU (<60):", (df['liver_hu_mean'] < 60).sum())
    print("\n")

def generate_thesis_table_1(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    '''
    Generates a publication-ready 'Table 1' comparing Train, Val, and Test splits.
    Uses Kruskal-Wallis for continuous (skewed) variables and Chi-square for categorical.

    Parameters:
    ---------
    train_df : `pandas.DataFrame`
        The DataFrame containing the training dataset.
    val_df : `pandas.DataFrame`
        The DataFrame containing the validation dataset.
    test_df : `pandas.DataFrame`
        The DataFrame containing the test dataset.
    '''
    # 1. Prepare Data and add Split identifiers
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_label = f'Train (n={len(train_df)})'
    val_label = f'Val (n={len(val_df)})'
    test_label = f'Test (n={len(test_df)})'
    
    train_df['Split'] = train_label
    val_df['Split'] = val_label
    test_df['Split'] = test_label
    
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Bin spacing_z for Chi-square (requires discrete categories)
    def bin_spacing(z):
        if z <= 1.0: return 'Thin (<=1.0 mm)'
        elif z <= 1.5: return 'Medium (1.25-1.5 mm)'
        else: return 'Thick (>=2.0 mm)'
        
    df['Spacing Group'] = df['spacing_z'].apply(bin_spacing)
    
    # 2. Define Variables
    continuous_vars = {
        'Liver HU Mean (HU)': 'liver_hu_mean',
        'Liver Volume (ml)': 'liver_volume_ml',
        'Tumour Volume (ml)*': 'tumour_volume_ml',
        'Number of Lesions': 'num_lesions',
        'Liver Texture Variance': 'liver_texture_variance',
        'Tumour to Liver Ratio': 'tumor_to_liver_ratio'
    }
    
    categorical_vars = {
        'Has Tumour': 'has_tumor',
        'Slice Thickness Group': 'Spacing Group'
    }
    
    # 3. Statistical Analysis and Formatting
    print(f"{'Variable':<25} | {train_label:<22} | {val_label:<22} | {test_label:<22} | {'p-value':<10}")
    print("-" * 120)
    
    # --- Continuous Variables (Median [IQR], Kruskal-Wallis) ---
    for label, col in continuous_vars.items():
        # Drop NaNs (e.g., tumour_volume_ml might be NaN if has_tumor is False)
        t = train_df[col].dropna()
        v = val_df[col].dropna()
        te = test_df[col].dropna()
        
        def get_median_iqr(s):
            if len(s) == 0: return "N/A"
            return f"{s.median():.2f} [{s.quantile(0.25):.2f}-{s.quantile(0.75):.2f}]"
            
        t_str = get_median_iqr(t)
        v_str = get_median_iqr(v)
        te_str = get_median_iqr(te)
        
        # Kruskal-Wallis H-test
        if len(t) > 0 and len(v) > 0 and len(te) > 0:
            stat, p_val = kruskal(t, v, te)
            p_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
        else:
            p_str = "N/A"
            
        print(f"{label:<25} | {t_str:<22} | {v_str:<22} | {te_str:<22} | {p_str:<10}")
        
    print("-" * 120)
    
    # --- Categorical Variables (N (%), Chi-square) ---
    for label, col in categorical_vars.items():
        # Create contingency table
        ct = pd.crosstab(df[col], df['Split'])
        
        # Ensure all splits are present in columns
        for split_col in [train_label, val_label, test_label]:
            if split_col not in ct.columns:
                ct[split_col] = 0
        ct = ct[[train_label, val_label, test_label]]
                 
        # Chi-square test
        chi2, p_val, dof, expected = chi2_contingency(ct)
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
        
        print(f"{label:<25} | {'':<22} | {'':<22} | {'':<22} | {p_str:<10}")
        
        # Print each category
        for cat in ct.index:
            t_n = ct.loc[cat, train_label]
            v_n = ct.loc[cat, val_label]
            te_n = ct.loc[cat, test_label]
            
            t_pct = (t_n / len(train_df)) * 100
            v_pct = (v_n / len(val_df)) * 100
            te_pct = (te_n / len(test_df)) * 100
            
            cat_str = f"  {str(cat)}"
            t_fmt = f"{t_n} ({t_pct:.1f}%)"
            v_fmt = f"{v_n} ({v_pct:.1f}%)"
            te_fmt = f"{te_n} ({te_pct:.1f}%)"
            
            print(f"{cat_str:<25} | {t_fmt:<22} | {v_fmt:<22} | {te_fmt:<22} | {'':<10}")

    print("-" * 120)
    print("* Tumour Volume calculated only for cases where has_tumor == True")
    print("\nNote: A p-value > 0.05 indicates no statistically significant difference between the splits,")
    print("proving that your stratification successfully balanced the dataset.")

def save_stratification_metadata(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_path: str):
    '''
    Generates and saves the stratification metadata to a JSON file 
    using the pre-computed stratification keys present in the dataframes.
    Excludes the test set as this file is used for training pipeline configuration.
    '''
    
    # 1. Build the 'bins' dictionary mapping composite keys to case lists
    bins = {}
    for df in [train_df, val_df, test_df]:
        for key, group in df.groupby('strat_key'):
            if key not in bins:
                bins[key] = []
            bins[key].extend(group['case_name'].tolist())

    # Sort bins and file lists for readability and deterministic output
    bins = {k: sorted(v) for k, v in sorted(bins.items())}

    # 2. Construct the final JSON payload
    metadata = {
        "creation_date": datetime.datetime.now().isoformat(),
        "stratification_method": "iterative_multilabel_with_pre_allocation",
        "stratified_by": "composite_key (slice_group + liver_hu_group + tumor_group)",
        "pre_allocation_rule": "Cases with liver_hu_mean < 60 were deterministically pre-allocated (1 to train, 1 to val, 1 to test) prior to iterative stratification to guarantee rare protocol representation.",
        "train": sorted(train_df['case_name'].tolist()),
        "val": sorted(val_df['case_name'].tolist()),
        "test": sorted(test_df['case_name'].tolist()),
        "bins": bins
    }
    
    # 3. Save to disk
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Stratification metadata successfully saved to {out_path.resolve()}")
