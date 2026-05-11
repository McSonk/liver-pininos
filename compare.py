#!/usr/bin/env python3
"""
3D-IRCADb to LiTS CT Identity Verification Script
==================================================
Verifies that CT volumes from 3D-IRCADb and LiTS datasets represent the same patient
using invariant anatomical fingerprints and intensity statistics.

A paper originally claimed that 3D-IRCADb and LiTS share same patients from 27 to 48,
however this is wrong as demonstrated in this script.

LiTS files 27 and 46 doesn't have a match in 3D-IRCADb-01

https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2021.697178/full

Usage:
    # In Jupyter:
    from compare import verify_case_pair
    verify_case_pair(ircadb_case_num=1, lits_case_num=27)
    
    # From terminal:
    python compare.py --ircadb 1 --lits 27
"""

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
import argparse
from typing import Dict, Tuple


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_ircadb_ct(case_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IRCADB CT volume from DICOM series.
    
    Args:
        case_num: IRCADB case number (1-20)
        
    Returns:
        Tuple of (ct_array in Z,Y,X order, spacing in z,y,x order in mm)
    """
    folder = f"/nvm-external/ct-scans/3Dircadb/3Dircadb2.{case_num}/PATIENT_DICOM"
    
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"IRCADB directory not found: {folder}")
    
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(names)
    img_sitk = reader.Execute()
    
    ct_array = sitk.GetArrayFromImage(img_sitk)  # (Z, Y, X)
    spacing = np.array(img_sitk.GetSpacing())[::-1]  # (z, y, x) in mm
    
    return ct_array, spacing


def load_lits_ct(case_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load LiTS CT volume from NIfTI file.
    
    Args:
        case_num: LiTS case number (27-48 for test set)
        
    Returns:
        Tuple of (ct_array in Z,Y,X order, spacing in z,y,x order in mm)
    """
    path = f"/nvm-external/ct-scans/lits/test/volume-{case_num}.nii.gz"
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LiTS file not found: {path}")
    
    img_nib = nib.load(path)
    ct_array = img_nib.get_fdata(dtype=np.float32)  # (X, Y, Z)
    spacing = np.abs(np.diag(img_nib.affine)[:3])  # (x, y, z) in mm
    
    # Convert to (Z, Y, X) order to match IRCADB
    ct_array = np.transpose(ct_array, (2, 1, 0))
    spacing = spacing[::-1]  # (z, y, x)
    
    return ct_array, spacing


# ============================================================================
# FINGERPRINT COMPUTATION
# ============================================================================

def compute_ct_fingerprint(ct_array: np.ndarray, spacing_mm: np.ndarray, 
                           target_shape: Tuple[int, int, int] = (64, 64, 64)) -> Dict:
    """
    Compute invariant features of a CT volume for identity verification.
    
    Args:
        ct_array: 3D CT volume in (Z, Y, X) order
        spacing_mm: Voxel spacing in (z, y, x) order in mm
        target_shape: Target shape for coarse spatial hash
        
    Returns:
        Dictionary containing coverage, histogram, spatial hash, and intensity stats
    """
    # 1. Physical coverage
    coverage_mm = np.array(ct_array.shape) * spacing_mm
    
    # 2. Intensity histogram (clip to valid HU range)
    ct_flat = ct_array.flatten()
    ct_flat = ct_flat[(ct_flat >= -1000) & (ct_flat <= 3000)]
    hist, _ = np.histogram(ct_flat, bins=256, range=(-1000, 3000))
    hist_norm = hist / hist.sum()
    
    # 3. Downsampled spatial hash (robust to moderate resampling)
    zoom_factors = np.array(target_shape) / np.array(ct_array.shape)
    ct_coarse = zoom(ct_array, zoom_factors, order=0)
    ct_coarse = np.ascontiguousarray(ct_coarse, dtype=np.int16)
    spatial_hash = hashlib.sha256(ct_coarse.tobytes()).hexdigest()
    
    # 4. Global intensity stats
    mean_hu = np.mean(ct_flat)
    std_hu = np.std(ct_flat)
    
    return {
        "coverage_mm": np.round(coverage_mm, 1).tolist(),
        "histogram": hist_norm,
        "spatial_hash": spatial_hash,
        "mean_hu": round(mean_hu, 1),
        "std_hu": round(std_hu, 1),
        "original_shape": ct_array.shape,
        "original_spacing": np.round(spacing_mm, 3).tolist(),
    }


# ============================================================================
# COMPARISON LOGIC
# ============================================================================

def compare_fingerprints(fp_ircadb: Dict, fp_lits: Dict) -> Tuple[Dict, bool]:
    """
    Compare fingerprints from IRCADB and LiTS volumes.
    
    Prioritizes intensity statistics over metadata-derived coverage,
    as LiTS NIfTI files may have simplified affine matrices.
    
    Args:
        fp_ircadb: Fingerprint from IRCADB volume
        fp_lits: Fingerprint from LiTS volume
        
    Returns:
        Tuple of (results_dict, is_same_patient_bool)
    """
    results = {}
    
    # 1. Coverage (mm) - allow ±10 mm per axis
    # Note: This may fail due to LiTS identity affine, so it's informational only
    cov_match = all(abs(a - b) <= 10.0 for a, b in 
                    zip(fp_ircadb["coverage_mm"], fp_lits["coverage_mm"]))
    results["coverage_match"] = cov_match
    
    # 2. Histogram similarity (Pearson correlation) - PRIMARY METRIC
    hist_corr = np.corrcoef(fp_ircadb["histogram"], fp_lits["histogram"])[0, 1]
    hist_match = hist_corr > 0.99  # Stricter threshold
    results["histogram_corr"] = hist_corr
    results["histogram_match"] = hist_match
    
    # 3. Spatial hash - informational (may differ due to floating-point drift)
    hash_match = fp_ircadb["spatial_hash"] == fp_lits["spatial_hash"]
    results["hash_match"] = hash_match
    
    # 4. Intensity stats - PRIMARY METRIC
    mean_diff = abs(fp_ircadb["mean_hu"] - fp_lits["mean_hu"])
    std_diff = abs(fp_ircadb["std_hu"] - fp_lits["std_hu"])
    mean_match = mean_diff < 1.0  # Very strict: < 1 HU difference
    std_match = std_diff < 5.0    # Strict: < 5 HU difference
    results["mean_diff"] = mean_diff
    results["std_diff"] = std_diff
    results["mean_match"] = mean_match
    results["std_match"] = std_match
    
    # 5. Shape match (after orientation correction)
    shape_match = fp_ircadb["original_shape"] == fp_lits["original_shape"]
    results["shape_match"] = shape_match
    
    # FINAL VERDICT: Prioritize intensity statistics
    # If histogram correlation > 0.99 AND mean difference < 1 HU, same patient
    is_same_patient = (hist_match and mean_match)
    
    return results, is_same_patient


# ============================================================================
# VISUALIZATION
# ============================================================================

def window_liver(ct_slice: np.ndarray) -> np.ndarray:
    """
    Apply liver windowing [-175, 250] HU and normalize to [0, 1].
    
    Args:
        ct_slice: 2D CT slice
        
    Returns:
        Windowed and normalized slice
    """
    windowed = np.clip(ct_slice, -175, 250)
    return (windowed + 175) / 425


def visualize_comparison(ircadb_array: np.ndarray, lits_array: np.ndarray,
                        ircadb_case: int, lits_case: int) -> None:
    """
    Display side-by-side comparison of middle axial slices.
    
    Args:
        ircadb_array: IRCADB CT volume (Z, Y, X) - already flipped
        lits_array: LiTS CT volume (Z, Y, X)
        ircadb_case: IRCADB case number
        lits_case: LiTS case number
    """
    mid_z_ircadb = ircadb_array.shape[0] // 2
    mid_z_lits = lits_array.shape[0] // 2
    
    slice_ircadb = window_liver(ircadb_array[mid_z_ircadb])
    slice_lits = window_liver(lits_array[mid_z_lits])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im0 = axes[0].imshow(slice_ircadb, cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[0].set_title(f"IRCADB Case {ircadb_case} (Y-Flipped)\n"
                     f"Slice {mid_z_ircadb}/{ircadb_array.shape[0]-1}\n"
                     f"Shape: {ircadb_array.shape}")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(slice_lits, cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1].set_title(f"LiTS Case {lits_case}\n"
                     f"Slice {mid_z_lits}/{lits_array.shape[0]-1}\n"
                     f"Shape: {lits_array.shape}")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"CT Identity Verification: IRCADB-{ircadb_case} ↔ LiTS-{lits_case}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Anatomical sanity check
    print("\n=== Anatomical Sanity Check ===")
    print("In axial CT: image LEFT = patient RIGHT. Liver should appear on image LEFT side.")
    print(f"IRCADB slice mean intensity (left half):  {slice_ircadb[:, :slice_ircadb.shape[1]//2].mean():.3f}")
    print(f"IRCADB slice mean intensity (right half): {slice_ircadb[:, slice_ircadb.shape[1]//2:].mean():.3f}")
    print(f"LiTS slice mean intensity (left half):    {slice_lits[:, :slice_lits.shape[1]//2].mean():.3f}")
    print(f"LiTS slice mean intensity (right half):   {slice_lits[:, slice_lits.shape[1]//2:].mean():.3f}")


# ============================================================================
# MAIN VERIFICATION FUNCTION
# ============================================================================

def verify_case_pair(ircadb_case_num: int, lits_case_num: int, 
                    verbose: bool = True) -> Tuple[Dict, bool]:
    """
    Verify that IRCADB and LiTS cases represent the same patient.
    
    This is the main entry point for verification. It loads both volumes,
    computes fingerprints, compares them, and displays visual confirmation.
    
    Args:
        ircadb_case_num: IRCADB case number (1-20)
        lits_case_num: LiTS case number (27-48 for test set)
        verbose: Whether to print detailed comparison results
        
    Returns:
        Tuple of (results_dict, is_same_patient_bool)
    """
    print(f"\n{'='*80}")
    print(f"Verifying: IRCADB Case {ircadb_case_num} ↔ LiTS Case {lits_case_num}")
    print(f"{'='*80}")
    
    # Load data
    print("\n[1/4] Loading data...")
    ircadb_ct, ircadb_spacing = load_ircadb_ct(ircadb_case_num)
    lits_ct, lits_spacing = load_lits_ct(lits_case_num)
    
    # Fix IRCADB orientation (flip Y-axis to match RAS+ convention)
    print("[2/4] Correcting orientation (flipping IRCADB Y-axis)...")
    ircadb_ct = np.flip(ircadb_ct, axis=1)
    
    # Compute fingerprints
    print("[3/4] Computing fingerprints...")
    fp_ircadb = compute_ct_fingerprint(ircadb_ct, ircadb_spacing)
    fp_lits = compute_ct_fingerprint(lits_ct, lits_spacing)
    
    # Compare
    print("[4/4] Comparing fingerprints...")
    results, is_same_patient = compare_fingerprints(fp_ircadb, fp_lits)
    
    # Display results
    if verbose:
        print(f"\n{'Feature':<25} | {'IRCADB':>20} | {'LiTS':>20} | {'Status'}")
        print("-" * 80)
        
        print(f"Shape                    | {str(fp_ircadb['original_shape']):>20} | {str(fp_lits['original_shape']):>20} | "
              f"{'✓' if results['shape_match'] else '✗'}")
        
        print(f"Coverage (mm)            | {str(fp_ircadb['coverage_mm']):>20} | {str(fp_lits['coverage_mm']):>20} | "
              f"{'✓' if results['coverage_match'] else '✗'} (informational)")
        
        print(f"Histogram correlation    | {results['histogram_corr']:>20.4f} | {results['histogram_corr']:>20.4f} | "
              f"{'✓ PRIMARY' if results['histogram_match'] else '✗'}")
        
        print(f"Mean HU                  | {fp_ircadb['mean_hu']:>20.1f} | {fp_lits['mean_hu']:>20.1f} | "
              f"{'✓ PRIMARY' if results['mean_match'] else '✗'}")
        
        print(f"Std HU                   | {fp_ircadb['std_hu']:>20.1f} | {fp_lits['std_hu']:>20.1f} | "
              f"{'✓' if results['std_match'] else '✗'}")
        
        print(f"Mean diff (HU)           | {results['mean_diff']:>20.4f} | {'< 1.0 required':>20} | "
              f"{'✓' if results['mean_match'] else '✗'}")
        
        print(f"Spatial hash match       | {fp_ircadb['spatial_hash'][:12]}... | {fp_lits['spatial_hash'][:12]}... | "
              f"{'✓' if results['hash_match'] else '✗'} (informational)")
        
        print(f"\nMetadata:")
        print(f"  IRCADB: shape={fp_ircadb['original_shape']}, spacing={fp_ircadb['original_spacing']} mm")
        print(f"  LiTS:   shape={fp_lits['original_shape']}, spacing={fp_lits['original_spacing']} mm")
        
        print("\n" + "-" * 80)
        if is_same_patient:
            print("✅ RESULT: SAME PATIENT")
            print("   Intensity statistics match within strict tolerances.")
            print("   Histogram correlation > 0.99 and mean difference < 1 HU.")
        else:
            print("❌ RESULT: DIFFERENT PATIENT or PREPROCESSING MISMATCH")
            print("   Primary metrics (histogram correlation, mean HU) do not match.")
        
        print("=" * 80)
    
    # Visual comparison
    visualize_comparison(ircadb_ct, lits_ct, ircadb_case_num, lits_case_num)
    
    return results, is_same_patient


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify identity between 3D-IRCADb and LiTS CT volumes"
    )
    parser.add_argument("--ircadb", type=int, required=True, 
                       help="IRCADB case number (1-20)")
    parser.add_argument("--lits", type=int, required=True,
                       help="LiTS case number (27-48 for test set)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    
    args = parser.parse_args()
    
    results, is_same = verify_case_pair(args.ircadb, args.lits, verbose=not args.quiet)
    
    # Exit code for scripting
    exit(0 if is_same else 1)