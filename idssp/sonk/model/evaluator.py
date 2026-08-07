"""
Statistics and metrics evaluation module for automated tumour segmentation.
Loads previously generated inference images and computes metrics (raw and post-processed).
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from scipy import ndimage

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)

def _post_process_class_map(pred_np: np.ndarray) -> np.ndarray:
    """
    Keep largest connected component for liver (class 1),
    remove tumour (class 2) outside retained liver.

    Note: Uses 6-connectivity (default in scipy.ndimage.label) for 3D volumes,
    which is standard for medical segmentation tasks.

    Args:
        pred_np: 3D numpy array of class indices (0=bg, 1=liver, 2=tumour)
    Returns:
        Post-processed class map (same shape)
    """
    result = pred_np.copy()

    # 1. Identify the largest connected component of the liver (class 1)
    liver_mask = (result == 1).astype(np.uint8)
    labelled_liver, num_liver = ndimage.label(liver_mask)

    if num_liver > 0:
        sizes = ndimage.sum(liver_mask, labelled_liver, range(1, num_liver + 1))
        largest_liver_label = np.argmax(sizes) + 1
        liver_lcc = (labelled_liver == largest_liver_label)
    else:
        liver_lcc = np.zeros_like(liver_mask, dtype=bool)

    # 2. Remove stray liver voxels
    result[(result == 1) & ~liver_lcc] = 0

    # 3. Retain only tumours connected to the main liver component
    # We create a combined mask and find which connected component contains the main liver.
    anatomical_mask = ((result == 1) | (result == 2)).astype(np.uint8)
    labelled_anat, num_anat = ndimage.label(anatomical_mask)

    if num_anat > 0 and liver_lcc.any():
        # Find the anatomical label that overlaps most with the retained liver
        anat_labels_in_liver = labelled_anat[liver_lcc]
        valid_labels = anat_labels_in_liver[anat_labels_in_liver > 0]
        if len(valid_labels) > 0:
            main_anat_label = np.bincount(valid_labels).argmax()
            main_anat_mask = (labelled_anat == main_anat_label)
        else:
            main_anat_mask = liver_lcc
    else:
        main_anat_mask = liver_lcc

    # Remove tumours outside this main anatomical component
    result[(result == 2) & ~main_anat_mask] = 0

    # 4. Warning logic for fragmented predictions
    liver_voxels_before = (pred_np == 1).sum()
    liver_voxels_after = (result == 1).sum()
    if liver_voxels_before > 0 and liver_voxels_after / liver_voxels_before < 0.5:
        logger.warning(
            "LCC post-processing discarded %.1f%% of predicted liver voxels. "
            "This may indicate fragmented predictions or atypical anatomy.",
            100 * (1 - liver_voxels_after / liver_voxels_before)
        )

    return result

class MetricsEvaluator:
    """
    Handles metric computation and result export for test datasets.
    Can evaluate both raw and post-processed predictions saved on disk.
    """
    def __init__(self, post_process: bool = False):
        self.config = config.get()
        self.post_process = post_process

        # Metrics expect decollated lists of tensors
        self.dice_metric = DiceMetric(include_background=False, reduction="none")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, reduction="none", percentile=95.0, distance_metric="euclidean"
        )

        logger.info("MetricsEvaluator initialised. Post-processing: %s", self.post_process)

    def evaluate(self, test_files: List[Dict[str, str]], pred_dir: Path) -> pd.DataFrame:
        """
        Load predictions and labels from disk, apply optional post-processing,
        and compute metrics in the original scanner space.
        Returns a DataFrame with per-case metrics.
        """
        results = []
        
        logger.info("Starting evaluation on %d test volumes...", len(test_files))
        start_time = time.time()

        for case_dict in test_files:
            img_path = Path(case_dict["image"])
            lbl_path = Path(case_dict["label"])
            
            case_name = img_path.stem
            if case_name.endswith(".nii"):
                case_name = case_name[:-4]
                
            pred_path = pred_dir / f"{case_name}_pred.nii.gz"
            
            if not pred_path.exists():
                logger.warning("Prediction not found for %s. Skipping.", case_name)
                continue
                
            pred_nib = nib.load(pred_path)
            label_nib = nib.load(lbl_path)
            
            pred_np = np.asarray(pred_nib.dataobj).astype(np.int32)
            label_np = np.asarray(label_nib.dataobj).astype(np.int32)

            # === SHAPE VALIDATION & ALIGNMENT ===
            if pred_np.shape != label_np.shape:
                raise ValueError(
                    f"Prediction and label spatial shapes differ for case {case_name}: "
                    f"pred={pred_np.shape}, label={label_np.shape}"
                )
            # ======

            if self.post_process and self.config.NUM_CLASSES == 3:
                logger.debug("Applying largest-connected-component post-processing to %s", case_name)
                pred_np = _post_process_class_map(pred_np)


            # Convert to one-hot (C, D, H, W) for MONAI metrics,
            # then add a batch dimension → (1, C, D, H, W).
            # MONAI metrics expect batch-first tensors; without this, the
            # channel axis is misread as the batch axis (img_dim becomes 2).
            pred_tensor = torch.nn.functional.one_hot(
                torch.from_numpy(pred_np).long(), num_classes=self.config.NUM_CLASSES
            ).permute(3, 0, 1, 2).float().unsqueeze(0)

            label_tensor = torch.nn.functional.one_hot(
                torch.from_numpy(label_np).long(), num_classes=self.config.NUM_CLASSES
            ).permute(3, 0, 1, 2).float().unsqueeze(0)

            # Wrap in MetaTensor with original affine for correct spatial representation
            affine = pred_nib.affine
            pred_meta = MetaTensor(pred_tensor, affine=affine)
            label_meta = MetaTensor(label_tensor, affine=affine)

            # Extract physical voxel spacing (mm) from the NIfTI affine matrix.
            # MONAI's HausdorffDistanceMetric does NOT automatically derive spacing 
            # from MetaTensor.affine; it defaults to unit spacing (1.0 mm) if not 
            # explicitly passed via kwargs.
            spacing = nib.affines.voxel_sizes(affine)

            # Dice is dimensionless (voxel overlap ratio), so spacing is not required.
            self.dice_metric(y_pred=pred_meta, y=label_meta)

            # HD95 measures physical surface distance; explicit spacing is mandatory 
            # to ensure true millimetre calculations on anisotropic grids.
            self.hd95_metric(y_pred=pred_meta, y=label_meta, spacing=spacing)

            case_dice = self.dice_metric.aggregate().cpu().numpy().flatten()
            case_hd95 = self.hd95_metric.aggregate().cpu().numpy().flatten()

            row = {"case_name": case_name}
            if self.config.NUM_CLASSES == 3:
                row["dice_liver"] = float(case_dice[0]) if not np.isnan(case_dice[0]) else None
                row["dice_tumour"] = float(case_dice[1]) if not np.isnan(case_dice[1]) else None
                row["hd95_liver_mm"] = float(case_hd95[0]) if not np.isnan(case_hd95[0]) else None
                row["hd95_tumour_mm"] = float(case_hd95[1]) if not np.isnan(case_hd95[1]) else None
            else:  # binary mode
                row["dice_tumour"] = float(case_dice[0]) if not np.isnan(case_dice[0]) else None
                row["hd95_tumour_mm"] = float(case_hd95[0]) if not np.isnan(case_hd95[0]) else None
                
            results.append(row)
            self.dice_metric.reset()
            self.hd95_metric.reset()
            
        elapsed = time.time() - start_time
        logger.info("Evaluation completed in %.1f s (%.2f s/volume)", elapsed, elapsed / len(test_files))
        return pd.DataFrame(results)

    def generate_report(self, df: pd.DataFrame, output_dir: Optional[Path] = None) -> str:
        """Aggregate metrics, print thesis-ready table, and export CSV."""
        if df.empty:
            raise ValueError("No evaluation results to report. DataFrame is empty.")
        out_path = Path(output_dir) if output_dir else self.config.RUN_DIR / "reports"
        out_path.mkdir(parents=True, exist_ok=True)

        # Aggregate statistics (mean ± std)
        agg_metrics = []
        class_names = ["liver", "tumour"] if self.config.NUM_CLASSES == 3 else ["tumour"]
        for name in class_names:
            d_dice = df[f"dice_{name}"].dropna()
            d_hd = df[f"hd95_{name}_mm"].replace([np.inf, -np.inf], np.nan).dropna()
            agg_metrics.append({
                "structure": name.capitalize(),
                "dice_mean": d_dice.mean(),
                "dice_std": d_dice.std(),
                "hd95_mean_mm": d_hd.mean(),
                "hd95_std_mm": d_hd.std(),
                "volumes_evaluated": len(d_dice)
            })
        agg_df = pd.DataFrame(agg_metrics)

        suffix = "pp" if self.post_process else "raw"
        
        # Export CSV
        csv_path = out_path / f"test_evaluation_results_{suffix}.csv"
        df.to_csv(csv_path, index=False)
        agg_csv_path = out_path / f"test_aggregated_metrics_{suffix}.csv"
        agg_df.to_csv(agg_csv_path, index=False)
        logger.info("Per-case results exported to: %s", csv_path)
        logger.info("Aggregated metrics exported to: %s", agg_csv_path)

        # Print thesis-ready table
        self._print_thesis_table(agg_df, suffix)
        return str(agg_csv_path)

    def _print_thesis_table(self, agg_df: pd.DataFrame, suffix: str):
        """Prints a formatted table suitable for direct inclusion in thesis chapters."""
        print("\n" + "="*60)
        print(f"TEST DATASET EVALUATION SUMMARY ({'POST-PROCESSED' if suffix == 'pp' else 'RAW'})")
        print("="*60)
        print(f"{'Structure':<12} | {'Dice (mean±std)':<18} | {'HD95 (mm) (mean±std)':<22} | {'N':<5}")
        print("-"*60)
        for _, row in agg_df.iterrows():
            dice_str = f"{row['dice_mean']:.3f} ± {row['dice_std']:.3f}"
            hd_str = f"{row['hd95_mean_mm']:.2f} ± {row['hd95_std_mm']:.2f}" if not pd.isna(row['hd95_mean_mm']) else "N/A"
            print(f"{row['structure']:<12} | {dice_str:<18} | {hd_str:<22} | {row['volumes_evaluated']:<5}")
        print("="*60 + "\n")
