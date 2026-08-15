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
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
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
    Computes both raw and post-processed metrics in a single pass to minimise disk I/O.
    """
    def __init__(self):
        self.config = config.get()

        # Metrics expect decollated lists of tensors
        self.dice_metric = DiceMetric(include_background=False, reduction="none")
        self.iou_metric = MeanIoU(include_background=False, reduction="none")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, reduction="none", percentile=95.0, distance_metric="euclidean"
        )

        logger.info("MetricsEvaluator initialised. Will compute both raw and post-processed metrics.")

    def evaluate(self, test_files: List[Dict[str, str]], pred_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load predictions and labels from disk, apply optional post-processing,
        and compute metrics in the original scanner space.
        Returns a dictionary of DataFrames with per-case metrics for each mode.
        """
        results_raw = []
        # Only track post-processed results if we have the 3-class layout 
        # required for anatomical post-processing (liver masking tumour).
        results_pp = [] if self.config.NUM_CLASSES == 3 else None

        logger.info("Starting evaluation on %d test volumes...", len(test_files))
        start_time = time.time()

        for case_dict in test_files:
            img_path = Path(case_dict["image"])
            lbl_path = Path(case_dict["label"])

            logger.info("Processing case: %s", img_path.name)
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

            # --- RAW METRICS ---
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

            # Intersection Over Union (IoU) is also dimensionless; spacing is not required.
            self.iou_metric(y_pred=pred_meta, y=label_meta)

            # HD95 measures physical surface distance; explicit spacing is mandatory 
            # to ensure true millimetre calculations on anisotropic grids.
            self.hd95_metric(y_pred=pred_meta, y=label_meta, spacing=spacing)

            case_dice = self.dice_metric.aggregate().cpu().numpy().flatten()
            case_iou = self.iou_metric.aggregate().cpu().numpy().flatten()
            case_hd95 = self.hd95_metric.aggregate().cpu().numpy().flatten()

            row_raw = {"case_name": case_name}
            if self.config.NUM_CLASSES == 3:
                row_raw["dice_liver"] = float(case_dice[0]) if not np.isnan(case_dice[0]) else None
                row_raw["dice_tumour"] = float(case_dice[1]) if not np.isnan(case_dice[1]) else None
                row_raw["iou_liver"] = float(case_iou[0]) if not np.isnan(case_iou[0]) else None
                row_raw["iou_tumour"] = float(case_iou[1]) if not np.isnan(case_iou[1]) else None
                row_raw["hd95_liver_mm"] = float(case_hd95[0]) if not np.isnan(case_hd95[0]) else None
                row_raw["hd95_tumour_mm"] = float(case_hd95[1]) if not np.isnan(case_hd95[1]) else None
            else:  # binary mode
                row_raw["dice_tumour"] = float(case_dice[0]) if not np.isnan(case_dice[0]) else None
                row_raw["iou_tumour"] = float(case_iou[0]) if not np.isnan(case_iou[0]) else None
                row_raw["hd95_tumour_mm"] = float(case_hd95[0]) if not np.isnan(case_hd95[0]) else None

            results_raw.append(row_raw)

            self.dice_metric.reset()
            self.iou_metric.reset()
            self.hd95_metric.reset()

            # --- POST-PROCESSED METRICS ---
            if self.config.NUM_CLASSES == 3:
                logger.debug("Applying largest-connected-component post-processing to %s", case_name)
                pred_np_pp = _post_process_class_map(pred_np)

                pred_tensor_pp = torch.nn.functional.one_hot(
                    torch.from_numpy(pred_np_pp).long(), num_classes=self.config.NUM_CLASSES
                ).permute(3, 0, 1, 2).float().unsqueeze(0)

                pred_meta_pp = MetaTensor(pred_tensor_pp, affine=affine)

                self.dice_metric(y_pred=pred_meta_pp, y=label_meta)
                self.iou_metric(y_pred=pred_meta_pp, y=label_meta)
                self.hd95_metric(y_pred=pred_meta_pp, y=label_meta, spacing=spacing)

                case_dice_pp = self.dice_metric.aggregate().cpu().numpy().flatten()
                case_iou_pp = self.iou_metric.aggregate().cpu().numpy().flatten()
                case_hd95_pp = self.hd95_metric.aggregate().cpu().numpy().flatten()

                row_pp = {"case_name": case_name}
                row_pp["dice_liver"] = float(case_dice_pp[0]) if not np.isnan(case_dice_pp[0]) else None
                row_pp["dice_tumour"] = float(case_dice_pp[1]) if not np.isnan(case_dice_pp[1]) else None
                row_pp["iou_liver"] = float(case_iou_pp[0]) if not np.isnan(case_iou_pp[0]) else None
                row_pp["iou_tumour"] = float(case_iou_pp[1]) if not np.isnan(case_iou_pp[1]) else None
                row_pp["hd95_liver_mm"] = float(case_hd95_pp[0]) if not np.isnan(case_hd95_pp[0]) else None
                row_pp["hd95_tumour_mm"] = float(case_hd95_pp[1]) if not np.isnan(case_hd95_pp[1]) else None

                results_pp.append(row_pp)

                self.dice_metric.reset()
                self.iou_metric.reset()
                self.hd95_metric.reset()

            else:
                logger.debug("Skipping post-processing metrics for binary mode (NUM_CLASSES=2).")

        elapsed = time.time() - start_time
        logger.info("Evaluation completed in %.1f s (%.2f s/volume)", elapsed, elapsed / len(test_files))

        out_dict = {"raw": pd.DataFrame(results_raw)}
        if results_pp is not None:
            out_dict["pp"] = pd.DataFrame(results_pp)
        return out_dict

    def generate_report(self, results_dict: Dict[str, pd.DataFrame], output_dir: Optional[Path] = None) -> str:
        """Aggregate metrics, print thesis-ready tables, and export CSVs for all computed modes."""
        if not results_dict:
            raise ValueError("No evaluation results to report. Dictionary is empty.")
        out_path = Path(output_dir) if output_dir else self.config.RUN_DIR / "reports"
        out_path.mkdir(parents=True, exist_ok=True)

        for suffix, df in results_dict.items():
            if df.empty:
                logger.warning("No evaluation results for '%s'. Skipping report generation for this mode.", suffix)
                continue

            df_sorted = df.sort_values("case_name").reset_index(drop=True)

            # Aggregate statistics (mean ± std)
            agg_metrics = []
            class_names = ["liver", "tumour"] if self.config.NUM_CLASSES == 3 else ["tumour"]
            for name in class_names:
                d_dice = df_sorted[f"dice_{name}"].dropna()
                d_iou = df_sorted[f"iou_{name}"].dropna()
                d_hd = df_sorted[f"hd95_{name}_mm"].replace([np.inf, -np.inf], np.nan).dropna()
                agg_metrics.append({
                    "structure": name.capitalize(),
                    "dice_mean": d_dice.mean(),
                    "dice_std": d_dice.std(),
                    "iou_mean": d_iou.mean(),
                    "iou_std": d_iou.std(),
                    "hd95_mean_mm": d_hd.mean(),
                    "hd95_std_mm": d_hd.std(),
                    "volumes_evaluated": len(d_dice)
                })
            agg_df = pd.DataFrame(agg_metrics)

            # Export CSV
            csv_path = out_path / f"test_evaluation_results_{suffix}.csv"
            df_sorted.to_csv(csv_path, index=False)
            agg_csv_path = out_path / f"test_aggregated_metrics_{suffix}.csv"
            agg_df.to_csv(agg_csv_path, index=False)
            logger.info("Per-case results exported to: %s", csv_path)
            logger.info("Aggregated metrics exported to: %s", agg_csv_path)

            # Print thesis-ready table
            self._print_thesis_table(agg_df, suffix)
            
        return str(out_path)

    def _print_thesis_table(self, agg_df: pd.DataFrame, suffix: str):
        """Prints a formatted table suitable for direct inclusion in thesis chapters."""
        logger.info("\n" + "="*60)
        logger.info("TEST DATASET EVALUATION SUMMARY (%s)", 'POST-PROCESSED' if suffix == 'pp' else 'RAW')
        logger.info("="*60)
        logger.info(f"{'Structure':<12} | {'Dice (mean±std)':<18} | {'HD95 (mm) (mean±std)':<22} | {'N':<5}")
        logger.info("-"*60)
        for _, row in agg_df.iterrows():
            dice_str = f"{row['dice_mean']:.3f} ± {row['dice_std']:.3f}"
            iou_str = f"{row['iou_mean']:.3f} ± {row['iou_std']:.3f}" if not pd.isna(row['iou_mean']) else "N/A"
            hd_str = f"{row['hd95_mean_mm']:.2f} ± {row['hd95_std_mm']:.2f}" if not pd.isna(row['hd95_mean_mm']) else "N/A"
            logger.info(f"{row['structure']:<12} | {dice_str:<18} | {hd_str:<22} | {row['volumes_evaluated']:<5}")
        logger.info("="*60 + "\n")
