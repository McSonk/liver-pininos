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

def _sanitise_hd95(hd95_val, dice_val, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    """
    Enforce the HD95 omission rule (AGENTS.md, Data Handling Rules).

    MONAI 1.5.2 returns inf when exactly one mask is empty and a finite
    (large) value for disjoint non-empty masks, so nan checks alone do not
    enforce the rule.
    """
    if pred_mask.sum() == 0 or gt_mask.sum() == 0 or dice_val is None or not dice_val > 0.0:
        return None
    hd95_val = float(hd95_val)
    return hd95_val if np.isfinite(hd95_val) else None

class MetricsEvaluator:
    """
    Handles metric computation and result export for test datasets.
    Computes raw metrics and, for 3-class configurations, post-processed metrics.
    """
    def __init__(self):
        self.config = config.get()

        # Metrics expect decollated lists of tensors
        self.dice_metric = DiceMetric(include_background=False, reduction="none")
        self.iou_metric = MeanIoU(include_background=False, reduction="none")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, reduction="none", percentile=95.0, distance_metric="euclidean"
        )

        modes = "raw and post-processed" if self.config.NUM_CLASSES == 3 else "raw"

        logger.info("MetricsEvaluator initialised. Will compute %s metrics.", modes)

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

            # Pre-compute masks for HD95 sanitisation
            pred_tumour_mask = pred_np == self.config.TUMOUR_CLASS_INDEX
            label_tumour_mask = label_np == self.config.TUMOUR_CLASS_INDEX

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
                pred_liver_mask = pred_np == 1
                label_liver_mask = label_np == 1

                d_liver = float(case_dice[0]) if not np.isnan(case_dice[0]) else None
                d_tumour = float(case_dice[1]) if not np.isnan(case_dice[1]) else None

                row_raw["dice_liver"] = d_liver
                row_raw["dice_tumour"] = d_tumour
                row_raw["iou_liver"] = float(case_iou[0]) if not np.isnan(case_iou[0]) else None
                row_raw["iou_tumour"] = float(case_iou[1]) if not np.isnan(case_iou[1]) else None

                row_raw["hd95_liver_mm"] = _sanitise_hd95(case_hd95[0], d_liver, pred_liver_mask, label_liver_mask)
                row_raw["hd95_tumour_mm"] = _sanitise_hd95(case_hd95[1], d_tumour, pred_tumour_mask, label_tumour_mask)
            else:  # binary mode
                d_tumour = float(case_dice[0]) if not np.isnan(case_dice[0]) else None

                row_raw["dice_tumour"] = d_tumour
                row_raw["iou_tumour"] = float(case_iou[0]) if not np.isnan(case_iou[0]) else None
                row_raw["hd95_tumour_mm"] = _sanitise_hd95(case_hd95[0], d_tumour, pred_tumour_mask, label_tumour_mask)

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

                pred_tumour_mask_pp = pred_np_pp == self.config.TUMOUR_CLASS_INDEX
                pred_liver_mask_pp = pred_np_pp == 1

                d_liver_pp = float(case_dice_pp[0]) if not np.isnan(case_dice_pp[0]) else None
                d_tumour_pp = float(case_dice_pp[1]) if not np.isnan(case_dice_pp[1]) else None

                row_pp["dice_liver"] = d_liver_pp
                row_pp["dice_tumour"] = d_tumour_pp
                row_pp["iou_liver"] = float(case_iou_pp[0]) if not np.isnan(case_iou_pp[0]) else None
                row_pp["iou_tumour"] = float(case_iou_pp[1]) if not np.isnan(case_iou_pp[1]) else None

                row_pp["hd95_liver_mm"] = _sanitise_hd95(
                    case_hd95_pp[0],
                    d_liver_pp,
                    pred_liver_mask_pp,
                    label_liver_mask,
                )
                row_pp["hd95_tumour_mm"] = _sanitise_hd95(
                    case_hd95_pp[1],
                    d_tumour_pp,
                    pred_tumour_mask_pp,
                    label_tumour_mask,
                )

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
        """Export per-case CSV results for all computed modes."""
        if not results_dict:
            raise ValueError("No evaluation results to report. Dictionary is empty.")
        out_path = Path(output_dir) if output_dir else self.config.RUN_DIR / "reports"
        out_path.mkdir(parents=True, exist_ok=True)

        for suffix, df in results_dict.items():
            if df.empty:
                logger.warning("No evaluation results for '%s'. Skipping report generation for this mode.", suffix)
                continue

            # Sort per-case results for deterministic CSV output.
            df_sorted = df.sort_values("case_name").reset_index(drop=True)

            # Only per-case results are exported; aggregation is left to manual analysis.
            csv_path = out_path / f"test_evaluation_results_{suffix}.csv"
            df_sorted.to_csv(csv_path, index=False)
            logger.info("Per-case results exported to: %s", csv_path)

        return str(out_path)
