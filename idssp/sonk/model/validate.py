"""
Test-time evaluation module for automated tumour segmentation.
Designed for full-volume inference, metric aggregation, and NIfTI export.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import KeepLargestConnectedComponent

from idssp.sonk import config
from idssp.sonk.model.models import get_model
from idssp.sonk.model.transforms import (get_activations_transforms,
                                         get_label_transform,
                                         get_validation_transforms)
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)


def _post_process_class_map(pred_np: np.ndarray) -> np.ndarray:
    """
    Keep largest connected component for liver (class 1),
    remove tumour (class 2) outside retained liver.
    
    Args:
        pred_np: 3D numpy array of class indices (0=bg, 1=liver, 2=tumour)
    Returns:
        Post-processed class map (same shape)
    """
    from scipy import ndimage
    
    result = pred_np.copy()
    
    # 1. Keep largest liver component
    liver_mask = (result == 1).astype(np.uint8)
    labelled, num = ndimage.label(liver_mask)
    
    if num > 0:
        sizes = ndimage.sum(liver_mask, labelled, range(1, num + 1))
        largest_label = np.argmax(sizes) + 1  # +1 because label 0 is background
        liver_lcc = (labelled == largest_label).astype(np.uint8)
    else:
        liver_lcc = np.zeros_like(liver_mask)
    
    # 2. Replace liver with LCC
    result[(result == 1) & (liver_lcc == 0)] = 0
    
    # 3. Remove tumour outside retained liver
    result[(result == 2) & (liver_lcc == 0)] = 0
    
    return result

class TestEvaluator:
    """
    Handles checkpoint loading, full-volume inference, metric computation,
    and result export for test datasets.
    """
    def __init__(self, checkpoint_path: str):
        self.config = config.get()
        self.device = torch.device(self.config.DEVICE)
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.inferer = None
        self.test_transforms = None
        self.pred_postprocess = None

         # EXACT post-processing used in training.py
        self.pred_transform = get_activations_transforms(self.config)
        self.label_transform = get_label_transform(self.config)

        # Metrics expect decollated lists of tensors
        self.dice_metric = DiceMetric(include_background=False, reduction="none")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, reduction="none", percentile=95.0, distance_metric="euclidean"
        )

        logger.info("TestEvaluator initialised. Device: %s", self.device)

    def load_checkpoint(self):
        """Restore model weights and verify config alignment."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        logger.info("Loading checkpoint: %s", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)

        if "config_snapshot" in checkpoint:
            ckpt_config = checkpoint["config_snapshot"]
            curr_config = self.config

            # Keys to verify for consistency between training and inference
            keys_to_check = ["NUM_CLASSES", "ISO_SPACING", "HU_WINDOW_MIN", "HU_WINDOW_MAX"]

            for key in keys_to_check:
                ckpt_val = ckpt_config.get(key)
                curr_val = getattr(curr_config, key, None)

                if ckpt_val is None or curr_val is None:
                    logger.warning(
                        "Config key '%s' missing in checkpoint or current config. Skipping check.",
                        key)
                    continue

                # Normalise values to lists for comparison to handle both scalars and tuples/lists
                # e.g. NUM_CLASSES (int) -> [3], ISO_SPACING (tuple) -> [1.0, 1.0, 1.0]
                def to_list(val):
                    if isinstance(val, (list, tuple)):
                        return list(val)
                    else:
                        return [val]

                if to_list(ckpt_val) != to_list(curr_val):
                    logger.warning(
                        "Config mismatch detected for '%s': Checkpoint=%s, Current=%s. "
                        "This may cause errors if architecture or preprocessing differs.",
                        key, ckpt_val, curr_val
                    )
                else:
                    logger.debug("Config match for '%s': %s", key, curr_val)

        # Initialise model architecture matching training
        # Ensure get_model() uses the current config to build the right architecture
        self.model = get_model().to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Model loaded successfully. Best Dice (train): %.4f",
                    checkpoint.get("best_dice", -1.0))

        # Sliding window inferer (must match training patch size)
        self.inferer = SlidingWindowInferer(
            roi_size=self.config.TRAIN_PATCH_SIZE,
            sw_batch_size=16,
            overlap=0.5,
            mode="gaussian",
            device=self.device,
            progress=False
        )

    def run_inference(self, test_files: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Run full-volume inference on test dataset.
        Returns a DataFrame with per-case metrics.
        """
        self.test_transforms = get_validation_transforms(self.config)
        test_ds = Dataset(data=test_files, transform=self.test_transforms)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        results = []
        save_path = self.config.OUTPUT_DIR / "test_predictions"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("Starting full-volume inference on %d test volumes...", len(test_files))
        start_time = time.time()

        with torch.inference_mode():
            for batch_idx, batch in enumerate(test_dl):
                case_name = Path(batch["image"].meta["filename_or_obj"][0]).stem
                logger.info("[%d/%d] Processing: %s", batch_idx + 1, len(test_dl), case_name)

                if batch_idx % 5 == 0:
                    logger.debug("Information of batch %d:", batch_idx)
                    logger.debug("Batch image shape: %s", batch["image"].shape)
                    logger.debug("MONAI meta affine shape:%s", batch["image"].meta["affine"].shape)
                    logger.debug("MONAI meta affine:\n%s", batch["image"].meta["affine"][0])

                images = batch["image"].to(self.device)

                # Full-volume sliding window inference
                with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                    preds = self.inferer(inputs=images, network=self.model)
                labels = batch["label"].to(self.device)

                # Decollate to per-volume tensors
                val_preds = decollate_batch(preds)
                val_labels = decollate_batch(labels)

                # Apply MONAI post-processing (matches training.py exactly)
                val_preds = [self.pred_transform(p) for p in val_preds]
                val_labels = [self.label_transform(l) for l in val_labels]

                # === SHAPE VALIDATION & ALIGNMENT ===
                for i, (pred, label) in enumerate(zip(val_preds, val_labels)):
                    pred_spatial = pred.shape[1:]  # Exclude channel dimension
                    label_spatial = label.shape[1:]

                    if pred_spatial != label_spatial:
                        logger.warning(
                            "Shape mismatch for sample %d: pred=%s, label=%s. "
                            "Resampling label to match prediction spatial dimensions.",
                            i, pred.shape, label.shape
                        )
                        # Use nearest-neighbour interpolation to preserve integer class labels
                        label_resampled = torch.nn.functional.interpolate(
                            label.unsqueeze(0).float(),  # Add batch dim for interpolate
                            size=pred_spatial,
                            mode='nearest'
                        ).squeeze(0).to(label.dtype)  # Remove batch dim, restore dtype
                        val_labels[i] = label_resampled
                # ======

                # === APPLY POST-PROCESSING BEFORE METRICS ===
                logger.debug("Applying largest-connected-component post-processing to predictions")
                processed_preds = []
                for pred in val_preds:
                    # pred: one-hot tensor (C, D, H, W)
                    pred_class = pred.argmax(dim=0)  # → (D, H, W) class indices
                    
                    # Apply post-processing on CPU (numpy)
                    pred_np = pred_class.cpu().numpy().astype(np.int32)
                    pred_post = _post_process_class_map(pred_np)
                    
                    # Convert back to one-hot for MONAI metrics
                    pred_onehot = torch.nn.functional.one_hot(
                        torch.from_numpy(pred_post).long(),
                        num_classes=self.config.NUM_CLASSES
                    ).permute(3, 0, 1, 2).float().to(pred.device)  # → (C, D, H, W)
                    
                    processed_preds.append(pred_onehot)

                # Compute metrics on post-processed predictions
                self.dice_metric(y_pred=processed_preds, y=val_labels)
                self.hd95_metric(y_pred=processed_preds, y=val_labels)

                # === END POST-PROCESSING ===

                # Save prediction (argmax back to class labels)
                pred_class_map = processed_preds[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
                pred_nib = nib.Nifti1Image(pred_class_map, affine=batch["image"].affine[0].numpy())
                nib.save(pred_nib, str(save_path / f"{Path(batch['image'].meta['filename_or_obj'][0]).stem}_pred.nii.gz"))
                
                # Aggregate & reset per-case
                case_dice = self.dice_metric.aggregate().cpu().numpy().flatten()
                case_hd95 = self.hd95_metric.aggregate().cpu().numpy().flatten()

                # --- FIX: Collect per-case results ---
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
                # --- END FIX ---

                self.dice_metric.reset()
                self.hd95_metric.reset()

        elapsed = time.time() - start_time
        logger.info("Inference completed in %.1f s (%.2f s/volume)", elapsed, elapsed / len(test_files))
        return pd.DataFrame(results)

    def generate_report(self, df: pd.DataFrame, output_dir: Optional[str] = None) -> str:
        """Aggregate metrics, print thesis-ready table, and export CSV."""
        out_path = Path(output_dir) if output_dir else self.config.OUTPUT_DIR / "reports"
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

        # Export CSV
        csv_path = out_path / "test_evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        agg_csv_path = out_path / "test_aggregated_metrics.csv"
        agg_df.to_csv(agg_csv_path, index=False)
        logger.info("Per-case results exported to: %s", csv_path)
        logger.info("Aggregated metrics exported to: %s", agg_csv_path)

        # Print thesis-ready table
        self._print_thesis_table(agg_df)
        return str(agg_csv_path)

    def _print_thesis_table(self, agg_df: pd.DataFrame):
        """Prints a formatted table suitable for direct inclusion in thesis chapters."""
        print("\n" + "="*60)
        print("TEST DATASET EVALUATION SUMMARY")
        print("="*60)
        print(f"{'Structure':<12} | {'Dice (mean±std)':<18} | {'HD95 (mm) (mean±std)':<22} | {'N':<5}")
        print("-"*60)
        for _, row in agg_df.iterrows():
            dice_str = f"{row['dice_mean']:.3f} ± {row['dice_std']:.3f}"
            hd_str = f"{row['hd95_mean_mm']:.2f} ± {row['hd95_std_mm']:.2f}" if not pd.isna(row['hd95_mean_mm']) else "N/A"
            print(f"{row['structure']:<12} | {dice_str:<18} | {hd_str:<22} | {row['volumes_evaluated']:<5}")
        print("="*60 + "\n")
