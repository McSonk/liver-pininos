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
from monai.networks.nets import UNet
from monai.transforms import (Activations, AsDiscrete, Compose, EnsureTyped,
                              LoadImaged, Orientationd, ScaleIntensityRanged,
                              Spacingd)
from monai.utils import set_determinism

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)

class TestEvaluator:
    """
    Handles checkpoint loading, full-volume inference, metric computation,
    and result export for test datasets.
    """
    def __init__(self, checkpoint_path: str):
        self.device = torch.device(config.DEVICE)
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.inferer = None
        self.test_transforms = None
        self.pred_postprocess = None

         # EXACT post-processing used in training.py
        self.pred_transform = Compose([
            Activations(softmax=True),
            AsDiscrete(argmax=True, to_onehot=config.NUM_CLASSES)
        ])
        self.label_transform = AsDiscrete(to_onehot=config.NUM_CLASSES)

        # Metrics expect decollated lists of tensors
        self.dice_metric = DiceMetric(include_background=False, reduction="none")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, reduction="none", percentile=95.0, distance_metric="euclidean"
        )

        set_determinism(seed=config.RANDOM_SEED)
        logger.info("TestEvaluator initialised. Device: %s", self.device)

    def load_checkpoint(self):
        """Restore model weights and verify config alignment."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        logger.info("Loading checkpoint: %s", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if "config_snapshot" in checkpoint:
            for key in ["NUM_CLASSES", "ISO_SPACING", "HU_WINDOW_MIN", "HU_WINDOW_MAX"]:
                if checkpoint["config_snapshot"].get(key) != getattr(config, key):
                    logger.warning("Config mismatch: %s (ckpt=%s, current=%s)",
                                key, checkpoint["config_snapshot"].get(key), getattr(config, key))


        # Initialise model architecture matching training
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=config.NUM_CLASSES,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=1 if config.is_limited_env() else 2,
            norm="INSTANCE",
            act="PRELU"
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Model loaded successfully. Best Dice (train): %.4f", checkpoint.get("best_dice", -1.0))

        # Sliding window inferer (must match training patch size)
        self.inferer = SlidingWindowInferer(
            roi_size=config.TRAIN_PATCH_SIZE,
            sw_batch_size=16,
            overlap=0.5,
            mode="gaussian",
            device=self.device,
            progress=False
        )

    def _get_test_transforms(self) -> Compose:
        """
        Deterministic preprocessing for test data.
        Ensures image/label spatial alignment via reference-key resampling.
        """
        return Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Orientationd(keys=["image", "label"], axcodes="LAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=config.ISO_SPACING,
                mode=("bilinear", "nearest"),
                recompute_affine=True,  # Ensure output affines are updated
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.HU_WINDOW_MIN,
                a_max=config.HU_WINDOW_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            EnsureTyped(keys=["image", "label"])
        ])

    def run_inference(self, test_files: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Run full-volume inference on test dataset.
        Returns a DataFrame with per-case metrics.
        """
        self.test_transforms = self._get_test_transforms()
        test_ds = Dataset(data=test_files, transform=self.test_transforms)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        results = []
        save_path = config.OUTPUT_DIR / "test_predictions"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("Starting full-volume inference on %d test volumes...", len(test_files))
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dl):
                case_name = Path(batch["image"].meta["filename_or_obj"][0]).stem
                logger.info("[%d/%d] Processing: %s", batch_idx + 1, len(test_dl), case_name)

                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

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

                # Compute metrics on one-hot tensors
                self.dice_metric(y_pred=val_preds, y=val_labels)
                self.hd95_metric(y_pred=val_preds, y=val_labels)
                
                # Save prediction (argmax back to class labels)
                pred_class_map = val_preds[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
                pred_nib = nib.Nifti1Image(pred_class_map, affine=batch["image"].affine[0].numpy())
                nib.save(pred_nib, str(save_path / f"{Path(batch['image'].meta['filename_or_obj'][0]).stem}_pred.nii.gz"))
                
                # Aggregate & reset per-case
                case_dice = self.dice_metric.aggregate().cpu().numpy().flatten()
                case_hd95 = self.hd95_metric.aggregate().cpu().numpy().flatten()

                # --- FIX: Collect per-case results ---
                row = {"case_name": case_name}
                if config.NUM_CLASSES == 3:
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
        out_path = Path(output_dir) if output_dir else config.OUTPUT_DIR / "reports"
        out_path.mkdir(parents=True, exist_ok=True)

        # Aggregate statistics (mean ± std)
        agg_metrics = []
        class_names = ["liver", "tumour"] if config.NUM_CLASSES == 3 else ["tumour"]
        for name in class_names:
            d_dice = df[f"dice_{name}"].dropna()
            d_hd = df[f"hd95_{name}_mm"].dropna()
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
