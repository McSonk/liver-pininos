"""
Inference generation module for automated tumour segmentation.
Designed for full-volume inference and NIfTI export in the original scanner space.
"""
import dataclasses
import time
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, MetaTensor, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.transforms import Invertd

from idssp.sonk import config
from idssp.sonk.model.models import get_model
from idssp.sonk.model.transforms import (get_activations_transforms,
                                         get_validation_transforms)
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)

class InferenceEngine:
    """
    Handles checkpoint loading, full-volume inference, and NIfTI export
    for test datasets.
    """
    def __init__(self, checkpoint_path: Path):
        self.config = config.get()
        self.device = torch.device(self.config.DEVICE)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.inferer: SlidingWindowInferer = None
        self.fallback_inferer: SlidingWindowInferer = None
        self.test_transforms = None

        self.pred_transform = get_activations_transforms(self.config.NUM_CLASSES)

        logger.info("InferenceEngine initialised. Device: %s", self.device)

    def load_checkpoint(self):
        """Restore model weights and verify config alignment."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        logger.info("Loading checkpoint: %s", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)

        if "config_snapshot" in checkpoint:
            ckpt_config = checkpoint["config_snapshot"]
            # STRICT INHERITANCE: Preprocessing & Architecture
            # These MUST match the training environment exactly for valid inference.
            strict_keys = [
                "NUM_CLASSES", "ISO_SPACING", "HU_WINDOW_MIN", "HU_WINDOW_MAX",
                "TRAIN_PATCH_SIZE", "MODEL", "TUMOUR_CLASS_INDEX"
            ]

            updates = {}
            for key in strict_keys:
                if key in ckpt_config:
                    val = ckpt_config[key]

                    # Cast lists back to tuples for spatial dimensions
                    # (JSON/dict saves them as lists)
                    list_properties = ["ISO_SPACING", "TRAIN_PATCH_SIZE", "VAL_PATCH_SIZE"]
                    if key in list_properties and isinstance(val, list):
                        val = tuple(val)

                    # Cast string back to Enum for MODEL
                    if key == "MODEL" and isinstance(val, str):
                        val = config.AvailableModels(val)

                    updates[key] = val

            if updates:
                logger.info("Aligning inference config with checkpoint training "
                            "parameters: %s", updates)
                # Config is a frozen dataclass, so we create a new instance with the updated fields
                self.config = dataclasses.replace(self.config, **updates)
                logger.info("NEW CONFIG CREATED: %s", config.to_dict(self.config))
                logger.debug('Reloading transforms with updated config...')
                self.pred_transform = get_activations_transforms(self.config.NUM_CLASSES)

            # 2. WARNINGS: Non-critical mismatches
            # These do not break the pipeline but might affect performance or logging.
            warn_keys = ["SLIDING_WINDOW_BATCH_SIZE", "RAND_CROP_NUM_SAMPLES"]
            for key in warn_keys:
                ckpt_val = ckpt_config.get(key)
                curr_val = getattr(self.config, key, None)
                if ckpt_val is not None and curr_val is not None and ckpt_val != curr_val:
                    logger.warning(
                        "Config mismatch for '%s': Checkpoint=%s, Current=%s. "
                        "Using current environment settings.",
                        key, ckpt_val, curr_val
                    )

        # Initialise model architecture matching training
        # model will now use the aligned self.config (e.g. correct MODEL and NUM_CLASSES)
        self.model = get_model(self.config).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Model loaded successfully. Best Dice (train): %.4f",
                    checkpoint.get("best_dice", -1.0))

        # Sliding window inferer (must match training patch size)
        self.inferer = SlidingWindowInferer(
            roi_size=self.config.TRAIN_PATCH_SIZE,
            sw_batch_size=self.config.SLIDING_WINDOW_BATCH_SIZE,
            overlap=0.5,
            mode="gaussian",
            device=self.device,
            progress=False
        )

        # Fallback inferer for limited GPU memory (smaller batch size)
        self.fallback_inferer = SlidingWindowInferer(
            roi_size=self.config.TRAIN_PATCH_SIZE,
            sw_batch_size=min(2, self.config.SLIDING_WINDOW_BATCH_SIZE // 4),
            overlap=0.5,
            mode="gaussian",
            device=self.device,
            progress=False
        )

    def run_inference(self, test_files: List[Dict[str, str]], save_path: Path) -> None:
        """
        Run full-volume inference on test dataset and save raw predictions.
        """
        self.test_transforms = get_validation_transforms(self.config)

        # TODO: add a CLI argument to run a dummy test for debugging purposes

        # Invertd requires the full deterministic pipeline (no random crops).
        # In limited environments, get_validation_transforms injects
        # RandCropByPosNegLabeld, which produces patches whose transform
        # traces cannot be inverted back to the original volume space.
        if config.is_limited_env(config=self.config, include_vram=False):
            raise RuntimeError(
                "Full-volume inference with Invertd is not supported in limited "
                "environments. The validation transforms include "
                "RandCropByPosNegLabeld, which prevents correct inversion of "
                "predictions back to the original image space. "
                "Run this script on a GPU environment (is_limited_env() == False)."
            )

        test_ds = Dataset(data=test_files, transform=self.test_transforms)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        save_path.mkdir(parents=True, exist_ok=True)

        logger.info("Starting full-volume inference on %d test volumes...", len(test_files))
        start_time = time.time()

        # PROBLEM:
        # Part of the pipeline consists of applying spatial transformations
        # (e.g. resampling, cropping) to the input images before feeding them
        # to the model. At inference time, the predictions therefore have
        # different affine matrices and shapes from the original images.
        # To save predictions in the original scanner space, we invert the
        # preprocessing transforms using MONAI's Invertd.

        # Inverter to map predictions back to the original raw space before saving
        inverter = Invertd(
            keys="pred",
            transform=self.test_transforms,
            orig_keys="image",    # The dictionary key holding the MetaTensor with the trace
            nearest_interp=True,  # Prevents class blurring during inverse resampling
            to_tensor=False,      # Return numpy arrays (convenient for nibabel)
            device="cpu"
        )

        # ---- Validation ---
        # Validate that Invertd will work before processing all volumes
        _probe_batch = next(iter(test_dl))
        _probe_images = decollate_batch(_probe_batch["image"])
        _probe_img = _probe_images[0]
        if not hasattr(_probe_img, "applied_operations") or len(_probe_img.applied_operations) == 0:
            raise RuntimeError(
                "MetaTensor has no applied_operations trace after decollate_batch. "
                "Invertd will silently return predictions in preprocessed space. "
                "Check MONAI version or DataLoader num_workers setting."
            )

        # Explicit conversion to MetaTensor (pure monai sometimes removes the MetaTensor
        # wrapper after decollate_batch)
        _probe_pred = MetaTensor(torch.zeros(1, *_probe_img.shape[1:], dtype=torch.float32))
        _probe_inverted = inverter({"pred": _probe_pred, "image": _probe_img})["pred"]
        _probe_inverted = np.asarray(_probe_inverted)

        if _probe_inverted.ndim == 4:
            _probe_inverted = _probe_inverted[0]

        _probe_original_path = _probe_img.meta.get("filename_or_obj")
        if _probe_original_path is not None:
            _probe_original_shape = nib.load(_probe_original_path).shape[:3]

            if _probe_inverted.shape != _probe_original_shape:
                raise RuntimeError(
                    "Invertd probe failed: inverted shape "
                    f"{_probe_inverted.shape} does not match original shape "
                    f"{_probe_original_shape}. Invertd is not applying the trace."
                )
        logger.info(
            "Invertd trace check passed: %d operations found on MetaTensor.",
            len(_probe_img.applied_operations)
        )
        # Reset the DataLoader iterator
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
        # --- End validation ---

        with torch.inference_mode():
            for batch_idx, batch in enumerate(test_dl):
                filename = Path(batch["image"].meta["filename_or_obj"][0]).name

                if filename.endswith(".nii.gz"):
                    case_name = filename[:-7]
                elif filename.endswith(".nii"):
                    case_name = filename[:-4]
                else:
                    case_name = Path(filename).stem

                logger.info("[%d/%d] Processing: %s", batch_idx + 1, len(test_dl), case_name)

                if batch_idx % 5 == 0:
                    logger.debug("Information of batch %d:", batch_idx)
                    logger.debug("Batch image shape: %s", batch["image"].shape)
                    logger.debug("MONAI meta affine shape:%s", batch["image"].meta["affine"].shape)
                    logger.debug("MONAI meta affine:\n%s", batch["image"].meta["affine"][0])

                images = batch["image"].to(self.device)

                logger.debug("Input image shape (after to device): %s", images.shape)

                # Full-volume sliding window inference
                with torch.amp.autocast(
                                    device_type="cuda",
                                    enabled=self.device.type == "cuda"):
                    try:
                        preds = self.inferer(inputs=images, network=self.model)
                    except torch.cuda.OutOfMemoryError as e:
                        torch.cuda.empty_cache()
                        logger.warning("CUDA OOM during inference on case %s. Error: %s",
                                       case_name, str(e))
                        logger.warning("Retrying inference with smaller batch size "
                                       "(fallback inferer).")
                        preds = self.fallback_inferer(inputs=images, network=self.model)

                # Decollate = split batched tensor into a list of individual
                # samples, each retaining its MetaTensor metadata (affine,
                # transform trace). We decollate the CPU batch (`batch["image"]`), not the GPU
                # copy, because Invertd only needs the metadata and runs on CPU.
                val_preds = decollate_batch(preds)
                val_images = decollate_batch(batch["image"])
                # Note that each of the val_elements is a list of individual
                # samples. Also, each contains the meta information (affine, spacing, etc.)
                # and applied transformations.

                # Apply MONAI post-processing (matches training.py exactly)
                val_preds = [self.pred_transform(p) for p in val_preds]

                # Save prediction in original raw space
                for i, (pred, image) in enumerate(zip(val_preds, val_images)):
                    # Invertd expects a channel dimension: (1, D, H, W)
                    pred_indices = pred.argmax(dim=0, keepdim=True).cpu()

                    # MONAI 1.5.2: Invertd silently skips inversion if this is not a MetaTensor.
                    pred_indices = MetaTensor(pred_indices)

                    # Invert spatial transforms using the image's stored trace
                    inverted_data = inverter({"pred": pred_indices, "image": image})
                    inverted_pred = inverted_data["pred"]

                    if isinstance(inverted_pred, torch.Tensor):
                        inverted_pred = inverted_pred.cpu().numpy()

                    inverted_pred = np.asarray(inverted_pred)
                    if inverted_pred.ndim == 4:
                        inverted_pred = inverted_pred[0]  # Remove channel dim → (D, H, W)

                    # Ensure discrete integer classes (make sure 1.999 -> 2, not 1,
                    # as with direct casting)
                    if np.issubdtype(inverted_pred.dtype, np.floating):
                        inverted_pred = np.rint(inverted_pred)

                    # Clip to valid class range and convert to uint8 for NIfTI saving
                    pred_class_map = np.clip(
                        inverted_pred,
                        0,
                        self.config.NUM_CLASSES - 1
                    ).astype(np.uint8, copy=False)

                    # Sanity check: only valid class indices should be present
                    unique_values = np.unique(pred_class_map)
                    valid_values = np.arange(self.config.NUM_CLASSES, dtype=np.uint8)

                    if not np.all(np.isin(unique_values, valid_values)):
                        logger.warning(
                            "Case %s: unexpected class values after inversion: %s",
                            case_name,
                            unique_values
                        )

                    # Sanity check: shape should match the original NIfTI on disk
                    original_path = image.meta.get("filename_or_obj")
                    if original_path is not None:
                        original_nib = nib.load(original_path)
                        original_shape = original_nib.shape[:3]

                        if pred_class_map.shape != original_shape:
                            logger.error(
                                "Inverted prediction shape mismatch for case %s: "
                                "prediction=%s, original=%s",
                                case_name,
                                pred_class_map.shape,
                                original_shape
                            )
                            raise ValueError(
                                f"Inverted prediction shape mismatch for case {case_name}: "
                                f"prediction={pred_class_map.shape}, original={original_shape}"
                            )

                    # LoadImaged stores the raw scanner affine in 'original_affine'
                    original_affine = np.asarray(
                        image.meta.get("original_affine", image.affine)
                    )

                    if original_affine.shape != (4, 4):
                        raise ValueError(
                            f"Invalid affine shape for case {case_name}: {original_affine.shape}"
                        )

                    pred_nib = nib.Nifti1Image(pred_class_map, affine=original_affine)
                    nib.save(pred_nib, str(save_path / f"{case_name}_pred.nii.gz"))

        elapsed = time.time() - start_time
        logger.info("Inference completed in %.1f s (%.2f s/volume)", elapsed, elapsed / len(test_files))
