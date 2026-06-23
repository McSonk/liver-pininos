import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from monai.data import (CacheDataset, DataLoader, Dataset, PersistentDataset,
                        decollate_batch)
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

from idssp.sonk import config
from idssp.sonk.model.models import get_model
from idssp.sonk.model.transforms import (get_activations_transforms,
                                         get_deterministic_transforms,
                                         get_label_transform,
                                         get_random_transforms,
                                         get_validation_transforms)
from idssp.sonk.utils.logger import get_logger, log_memory_usage
from idssp.sonk.utils.notifications import send_alert
from idssp.sonk.view.utils import log_segmentation_overlay

logger = get_logger(__name__)

class AugmentedDataset(Dataset):
    '''
    A wrapper around a base dataset that applies additional random transforms on-the-fly.
    This is used to apply random cropping and flipping during training when using
    Persistent or Cached Dataset,
    which applies the main transforms once and caches the results. By separating the random
    transforms into a second Compose, we can ensure that we still get data augmentation
    benefits without losing the caching advantages.
    '''
    def __init__(self, base_ds, random_transform):
        '''
        Params
        ------
        `base_ds`: Dataset
            The base dataset (e.g. PersistentDataset) that applies the main
            deterministic transforms.
        `random_transform`: Compose
            A Compose object containing the random transforms to apply on-the-fly during training.
        '''
        super().__init__(base_ds)
        self.base_ds = base_ds
        self.aug = random_transform
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, i):
        return self.aug(self.base_ds[i])

class ModelBuilder:
    '''
    This class encapsulates the entire model training pipeline, including:
    - Data loading and transformation
    - Model initialization
    - Training and validation loops
    - Checkpointing and logging
    '''
    def __init__(self):
        self.train_dl = None
        self.val_dl = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.inferer: SlidingWindowInferer = None
        self.config = config.get()
        self.device = torch.device(self.config.DEVICE)
        self._overlay_batch = None
        '''This will store a fixed validation image for logging to tensorboard'''

        # Post-processing & Metrics
        self.pred_trans = get_activations_transforms(self.config)
        '''`Compose` of transforms applied to model predictions so we have
        a probability distribution [0-1] for each voxel per class (`config.NUM_CLASSES`).
        To be used in validation step.
        
        1. `Activations(softmax=True)`: Applies the softmax function to the raw model
           outputs (logits) to convert them into class probabilities for each voxel.
        
        2. `AsDiscrete(argmax=True, to_onehot=self.config.NUM_CLASSES)`: This transform
           first applies argmax to select the class with the highest probability
           for each voxel, then it converts these class labels into one-hot
           encoding format.
        '''

        self.label_trans = get_label_transform(self.config)
        '''Transform applied to ground truth labels so they're represented as one-hot
           encoded tensors to each class before metric calculation.
           It only contains `AsDiscrete(to_onehot=self.config.NUM_CLASSES)`, which converts
           the integer class labels into one-hot encoding format.
           
           To be used in validation step.'''

        # include_background=False is standard for multi-class segmentation
        # to avoid background dominating the metric.
        # reduction="none" keeps per-sample/per-class Dice values so any averaging
        # can be handled explicitly elsewhere in the validation epoch.
        self.dice_metric = DiceMetric(include_background=False, reduction="none")
        '''Stores DiceMetric scores with background excluded. When `.aggregate()`
           is called, it returns unreduced Dice values rather than a single global
           mean. The aggregated result contains per-sample/per-class foreground
           scores (typically one value for each validation item and non-background
           class, e.g. shape `[batch_size, self.config.NUM_CLASSES - 1]`).

           To be used in validation step.'''

        self.history = {"train_loss": [], "val_loss": [], "val_dice": []}

        # So we can use float16 mixed precision on CUDA
        # (Multiplies loss by a scale factor to prevent underflow, and unscales
        # gradients before the optimizer step)
        self.scaler = GradScaler('cuda') if self.config.DEVICE == 'cuda' else None
        '''Mixed precision training scaler, enabled only on CUDA for potential speed up.'''

        # tensorboard writer for logging training metrics
        logger.debug("Writing initial hyperparameters to TensorBoard: \n%s", config.to_param_dict())
        # TODO: review the use of all writers to tensorboard
        self.writer = SummaryWriter(log_dir=str(self.config.TENSORBOARD_DIR))
        self.writer.add_hparams(
            config.to_param_dict(),
            { # metric dict
                 "Metrics/Dice": 0.0,
                "val/dice_liver": 0.0,
                "val/dice_tumour": 0.0,
                "Loss/Train": 0.0,
            }
        )
        logger.info("TensorBoard writer initialised at: %s", self.writer.log_dir)

        logger.info("ModelBuilder initialized. Device set to: %s", self.device)

    def get_train_transforms(self) -> tuple[Compose, Compose]:
        '''
        Returns the transforms for the training data. Note that the result will
        differ based on the environment:
        - In a limited environment (e.g. CPU) or when using CacheDataset
          (`config.USE_CACHE_(TRAIN/VAL)_DATASET`), all transforms (including random cropping)
          are included in the main deterministic pipeline
        - In a GPU environment with PersistentDataset, the random cropping is separated into
            a second Compose that is applied on-the-fly. This due to the fact that
            PersistentDataset applies transforms once and caches the results, so
            we can't include random cropping in the main pipeline without losing variability.

        Returns
        -------
        `tuple[Compose, Compose]`
            A tuple containing two Compose objects:
            - The first Compose contains the deterministic transforms that are
              applied to all training samples.
            - The second Compose contains the random transforms that are applied
              on-the-fly during training for
              data augmentation. This may be empty if we're in a limited environment
              or using CacheDataset, in which case the random transforms are included
              in the deterministic pipeline.
        '''
        deterministic_transforms = get_deterministic_transforms(self.config)

        random_transforms = get_random_transforms(self.config)

        if config.is_limited_env():
            logger.debug("Using random crop in main transform pipeline for limited "
            "environment")
            # When we won't use Persistent or CacheDataset (which applies transforms once
            # and caches the results), we can include the random cropping in the
            # main transform pipeline.
            deterministic_transforms.extend(random_transforms)
            random_transforms = []  # No separate random transforms needed

        return Compose(deterministic_transforms), Compose(random_transforms)

    def init_data_loaders(self, train_files: list, val_files: list):
        '''
        Initializes the training and validation data loaders.
        This includes creating the datasets with the appropriate transforms and
        then wrapping them in DataLoader objects.

        Params
        -----
        `train_files`: list
            A list of file paths for the training data. Each item in the list
            should be a dictionary with keys "image" and "label" pointing to the
            respective file paths.
        `val_files`: list
            A list of file paths for the validation data. Each item in the list
            should be a dictionary with keys "image" and "label" pointing to the
            respective file paths.
        '''
        logger.debug("Creating training transforms object...")
        # train_ran_trans will be empty if we're in a limited environment or using CacheDataset
        train_det_trans, train_ran_trans = self.get_train_transforms()

        logger.debug("Deterministic transforms for training (%d steps):",
                     len(train_det_trans.transforms))
        for i, t in enumerate(train_det_trans.transforms, 1):
            logger.debug("  %2d. %s", i, t.__class__.__name__)

        logger.debug("Random transforms for training (%d steps):", len(train_ran_trans.transforms))
        for i, t in enumerate(train_ran_trans.transforms, 1):
            logger.debug("  %2d. %s", i, t.__class__.__name__)

        logger.debug("Creating validation transforms object...")
        val_transforms = get_validation_transforms(self.config)

        logger.debug("Validation transforms (%d steps):", len(val_transforms.transforms))
        for i, t in enumerate(val_transforms.transforms, 1):
            logger.debug("  %2d. %s", i, t.__class__.__name__)

        logger.info("Initializing training and validation datasets...")
        if config.is_limited_env():
            logger.debug("Limited environment detected. Using plain MONAI Dataset.")
            # With a Dataset the transformations are executed every iteration
            # so the process will be slow (but less memory intensive)
            train_ds = Dataset(data=train_files, transform=train_det_trans)
            val_ds = Dataset(data=val_files, transform=val_transforms)
        else:
            time_at_start = time.time()
            if self.config.USE_CACHE_TRAIN_DATASET:
                logger.debug("[Train] Sufficient resources detected. Using MONAI CacheDataset.")
                train_ds = AugmentedDataset(
                    CacheDataset(
                        data=train_files,
                        transform=train_det_trans,
                        cache_rate=self.config.CACHE_DATASET_RATE,
                        num_workers=self.config.CACHE_NUM_WORKERS,
                    ),
                    train_ran_trans
                )
            else:
                logger.info("[Train] Sufficient resources detected. Using PersistentDataset.")
                train_ds = AugmentedDataset(
                    PersistentDataset(
                        data=train_files,
                        transform=train_det_trans,
                        cache_dir=str(self.config.PERSISTENT_DATASET_DIR /
                                      f"hmin_{self.config.HU_WINDOW_MIN}"
                                      f"_hmax_{self.config.HU_WINDOW_MAX}_train_cache")
                    ),
                    train_ran_trans
                )
            # end if-else cache train dataset

            if self.config.USE_CACHE_VAL_DATASET:
                logger.debug("[Val] Using MONAI CacheDataset.")
                val_ds = CacheDataset(
                    data=val_files,
                    transform=val_transforms,
                    cache_rate=self.config.CACHE_DATASET_RATE,
                    num_workers=self.config.CACHE_NUM_WORKERS,
                )
            else:
                # TODO: implement a hashing mechanism to detect changes in transforms
                # (use the hash as dir name)
                logger.info("[Val] Sufficient resources detected. Using PersistentDataset.")
                val_ds = PersistentDataset(
                    data=val_files,
                    transform=val_transforms,
                    cache_dir=str(self.config.PERSISTENT_DATASET_DIR /
                                  f"hmin_{self.config.HU_WINDOW_MIN}"
                                  f"_hmax_{self.config.HU_WINDOW_MAX}_val_cache")
                )
            # end if-else cache val dataset

            logger.info("Datasets initialized in %.2f seconds.", time.time() - time_at_start)
        # end if-else for is limited environment

        logger.debug("Creating training dataloader...")
        self.train_dl = DataLoader(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.DL_NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            # This will make workers persistent across epochs
            # (so workers are no recreated every epoch as usual)
            # This prevents run-first memory problems on shared computer
            # but also takes more memory (which is never freed until the end of training)
            persistent_workers=self.config.DL_NUM_WORKERS > 0,
        )

        logger.debug("Creating validation dataloader...")
        self.val_dl = DataLoader(
            val_ds,
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.DL_NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            persistent_workers=self.config.DL_NUM_WORKERS > 0,
        )

        # Initialise a fixed validation batch for logging overlays during training
        for batch in self.val_dl:
            # Check if current image has tumour voxels (label == 2)
            if (batch["label"] == self.config.TUMOUR_CLASS_INDEX).any():
                self._overlay_batch = batch
                logger.debug(
                    "Overlay volume selected — label shape: %s, tumour voxels: %d",
                    batch["label"].shape,
                    (batch["label"] == self.config.TUMOUR_CLASS_INDEX).sum().item()
                )
                break

        if self._overlay_batch is None:
            raise ValueError("No validation sample with tumour voxels was found")

        logger.debug("Data loaders initialized successfully.")
        logger.info(
            "Training DataLoader: %d batches | effective batch shape: torch.Size([%d, 1, %s])",
            len(self.train_dl),
            self.config.BATCH_SIZE * self.config.RAND_CROP_NUM_SAMPLES,
            ", ".join(str(d) for d in self.config.TRAIN_PATCH_SIZE),
        )

    def init_model(self):
        '''Initializes the model, loss function, and optimizer.'''
        logger.debug("Initializing model...")
        self.model = get_model().to(self.device)

        log_memory_usage(logger, prefix="After model initialization: ")

        # Cross entropy: voxel-wise classification (smooth gradients)
        # Dice: measures the overlap between predicted and true segmentation masks.
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            # to match `self.dice_metric`
            include_background=False,
            # dice weight
            lambda_dice=1.0,
            # CE weight
            lambda_ce=1.0,
            # weight is vector penalisation (how aggressive the penalisation)
            # is for that class. [background_weight, liver_weight, tumour_weight]
            # NOTE: weight should be ce_weight, but apparently pytorch version 2.11.0
            # doesn't expose it yet 
            weight=torch.tensor(self.config.DICE_CE_WEIGHTS, device=self.device)
        )
        self.optimizer = optim.AdamW(
            # weights and biases
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-5
        )

        logger.info("DiceCELoss initialized with weights: %s", self.config.DICE_CE_WEIGHTS)

        if not config.is_limited_env():
            # Initialize sliding window inferer (run validation on full volumes
            # via sliding window patches)
            self.inferer = SlidingWindowInferer(
                # MUST be the same as the training patch size
                roi_size=self.config.TRAIN_PATCH_SIZE,
                # Process 16 patches in parallel
                sw_batch_size=self.config.SLIDING_WINDOW_BATCH_SIZE,
                # Generate overlapping patches (reduces the step size)
                # to smooth out predictions at patch borders
                # (25% is a common choice, but 50% can further reduce border
                # artifacts at the cost of more computation)
                overlap=0.5,
                # Use a Gaussian weighting function to give more importance to
                # the centre of the patch in the predictions while ensuring overlapping
                # regions blend smoothly
                mode="gaussian",
                device=self.device,
                # The process will be run via jobs, so progress bar won't be watched anyway
                progress=False
            )
            logger.debug("Using SlidingWindowInferer for full-volume inference"
                         "with params: "
                         "roi_size=%s, "
                         "sw_batch_size=%d, "
                         "overlap=%.2f, "
                         "mode=%s, "
                         "device=%s, "
                         "progress=%s",
                         self.config.TRAIN_PATCH_SIZE,
                         self.config.SLIDING_WINDOW_BATCH_SIZE,
                         0.5,
                         "gaussian",
                         self.device,
                         False)

        logger.info("Model initialized on %s", self.device)
        logger.info("Optimizer: AdamW | LR: %f | Weight Decay: 1e-5", self.config.LEARNING_RATE)


        # During training scheduler follows a cosine curve between LEARNING_RATE
        # over NUM_EPOCHS epochs
        warm_epochs = self.config.WARMUP_EPOCHS
        if warm_epochs <= 0:
            warm_epochs = 1  # Avoid invalid total_iters for LinearLR

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warm_epochs - 1
        )
        # TODO: UNDERSTAND THIS
        t_max = self.config.NUM_EPOCHS - self.config.WARMUP_EPOCHS
        if t_max <= 0:
            logger.info("WARMUP_EPOCHS (%d) is greater than or equal to NUM_EPOCHS (%d). "
            "Cosine annealing will not be applied.", self.config.WARMUP_EPOCHS, self.config.NUM_EPOCHS)
            t_max = 1  # Avoid invalid T_max for CosineAnnealingLR
            logger.info(
                "Using fallback T_max=%d only to keep CosineAnnealingLR valid.",
                t_max
            )
        else:
            logger.info(
                "Cosine annealing will be applied for %d epochs after a warmup of %d epochs.",
                t_max,
                self.config.WARMUP_EPOCHS
            )
            logger.info("Eta min for cosine annealing: %e", self.config.COSINE_ETA_MIN)
        logger.info("Warmup scheduler: LinearLR (start_factor=0.1, end_factor=1.0, total_iters=%d)", warm_epochs - 1)
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=self.config.COSINE_ETA_MIN
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.WARMUP_EPOCHS]
        )

        logger.info("Scheduler initialized: CosineAnnealingLR (T_max=%d, eta_min=%e)",
                    t_max, self.config.COSINE_ETA_MIN)

    def _validate_checkpoint(self, checkpoint: dict, checkpoint_path: Path) -> None:
        '''
        Validates a checkpoint before loading to ensure compatibility and prevent
        corrupted or incompatible resumes.

        Validation rules:
        - Required keys must exist (model_state_dict at minimum)
        - MODEL and NUM_CLASSES must match exactly (hard-fail on mismatch)
        - Preprocessing keys (ISO_SPACING, HU_WINDOW_MIN, HU_WINDOW_MAX) warn on mismatch
        - Checkpoint epoch must be < current NUM_EPOCHS config
        - torch.load errors are caught and re-raised with clear messages

        Params
        ------
        `checkpoint`: dict
            The loaded checkpoint dictionary to validate.
        `checkpoint_path`: Path
            Path to the checkpoint file (for error messages).

        Raises
        ------
        ValueError
            If required keys are missing or if MODEL/NUM_CLASSES mismatch.
        RuntimeError
            If checkpoint is corrupted or epoch >= NUM_EPOCHS.
        '''
        logger.info("Validating checkpoint: %s", checkpoint_path)

        # Step 1: Validate required keys exist
        required_keys = ["model_state_dict"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(
                    f"Checkpoint validation failed: Required key '{key}' not found in "
                    f"checkpoint {checkpoint_path}. "
                    f"Checkpoint may be corrupted or from an incompatible version."
                )

        # Step 2: Validate model architecture compatibility (hard-fail)
        if "config_snapshot" in checkpoint:
            ckpt_config = checkpoint["config_snapshot"]
            curr_config = self.config

            # MODEL check - hard fail on mismatch
            ckpt_model = ckpt_config.get("MODEL")
            curr_model = curr_config.MODEL.value
            if ckpt_model is not None and ckpt_model != curr_model:
                raise ValueError(
                    f"Checkpoint validation failed: MODEL mismatch. "
                    f"Checkpoint has MODEL='{ckpt_model}', but current config has MODEL='{curr_model}'. "
                    f"Cannot resume training with a different model architecture."
                )

            # NUM_CLASSES check - hard fail on mismatch
            ckpt_num_classes = ckpt_config.get("NUM_CLASSES")
            curr_num_classes = curr_config.NUM_CLASSES
            if ckpt_num_classes is not None and ckpt_num_classes != curr_num_classes:
                raise ValueError(
                    f"Checkpoint validation failed: NUM_CLASSES mismatch. "
                    f"Checkpoint has NUM_CLASSES={ckpt_num_classes}, but current config has NUM_CLASSES={curr_num_classes}. "
                    f"Cannot resume training with a different number of classes."
                )

            # Preprocessing checks - warn on mismatch but allow resume
            preprocessing_keys = ["ISO_SPACING", "HU_WINDOW_MIN", "HU_WINDOW_MAX"]
            for key in preprocessing_keys:
                ckpt_val = ckpt_config.get(key)
                curr_val = getattr(curr_config, key, None)

                if ckpt_val is None or curr_val is None:
                    logger.debug(
                        "Config key '%s' missing in checkpoint or current config. Skipping check.",
                        key)
                    continue

                def to_list(val):
                    if isinstance(val, (list, tuple)):
                        return list(val)
                    else:
                        return [val]

                if to_list(ckpt_val) != to_list(curr_val):
                    logger.warning(
                        "Preprocessing mismatch for '%s': Checkpoint=%s, Current=%s. "
                        "Resume will proceed but results may differ due to preprocessing changes.",
                        key, ckpt_val, curr_val
                    )
                else:
                    logger.debug("Preprocessing match for '%s': %s", key, curr_val)
        else:
            logger.warning(
                "Checkpoint does not contain 'config_snapshot'. "
                "Skipping config compatibility validation. "
                "Ensure checkpoint was created with compatible settings."
            )

        # Step 3: Validate epoch is within valid range
        saved_epoch = checkpoint.get("epoch", 0)
        if saved_epoch >= self.config.NUM_EPOCHS:
            raise RuntimeError(
                f"Checkpoint validation failed: Checkpoint epoch ({saved_epoch}) >= "
                f"current NUM_EPOCHS ({self.config.NUM_EPOCHS}). "
                f"Training has already completed or NUM_EPOCHS was reduced."
            )

        logger.info(
            "Checkpoint validation passed: epoch=%d/%d, config compatible.",
            saved_epoch, self.config.NUM_EPOCHS - 1
        )

    def _resume_from_checkpoint(self, checkpoint_path: Path, early_stopper: 'EarlyStopper') -> None:
        '''
        Resumes training from a checkpoint file. Loads model weights, optimizer state,
        scaler state, scheduler state (if available), RNG states, and seeds the EarlyStopper
        with saved metrics. Includes strict validation before loading.

        Params
        ------
        `checkpoint_path`: Path
            Path to the checkpoint file (.pth) to resume from.
        `early_stopper`: EarlyStopper
            The EarlyStopper instance to seed with saved best metrics.

        Raises
        ------
        ValueError
            If checkpoint validation fails (missing keys, MODEL/NUM_CLASSES mismatch).
        RuntimeError
            If checkpoint is corrupted or incompatible.
        '''
        logger.info("Loading checkpoint for resume: %s", checkpoint_path)
        # Step 0: Load checkpoint with error handling
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint {checkpoint_path}: {type(e).__name__}: {e}. "
                f"Checkpoint may be corrupted or invalid."
            ) from e

        # Step 1: Validate checkpoint before loading any state
        self._validate_checkpoint(checkpoint, checkpoint_path)

        # Step 2: Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model weights loaded successfully.")

        # Step 3: Load optimizer state if present
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded.")
        else:
            logger.warning("Optimizer state not found in checkpoint. Training will restart with fresh optimizer.")

        # Step 4: Load scaler state if present
        if checkpoint.get("scaler_state_dict") is not None and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("Scaler state loaded.")

        # Step 5: Load scheduler state if present (backward compatible)
        saved_epoch = checkpoint.get("epoch", 0)
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Scheduler state loaded from checkpoint.")
        else:
            # Fallback: recreate scheduler and step to saved epoch (for older checkpoints)
            logger.warning(
                "Scheduler state not found in checkpoint. Recreating scheduler and advancing to epoch %d.",
                saved_epoch
            )
            warm_epochs = self.config.WARMUP_EPOCHS
            if warm_epochs <= 0:
                warm_epochs = 1
            t_max = self.config.NUM_EPOCHS - self.config.WARMUP_EPOCHS
            if t_max <= 0:
                t_max = 1

            # Re-create schedulers with current optimizer
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warm_epochs - 1
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=self.config.COSINE_ETA_MIN
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.config.WARMUP_EPOCHS]
            )

            # Step scheduler to align with saved epoch
            for _ in range(saved_epoch + 1):
                self.scheduler.step()
            logger.info("Scheduler recreated and advanced to epoch %d.", saved_epoch)

        # Step 6: Restore RNG states for deterministic restart
        if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
            rng = checkpoint["rng_state"]
            if rng.get("torch_cpu") is not None:
                torch.set_rng_state(rng["torch_cpu"])
                logger.debug("Torch CPU RNG state restored.")
            if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
                logger.debug("Torch CUDA RNG state restored.")
            if rng.get("numpy") is not None:
                try:
                    np_state = rng["numpy"]
                    # Reconstruct the tuple with a numpy array for np.random.set_state
                    restored_state = (
                        np_state[0],
                        np.array(np_state[1], dtype=np.uint32),
                        np_state[2],
                        np_state[3],
                        np_state[4]
                    )
                    np.random.set_state(restored_state)
                    logger.debug("NumPy RNG state restored.")
                except Exception as e:
                    logger.debug("Failed to restore NumPy RNG state: %s", e)
            if rng.get("python") is not None:
                random.setstate(rng["python"])
                logger.debug("Python RNG state restored.")
        else:
            logger.warning("RNG states not found in checkpoint. "
                           "Deterministic restart not guaranteed.")

        # Step 7: Seed EarlyStopper with saved metrics
        early_stopper.best_mean_dice = checkpoint.get("best_dice", -1.0)
        early_stopper.best_liver_dice = checkpoint.get("best_liver_dice", -1.0)
        early_stopper.best_tumour_dice = checkpoint.get("best_tumour_dice", -1.0)
        early_stopper.best_epoch = saved_epoch
        logger.info(
            "EarlyStopper seeded: best_epoch=%d, best_dice=%.4f,"
            " best_liver_dice=%.4f, best_tumour_dice=%.4f",
            saved_epoch,
            early_stopper.best_mean_dice,
            early_stopper.best_liver_dice,
            early_stopper.best_tumour_dice
        )

        # Step 8: Optionally restore history for audit/debug (non-critical)
        if "history" in checkpoint and checkpoint["history"] is not None:
            self.history = checkpoint["history"]
            logger.debug("Training history restored from checkpoint.")

        if "current_lr" in checkpoint and checkpoint["current_lr"] is not None:
            logger.debug("Checkpoint LR at save: %.6f", checkpoint["current_lr"])

    def back_propagate(self, loss, is_update_step: bool = True):
        '''
        Performs backpropagation with optional mixed precision scaling.

        Params
        ------
        `loss`: torch.Tensor
            The computed loss for the current batch, which will be back-propagated.
        '''
        if self.scaler is not None:
            # Multiplies the loss by the scale factor to prevent underflow during backpropagation
            # scaled_loss = original_loss * scale_factor
            self.scaler\
                    .scale(loss)\
                    .backward()
            # ^ And then back propagate (∇(scaled_loss) )

            if is_update_step:
                # Un-scales gradients (∇(original_loss) = ∇(scaled_loss) / scale_factor)
                # and checks for inf/NaN values.
                # ("_" means in-place operation)
                self.scaler.unscale_(self.optimizer)

                # Limit the magnitude of the gradients to prevent them from exploding
                # This will compute a global norm over all parameters
                clip_grad_norm_(
                    self.model.parameters(),
                    # If global norm exceeds 1.0, all gradients are scaled down
                    max_norm=1.0
                )

                # Now we have safe gradient values. Optimiser will do safe updates.
                # Step the optimiser with scaled gradients
                # Scaler checks if gradients has inf/NaN values. If they do,
                # optimizer step is skipped to avoid corrupting the model weights.
                self.scaler.step(self.optimizer)

                # Updates scaling factor. It can either duplicate, halve or keep it as is.
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

        else:
            # Standard backpropagation for CPU or if mixed precision is not enabled
            # calc gradients
            loss.backward()
            if is_update_step:
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

    def _train_epoch(self):
        '''Performs one epoch of training over the entire training dataset.
        
        Returns
        -------
        `float`
            The average training loss over the epoch.
        '''
        # Sets model to train mode
        self.model.train()
        train_loss = 0

        # Zero gradients at the start of the epoch
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_dl):
            # Batch is a dictionary with keys "image" and "label", each containing a tensor of shape
            # (batch_size, channels, depth, height, width).

            # "batch" will contain conf.RAND_CROP_NUM_SAMPLES * conf.BATCH_SIZE
            # patches in total (e.g. batch_size=16 if BATCH_SIZE=2 and RAND_CROP_NUM_SAMPLES=8).
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Mixed precision training for potential speed up on CUDA
            with autocast(device_type="cuda", enabled=self.config.DEVICE == "cuda"):
                predictions = self.model(images)
                loss = self.loss_fn(predictions, labels)

            # Scale loss to ensure gradient magnitude remains identical
            loss = loss / self.config.ACCUMULATION_STEPS

            # Determine if we should update weights on this step
            is_update_step = ((step + 1) % self.config.ACCUMULATION_STEPS == 0) or \
                             (step + 1 == len(self.train_dl))

            self.back_propagate(loss, is_update_step)

            # Accumulate unscaled loss for logging
            train_loss += loss.item() * self.config.ACCUMULATION_STEPS

        return train_loss / len(self.train_dl)

    def _should_log_overlay(self, epoch: int) -> bool:
        """Checks if we should log the segmentation overlay for the current epoch
        based on the defined interval:

        - Every epoch for the first 10 epochs to closely monitor initial learning dynamics.
        - Every 5 epochs during the rapid improvement phase (epochs 11-30)
        - Every 10 epochs once the model performance stabilizes (epoch 31+)
        
        """
        if epoch <= 10:
            return True          # every epoch for the first 10
        if epoch <= 30:
            return epoch % 5 == 0    # every 5 epochs during rapid improvement
        return epoch % 10 == 0   # every 10 epochs once stable

    def _should_notify(self, epoch: int) -> bool:
        """Checks if we should send a notification for the current epoch based on the defined interval:

        - Every 5 epochs for the first 50 epochs to closely monitor early training progress.
        - Every 10 epochs during the middle phase (epochs 51-100) when improvements are more gradual.
        - Every 20 epochs once the model performance stabilizes (epoch 101+)
        
        """
        if epoch <= 50:
            return epoch % 5 == 0    # every 5 epochs for the first 50
        if epoch <= 100:
            return epoch % 10 == 0   # every 10 epochs during middle phase
        return epoch % 20 == 0       # every 20 epochs once stable

    def _run_small_inference(self, image: torch.Tensor) -> torch.Tensor:
        """Run full-volume sliding window inference on a single batch.
        Called from within torch.inference_mode() and autocast contexts in _validate.
        """
        with autocast(device_type="cuda", enabled=self.config.DEVICE == "cuda"):
            if self.inferer is not None:
                return self.inferer(inputs=image, network=self.model)
            else:
                logger.debug(
                    "(Tensorboard image) Inferer not available, running "
                    "direct inference on CPU."
                )
                return self.model(image)

    def _run_val_epoch(self, epoch: int):
        val_loss = 0

        for batch in self.val_dl:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(device_type="cuda", enabled=self.config.DEVICE == "cuda"):
                if self.inferer is not None:
                    try:
                        # GPU: Process full volume via sliding window
                        preds = self.inferer(images, self.model)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        logger.warning("OOM during validation, falling back to sw_batch_size=1")
                        # Fallback on low memory
                        fallback = SlidingWindowInferer(
                            roi_size=self.config.TRAIN_PATCH_SIZE,
                            sw_batch_size=1, # (original: 4)
                            overlap=0.25, # (original: 0.5)
                            mode="gaussian",
                            device=self.device,
                            progress=False
                        )
                        preds = fallback(images, self.model)
                else:
                    # CPU: Images are already cropped to VAL_PATCH_SIZE, run directly
                    preds = self.model(images)

            # Cast predictions to float32 before loss calculation.
            # Mixed precision (float16) can cause numerical instability and NaNs
            # during operations like Dice or cross-entropy. Using float32 ensures
            # stable loss computation while still allowing mixed precision in the
            # forward pass for performance.
            preds = preds.float()

            # Labels need to be long for the loss function, so we do nothing


            # Get the loss of the current batch
            batch_loss = self.loss_fn(preds, labels).item()

            # Accumulate validation loss for the epoch
            # (This works for model learning, as opposed to `dice_metric`)
            val_loss += batch_loss

            # MONAI best practice: decollate batch before applying per-sample transforms
            # (Creates a view of the batch tensors as lists of individual samples)
            # This enables per-sample metric calculation and metadata
            # Note that decollate_batch returns a `list[torch.Tensor]`
            # where each tensor is a single sample (e.g. shape (channels, depth, height, width))
            val_preds: list[torch.Tensor] = decollate_batch(preds)
            val_labels: list[torch.Tensor] = decollate_batch(labels)

            # apply per sample post processing
            val_preds = [self.pred_trans(p) for p in val_preds]
            val_labels = [self.label_trans(l) for l in val_labels]

            # Now both val_preds and val_labels are lists of one-hot encoded
            # tensors representing the predicted and true segmentation masks
            # for each sample in the batch.
            # Calculate and accumulate metrics (this works for human reading)
            self.dice_metric(y_pred=val_preds, y=val_labels)

        # end for batch

        avg_val_loss = val_loss / len(self.val_dl)

        if self._should_log_overlay(epoch):
            img = self._overlay_batch["image"].to(self.device)
            log_segmentation_overlay(
                self.writer,
                epoch,
                img,
                self._overlay_batch["label"].to(self.device),
                pred = self._run_small_inference(img)
            )
        # end if
        return avg_val_loss


    def _validate(self, epoch: int, best_dice: float = None) -> tuple[float, float, float, float]:
        # Activate inference mode
        self.model.eval()
        # Reset metric from previous epochs
        self.dice_metric.reset()

        with torch.inference_mode():
            avg_val_loss = self._run_val_epoch(epoch)

        # === DYNAMIC PER-CLASS DICE REPORTING ===
        # Perform average computation and extract per sample raw scores
        # Shape: (num_samples, num_foreground_classes)
        per_sample_dice: torch.Tensor = self.dice_metric.aggregate()

        # Count valid (non-NaN) samples per class to report effective sample size
        valid_mask = ~torch.isnan(per_sample_dice)
        valid_counts_per_class = valid_mask.sum(dim=0).cpu().tolist()

        # We compute the class means and then move the tensor to CPU, as it is no
        # longer needed for GPU computation
        per_class_dice: list = torch.nanmean(per_sample_dice, dim=0).cpu().tolist()
        # Also compute the global mean dice
        mean_dice: float = torch.nanmean(per_sample_dice).item()
        liver_dice: float = None
        if self.config.NUM_CLASSES == 3:
            liver_dice = per_class_dice[self.config.TUMOUR_CLASS_INDEX - 2]
        # -1 because per_class_dice doesn't include background class
        tumour_dice: float = per_class_dice[self.config.TUMOUR_CLASS_INDEX - 1]

        #    OR macro-average (unweighted across classes, often preferred for tumour papers):
        # mean_dice = float(torch.nanmean(torch.tensor(per_class_dice)).item())

        # Map foreground indices to names based on current config
        if self.config.NUM_CLASSES == 3:
            class_map = {0: "liver", 1: "tumour"}
        else:  # self.config.NUM_CLASSES == 2
            class_map = {0: "tumour"}

        log_parts = []
        count_parts = []
        for idx, name in class_map.items():
            dice_val = per_class_dice[idx]
            valid_n = valid_counts_per_class[idx]
            # for logger print
            log_parts.append("Dice %s: %.4f" % (name, dice_val))
            count_parts.append(f"{name} n={valid_n}")
            # for tensorboard
            self.writer.add_scalar(f"val/dice_{name}", dice_val, epoch)
            self.writer.add_scalar(f"val/valid_samples_{name}", valid_n, epoch)

        if self._should_notify(epoch):
            send_alert(
                title=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} completed",
                message="\n".join([
                    f"Validation loss: {avg_val_loss:.4f}",
                    f"Mean Dice: {mean_dice:.4f}",
                    *log_parts,
                    f"Best tumour Dice: {best_dice:.4f}",
                ])
            )

        logger.info(
            "(Val) Epoch %d -> Loss: %.4f | Dice Mean: %.4f | %s | Valid samples: %s",
            epoch + 1, avg_val_loss, mean_dice, " | ".join(log_parts), " | ".join(count_parts)
        )
        # ========================================

        self.scheduler.step()

        return avg_val_loss, mean_dice, liver_dice, tumour_dice


    def _run_epoch(self, epoch: int, best_dice: float = None) -> float:
        """Runs one full training + validation epoch.

        Params
        ------
            `epoch`: int
                Zero-indexed epoch number.

        Returns:
        ------
        `epoch_dice`: float
            Mean validation Dice score for the epoch.
        """
        epoch_start_time = time.time()

        logger.info("======Starting epoch %d/%d ======", epoch+1, self.config.NUM_EPOCHS)
        log_memory_usage(logger)

        # Train and validate one epoch
        avg_train_loss = self._train_epoch()
        avg_val_loss, epoch_dice, liver_dice, tumour_dice = self._validate(epoch, best_dice)


        # Get current learning rate for logging
        current_lr = self.optimizer.param_groups[0]['lr']

        # Calculate epoch duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # log to tensorboard
        self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        self.writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        self.writer.add_scalar("Metrics/Dice", epoch_dice, epoch)
        self.writer.add_scalar("Hyperparams/LR", current_lr, epoch)
        self.writer.add_scalar("Time/Epoch_Duration", epoch_duration, epoch)

        logger.info("  Train Loss: %f | Val Loss: %f | Dice: %f | LR: %f",
                    avg_train_loss, avg_val_loss, epoch_dice, current_lr)
        logger.info("  Epoch Time: %f seconds", epoch_duration)

        # Track history
        self.history["train_loss"].append(avg_train_loss)
        self.history["val_loss"].append(avg_val_loss)
        self.history["val_dice"].append(epoch_dice)

        return epoch_dice, liver_dice, tumour_dice

    def train(self, resume_path: Optional[Path] = None) -> None:
        '''
        Main training loop that iterates over epochs, performs training and validation,
        logs metrics, and handles checkpointing and early stopping.

        Params
        -----
        `resume_path`: Optional[Path]
            Path to a checkpoint file to resume training from. If None, starts from scratch.
        '''
        early_stopper = EarlyStopper(self)
        start_epoch = 0

        # Resume from checkpoint if provided
        # NOTE: Not other checkpoint is saved (e.g. for each epoch), so resume_path
        # is expected to be the best checkpoint path
        if resume_path is not None:
            self._resume_from_checkpoint(resume_path, early_stopper)
            start_epoch = early_stopper.best_epoch + 1
            logger.info("Resuming training from epoch %d (checkpoint had %d epochs)",
                        start_epoch, early_stopper.best_epoch)


        total_start_time = time.time()
        logger.info("Starting training for %d epochs...", self.config.NUM_EPOCHS)

        try:
            for epoch in range(start_epoch, self.config.NUM_EPOCHS):
                epoch_dice, liver_dice, tumour_dice = self._run_epoch(epoch, early_stopper.best_tumour_dice)

                # Returns true if the model hasn't improved for
                # `self.config.EARLY_STOPPING_PATIENCE` epochs
                if early_stopper(epoch, epoch_dice, liver_dice, tumour_dice):
                    break
            # End epoch loop
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Saving current model...")
            early_stopper.save_checkpoint(is_best=False, current_epoch=epoch)
            raise

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Convert seconds to hours/minutes for readability
        hours, rem = divmod(total_duration, 3600)
        minutes, seconds = divmod(rem, 60)

        self.writer.add_scalar("Final/Best_Mean_Dice", early_stopper.best_mean_dice, 0)
        self.writer.add_scalar("Final/Best_Liver_Dice", early_stopper.best_liver_dice, 0)
        self.writer.add_scalar("Final/Best_Tumour_Dice", early_stopper.best_tumour_dice, 0)
        self.writer.add_scalar("Final/Total_Duration_Hours", total_duration / 3600, 0)

        self.writer.close()

        logger.info("\n%s", "="*40)
        logger.info("Training Complete!")
        logger.info("Best Validation Tumour Dice: %f", early_stopper.best_tumour_dice)
        logger.info("Best Validation Liver Dice: %f", early_stopper.best_liver_dice)
        logger.info("Best Validation Mean Dice: %f", early_stopper.best_mean_dice)
        logger.info("Total Training Time: %dh %dm %ds", int(hours), int(minutes), seconds)
        logger.info("Checkpoint saved to: %s", early_stopper.checkpoint_path)
        logger.info("\n%s", "="*40)

class EarlyStopper:
    '''Helper class to manage early stopping logic and checkpoint saving.'''
    def __init__(self, builder: ModelBuilder):
        self.config = config.get()
        self.best_mean_dice = -1.0
        self.best_liver_dice = -1.0
        self.best_tumour_dice = -1.0
        self.best_epoch = -1
        self.epochs_no_improve = 0
        self.builder = builder
        self.checkpoint_path = self.config.CHECKPOINT_DIR / "best_model.pth"
        logger.info("EarlyStopper initialized with patience=%d and min_delta=%.4f",
                    self.config.EARLY_STOPPING_PATIENCE, self.config.EARLY_STOPPING_MIN_DELTA)

    def __call__(self, epoch: int, mean_dice: float, liver_dice: float, tumour_dice: float) -> bool:
        '''Checks if the current epoch's tumour dice score shows an improvement over
           the best recorded within a window.

        Params
        -----
        `epoch_dice`: float
            The mean Dice score for the current epoch.
        `liver_dice`: float
            The liver Dice score for the current epoch (if applicable).
        `tumour_dice`: float
            The tumour Dice score for the current epoch.

        Returns
        -----
        `bool`
            True if the model hasn't improved for at least
            `self.config.EARLY_STOPPING_PATIENCE` epochs
        '''
        if tumour_dice > self.best_tumour_dice + self.config.EARLY_STOPPING_MIN_DELTA:
            self.best_tumour_dice = tumour_dice
            self.best_mean_dice = mean_dice
            self.best_liver_dice = liver_dice
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            self.builder.writer.add_scalar("Improvement/Mean_Dice", mean_dice, epoch)
            self.builder.writer.add_scalar("Improvement/Liver_Dice", liver_dice, epoch)
            self.builder.writer.add_scalar("Improvement/Tumour_Dice", tumour_dice, epoch)
            self.save_checkpoint()
            return False  # Indicates improvement

        if self.epochs_no_improve < self.config.EARLY_STOPPING_PATIENCE:
            self.epochs_no_improve += 1
            logger.info("  -> No improvement (%d/%d epochs)",
                        self.epochs_no_improve, self.config.EARLY_STOPPING_PATIENCE)
            return False

        logger.info("  -> Early stopping triggered at epoch %d", epoch + 1)
        return True  # No improvement

    def save_checkpoint(self, is_best: bool = True, current_epoch: int = None):
        '''Saves a checkpoint of the current model state.

        Checkpoint schema includes:
        - Core: version, epoch, model_state_dict, optimizer_state_dict
        - Training state: scaler_state_dict, scheduler_state_dict (if available)
        - Metrics: best_dice, best_liver_dice, best_tumour_dice, epochs_no_improve
        - Reproducibility: RNG states (python, numpy, torch cpu/cuda)
        - Config: config_snapshot for compatibility checking
        - Optional: history, current_lr for audit/debug

        When extending this checkpoint dict in the future, consider adding:
        - New optimizer state (e.g., SAM, AdamW variants with additional buffers)
        - Loss component states (if using trainable loss weights or auxiliary losses)
        - Transform states (if using stateful augmentations like RandAugment)
        - Distributed training state (DDP/FSDP shards, world_size, rank)
        - Custom scheduler state (if not using standard PyTorch schedulers)
        '''
        path: Path = None
        if is_best:
            path = self.checkpoint_path
        else:
            path = self.config.CHECKPOINT_DIR / "last_epoch.pth"

        before_save_time = time.time()

        # Capture all CUDA RNG states for multi-GPU safety
        cuda_rng_states = None
        if torch.cuda.is_available():
            try:
                cuda_rng_states = torch.cuda.get_rng_state_all()
            except RuntimeError:
                # Fallback if get_rng_state_all fails in some environments
                cuda_rng_states = [torch.cuda.get_rng_state()]

        # np.random.get_state() returns a tuple containing a numpy array, which 
        # triggers a WeightsUnpickler error with weights_only=True.
        # We convert the array to a standard Python list for safe serialisation.
        np_state = np.random.get_state()
        safe_np_state = (
            np_state[0],
            np_state[1].tolist(),  # Convert uint32 array to list
            np_state[2],
            np_state[3],
            np_state[4]
        )

        torch.save({
            "version": self.config.VERSION,
            "epoch": self.best_epoch if is_best else current_epoch,
            # Weights and biases
            "model_state_dict": self.builder.model.state_dict(),
            # Variance, step counters, etc
            "optimizer_state_dict": self.builder.optimizer.state_dict(),
            "best_dice": self.best_mean_dice,
            "best_liver_dice": self.best_liver_dice,
            "best_tumour_dice": self.best_tumour_dice,
            # Counters and others for AMP
            "scaler_state_dict":
                self.builder.scaler.state_dict() if self.builder.scaler is not None else None,
            # Scheduler state for exact LR schedule resume (added in v2.3.2+)
            "scheduler_state_dict":
                self.builder.scheduler.state_dict() if self.builder.scheduler is not None else None,
            # early stopping state
            "epochs_no_improve": self.epochs_no_improve,
            # Config snapshot for reproducibility (can be used to log the exact
            # config that led to the best model)
            "config_snapshot": config.to_dict(),
            "interrupted": not is_best,
            # RNG states for deterministic restart (added in v2.3.2+)
            "rng_state": {
                "python": random.getstate(),
                "numpy": safe_np_state,
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": cuda_rng_states,
            },
            # Optional audit/debug fields (added in v2.3.2+)
            "history": self.builder.history,
            "current_lr": self.builder.optimizer.param_groups[0]["lr"] if self.builder.optimizer else None,
        }, path)
        after_save_time = time.time()
        save_duration = after_save_time - before_save_time
        if is_best:
            logger.info(
                "  -> New Best Model Saved (Dice: %.4f, Liver: %.4f, Tumour: %.4f) in %s (%.2f seconds)",
                self.best_mean_dice,
                self.best_liver_dice,
                self.best_tumour_dice,
                path,
                save_duration)
        else:
            logger.info("  -> Last Epoch Model Saved in %s (%.2f seconds)", path, save_duration)
