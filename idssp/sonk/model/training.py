import time

import torch
import torch.optim as optim
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (Activations, AsDiscrete, Compose, EnsureTyped,
                              LoadImaged, RandCropByPosNegLabeld, RandFlipd,
                              ScaleIntensityRanged)
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger, log_memory_usage

logger = get_logger(__name__)

class ModelBuilder:
    def __init__(self):
        self.train_dl = None
        self.val_dl = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device(config.DEVICE)

        # Post-processing & Metrics
        self.pred_trans = Compose([
            Activations(softmax=True),
            AsDiscrete(argmax=True, to_onehot=config.NUM_CLASSES)
        ])
        self.label_trans = AsDiscrete(to_onehot=config.NUM_CLASSES)
        
        # include_background=False is standard for multi-class segmentation 
        # to avoid background dominating the metric.
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.history = {"train_loss": [], "val_loss": [], "val_dice": []}

        self.scaler = GradScaler(config.DEVICE) if self.device.type == 'cuda' else None
        '''Mixed precision training scaler, enabled only on CUDA for potential speed up.'''

        logger.info("ModelBuilder initialized. Device set to: %s", self.device)

    def get_train_transforms(self):
        '''Returns the transforms for the training data.'''
        return Compose([
            LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                # We clip the HU values to the defined liver/tumour range
                a_min=config.HU_WINDOW_MIN, a_max=config.HU_WINDOW_MAX,
                # We then scale that range to [0, 1] for better training stability.
                b_min=0.0,  b_max=1.0,
                # Values outside liver/tumour range are clipped.
                clip=True,
            ),

            # Sample patches with a balanced ratio of positive (tumor) and
            # negative (background) examples.
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.TRAIN_PATCH_SIZE,
                pos=1,   # sample from foreground (tumor) regions
                neg=1,   # sample from background regions
                num_samples=2,  # number of samples to generate per volume
                image_key="image",
                image_threshold=0,  # consider non-zero pixels as foreground for sampling
            ),

            # Randomly flip the image and label horizontally, vertically and
            # depth-wise with a 50% chance each to augment the data and improve generalization.
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            # Ensure the data is in the correct tensor format
            EnsureTyped(keys=["image"]),
            # Labels must be long for the loss function
            # this ensures that the dtype is consistent across all transforms
            EnsureTyped(keys=["label"], dtype=torch.long),
        ])

    def get_val_transforms(self):
        '''Returns the transforms for the validation data.'''
        compose_list = [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.HU_WINDOW_MIN, a_max=config.HU_WINDOW_MAX,
                b_min=0.0,  b_max=1.0,
                clip=True,
            ),
        ]

        if config.is_limited_env():
            logger.debug("Validation transforms: Using random crop for limited environment.")
            compose_list.extend([
                # To make sure we have some positive examples in the validation set
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=config.VAL_PATCH_SIZE,
                    pos=1,
                    neg=0,       # always sample from foreground for val
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                )
            ])
        # In GPU we can afford to run inference on the full volume, so we skip the cropping.

        compose_list.extend([
            EnsureTyped(keys=["image"]),
            EnsureTyped(keys=["label"], dtype=torch.long),
        ])
        return Compose(compose_list)

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
        logger.info("Creating training transforms object...")
        train_transforms = self.get_train_transforms()

        logger.info("Creating validation transforms object...")
        val_transforms = self.get_val_transforms()

        logger.info("Initializing training and validation datasets...")
        if config.is_limited_env():
            logger.info("Limited environment detected. Using regular Dataset.")
            train_ds = Dataset(data=train_files, transform=train_transforms)
            val_ds = Dataset(data=val_files, transform=val_transforms)
        else:
            logger.info("Sufficient resources detected. Using CacheDataset.")
            train_ds = CacheDataset(
                data=train_files,
                transform=train_transforms,
                cache_rate=1.0,
                num_workers=config.NUM_WORKERS
            )
            val_ds = CacheDataset(
                data=val_files,
                transform=val_transforms,
                cache_rate=1.0,
                num_workers=config.NUM_WORKERS
            )

        logger.info("Creating training dataloader...")
        self.train_dl = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        logger.info("Creating validation dataloader...")
        self.val_dl = DataLoader(
            val_ds,
            batch_size=config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        logger.info("Data loaders initialized successfully.")

    def init_model(self):
        '''Initializes the model (uNet), loss function, and optimizer.'''
        logger.debug("Initializing model...")
        self.model = UNet(
            spatial_dims=3,
            # Just 1 channel for the grayscale CT image. For RGB images, this would be 3.
            in_channels=1,
            out_channels=config.NUM_CLASSES,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            # TODO: This is temporal for low-resource environments.
            num_res_units=1
        ).to(self.device)

        log_memory_usage(logger, prefix="After model initialization: ")

        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            lambda_dice=1.0,
            lambda_ce=1.0
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )

        logger.info("Model initialized on %s", self.device)
        logger.info("Optimizer: AdamW | LR: %f | Weight Decay: 1e-5", config.LEARNING_RATE)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            # Minimise the validation loss, not maximise the dice score
            mode='min',
            factor=0.5,
            # Wait for at least 5 epochs without improvement before reducing LR
            patience=5
        )

        logger.info("Scheduler initialized: ReduceLROnPlateau (patience=5, factor=0.5)")

    def back_propagate(self, loss):
        '''
        Performs backpropagation with optional mixed precision scaling.

        Params
        ------
        `loss`: torch.Tensor
            The computed loss for the current batch, which will be back-propagated.
        '''
        if self.scaler is not None:
            # backpropagation with mixed precision scaling if enabled
            self.scaler.scale(loss).backward()

            # Unscales gradients for clipping
            self.scaler.unscale_(self.optimizer)

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Step the optimizer with scaled gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # Standard backpropagation for CPU or if mixed precision is not enabled
            loss.backward()
            self.optimizer.step()

    def train_epoch(self):
        self.model.train()
        train_loss = 0

        for batch in self.train_dl:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision training for potential speed up on CUDA
            with autocast(config.DEVICE, enabled=config.DEVICE == "cuda"):
                preds = self.model(images)
                loss = self.loss_fn(preds, labels)

            self.back_propagate(loss)

            train_loss += loss.item()

        return train_loss / len(self.train_dl)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0
        self.dice_metric.reset()
        inferer: SlidingWindowInferer = None

        if not config.is_limited_env():
            # Initialize sliding window inferer (run once outside loop)
            inferer = SlidingWindowInferer(
                roi_size=config.TRAIN_PATCH_SIZE,  # (96, 96, 96)
                sw_batch_size=4,      # Process 4 patches in parallel during inference
                overlap=0.25,         # 25% overlap for smooth stitching
                mode="gaussian",      # Weight centre of patch more heavily
                device=self.device,
                progress=False
            )
            logger.debug("Validation: Using SlidingWindowInferer for full-volume inference")

        with torch.no_grad():
            for batch in self.val_dl:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                with autocast(config.DEVICE, enabled=config.DEVICE == "cuda"):
                    if inferer is not None:
                        # GPU: Process full volume via sliding window
                        preds = inferer(images, self.model)
                    else:
                        # CPU: Images are already cropped to VAL_PATCH_SIZE, run directly
                        preds = self.model(images)

                # Cast predictions and labels to float32 before loss calculation.
                # Mixed precision (float16) can cause numerical instability and NaNs
                # during operations like Dice or cross-entropy. Using float32 ensures
                # stable loss computation while still allowing mixed precision in the
                # forward pass for performance.
                preds = preds.float()
                labels = labels.float()

                val_loss += self.loss_fn(preds, labels).item()

                # MONAI best practice: decollate batch before applying per-sample transforms
                val_preds = decollate_batch(preds)
                val_labels = decollate_batch(labels)

                val_preds = [self.pred_trans(p) for p in val_preds]
                val_labels = [self.label_trans(l) for l in val_labels]

                # Accumulate metrics
                self.dice_metric(y_pred=val_preds, y=val_labels)

        avg_val_loss = val_loss / len(self.val_dl)
        epoch_dice = self.dice_metric.aggregate().item()

        # Step scheduler based on validation loss
        self.scheduler.step(avg_val_loss)

        return avg_val_loss, epoch_dice

    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS

        best_val_dice = -1.0
        best_ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"

        # --- START TOTAL TIMER ---
        total_start_time = time.time()
        logger.info("Starting training for %d epochs...", num_epochs)

        for epoch in range(num_epochs):
            # --- START EPOCH TIMER ---
            epoch_start_time = time.time()

            logger.info("Starting epoch %d/%d", epoch+1, num_epochs)
            log_memory_usage(logger)

            avg_train_loss = self.train_epoch()
            avg_val_loss, epoch_dice = self.validate_epoch()

            # Get current learning rate for logging
            current_lr = self.optimizer.param_groups[0]['lr']

            # --- CALCULATE EPOCH TIME ---
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            logger.info("  Train Loss: %f | Val Loss: %f | Dice: %f | LR: %f", avg_train_loss, avg_val_loss, epoch_dice, current_lr)
            logger.info("  Epoch Time: %f seconds", epoch_duration)

            # Track history
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_dice"].append(epoch_dice)

            # Checkpoint saving
            if epoch_dice > best_val_dice:
                best_val_dice = epoch_dice
                # TODO: add load_checkpoint method
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_dice": best_val_dice,
                }, best_ckpt_path)
                logger.info("  -> New Best Model Saved (Dice: %f)", best_val_dice)

        # --- END TOTAL TIMER ---
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Convert seconds to hours/minutes for readability
        hours, rem = divmod(total_duration, 3600)
        minutes, seconds = divmod(rem, 60)

        logger.info("\n%s", "="*40)
        logger.info("Training Complete!")
        logger.info("Best Validation Dice: %f", best_val_dice)
        logger.info("Total Training Time: %dh %dm %ds", int(hours), int(minutes), seconds)
        logger.info("Checkpoint saved to: %s", best_ckpt_path)
        logger.info("\n%s", "="*40)
