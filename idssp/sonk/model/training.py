import datetime
import time
from pathlib import Path

import torch
import torch.optim as optim
from monai.data import (CacheDataset, DataLoader, Dataset, PersistentDataset,
                        decollate_batch)
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (Activations, AsDiscrete, Compose,
                              CropForegroundd, EnsureTyped, LoadImaged,
                              Orientationd, RandCropByPosNegLabeld, RandFlipd,
                              ScaleIntensityRanged, Spacingd, SpatialPadd,
                              Transform)
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger, log_memory_usage
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
        self.device = torch.device(config.DEVICE)
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Post-processing & Metrics
        self.pred_trans = Compose([
            # Apply softmax Transform to get class probabilities for each voxel
            Activations(softmax=True),
            # Select the class with the highest probability for each voxel
            # and convert to one-hot encoding
            AsDiscrete(argmax=True, to_onehot=config.NUM_CLASSES)
        ])
        '''`Compose` of transforms applied to model predictions so we have
        a probability distribution [0-1] for each voxel per class (`config.NUM_CLASSES`).
        To be used in validation step.
        
        1. `Activations(softmax=True)`: Applies the softmax function to the raw model
           outputs (logits) to convert them into class probabilities for each voxel.
        
        2. `AsDiscrete(argmax=True, to_onehot=config.NUM_CLASSES)`: This transform
           first applies argmax to select the class with the highest probability
           for each voxel, then it converts these class labels into one-hot
           encoding format.
        '''

        self.label_trans = AsDiscrete(to_onehot=config.NUM_CLASSES)
        '''Transform applied to ground truth labels so they're represented as one-hot
           encoded tensors to each class before metric calculation.
           It only contains `AsDiscrete(to_onehot=config.NUM_CLASSES)`, which converts
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
           class, e.g. shape `[batch_size, config.NUM_CLASSES - 1]`).

           To be used in validation step.'''

        self.history = {"train_loss": [], "val_loss": [], "val_dice": []}

        # So we can use float16 mixed precision on CUDA
        # (Multiplies loss by a scale factor to prevent underflow, and unscales
        # gradients before the optimizer step)
        self.scaler = GradScaler('cuda') if config.DEVICE == 'cuda' else None
        '''Mixed precision training scaler, enabled only on CUDA for potential speed up.'''

        # tensorboard writer for logging training metrics
        # TODO: Use run_id to store global logs (tensorboard, log, checkpoints)
        self.writer = SummaryWriter(log_dir=
                                    str(config.LOG_DIR / "tensorboard" / self.run_id))
        self.writer.add_hparams(
            { # h param dict
                "environment": config.ENV,
                "batch_size": config.BATCH_SIZE,
                "num_classes": config.NUM_CLASSES,
                "precision": "float16" if self.scaler is not None else "float32",
            }, { # metric dict
                "val/dice_mean": 0.0,
                "val/dice_liver": 0.0,
                "val/dice_tumour": 0.0,
                "train/loss": 0.0,
            }
        )
        logger.info("TensorBoard writer initialised at: %s", self.writer.log_dir)

        logger.info("ModelBuilder initialized. Device set to: %s", self.device)

    def _get_deterministic_transforms(self) -> list[Transform]:
        '''
        Returns the deterministic transforms that are applied to all training samples.
        This includes loading, orientation, resampling, and intensity scaling.
        These transforms are applied once and can be cached when using PersistentDataset.
        '''
        # WARNING: If you modify the transforms and you're on a GPU environment,
        # make sure to clear (delete) the PersistentDataset folder
        # to avoid issues with cached data that doesn't match the new transforms.
        return [
            LoadImaged(keys=['image', 'label'], ensure_channel_first=True),

            # Ensure consistent orientation (LAS)
            Orientationd(keys=["image", "label"], axcodes="LAS", labels=None),

            # We standardise (resample) spacing across all volumes so a tumour of a given
            # voxel size appears at the same scale regardless of the original scan resolution.
            Spacingd(
                keys=["image", "label"],
                pixdim=config.ISO_SPACING,
                # bilinear (average) interpolation for CT
                # nearest for labels to avoid creating non-integer class values
                mode=("bilinear", "nearest")
            ),

            # Changes CT intensity scale to something more meaningful for the model
            ScaleIntensityRanged(
                keys=["image"],
                # We clip the HU values to the defined liver/tumour range
                a_min=config.HU_WINDOW_MIN, a_max=config.HU_WINDOW_MAX,
                # We then scale that range to [0, 1] for better training stability.
                b_min=0.0,  b_max=1.0,
                # Values outside liver/tumour range are clipped.
                clip=True,
            ),

            # Remove excess background to reduce memory usage and speed up training.
            # (It effectively crops the image to a bounding box around the
            # liver + tumour region, with a margin of 10 voxels.)
            # THIS WILL CHANGE THE SHAPE OF THE INPUT DATA
            # Remove on inference if fixed-size validation is needed
            # TODO: Consider using "image" as source_key (edge case: small peripheral tumours)
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                # Background is 0
                select_fn=lambda x: x > 0,
                margin=10,
                allow_smaller=True
            ),

            # After cropping foreground we might end up with volumes smaller
            # than `config.TRAIN_PATCH_SIZE`, which will cause issues
            # with `RandCropByPosNegLabeld`. To avoid that, we 0-pad the volumes to ensure
            # they are at least as large as the training patch size.
            # (it is conceptually just adding air)
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=config.TRAIN_PATCH_SIZE,
                mode="constant",
                value=0
            )
        ]

    def get_train_transforms(self) -> tuple[Compose, Compose]:
        '''
        Returns the transforms for the training data. Note that the result will
        differ based on the environment:
        - In a limited environment (e.g. CPU) or when using CacheDataset
          (`config.USE_CACHE_DATASET`), all transforms (including random cropping)
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
        deterministic_transforms = self._get_deterministic_transforms()

        random_transforms = [
            # Sample patches with a 2:1 ratio of positive (tumor/liver) and
            # negative (background) examples.
            # This is because of voxel imbalance.
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.TRAIN_PATCH_SIZE,
                pos=2,
                neg=1,
                # number of samples to generate per volume
                num_samples=config.RAND_CROP_NUM_SAMPLES,
                image_key="image",
                # Negative samples are taken on tissue ( HU > 0). Used with image_key
                image_threshold=0,
            ),

            # Randomly flip the image and label horizontally, vertically and
            # depth-wise with a 50% chance each to augment the data and improve generalization.
            # NOTE: Liver sits always on the left side of the image. 
            # Flipping along x-axis would create unrealistic samples.
            # RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            # Converts data to PyTorch tensors
            EnsureTyped(keys=["image"]),
            # Labels must be long for the loss function
            # (Sometimes it is loaded as float)
            EnsureTyped(keys=["label"], dtype=torch.long),
        ]

        if config.is_limited_env():
            logger.debug("Using random crop in main transform pipeline for limited "
            "environment")
            # When we won't use Persistent or CacheDataset (which applies transforms once
            # and caches the results), we can include the random cropping in the
            # main transform pipeline.
            deterministic_transforms.extend(random_transforms)
            random_transforms = []  # No separate random transforms needed

        return Compose(deterministic_transforms), Compose(random_transforms)

    def get_val_transforms(self) -> Compose:
        '''Returns the transforms for the validation data.'''
        compose_list = self._get_deterministic_transforms()

        if config.is_limited_env():
            logger.warning("Validation transforms: Using random crop for limited environment.")
            compose_list.extend([
                # On limited GPU or CPU we apply cropping so we don't overload memory.
                # ByPosNeg to make sure we have some positive examples in the validation set
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=config.VAL_PATCH_SIZE,
                    pos=1,
                    neg=0,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                )
            ])

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
        val_transforms = self.get_val_transforms()

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
            if config.USE_CACHE_DATASET:
                logger.debug("Sufficient resources detected. Using MONAI CacheDataset.")
                train_ds = AugmentedDataset(
                    CacheDataset(
                        data=train_files,
                        transform=train_det_trans,
                        cache_rate=1.0,
                        num_workers=config.NUM_WORKERS,
                    ),
                    train_ran_trans
                )
                val_ds = CacheDataset(
                    data=val_files,
                    transform=val_transforms,
                    cache_rate=1.0,
                    num_workers=config.NUM_WORKERS,
                )
            else:
                # TODO: implement a hashing mechanism to detect changes in transforms
                # (use the hash as dir name)
                logger.info("Sufficient resources detected. Using PersistentDataset.")
                train_ds = AugmentedDataset(
                    PersistentDataset(
                        data=train_files,
                        transform=train_det_trans,
                        cache_dir=str(config.PERSISTENT_DATASET_DIR /
                                      f"hmin_{config.HU_WINDOW_MIN}"
                                      f"_hmax_{config.HU_WINDOW_MAX}_train_cache")
                    ),
                    train_ran_trans
                )
                val_ds = PersistentDataset(
                    data=val_files,
                    transform=val_transforms,
                    cache_dir=str(config.PERSISTENT_DATASET_DIR /
                                  f"hmin_{config.HU_WINDOW_MIN}"
                                  f"_hmax_{config.HU_WINDOW_MAX}_val_cache")
                )

        logger.debug("Creating training dataloader...")
        self.train_dl = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

        logger.debug("Creating validation dataloader...")
        self.val_dl = DataLoader(
            val_ds,
            batch_size=config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

        logger.debug("Data loaders initialized successfully.")

    def init_model(self):
        '''Initializes the model (uNet), loss function, and optimizer.'''
        logger.debug("Initializing model...")
        # TODO: change for a SegResNet
        self.model = UNet(
            spatial_dims=3,
            # Just 1 channel for the grayscale CT image.
            in_channels=1,
            out_channels=config.NUM_CLASSES,
            # channels=(64, 128, 256, 512)
            channels=(32, 64, 128, 256),
            # One stride per downsampling transition: len(strides) == len(channels) - 1
            strides=(2, 2, 2),
            num_res_units=1 if config.is_limited_env() else 2,
            # batch norm isn't useful for 3d images (usually batches aren't longer than 4)
            norm="INSTANCE",
            act="PRELU"
        ).to(self.device)

        log_memory_usage(logger, prefix="After model initialization: ")

        # Cross entropy: voxel-wise classification (smooth gradients)
        # Dice: measures the overlap between predicted and true segmentation masks.
        # TODO: consider adding ce_weight=[0.0, 1.0, 3.0]
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            # to match `self.dice_metric`
            include_background=False,
            # dice weight
            lambda_dice=1.0,
            # CE weight
            lambda_ce=1.0
        )
        self.optimizer = optim.AdamW(
            # weights and biases
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )

        if not config.is_limited_env():
            # Initialize sliding window inferer (run validation on full volumes
            # via sliding window patches)
            self.inferer = SlidingWindowInferer(
                # MUST be the same as the training patch size
                roi_size=config.TRAIN_PATCH_SIZE,
                # Process 4 patches in parallel
                sw_batch_size=4,
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
            logger.debug("Using SlidingWindowInferer for full-volume inference")

        logger.info("Model initialized on %s", self.device)
        logger.info("Optimizer: AdamW | LR: %f | Weight Decay: 1e-5", config.LEARNING_RATE)

        # Change to cosine annealing scheduler with warm restarts on ResUNETR
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            # Minimise the validation loss, not maximise the dice score
            mode='min',
            # Halves the learning rate when the validation loss plateaus
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
            # Multiplies the loss by the scale factor to prevent underflow during backpropagation
            # scaled_loss = original_loss * scale_factor
            self.scaler\
                    .scale(loss)\
                    .backward()
            # ^ And then back propagate (∇(scaled_loss) )

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

        else:
            # Standard backpropagation for CPU or if mixed precision is not enabled
            # calc gradients
            loss.backward()
            # Update weights
            self.optimizer.step()

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

        for i, batch in enumerate(self.train_dl):
            # Batch is a dictionary with keys "image" and "label", each containing a tensor of shape
            # (batch_size, channels, depth, height, width).

            # "batch" will contain conf.RAND_CROP_NUM_SAMPLES * conf.BATCH_SIZE
            # patches in total (e.g. batch_size=16 if BATCH_SIZE=2 and RAND_CROP_NUM_SAMPLES=8).

            logger.debug("Training batch: %d/%d. Batch (image) shape: %s",
                         i + 1, len(self.train_dl),
                         batch["image"].shape)
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Set gradients to None
            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision training for potential speed up on CUDA
            with autocast(device_type="cuda", enabled=config.DEVICE == "cuda"):
                predictions = self.model(images)
                loss = self.loss_fn(predictions, labels)

            self.back_propagate(loss)

            train_loss += loss.item()
            logger.debug("  Batch Loss: %f | Cumulative Loss: %f", loss.item(), train_loss)

        return train_loss / len(self.train_dl)

    def _run_val_epoch(self, epoch: int):
        val_loss = 0

        for batch_idx, batch in enumerate(self.val_dl):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logger.debug("Validation batch: %d/%d. Batch (image) shape: %s",
                        batch_idx + 1, len(self.val_dl),
                        batch["image"].shape)

            with autocast(device_type="cuda", enabled=config.DEVICE == "cuda"):
                if self.inferer is not None:
                    try:
                        # GPU: Process full volume via sliding window
                        preds = self.inferer(images, self.model)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        logger.warning("OOM during validation, falling back to sw_batch_size=1")
                        fallback = SlidingWindowInferer(
                            roi_size=config.TRAIN_PATCH_SIZE,
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
            logger.debug("  Batch Loss: %f | Cumulative Loss: %f", batch_loss, val_loss)

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

            # --- LOG OVERLAY ONCE PER EPOCH (first batch only) ---
            # Log every config.FIGURE_EPOCH_INTERVAL epochs
            if epoch % config.FIGURE_EPOCH_INTERVAL == 0 and batch_idx == 0:
                log_segmentation_overlay(self.writer, epoch, images, labels, preds)
            # -----------------------------------------------------
        # end for batch
        return val_loss / len(self.val_dl)


    def _validate(self, epoch: int):
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

        #    OR macro-average (unweighted across classes, often preferred for tumour papers):
        # mean_dice = float(torch.nanmean(torch.tensor(per_class_dice)).item())

        # Map foreground indices to names based on current config
        if config.NUM_CLASSES == 3:
            class_map = {0: "liver", 1: "tumour"}
        else:  # config.NUM_CLASSES == 2
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

        logger.info(
            "(Val) Epoch %d -> Loss: %.4f | Dice Mean: %.4f | %s | Val samples: %s",
            epoch + 1, avg_val_loss, mean_dice, " | ".join(log_parts), " | ".join(count_parts)
        )
        # ========================================

        # Step scheduler based on validation loss
        # TODO: Check if it's better to step based on tumour DiceCE score
        # (instead of the average loss)

        # Claude comment:
        # Your scheduler steps on avg_val_loss, and early stopping triggers on epoch_dice.
        # These will usually move in the same direction, but not always — particularly
        # on LiTS where tumour Dice can plateau while loss keeps improving slightly
        # due to the liver class. This is not a crash risk but it is a logical inconsistency.
        # Since you care most about tumour Dice, consider stepping the scheduler on
        # tumour Dice specifically (negated, since the scheduler is in mode='min'),
        # or switch to mode='max' and pass Dice directly.
        self.scheduler.step(avg_val_loss)

        return avg_val_loss, mean_dice


    def _run_epoch(self, epoch: int) -> float:
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

        logger.info("======Starting epoch %d/%d ======", epoch+1, config.NUM_EPOCHS)
        log_memory_usage(logger)

        # Train and validate one epoch
        avg_train_loss = self._train_epoch()
        avg_val_loss, epoch_dice = self._validate(epoch)

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

        return epoch_dice

    def train(self) -> None:
        '''
        Main training loop that iterates over epochs, performs training and validation,
        logs metrics, and handles checkpointing and early stopping.
        '''
        early_stopper = EarlyStopper(self)

        total_start_time = time.time()
        logger.info("Starting training for %d epochs...", config.NUM_EPOCHS)

        try:
            for epoch in range(config.NUM_EPOCHS):
                epoch_dice = self._run_epoch(epoch)

                # Returns true if the model hasn't improved for
                # `config.EARLY_STOPPING_PATIENCE` epochs
                if early_stopper(epoch, epoch_dice):
                    break
            # End epoch loop
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Saving current model...")
            # TODO: Add a --resume flag to load the last checkpoint and continue training
            # (This also works for broken training runs, e.g. due to OOM errors or preemption on a cluster)
            early_stopper.save_checkpoint(is_best=False, current_epoch=epoch)
            raise

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Convert seconds to hours/minutes for readability
        hours, rem = divmod(total_duration, 3600)
        minutes, seconds = divmod(rem, 60)

        self.writer.add_scalar("Final/Best_Dice", early_stopper.best_dice, 0)
        self.writer.add_scalar("Final/Total_Duration_Hours", total_duration / 3600, 0)

        self.writer.close()

        logger.info("\n%s", "="*40)
        logger.info("Training Complete!")
        logger.info("Best Validation Dice: %f", early_stopper.best_dice)
        logger.info("Total Training Time: %dh %dm %ds", int(hours), int(minutes), seconds)
        logger.info("Checkpoint saved to: %s", early_stopper.checkpoint_path)
        logger.info("\n%s", "="*40)

class EarlyStopper:
    '''Helper class to manage early stopping logic and checkpoint saving.'''
    def __init__(self, builder: ModelBuilder):
        self.best_dice = -1.0
        self.best_epoch = -1
        self.epochs_no_improve = 0
        self.builder = builder
        self.checkpoint_path = config.CHECKPOINT_DIR / (self.builder.run_id + "_best_model.pth")

    def __call__(self, epoch: int, epoch_dice: float) -> bool:
        '''Checks if the current epoch's Dice score shows an improvement over
           the best recorded within a window

        Params
        -----
        `epoch_dice`: float
            The mean Dice score for the current epoch.

        Returns
        -----
        `bool`
            True if the model hasn't improved for at least `config.EARLY_STOPPING_PATIENCE` epochs
        '''
        if epoch_dice > self.best_dice + config.EARLY_STOPPING_MIN_DELTA:
            self.best_dice = epoch_dice
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            self.save_checkpoint()
            return False  # Indicates improvement

        if self.epochs_no_improve < config.EARLY_STOPPING_PATIENCE:
            self.epochs_no_improve += 1
            logger.info("  -> No improvement (%d/%d epochs)",
                        self.epochs_no_improve, config.EARLY_STOPPING_PATIENCE)
            return False

        logger.info("  -> Early stopping triggered at epoch %d", epoch + 1)
        return True  # No improvement

    def save_checkpoint(self, is_best: bool = True, current_epoch: int = None):
        '''Saves a checkpoint of the current model state.'''
        path: Path = None
        if is_best:
            path = self.checkpoint_path
        else:
            path = config.CHECKPOINT_DIR / (self.builder.run_id + "_last_epoch.pth")

        before_save_time = time.time()
        torch.save({
            "epoch": self.best_epoch if is_best else current_epoch,
            # Weights and biases
            "model_state_dict": self.builder.model.state_dict(),
            # Variance, step counters, etc
            "optimizer_state_dict": self.builder.optimizer.state_dict(),
            "best_dice": self.best_dice,
            # Counters and others for AMP
            "scaler_state_dict":
                self.builder.scaler.state_dict() if self.builder.scaler is not None else None,
            # Config snapshot for reproducibility (can be used to log the exact
            # config that led to the best model)
            "config_snapshot": config.to_dict(),
            "interrupted": not is_best
        }, path)
        after_save_time = time.time()
        save_duration = after_save_time - before_save_time
        if is_best:
            logger.info("  -> New Best Model Saved (Dice: %.4f) in %s (%.2f seconds)", self.best_dice, path, save_duration)
        else:
            logger.info("  -> Last Epoch Model Saved in %s (%.2f seconds)", path, save_duration)
