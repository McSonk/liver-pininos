import time

import torch
import torch.optim as optim
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (Activations, AsDiscrete, CenterSpatialCropd,
                              Compose, EnsureTyped, LoadImaged, RandFlipd,
                              RandSpatialCropd, ScaleIntensityRanged,
                              SqueezeDimd)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from idssp.sonk import config


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

        print("ModelBuilder initialized. Device set to:", self.device)

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
            # TODO: this is temporal. Replace with RandCropByPosNegLabel
            # https://monai-dev.readthedocs.io/en/stable/transforms.html#monai.transforms.RandCropByPosNegLabel
            # By now, this just converts the 3D volume into a 2D slice by cropping the depth to 1
            # But it picks it randomly, so we get different slices each epoch.
            # (so it might pick a slice outside the liver/tumour region)
            # This is a simple way to augment our data and train on 2D slices
            # until we have GPU access for 3D training.
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(-1, -1, 1),   # full H×W, depth=1
                random_size=False,
            ),
            # Just to remove the extra dimension added by the previous transform
            SqueezeDimd(keys=["image", "label"], dim=-1),  # (C,H,W,1) → (C,H,W)
            # Randomly flip the image and label horizontally and vertically with
            # 50% probability each
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            # Ensure the data is in the correct tensor format
            EnsureTyped(keys=["image"]),
            # Labels must be long for the loss function
            # this ensures that the dtype is consistent across all transforms
            EnsureTyped(keys=["label"], dtype=torch.long),
        ])

    def get_val_transforms(self):
        '''Returns the transforms for the validation data.'''
        return Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.HU_WINDOW_MIN, a_max=config.HU_WINDOW_MAX,
                b_min=0.0,  b_max=1.0,
                clip=True,
            ),
            # Deterministic crop for reliable validation metrics
            CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(-1, -1, 1),
            ),
            SqueezeDimd(keys=["image", "label"], dim=-1),
            EnsureTyped(keys=["image"]),
            EnsureTyped(keys=["label"], dtype=torch.long),
        ])

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
        print("Creating training transforms object...")
        train_transforms = self.get_train_transforms()

        print("Creating validation transforms object...")
        val_transforms = self.get_val_transforms()

        print("Initializing training and validation datasets...")
        if config.is_limited_env():
            print("Limited environment detected. Using regular Dataset.")
            train_ds = Dataset(data=train_files, transform=train_transforms)
            val_ds = Dataset(data=val_files, transform=val_transforms)
        else:
            print("Sufficient resources detected. Using CacheDataset.")
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

        print("Creating training dataloader...")
        self.train_dl = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        print("Creating validation dataloader...")
        self.val_dl = DataLoader(
            val_ds,
            batch_size=config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        print("Data loaders initialized successfully.")

    def init_model(self):
        '''Initializes the model (uNet), loss function, and optimizer.'''
        print("Initializing model...")
        self.model = UNet(
            # TODO: By now (no GPU) we will train on 2D slices. Replace with 3D UNet when we have GPU access
            spatial_dims=2,
            # Just 1 channel for the grayscale CT image. For RGB images, this would be 3.
            in_channels=1,
            out_channels=config.NUM_CLASSES,
            # TODO: these are just example channel sizes. We can experiment with different configurations later.
            #channels=(16, 32, 64, 128),
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
            num_res_units=2
        ).to(self.device)

        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )

        print(f"Model initialized on {self.device}")
        print(f"Optimizer: AdamW | LR: {config.LEARNING_RATE} | Weight Decay: 1e-5")

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            # Minimise the validation loss, not maximise the dice score
            mode='min',
            factor=0.5,
            # Wait for at least 5 epochs without improvement before reducing LR
            patience=5
        )

        print("Scheduler initialized: ReduceLROnPlateau (patience=5, factor=0.5)")

    def train_epoch(self):
        self.model.train()
        train_loss = 0

        for batch in self.train_dl:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.loss_fn(preds, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / len(self.train_dl)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0
        self.dice_metric.reset()

        with torch.no_grad():
            for batch in self.val_dl:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                preds = self.model(images)
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
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # --- START EPOCH TIMER ---
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            avg_train_loss = self.train_epoch()
            avg_val_loss, epoch_dice = self.validate_epoch()
            
            # Get current learning rate for logging
            current_lr = self.optimizer.param_groups[0]['lr']

            # --- CALCULATE EPOCH TIME ---
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {epoch_dice:.4f} | LR: {current_lr:.6f}")
            print(f"  Epoch Time: {epoch_duration:.2f} seconds")

            # Track history
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_dice"].append(epoch_dice)

            # Checkpoint saving
            if epoch_dice > best_val_dice:
                best_val_dice = epoch_dice
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_dice": best_val_dice,
                }, best_ckpt_path)
                print(f"  -> New Best Model Saved (Dice: {best_val_dice:.4f})")

        # --- END TOTAL TIMER ---
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # Convert seconds to hours/minutes for readability
        hours, rem = divmod(total_duration, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"\n{'='*40}")
        print(f"Training Complete!")
        print(f"Best Validation Dice: {best_val_dice:.4f}")
        print(f"Total Training Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"Checkpoint saved to: {best_ckpt_path}")
        print(f"{'='*40}")
