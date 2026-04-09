import torch
import torch.optim as optim
from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandSpatialCropd, CenterSpatialCropd, SqueezeDimd, RandFlipd,
    EnsureTyped, Activations, AsDiscrete
)

from idssp.sonk.model.data import DataWrapper
from idssp.sonk import config

class ModelBuilder:
    def __init__(self):
        self.train_dl = None
        self.val_dl = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.device = torch.device(config.DEVICE)
        print("ModelBuilder initialized. Device set to:", self.device)

    def get_train_transforms(self):
        return Compose([
            LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
            # CTs are in Hounsfield Units: -1000 (air), 0 (water), 40-60 (soft tissues), 100+ (bone)
            # we just need liver and tumor, so we can clip the intensities to a smaller range
            # We then scale this range to [0, 1] for better training stability.
            # Values outside the range are clipped.
            # a -> input range, b -> output range
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config.HU_WINDOW_MIN, a_max=config.HU_WINDOW_MAX,
                b_min=0.0,  b_max=1.0,
                clip=True,
            ),
            # TODO: this is temporal. Replace with RandCropByPosNegLabel
            # By now, this just converts the 3D volume into a 2D slice by cropping the depth to 1
            # But it picks it randomly, so we get different slices each epoch. 
            # This is a simple way to augment our data and train on 2D slices until we have GPU access for 3D training.
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(-1, -1, 1),   # full H×W, depth=1
                random_size=False,
            ),
            # Just to remove the extra dimension added by the previous transform
            SqueezeDimd(keys=["image", "label"], dim=-1),  # (C,H,W,1) → (C,H,W)
            # Randomly flip the image and label horizontally and vertically with 50% probability each
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            # Ensure the data is in the correct tensor format
            EnsureTyped(keys=["image"]),
            # Labels must be long for the loss function
            # this ensures that the dtype is consistent across all transforms
            EnsureTyped(keys=["label"], dtype=torch.long),
        ])

    def get_val_transforms(self):
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

    def init_data_loaders(self, train_files, val_files):
        print("Creating training transforms object...")
        train_transforms = self.get_train_transforms()

        print("Creating validation transforms object...")
        val_transforms = self.get_val_transforms()

        print("Initializing training dataset...")
        # TODO: check if we can change this to be a CacheDataset
        train_ds = Dataset(data=train_files, transform=train_transforms)

        print("Initializing validation dataset...")
        val_ds = Dataset(data=val_files, transform=val_transforms)

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
            batch_size=config.BATCH_SIZE, # Keep val batch size 1 for safety
            shuffle=False, 
            num_workers=config.NUM_WORKERS, 
            pin_memory=config.PIN_MEMORY
        )

        print("Data loaders initialized successfully.")

    def init_model(self):
        print("Initializing model...")
        self.model = UNet(
            # TODO: By now (no GPU) we will train on 2D slices. Replace with 3D UNet when we have GPU access
            spatial_dims=2,
            in_channels=1,
            out_channels=3,  # background, liver, tumor
            #channels=(16, 32, 64, 128),
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
            num_res_units=2
        ).to(self.device)

        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS

        # 1. Post-processing for Predictions: Softmax -> Argmax -> One-Hot
        pred_trans = Compose([
            Activations(softmax=True),
            AsDiscrete(argmax=True, to_onehot=config.NUM_CLASSES)
        ])

        # 2. Post-processing for Labels: Index -> One-Hot
        # This converts labels like [0, 1, 2] into one-hot vectors [[1,0,0], [0,1,0], [0,0,1]]
        label_trans = AsDiscrete(to_onehot=config.NUM_CLASSES)

        # 3. Initialize Metric (No extra args needed here)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        best_val_dice = -1.0
        best_ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
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

            avg_train_loss = train_loss / len(self.train_dl)
            print(f"  Training loss: {avg_train_loss:.4f}")

            # Validation
            self.model.eval()
            val_loss = 0
            dice_metric.reset()
            
            with torch.no_grad():
                for batch in self.val_dl:
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)

                    preds = self.model(images)
                    val_loss += self.loss_fn(preds, labels).item()

                    # 4. Apply transformations before passing to metric
                    val_preds = pred_trans(preds)   # Model output -> One-Hot
                    val_labels = label_trans(labels) # Label indices -> One-Hot

                    # 5. Calculate Dice on matching one-hot tensors
                    dice_metric(y_pred=val_preds, y=val_labels)

            avg_val_loss = val_loss / len(self.val_dl)
            epoch_dice = dice_metric.aggregate().item()
            print(f"  Validation loss: {avg_val_loss:.4f} | Dice: {epoch_dice:.4f}")

            # Checkpoint saving
            if epoch_dice > best_val_dice:
                best_val_dice = epoch_dice
                print(f"  ✅ New best Dice: {best_val_dice:.4f}. Saving checkpoint...")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_dice": best_val_dice,
                }, best_ckpt_path)

        print(f"\nTraining complete. Best validation Dice: {best_val_dice:.4f}")
        print(f"Checkpoint saved to: {best_ckpt_path}")