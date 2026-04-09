import torch
import torch.nn as nn
import torch.optim as optim
from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (Compose, EnsureTyped, LoadImaged, RandFlipd,
                              RandSpatialCropd, ScaleIntensityd,
                              ScaleIntensityRanged, SqueezeDimd, ToTensord)

from idssp.sonk.model.data import DataWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelBuilder:
    def __init__(self):
        self.train_dl = None
        self.val_dl = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("ModelBuilder initialized. Device set to:", self.device)

    def get_train_transforms(self):
        # Liver window: [-175, 250] HU is standard
        # [-175, 250] covers the liver and tumor intensities well, while also removing
        # some of the irrelevant background noise.
        # TODO: verify if this range is good for our data. We can adjust it based on the actual intensity distribution of the liver and tumor in our dataset.
        HU_MIN, HU_MAX = -175, 250

        return Compose([
            LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
            # CTs are in Hounsfield Units: -1000 (air), 0 (water), 40-60 (soft tissues), 100+ (bone)
            # we just need liver and tumor, so we can clip the intensities to a smaller range
            # We then scale this range to [0, 1] for better training stability.
            # Values outside the range are clipped.
            # a -> input range, b -> output range
            ScaleIntensityRanged(
                keys=["image"],
                a_min=HU_MIN, a_max=HU_MAX,
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
            EnsureTyped(keys=["image", "label"]),
            ToTensord(keys=['image', 'label'])
        ])

    def get_val_transforms(self):
        return Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175, a_max=250,
                b_min=0.0,  b_max=1.0,
                clip=True,
            ),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(-1, -1, 1),
                random_size=False,
            ),
            SqueezeDimd(keys=["image", "label"], dim=-1),
            # EnsureTyped already converts to tensor, so we don't need ToTensord here
            EnsureTyped(keys=["image", "label"]),
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
        # TODO: check if we can change this to be a CacheDataset
        val_ds = Dataset(data=val_files, transform=val_transforms)

        print("Creating training dataloader...")
        # TODO: change the number of workers when we have GPU access.
        # TODO: remove pin_memory=False when we have GPU access, as it can speed up data transfer to GPU.
        self.train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

        print("Creating validation dataloader...")
        self.val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            self.model.train()
            train_loss = 0

            # Training
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
            with torch.no_grad():
                for batch in self.val_dl:
                    images = batch["image"].to(self.device)
                    # NIfTI labels are often stored as float, but we need them as long for the loss function
                    labels = batch["label"].to(self.device).long()

                    preds = self.model(images)
                    loss = self.loss_fn(preds, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_dl)
            print(f"  Validation loss: {avg_val_loss:.4f}")
