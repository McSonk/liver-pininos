import torch
from monai.transforms import (Compose, CropForegroundd, EnsureTyped, LoadImaged,
                              Orientationd, RandCropByPosNegLabeld, RandFlipd,
                              RandGaussianNoised, RandRotated,
                              RandScaleIntensityd, RandZoomd,
                              ScaleIntensityRanged, Spacingd, SpatialPadd, Activations, AsDiscrete,
                              Transform)
from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)

def get_activations_transforms(num_classes: int) -> Compose:
    '''Returns the transforms that are applied to the model's raw output logits
       during validation/test evaluation to convert them into class predictions

       Params
       ------
       `num_classes`: int
            The number of classes in the segmentation task. This is used to determine
            the number of channels in the one-hot encoded output.
       '''
    return Compose([
        # Apply softmax Transform to get class probabilities for each voxel
        Activations(softmax=True),
        # Select the class with the highest probability for each voxel
        # and convert to one-hot encoding
        AsDiscrete(argmax=True, to_onehot=num_classes)
    ])

def get_label_transform(num_classes: int) -> AsDiscrete:
    '''Returns the transforms that are applied to the ground truth labels
       during validation/test evaluation to ensure they are in the correct format
       for metric calculation.

       Params
       ------
       `num_classes`: int
            The number of classes in the segmentation task. This is used to determine
            the number of channels in the one-hot encoded output.
       '''
    return AsDiscrete(to_onehot=num_classes)


def get_deterministic_transforms(config_obj: config.Config, include_inference: bool = False) -> list[Transform]:
    '''
    Returns the deterministic transforms that are applied to all training samples.
    This includes loading, orientation, resampling, and intensity scaling.
    These transforms are applied once and can be cached when using PersistentDataset.

    Params
    ------
    `config_obj`: config.Config
         The configuration object containing the necessary parameters for the transforms.
    `include_inference`: bool, optional
         Whether to include inference data in the transforms, by default False
    '''
    # WARNING: If you modify the transforms and you're on a GPU environment,
    # make sure to clear (delete) the PersistentDataset folder
    # to avoid issues with cached data that doesn't match the new transforms.

    keys = ['image', 'label']
    if include_inference:
        keys.append('inference')

    modes = ['bilinear', 'nearest']
    if include_inference:
        # Inference is a label-like volume, so use nearest interpolation
        modes.append('nearest')
    return [
        LoadImaged(keys=keys, ensure_channel_first=True),

        # Ensure consistent orientation (LAS)
        Orientationd(keys=keys, axcodes="LAS", labels=None),

        # We standardise (resample) spacing across all volumes so a tumour of a given
        # voxel size appears at the same scale regardless of the original scan resolution.
        Spacingd(
            keys=keys,
            pixdim=config_obj.ISO_SPACING,
            # bilinear (average) interpolation for CT
            # nearest for labels to avoid creating non-integer class values
            mode=modes
        ),

        # Changes CT intensity scale to something more meaningful for the model
        ScaleIntensityRanged(
            keys=["image"],
            # We clip the HU values to the defined liver/tumour range
            a_min=config_obj.HU_WINDOW_MIN, a_max=config_obj.HU_WINDOW_MAX,
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
        CropForegroundd(
            keys=keys,
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
            keys=keys,
            spatial_size=config_obj.TRAIN_PATCH_SIZE,
            mode="constant",
            value=0
        )
    ]

def get_random_transforms(config_obj: config.Config) -> list[Transform]:
    '''Returns the random transforms that are applied to training samples on-the-fly.
       These transforms introduce variability into the training data and help
       improve generalization.
       
       Returns
       -------
         list[Transform]
              A list of MONAI Transform objects that are applied randomly to training samples.
              It already includes EnsureTyped to convert data to PyTorch tensors at the end,
              so the output of this function is ready to be fed into the model.'''
    return [
        # Sample patches with a given ratio of positive (tumor/liver) and
        # negative (background) examples.
        # This is because of voxel imbalance. (we want to maximise the likelihood
        # of sampling tumour voxels, which are the most important to learn,
        # while still including some negative samples to learn the background)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config_obj.TRAIN_PATCH_SIZE,
            pos=2,
            neg=1,
            # number of samples to generate per volume
            num_samples=config_obj.RAND_CROP_NUM_SAMPLES,
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
        # TODO: study this
        # Intensity augmentations: simulate CT scanner noise & protocol variations
        # Applied post-normalisation ([0,1] range) to act as implicit regularisers
        # and improve generalisation across heterogeneous clinical scanners.
        RandGaussianNoised(
            keys=["image"],
            mean=0.0,
            std=0.010,      # ~1.0% of the [0, 1] normalised range
            prob=0.10
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=0.05,    # ±5% intensity variation
            prob=0.10
        ),
        RandRotated(
            keys=["image", "label"],
            range_x=0.2, range_y=0.2, range_z=0.2,
            prob=0.3,
            mode=("bilinear", "nearest"),
            padding_mode=("zeros", "zeros")
        ),
        RandZoomd(
            keys=["image", "label"], 
            prob=0.3, 
            min_zoom=0.9, 
            max_zoom=1.1,
            mode=["trilinear", "nearest"]
        ),
        # Converts data to PyTorch tensors
        EnsureTyped(keys=["image"]),
        # Labels must be long for the loss function
        # (Sometimes it is loaded as float)
        EnsureTyped(keys=["label"], dtype=torch.long),
    ]

def get_validation_transforms(config_obj: config.Config) -> Compose:
    '''Returns the transforms that are applied to validation/test samples.
       These transforms should be deterministic and not include any random augmentations,
       to ensure consistent evaluation.'''
    # For validation, we typically want to apply the same deterministic transforms as training
    # (loading, orientation, resampling, intensity scaling, cropping/padding) but without
    # the random augmentations. This ensures that the model is evaluated on data that is
    # processed in the same way as training data, but without any additional variability.
    deterministic = get_deterministic_transforms(config_obj)
    if config.is_limited_env():
        logger.warning("Validation transforms: Using random crop for limited environment.")
        deterministic.extend([
            # On limited GPU or CPU we apply cropping so we don't overload memory.
            # ByPosNeg to make sure we have some positive examples in the validation set
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config_obj.VAL_PATCH_SIZE,
                pos=1,
                neg=0,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            )
        ])
    return Compose(deterministic + [
        # Converts data to PyTorch tensors
        EnsureTyped(keys=["image"]),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])
