from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR, UNet

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)

def _load_monai_pretrained_weights(model: torch.nn.Module, pretrained_path: Path) -> None:
    """
    Loads MONAI Model or external pretrained weights into a given model.
    Uses strict=False to accommodate different NUM_CLASSES (e.g. BTCV -> LiTS).
    """
    # MONAI checkpoints sometimes contain non-tensor objects or custom classes.
    # We attempt weights_only=True first for security, falling back to False if it fails.
    device = torch.device("cpu")  # Load on CPU to avoid GPU memory issues during loading
    try:
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
    except Exception:
        logger.warning("weights_only=True failed. Falling back to weights_only=False.")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        
    # Extract the actual state_dict from common wrapper formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Assume the dictionary itself is the state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean up 'module.' prefix from DDP/DataParallel if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "", 1) if k.startswith("module.") else k
        cleaned_state_dict[name] = v

    # strict=False is CRITICAL for fine-tuning on a new dataset with different NUM_CLASSES
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    
    logger.info("Loaded pretrained weights from %s", pretrained_path)
    if missing:
        logger.info("Missing keys (expected for the segmentation head due to different NUM_CLASSES): %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected)

def get_unet(config_obj: config.Config) -> UNet:
    '''Creates a 3D UNet model for medical image segmentation. The architecture
       is defined by the MONAI library's UNet implementation, with parameters
       set according to the current configuration.'''
    logger.info("Creating UNet model with %d output classes.", config_obj.NUM_CLASSES)
    s_dims = 3
    in_channels = 1
    out_channels = config_obj.NUM_CLASSES
    # channels=(64, 128, 256, 512)
    channels = (32, 64, 128, 256)
    strides = (2, 2, 2)
    num_res_units = 1 if config.is_limited_env() else 2
    norm="INSTANCE"
    act="PRELU"
    logger.debug("UNet architecture parameters: Spatial Dims: %d, In Channels: %d, "
                 "Out Channels: %d, Channels: %s, Strides: %s, Num Res Units: %d, "
                 "Norm: %s, Act: %s",
                 s_dims, in_channels, out_channels, channels, strides, num_res_units, norm, act)
    return UNet(
        spatial_dims=s_dims,
        # Just 1 channel for the grayscale CT image.
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        # One stride per downsampling transition: len(strides) == len(channels) - 1
        strides=strides,
        num_res_units=num_res_units,
        # batch norm isn't useful for 3d images (usually batches aren't longer than 4)
        norm=norm,
        act=act
    )

def get_seg_res_net(config_obj: config.Config) -> SegResNet:
    '''Creates a 3D SegResNet model for medical image segmentation. The architecture
       is defined by the MONAI library's SegResNet implementation, with parameters
       set according to the current configuration.'''
    logger.info("Creating SegResNet model with %d output classes.", config_obj.NUM_CLASSES)
    s_dims = 3
    in_channels = 1
    out_channels = config_obj.NUM_CLASSES
    init_filters = 32
    blocks_down = [1, 2, 2, 4]
    blocks_up = [1, 1, 1]
    norm = ('GROUP', {'num_groups': 8})
    act = 'RELU'
    logger.debug("SegResNet architecture parameters: Spatial Dims: %d, In Channels: %d, "
                 "Out Channels: %d, Blocks Down: %s, Blocks Up: %s, Norm: %s, "
                 "Act: %s, Init Filters: %d",
                 s_dims, in_channels, out_channels, blocks_down, blocks_up,
                 norm, act, init_filters)
    # TODO: Study this
    return SegResNet(
        spatial_dims=s_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        # init_filters is the number of filters in the first layer,
        # which is then doubled at each downsampling step.
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        norm=norm,
        act=act,
    )

def get_swin_unetr(config_obj: config.Config) -> SwinUNETR:
    '''Creates a 3D SwinUNETR model for medical image segmentation. The architecture
       is defined by the MONAI library's SwinUNETR implementation, with parameters
       set according to the current configuration.'''
    logger.info("Creating SwinUNETR model with %d output classes.", config_obj.NUM_CLASSES)
    train_patch_size = config_obj.TRAIN_PATCH_SIZE
    if len(train_patch_size) != 3:
        raise ValueError("TRAIN_PATCH_SIZE must contain exactly 3 spatial dimensions "
                         f"for SwinUNETR, got {train_patch_size}.")
    if any(dim % 32 != 0 for dim in train_patch_size):
        raise ValueError("All TRAIN_PATCH_SIZE dimensions must be divisible by 32 "
                         f"for SwinUNETR due to the architecture's downsampling "
                         f"steps, got {train_patch_size}.")
    # We explicitly define feature_size=48 (the "large" variant for better performance)
    # but leave depths, num_heads, patch_size, and window_size to MONAI's defaults.
    spatial_dims=3
    in_channels=1
    feature_size=48
    # CRITICAL: Trades ~20% compute time for ~40% VRAM savings
    use_checkpoint=True
    norm_name="instance"

    logger.debug("SwinUNETR architecture parameters: "
                 "Spatial Dims: %d, "
                 "In Channels: %d, "
                 "Out Channels: %s, "
                 "Feature Size: %d, "
                 "Use Checkpoint: %s, "
                 "Norm Name: %s",
                 spatial_dims,
                 in_channels,
                 config_obj.NUM_CLASSES,
                 feature_size,
                 use_checkpoint,
                 norm_name
                )
    model = SwinUNETR(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=config_obj.NUM_CLASSES,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        norm_name=norm_name,
    )

    return model

def get_swin_unetr_pretrain(config_obj: config.Config) -> SwinUNETR:
    '''Creates a SwinUNETR model and loads pretrained weights from the specified path.'''
    logger.info("Creating SwinUNETR model with pretrained weights from %s.",
                config_obj.PRE_TRAINED_MODEL_PATH)
    model = get_swin_unetr(config_obj)

    try:
        _load_monai_pretrained_weights(
            model=model,
            pretrained_path=config_obj.PRE_TRAINED_MODEL_PATH
        )
    except Exception as e:
        logger.error("Failed to load pretrained weights from %s: %s",
                     config_obj.PRE_TRAINED_MODEL_PATH, str(e))
        raise RuntimeError(f"Failed to load pretrained weights: {str(e)}")

    return model

def get_model() -> nn.Module:
    '''Factory function to create the segmentation model based
       on the current configuration.'''
    cfg = config.get()
    if cfg.MODEL == config.AvailableModels.U_NET:
        return get_unet(cfg)
    elif cfg.MODEL == config.AvailableModels.SEG_RES_NET:
        return get_seg_res_net(cfg)
    elif cfg.MODEL == config.AvailableModels.SWIN_UNETR:
        return get_swin_unetr(cfg)
    elif cfg.MODEL == config.AvailableModels.SWIN_UNETR_PRETRAIN:
        return get_swin_unetr_pretrain(cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.MODEL}")
