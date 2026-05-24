import torch.nn as nn
from monai.networks.nets import SegResNet, UNet

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)

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
    norm = 'GROUP'
    act = 'RELU'
    logger.debug("SegResNet architecture parameters: Spatial Dims: %d, In Channels: %d, "
                 "Out Channels: %d, Blocks Down: %s, Blocks Up: %s, Norm: %s, "
                 "Act: %s, Init Filters: %d",
                 s_dims, in_channels, out_channels, blocks_down, blocks_up,
                 norm, act, init_filters)
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

def get_model() -> nn.Module:
    '''Factory function to create the segmentation model based on the current configuration.'''
    cfg = config.get()
    if cfg.MODEL == config.AvailableModels.U_NET:
        return get_unet(cfg)
    elif cfg.MODEL == config.AvailableModels.SEG_RES_NET:
        return get_seg_res_net(cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.MODEL}")
