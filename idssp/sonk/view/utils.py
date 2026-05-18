import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import animation
from monai.visualize import blend_images

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)

# For the animation to display the full slices range
mpl.rcParams['animation.embed_limit'] = 50 * 1024 * 1024  # 50 MB


def print_image_plot(img_data, slice_index, include_axis=False, ax=None,
                     use_training_window=True):
    ''' Prints a single slice of the CT image with appropriate windowing.

    Params
    ------
    `img_data` : numpy.ndarray
        3D array containing the CT image data
    `slice_index` : int
        Index of the slice to display
    `include_axis` : bool, optional
        Whether to show axis ticks and labels (default: False)
    `ax` : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, uses current axes
    `use_training_window` : bool, optional
        If True, uses the training window (HU: -175 to 250).
        If False, uses general abdominal window (default: True)
    '''

    if use_training_window:
        # Window matching ScaleIntensityRanged preprocessing
        vmin = config.HU_WINDOW_MIN
        vmax = config.HU_WINDOW_MAX
        title_suffix = "(Training Window)"
    else:
        # General abdominal CT presets
        window = 400
        level = 50
        vmin = level - window/2  # -150
        vmax = level + window/2  # 250
        title_suffix = "(Abdominal Window)"

    if ax is None:
        ax = plt.gca()  # fallback to current axes

    img_obj = ax.imshow(img_data[:,:,slice_index], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'CT Image - Slice {slice_index} {title_suffix}')
    ax.axis('on' if include_axis else 'off')
    return img_obj

def print_mask_plot(mask_data, slice_index, include_axis=False, is_overlay=False, ax=None):
    '''
    Prints a single slice of the segmentation mask with appropriate colouring.
    Params
    ------
    `mask_data` : numpy.ndarray
        3D array containing the segmentation mask data
    `slice_index` : int
        Index of the slice to display
    `include_axis` : bool, optional
        Whether to show axis ticks and labels (default: False)
    `is_overlay` : bool, optional
        Whether this mask will be plotted as an overlay on the CT image (default: False)
    `ax` : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, uses current axes (default: None)
        (Useful for plotting multiple subplots in the animation function)
    '''
    if ax is None:
        ax = plt.gca()

    if is_overlay:
        cmap = mcolors.ListedColormap(['none','green','red'])
        alpha_value = 0.4
    else:
        cmap = mcolors.ListedColormap(['gray','green','red'])
        alpha_value = 1.0

    lbl_obj = ax.imshow(mask_data[:,:,slice_index], cmap=cmap, alpha=alpha_value)
    ax.set_title(f'Segmentation Mask - Slice {slice_index}')
    ax.axis('on' if include_axis else 'off')
    return lbl_obj

def plot_slice(img_data, mask_data, slice_index, include_axis=False):
    '''
    Plots a slice of the CT image and the corresponding segmentation mask.
    (2 subplots side by side)
    Params
    ------
    `img_data` : numpy.ndarray
        3D array containing the CT image data
    `mask_data` : numpy.ndarray
        3D array containing the segmentation mask data
    `slice_index` : int
        Index of the slice to display
    `include_axis` : bool, optional
        Whether to show axis ticks and labels (default: False)
    '''
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    print_image_plot(img_data, slice_index, include_axis)
    
    plt.subplot(1, 2, 2)
    print_mask_plot(mask_data, slice_index, include_axis)
    
    plt.show()



def plot_mixed_slice(img_data, mask_data, slice_index, include_axis=False):
    '''
    Plots a slice of the CT image and the corresponding segmentation mask overlay.
    Params
    ------
    `img_data` : numpy.ndarray
        3D array containing the CT image data
    `mask_data` : numpy.ndarray
        3D array containing the segmentation mask data
    `slice_index` : int
        Index of the slice to display
    `include_axis` : bool, optional
        Whether to show axis ticks and labels (default: False)
    '''
    plt.figure(figsize=(6,6))
    print_image_plot(img_data, slice_index, include_axis, use_training_window=False)
    print_mask_plot(mask_data, slice_index, include_axis, is_overlay=True)
    plt.title(f'Slice {slice_index} - combined')
    plt.axis('on' if include_axis else 'off')
    plt.show()


def plot_animation(img_data, mask_data, first_slice, last_slice, first_tumour_slice):
    '''Creates an animation of the CT slices with the segmentation mask overlay.
    Params
    -----
    img_data : numpy.ndarray
        3D array containing the CT image data
    mask_data : numpy.ndarray
        3D array containing the segmentation mask data
    first_slice : int
        Index of the first liver slice to display
    last_slice : int
        Index of the last liver slice to display
    first_tumour_slice : int
        Index of the first slice containing the tumour
    '''

    print(f"Creating animation for slices {first_slice} to {last_slice} (tumour starts at slice {first_tumour_slice})...")

    # Create three subplots: CT, mask, and overlay
    fig, (img_ax, lbl_ax, overlay_ax) = plt.subplots(1, 3, figsize=(15, 5))

    # CT only
    img_obj = print_image_plot(img_data, first_tumour_slice, ax=img_ax)
    img_ax.set_title('CT Image')

    # Mask only
    lbl_obj = print_mask_plot(mask_data, first_tumour_slice, ax=lbl_ax)
    lbl_ax.set_title('Segmentation Mask')

    # Overlay: CT + mask together
    overlay_img_obj = print_image_plot(img_data, first_tumour_slice, ax=overlay_ax)
    overlay_lbl_obj = print_mask_plot(mask_data, first_tumour_slice, ax=overlay_ax, is_overlay=True)
    overlay_ax.set_title('Overlay')

    index_text = fig.text(0.5, 0.05, f'Index: {first_tumour_slice}', ha='center')

    def update(i):
        # Update CT
        img_obj.set_array(img_data[:, :, i])
        # Update mask
        lbl_obj.set_array(mask_data[:, :, i])
        # Update overlay
        overlay_img_obj.set_array(img_data[:, :, i])
        overlay_lbl_obj.set_array(mask_data[:, :, i])
        # Update index text
        index_text.set_text(f'Index: {i}')
        return [img_obj, lbl_obj, overlay_img_obj, overlay_lbl_obj]

    ani = animation.FuncAnimation(fig, update, frames=range(first_slice, last_slice),
                                  interval=100, blit=True, repeat=False)

    plt.close(fig)
    return ani

# voxel coordinate (x=0, y=0, z=slice_index)
def slice_to_world_coordinates(image_object, slice_index):
    '''
    Converts a slice index to world coordinates. Useful for comparing with 
    visualisation tools like 3D Slicer that use world coordinates.
    Params
    ------
    `image_object` : nibabel.Nifti1Image
        The image object containing the affine transformation
    `slice_index` : int
        Index of the slice to convert
    Returns
    -------
    `world_coord` : float
        The world coordinate of the slice
    '''
    voxel_coord = np.array([0, 0, slice_index, 1])  # homogeneous coordinate
    world_coord = image_object.affine @ voxel_coord
    return world_coord[2]


def log_segmentation_overlay(
    writer,
    epoch: int,
    image: torch.Tensor,
    label: torch.Tensor,
    pred: torch.Tensor,
    slice_axis: int = 2,
    slice_idx: int = None
):
    """
    Logs a side-by-side GT vs prediction overlay to TensorBoard.
    Slice is chosen automatically as the axial slice with the most
    tumour voxels in the ground truth. Falls back to middle slice
    for tumour-negative volumes.

    Args:
        writer:     TensorBoard SummaryWriter.
        epoch:      Current epoch (used as global_step).
        image:      CT volume [B, 1, D, H, W] — full volume from SlidingWindowInferer input.
        label:      Ground truth [B, 1, D, H, W].
        pred:       Model logits [B, NUM_CLASSES, D, H, W] — full-volume inference output.
        slice_axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial).
        slice_idx:  Override automatic slice selection if provided.
    """
    image = image.detach().cpu()
    label = label.detach().cpu()
    pred  = pred.detach().cpu()

    # Take first sample in batch
    img_vol   = image[0, 0]    # [D, H, W]
    label_vol = label[0, 0]    # [D, H, W]
    pred_vol  = pred[0]        # [NUM_CLASSES, D, H, W]

    # --- Select most informative slice ---
    if slice_idx is None:
        # LAS orientation: sum over X (dim 0) and Y (dim 1) 
        # to get tumour voxel count per axial Z slice (dim 2)
        tumour_per_slice = (label_vol == config.TUMOUR_CLASS_INDEX).sum(dim=(0, 1))
        if tumour_per_slice.max() > 0:
            slice_idx = tumour_per_slice.argmax().item()
            logger.debug("Overlay: using tumour-centred slice %d", slice_idx)
        else:
            slice_idx = img_vol.shape[slice_axis] // 2
            logger.debug("Overlay: tumour-negative volume, using middle slice %d", slice_idx)

    # --- Extract 2D slices ---
    img_slice  = img_vol.select(slice_axis, slice_idx)             # [H, W]
    gt_slice   = label_vol.select(slice_axis, slice_idx)           # [H, W]
    pred_class = torch.argmax(pred_vol, dim=0)                     # [D, H, W]
    pred_slice = pred_class.select(slice_axis, slice_idx)          # [H, W]

    # --- Normalise CT to [0, 1] ---
    img_min, img_max = img_slice.min(), img_slice.max()
    img_norm = (img_slice - img_min) / (img_max - img_min + 1e-8)  # [H, W]
    img_norm = img_norm.unsqueeze(0)                                # [1, H, W]

    # --- Extract tumour masks ---
    gt_tumour_mask   = (gt_slice   == config.TUMOUR_CLASS_INDEX).float()  # [H, W]
    pred_tumour_mask = (pred_slice == config.TUMOUR_CLASS_INDEX).float()  # [H, W]

    # --- Spatial mismatch guard ---
    img_spatial = img_norm.shape[1:]
    for name, mask in [("gt", gt_tumour_mask), ("pred", pred_tumour_mask)]:
        if mask.shape != img_spatial:
            logger.warning(
                "Spatial mismatch in overlay: %s mask=%s vs image=%s — resizing.",
                name, mask.shape, img_spatial
            )

    gt_tumour_mask   = _resize_mask_if_needed(gt_tumour_mask,   img_spatial)
    pred_tumour_mask = _resize_mask_if_needed(pred_tumour_mask, img_spatial)

    # --- Build overlays ---
    gt_overlay   = blend_images(
        image=img_norm.unsqueeze(0),                       # [1, 1, H, W]
        label=gt_tumour_mask.unsqueeze(0).unsqueeze(0),    # [1, 1, H, W]
        alpha=0.4,
        cmap="hot"
    )  # [1, 3, H, W]

    pred_overlay = blend_images(
        image=img_norm.unsqueeze(0),
        label=pred_tumour_mask.unsqueeze(0).unsqueeze(0),
        alpha=0.4,
        cmap="hot"
    )  # [1, 3, H, W]

    # --- Log side by side ---
    writer.add_images(
        tag="Visualisation/GT_vs_Pred_Tumour",
        img_tensor=torch.cat([gt_overlay, pred_overlay], dim=0),  # [2, 3, H, W]
        global_step=epoch,
        dataformats="NCHW"
    )

    logger.debug(
        "Logged GT vs Pred overlay at epoch %d, slice %d "
        "(gt tumour px: %d, pred tumour px: %d)",
        epoch, slice_idx,
        int(gt_tumour_mask.sum()),
        int(pred_tumour_mask.sum())
    )


def _resize_mask_if_needed(mask: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """Resize [H, W] mask to target_size using nearest-neighbour if needed."""
    if mask.shape == target_size:
        return mask
    return F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, H, W]
        size=target_size,
        mode='nearest'
    ).squeeze(0).squeeze(0)