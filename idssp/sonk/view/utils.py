import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# For the animation to display the full slices range
mpl.rcParams['animation.embed_limit'] = 50 * 1024 * 1024  # 50 MB


def print_image_plot(img_data, slice_index, include_axis=False, ax=None):
    ''' Prints a single slice of the CT image with appropriate windowing for abdominal CT scans.
    Params
    ------
    `img_data` : numpy.ndarray
        3D array containing the CT image data
    `slice_index` : int
        Index of the slice to display
    `include_axis` : bool, optional
        Whether to show axis ticks and labels (default: False)
    `ax` : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, uses current axes (default: None)
        (Useful for plotting multiple subplots in the animation function)
    '''
    # abdominal CT presets
    window = 400
    level = 50
    vmin = level - window/2
    vmax = level + window/2

    if ax is None:
        ax = plt.gca()  # fallback to current axes

    img_obj = ax.imshow(img_data[:,:,slice_index], cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'CT Image - Slice {slice_index}')
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
    print_image_plot(img_data, slice_index, include_axis)
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
