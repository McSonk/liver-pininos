import nibabel as nib
import numpy as np

from idssp.sonk import config
from idssp.sonk.view import utils


class VolumeWrapper:
    def __init__(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path
        self.image = None
        self.image_data = None
        self.label = None
        self.label_data = None
        self.mask_unique_values = None
        self.slice_thresholds = None

    def load_data(self):
        '''
        Loads the image and label data for the volume.
        '''
        print("Loading data for volume...")
        self.image = nib.load(self.img_path)
        self.label = nib.load(self.label_path)

        self.image_data = self.image.get_fdata()
        self.label_data = self.label.get_fdata()

        print("Data loaded successfully.")
        print("Calculating unique values in the label data...")
        self.mask_unique_values = np.unique(self.label_data)
        self.convert_mask_to_long()
        print("Finding slice information...")
        self.find_slice_thresholds()
        print("done!")


    def find_slice_thresholds(self):
        '''
        Finds the threshold slices for liver and tumor regions.
        '''
        first_liver_slice = None
        first_tumor_slice = None

        last_liver_slice = None
        last_tumor_slice = None

        num_of_slices = self.image_data.shape[2]

        for i in range(num_of_slices):
            slice_mask = self.label_data[:, :, i]

            if np.any(slice_mask == 1):  # Check if there's any liver in the slice
                if first_liver_slice is None:
                    first_liver_slice = i
                last_liver_slice = i

            if np.any(slice_mask == 2):  # Check if there's any tumor in the slice
                if first_tumor_slice is None:
                    first_tumor_slice = i
                last_tumor_slice = i

        self.slice_thresholds = {
            "liver": {
                "first": first_liver_slice,
                "last": last_liver_slice
            },
            "tumor": {
                "first": first_tumor_slice,
                "last": last_tumor_slice
            }
        }

    def load_image(self):
        if self.image is None:
            self.image = nib.load(self.img_path)
        return self.image

    def load_label(self):
        if self.label is None:
            self.label = nib.load(self.label_path)
        return self.label

    def convert_mask_to_long(self):
        if self.label_data is not None:
            self.label_data = self.label_data.astype(np.uint8)
        else:
            print("Label data is not loaded. Please load the label data before converting to long.")

    def print_slice_summary(self):
        print(f"Volume has {self.image_data.shape[2]} slices.")
        print(f"Liver slices range from {self.slice_thresholds['liver']['first']} to {self.slice_thresholds['liver']['last']}")
        print(f"Tumor slices range from {self.slice_thresholds['tumor']['first']} to {self.slice_thresholds['tumor']['last']}")

class DataWrapper:
    def __init__(self):
        self.volume = None

    def set_volume(self, img_path: str, label_path: str):
        '''
        Sets the volume for the data wrapper.
        Params
        ------
        `img_path`: str
            The file path for the image volume.
        `label_path`: str
            The file path for the label volume.
        '''
        self.volume = VolumeWrapper(img_path, label_path)
        self.volume.load_data()


    def print_summary_of_volume(self):
        '''
        Prints a summary of the image and label files of a given volume ID, including
        their paths and shapes.
        Params
        ------
        `volume_id`: int
            The ID of the volume to print the summary for.
        '''
        if self.volume is None:
            raise ValueError("Volume is not set. Please set the volume using set_volume() before printing the summary.")

        print(f"Volume summary:")
        print("--------------------File paths--------------------")
        print(f"Image path: {self.volume.img_path}")
        print(f"Label path: {self.volume.label_path}")

        print("--------------------File shapes--------------------")
        print(f"Image shape: {self.volume.image.shape}")
        print(f"Label shape: {self.volume.label.shape}")

        # Check if the shapes match
        if self.volume.image.shape != self.volume.label.shape:
            print("Warning: Image and label shapes do not match!")

        print("--------------------Data arrays--------------------")
        print('Image data shape: %s' % str(self.volume.image_data.shape))
        print('Mask data shape: %s' % str(self.volume.label_data.shape))

        # TODO: use this information to decide on the necessary preprocessing steps (e.g., resampling, normalization, etc.)
        print("--------------------- value ranges--------------------")
        print("CT intensity range:", self.volume.image_data.min(), "to", self.volume.image_data.max())
        print("Mask intensity range:", self.volume.label_data.min(), "to", self.volume.label_data.max())

        print("Voxel dimensions (mm):", self.volume.image.header.get_zooms())

        print("--------------------Affine information--------------------")
        print("Image affine transformation matrix:\n", self.volume.image.affine)
        print("Human readable header affine:\n", nib.aff2axcodes(self.volume.image.affine))

        # Check the unique values in the label data to understand the classes present
        print("--------------------Unique labels in segmentation--------------------")
        print("Unique labels in segmentation:", self.volume.mask_unique_values)
        print("Check if the unique labels match the expected classes (e.g., 0 for background, 1 for liver, 2 for tumor).")

        print("---------------------Slice information---------------------")
        self.volume.print_slice_summary()

    def plot_slice(self, slice_index):
        '''
        Plots a specific slice of the image and its corresponding label for a given volume ID.
        Params
        ------
        `slice_index`: int
            The index of the slice to plot.
        '''
        if self.volume is None:
            raise ValueError("Volume is not set. Please set the volume using set_volume()")

        print(f"Plotting slice {slice_index} of volume...")
        utils.plot_slice(self.volume.image_data, self.volume.label_data, slice_index)
        utils.plot_mixed_slice(self.volume.image_data, self.volume.label_data, slice_index)


    def get_animation_motion(self):
        '''
        Creates an animation of the slices of the image and their corresponding labels for a given volume ID.
        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object that can be displayed in a Jupyter notebook or saved as a video file.
        '''
        if self.volume is None:
            raise ValueError("Volume is not set. Please set the volume using set_volume() before creating an animation.")

        print("Creating animation for volume...")
        ani = utils.plot_animation(
            self.volume.image_data,
            self.volume.label_data,
            first_slice=self.volume.slice_thresholds['liver']['first'],
            last_slice=self.volume.slice_thresholds['liver']['last'],
            first_tumour_slice=self.volume.slice_thresholds['tumor']['first']
            )
        return ani
