import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import skew

from idssp.sonk.utils.logger import get_logger
from idssp.sonk.view import utils

logger = get_logger(__name__)


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

    def load_data(self) -> str | None:
        '''
        Loads the image and label data for the volume.
        '''
        message = None
        logger.info("Loading data for volume...")
        self.image = nib.load(self.img_path)
        logger.debug("Loading label data for volume...")
        self.label = nib.load(self.label_path)

        logger.debug("Extracting data arrays from the loaded NIfTI files...")
        self.image_data = self.image.get_fdata()
        self.label_data = np.asanyarray(self.label.dataobj).astype(np.uint8)
        logger.info("Data loaded successfully.")

        if self.image_data.shape != self.label_data.shape:
            raise ValueError(
                f"Shape mismatch: Image {self.image_data.shape} vs "
                f"Label {self.label_data.shape}"
            )

        logger.info("Doing some basic checks...")
        if not np.allclose(self.image.affine, self.label.affine, atol=1e-2):
            logger.warning("Image and label affines do not match natively. ")
            message = "Warning: Image and label affines did not match. "

        logger.info("Calculating unique values in the label data...")
        self.mask_unique_values = np.unique(self.label_data)
        logger.info("Finding slice information...")
        self.find_slice_thresholds()
        logger.info("done!")

        return message


    def find_slice_thresholds(self):
        '''
        Finds the threshold slices for liver and tumor regions.
        '''
        first_liver_slice = None
        first_tumor_slice = None

        last_liver_slice = None
        last_tumor_slice = None

        # TODO: this function assumes LiTS. Update for general datasets.
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

    def print_slice_summary(self):
        logger.info("Volume has %d slices.", self.image_data.shape[2])
        logger.info("Liver slices range from %d to %d",
                    self.slice_thresholds['liver']['first'],
                    self.slice_thresholds['liver']['last'])
        logger.info("Tumor slices range from %d to %d",
                    self.slice_thresholds['tumor']['first'],
                    self.slice_thresholds['tumor']['last'])

    def get_volume_summary(self) -> Dict[str, Any]:
        '''
        Extracts a comprehensive summary of the volume as a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing volume metadata and statistics
        '''
        if self.image_data is None or self.label_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Basic shapes
        image_shape = self.image_data.shape
        label_shape = self.label_data.shape

        # CT intensity range
        ct_min = float(self.image_data.min())
        ct_max = float(self.image_data.max())

        # Voxel spacing
        # Voxel spacing (authoritative)
        voxel_sizes = nib.affines.voxel_sizes(self.image.affine)
        spacing_x, spacing_y, spacing_z = float(voxel_sizes[0]), float(voxel_sizes[1]), float(voxel_sizes[2])

        # Affine axis codes (eg: LAS)
        affine_codes = nib.aff2axcodes(self.image.affine)

        # Unique labels
        unique_labels = sorted([int(x) for x in self.mask_unique_values])

        # Slice thresholds
        liver_first = self.slice_thresholds['liver']['first']
        liver_last = self.slice_thresholds['liver']['last']
        tumor_first = self.slice_thresholds['tumor']['first']
        tumor_last = self.slice_thresholds['tumor']['last']

        # Voxel counts
        total_voxels = int(np.prod(label_shape))
        liver_voxels = int(np.sum(self.label_data == 1))
        tumor_voxels = int(np.sum(self.label_data == 2))

        # Ratios
        liver_to_total_ratio = liver_voxels / total_voxels if total_voxels > 0 else 0.0
        tumor_to_total_ratio = tumor_voxels / total_voxels if total_voxels > 0 else 0.0
        tumor_to_liver_ratio = tumor_voxels / liver_voxels if liver_voxels > 0 else 0.0

        # Compute robust HU bounds within liver mask (label == 1)
        liver_mask = self.label_data == 1
        if np.any(liver_mask):
            liver_hu = self.image_data[liver_mask]
            liver_hu_mean = liver_hu.mean()
            liver_hu_std = liver_hu.std()
            liver_hu_p005 = float(np.percentile(liver_hu, 0.5))
            liver_hu_p995 = float(np.percentile(liver_hu, 99.5))
            liver_hu_min = liver_hu.min()
            liver_hu_max = liver_hu.max()
        else:
            liver_hu_p005, liver_hu_p995 = None, None
            liver_hu_mean, liver_hu_std = None, None
            liver_hu_min, liver_hu_max = None, None
            logger.warning("No liver voxels found in volume. Cannot compute liver HU statistics.")

        # Tumour intensity statistics (parallel to liver stats)
        tumour_mask = self.label_data == 2
        if np.any(tumour_mask):
            tumour_hu = self.image_data[tumour_mask]
            tumour_hu_mean = tumour_hu.mean()
            tumour_hu_std = tumour_hu.std()
            tumour_hu_median = np.median(tumour_hu)
            tumour_hu_skew = skew(tumour_hu)
            tumour_hu_p005 = float(np.percentile(tumour_hu, 0.5))
            tumour_hu_p995 = float(np.percentile(tumour_hu, 99.5))
            tumour_hu_min = tumour_hu.min()
            tumour_hu_max = tumour_hu.max()
        else:
            tumour_hu_mean = tumour_hu_std = tumour_hu_median = tumour_hu_skew = None
            tumour_hu_p005 = tumour_hu_p995 = tumour_hu_min = tumour_hu_max = None

        # tuple of floats  representing the size in mm for the x, y, and z axes
        voxel_sizes = nib.affines.voxel_sizes(self.image.affine)
        # 1 voxel volume = x * y * z
        voxel_volume_mm3 = float(np.prod(voxel_sizes))

        # Volume in millilitres
        liver_volume_ml = liver_voxels * voxel_volume_mm3 / 1000.0
        tumour_volume_ml = tumor_voxels * voxel_volume_mm3 / 1000.0

        # Lesion-level metrics
        lesion_metrics = self._compute_lesion_metrics(self.label_data)

        # Simple liver texture variance (tumour-excluded)
        liver_only_mask = (self.label_data == 1)
        if np.any(liver_only_mask):
            liver_texture_variance = float(np.var(self.image_data[liver_only_mask]))
        else:
            liver_texture_variance = None

        # Noise estimate: std in homogeneous liver sub-region (optional refinement)
        # Using interquartile range within liver as robust noise proxy
        if np.any(liver_only_mask):
            liver_hu_vals = self.image_data[liver_only_mask]
            q1, q3 = np.percentile(liver_hu_vals, [25, 75])
            iqr = q3 - q1
            # Approximate noise as IQR / 1.35 (for Gaussian)
            liver_noise_estimate = iqr / 1.35 if iqr > 0 else None
        else:
            liver_noise_estimate = None

        return {
            'image_path': self.img_path,
            'label_path': self.label_path,
            'image_shape': image_shape,
            'label_shape': label_shape,
            'ct_min': ct_min,
            'ct_max': ct_max,
            'spacing_x': spacing_x,
            'spacing_y': spacing_y,
            'spacing_z': spacing_z,
            'affine_codes': affine_codes,
            'unique_labels': unique_labels,
            'liver_first': liver_first,
            'liver_last': liver_last,
            'tumor_first': tumor_first,
            'tumor_last': tumor_last,
            'liver_voxels': liver_voxels,
            'tumor_voxels': tumor_voxels,
            'liver_to_total_ratio': liver_to_total_ratio,
            'tumor_to_total_ratio': tumor_to_total_ratio,
            'tumor_to_liver_ratio': tumor_to_liver_ratio,
            'has_tumor': tumor_voxels > 0,
            'liver_hu_mean': liver_hu_mean,
            'liver_hu_std': liver_hu_std,
            'liver_hu_p005': liver_hu_p005,
            'liver_hu_p995': liver_hu_p995,
            'liver_hu_min': liver_hu_min,
            'liver_hu_max': liver_hu_max,
            # Tumour intensity statistics
            'tumour_hu_mean': tumour_hu_mean,
            'tumour_hu_std': tumour_hu_std,
            'tumour_hu_median': tumour_hu_median,
            'tumour_hu_skewness': tumour_hu_skew,
            'tumour_hu_p005': tumour_hu_p005,
            'tumour_hu_p995': tumour_hu_p995,
            'tumour_hu_min': tumour_hu_min,
            'tumour_hu_max': tumour_hu_max,

            # Volume in clinical units
            'voxel_volume_mm3': voxel_volume_mm3,
            'liver_volume_ml': liver_volume_ml,
            'tumour_volume_ml': tumour_volume_ml,

            # Lesion-level metrics
            'num_lesions': lesion_metrics['num_lesions'],
            'lesion_volumes_ml': lesion_metrics['lesion_volumes_ml'],
            'lesion_equiv_diameters_mm': lesion_metrics['lesion_equiv_diameters_mm'],
            'min_lesion_diameter_mm': lesion_metrics['min_lesion_diameter_mm'],
            'max_lesion_diameter_mm': lesion_metrics['max_lesion_diameter_mm'],
            'mean_lesion_diameter_mm': lesion_metrics['mean_lesion_diameter_mm'],

            # Liver texture and noise
            'liver_texture_variance': liver_texture_variance,
            'liver_noise_estimate': liver_noise_estimate,
        }

    def _compute_lesion_metrics(self, label_data: np.ndarray) -> dict:
        """
        Compute lesion-level metrics for tumour mask (label == 2).
        
        Returns
        -------
        dict
            Contains:
            - num_lesions: int
            - lesion_volumes_ml: list[float]
            - lesion_equiv_diameters_mm: list[float]
        """
        tumour_mask = (label_data == 2).astype(np.uint8)

        if not np.any(tumour_mask):
            return {
                'num_lesions': 0,
                'lesion_volumes_ml': [],
                'lesion_equiv_diameters_mm': [],
                'min_lesion_diameter_mm': None,
                'max_lesion_diameter_mm': None,
                'mean_lesion_diameter_mm': None
            }

        # Connected component labelling (26-connectivity for 3D)
        structure = ndimage.generate_binary_structure(3, 3)
        labelled, num = ndimage.label(tumour_mask, structure=structure)
        voxel_volume_mm3 = float(np.prod(nib.affines.voxel_sizes(self.image.affine)))

        lesion_volumes_ml = []
        lesion_equiv_diameters_mm = []

        for label_idx in range(1, num + 1):
            lesion_mask = (labelled == label_idx)
            volume_voxels = np.sum(lesion_mask)
            volume_ml = volume_voxels * voxel_volume_mm3 / 1000.0
            lesion_volumes_ml.append(volume_ml)

            # Equivalent diameter: diameter of sphere with same volume
            # V = (4/3)πr³ → r = (3V/4π)^(1/3) → d = 2r
            volume_mm3 = volume_voxels * voxel_volume_mm3
            equiv_diameter_mm = 2 * ((3 * volume_mm3) / (4 * np.pi)) ** (1/3)
            lesion_equiv_diameters_mm.append(equiv_diameter_mm)

        return {
            'num_lesions': num,
            'lesion_volumes_ml': lesion_volumes_ml,
            'lesion_equiv_diameters_mm': lesion_equiv_diameters_mm,
            'min_lesion_diameter_mm': min(lesion_equiv_diameters_mm) if lesion_equiv_diameters_mm else None,
            'max_lesion_diameter_mm': max(lesion_equiv_diameters_mm) if lesion_equiv_diameters_mm else None,
            'mean_lesion_diameter_mm': np.mean(lesion_equiv_diameters_mm) if lesion_equiv_diameters_mm else None
        }
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
            raise ValueError("Volume is not set. Please set the volume using "
                             "set_volume() before printing the summary.")

        print("Volume summary:")
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


class DatasetSummary:
    '''
    Dataset-wide analysis utility that iterates over all paired volumes
    discovered by DataCollector and produces per-case rows plus aggregate
    statistics for thesis analysis.
    
    Usage
    -----
    collector = DataCollector()
    collector.read_dir(config.CT_ROOT, ds_source='LiTS')
    collector.extract_images_and_labels()
    
    summary = DatasetSummary(collector.datasources)
    summary.analyse_all()
    summary.print_table()
    summary.export_csv('lits_dataset_summary.csv')
    agg = summary.get_aggregate_stats()
    '''

    def __init__(self, datasources: List[Dict[str, str]]):
        '''
        Initialize the dataset summary analyzer.
        
        Parameters
        ----------
        datasources : list of dict
            List of paired image/label paths from DataCollector.datasources
        '''
        self.datasources = datasources
        self.per_case_rows: List[Dict[str, Any]] = []
        self.aggregate_stats: Optional[Dict[str, Any]] = None

    def analyse_all(self) -> List[Dict[str, Any]]:
        '''
        Iterate over all paired volumes and extract per-case summaries.
        
        Parameters
        ----------
        verbose : bool
            If True, print progress messages during analysis.
        
        Returns
        -------
        list of dict
            List of per-case summary dictionaries (same as self.per_case_rows)
        '''
        self.per_case_rows = []

        for i, pair in enumerate(self.datasources):
            case_name = Path(pair['image']).name
            logger.debug("[%d/%d] Analysing %s...", i + 1, len(self.datasources), case_name)

            try:
                wrapper = VolumeWrapper(pair['image'], pair['label'])
                message = wrapper.load_data()
                row = wrapper.get_volume_summary()

                # Add case index for convenience
                row['case_index'] = i
                # Extract filename for easier reading
                row['case_name'] = Path(pair['image']).name
                row['status'] = 'success'
                row['obs'] = message

                self.per_case_rows.append(row)
            except Exception as e:
                logger.error("Failed to analyse %s: ", case_name)
                logger.error("Exception details:", exc_info=True)
                # Append a failure row so your CSV still accounts for this case
                self.per_case_rows.append({
                    'case_index': i,
                    'case_name': case_name,
                    'status': 'failed',
                    'obs': str(e)
                })

        logger.debug("Completed analysis. %d succeeded, %d failed.", 
                     sum(1 for r in self.per_case_rows if r.get('status') == 'success'),
                     sum(1 for r in self.per_case_rows if r.get('status') == 'failed'))
        return self.per_case_rows

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Recursively flatten a nested dictionary for CSV export.
        Lists are converted to semicolon-separated strings.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to semicolon-separated string for CSV compatibility
                items.append((new_key, ';'.join(map(str, v))))
            elif isinstance(v, (np.floating, np.integer)):
                # Convert numpy scalars to native Python types
                items.append((new_key, v.item()))
            else:
                items.append((new_key, v))
        return dict(items)

    def export_csv_auto(self, output_path: Path, exclude_keys: List[str] | None = None) -> None:
        """
        Export per-case rows to CSV using automatic flattening.
        
        Parameters
        ----------
        output_path : Path
            Path to the output CSV file.
        exclude_keys : list of str, optional
            Keys to exclude from export (e.g., large arrays, paths).
        """
        if not self.per_case_rows:
            raise ValueError("No data analysed. Call analyse_all() first.")

        # Exclude long paths by default; avoid mutating caller's list
        default_excludes = ['image_path', 'label_path']
        if exclude_keys is None:
            exclude_keys = default_excludes
        else:
            exclude_keys = list(exclude_keys) + default_excludes

        flattened_rows = []
        for r in self.per_case_rows:
            flat = self._flatten_dict(r)
            # Remove excluded keys
            for key in exclude_keys:
                flat.pop(key, None)
            flattened_rows.append(flat)

        # Use pandas for robust CSV handling
        df = pd.DataFrame(flattened_rows)
        df.to_csv(output_path, index=False)
        print(f"CSV exported to: {output_path} ({len(df)} rows, {len(df.columns)} columns)")

def analyse_dataset(
        datasources: List[Dict[str, str]],
        output_csv: Path
    ) -> DatasetSummary:
    '''
    Convenience function to run a complete LiTS dataset analysis.
    
    Parameters
    ----------
    datasources : list of dict
        Paired image/label paths from DataCollector
    output_csv : Path
        Path to export per-case rows to this CSV path

    Returns
    -------
    DatasetSummary
        The summary object with per_case_rows and aggregate_stats populated
    '''
    summary = DatasetSummary(datasources)
    summary.analyse_all()
    summary.export_csv_auto(output_csv)

    return summary
