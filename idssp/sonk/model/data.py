import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np

from idssp.sonk.view import utils
from idssp.sonk.utils.logger import get_logger

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

    def load_data(self):
        '''
        Loads the image and label data for the volume.
        '''
        logger.info("Loading data for volume...")
        self.image = nib.load(self.img_path)
        self.label = nib.load(self.label_path)

        self.image_data = self.image.get_fdata()
        self.label_data = self.label.get_fdata()

        logger.info("Data loaded successfully.")
        logger.info("Calculating unique values in the label data...")
        self.mask_unique_values = np.unique(self.label_data)
        self.convert_mask_to_long()
        logger.info("Finding slice information...")
        self.find_slice_thresholds()
        logger.info("done!")


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
            logger.warning("Label data is not loaded. Please load the label"
                           " data before converting to long.")

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
            Dictionary containing volume metadata and statistics:
            - image_path, label_path
            - image_shape, label_shape
            - ct_min, ct_max (CT intensity range)
            - spacing_x, spacing_y, spacing_z (voxel spacing in mm)
            - affine_codes (human-readable axis codes)
            - unique_labels
            - liver_first, liver_last, tumor_first, tumor_last (slice thresholds)
            - liver_voxels, tumor_voxels (voxel counts)
            - liver_ratio, tumor_ratio (foreground ratios)
            - has_tumor (boolean)
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
        spacing = self.image.header.get_zooms()
        spacing_x, spacing_y, spacing_z = float(spacing[0]), float(spacing[1]), float(spacing[2])

        # Affine axis codes
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
        liver_ratio = liver_voxels / total_voxels if total_voxels > 0 else 0.0
        tumor_ratio = tumor_voxels / total_voxels if total_voxels > 0 else 0.0

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
            'liver_ratio': liver_ratio,
            'tumor_ratio': tumor_ratio,
            'has_tumor': tumor_voxels > 0
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
    # From code:
    collector = DataCollector()
    collector.read_dir(config.CT_ROOT, ds_source='LiTS')
    collector.extract_images_and_labels()
    
    summary = DatasetSummary(collector.datasources)
    summary.analyze_all()
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

    def analyze_all(self, verbose: bool = False) -> List[Dict[str, Any]]:
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
            if verbose:
                logger.info("[%d/%d] Analysing %s...", i + 1, len(self.datasources), pair['image'])

            wrapper = VolumeWrapper(pair['image'], pair['label'])
            wrapper.load_data()
            row = wrapper.get_volume_summary()

            # Add case index for convenience
            row['case_index'] = i
            # Extract filename for easier reading
            row['case_name'] = Path(pair['image']).name

            self.per_case_rows.append(row)

        if verbose:
            logger.info("Completed analysis of %d volumes.", len(self.per_case_rows))

        return self.per_case_rows

    def get_aggregate_stats(self) -> Dict[str, Any]:
        '''
        Compute dataset-level aggregate statistics from analyzed per-case rows.
        Must call analyze_all() first.
        
        Returns
        -------
        dict
            Dictionary containing aggregate statistics:
            - num_volumes
            - shape_mean, shape_median, shape_std (per axis)
            - spacing_mean, spacing_std (per axis)
            - orientation_distribution (count per affine code tuple)
            - tumor_proportion (fraction of volumes with tumor)
            - slice_range stats (liver/tumor span mean/std)
            - foreground_imbalance (mean liver/tumor ratios)
            - ct_intensity_mean (mean min/max across volumes)
        '''
        if not self.per_case_rows:
            raise ValueError("No data analyzed. Call analyze_all() first.")
        
        rows = self.per_case_rows
        n = len(rows)
        
        # Shape statistics (D, H, W)
        shapes = np.array([r['image_shape'] for r in rows])
        shape_mean = shapes.mean(axis=0).tolist()
        shape_median = np.median(shapes, axis=0).tolist()
        shape_std = shapes.std(axis=0).tolist()
        
        # Spacing statistics
        spacing_x = [r['spacing_x'] for r in rows]
        spacing_y = [r['spacing_y'] for r in rows]
        spacing_z = [r['spacing_z'] for r in rows]
        
        spacing_mean = [np.mean(spacing_x), np.mean(spacing_y), np.mean(spacing_z)]
        spacing_std = [np.std(spacing_x), np.std(spacing_y), np.std(spacing_z)]
        
        # Orientation distribution
        orientation_counts: Dict[str, int] = {}
        for r in rows:
            key = str(r['affine_codes'])
            orientation_counts[key] = orientation_counts.get(key, 0) + 1
        
        # Tumor proportion
        num_with_tumor = sum(1 for r in rows if r['has_tumor'])
        tumor_proportion = num_with_tumor / n
        
        # Slice range statistics
        liver_spans = []
        tumor_spans = []
        for r in rows:
            if r['liver_first'] is not None and r['liver_last'] is not None:
                liver_spans.append(r['liver_last'] - r['liver_first'] + 1)
            if r['tumor_first'] is not None and r['tumor_last'] is not None:
                tumor_spans.append(r['tumor_last'] - r['tumor_first'] + 1)
        
        liver_span_mean = np.mean(liver_spans) if liver_spans else 0.0
        liver_span_std = np.std(liver_spans) if liver_spans else 0.0
        tumor_span_mean = np.mean(tumor_spans) if tumor_spans else 0.0
        tumor_span_std = np.std(tumor_spans) if tumor_spans else 0.0
        
        # Foreground imbalance metrics
        liver_ratios = [r['liver_ratio'] for r in rows]
        tumor_ratios = [r['tumor_ratio'] for r in rows]
        liver_ratio_mean = np.mean(liver_ratios)
        liver_ratio_std = np.std(liver_ratios)
        tumor_ratio_mean = np.mean(tumor_ratios)
        tumor_ratio_std = np.std(tumor_ratios)
        
        # CT intensity statistics
        ct_mins = [r['ct_min'] for r in rows]
        ct_maxs = [r['ct_max'] for r in rows]
        ct_min_mean = np.mean(ct_mins)
        ct_max_mean = np.mean(ct_maxs)
        
        self.aggregate_stats = {
            'num_volumes': n,
            'shape_mean': shape_mean,
            'shape_median': shape_median,
            'shape_std': shape_std,
            'spacing_mean': spacing_mean,
            'spacing_std': spacing_std,
            'orientation_distribution': orientation_counts,
            'tumor_proportion': tumor_proportion,
            'num_with_tumor': num_with_tumor,
            'liver_span_mean': liver_span_mean,
            'liver_span_std': liver_span_std,
            'tumor_span_mean': tumor_span_mean,
            'tumor_span_std': tumor_span_std,
            'liver_ratio_mean': liver_ratio_mean,
            'liver_ratio_std': liver_ratio_std,
            'tumor_ratio_mean': tumor_ratio_mean,
            'tumor_ratio_std': tumor_ratio_std,
            'ct_min_mean': ct_min_mean,
            'ct_max_mean': ct_max_mean
        }
        
        return self.aggregate_stats

    def print_table(self) -> None:
        '''
        Print a terminal table-like summary of the dataset analysis.
        '''
        if not self.per_case_rows:
            logger.warning("No data analyzed. Call analyze_all() first.")
            return

        logger.info("")
        logger.info("=" * 50)
        logger.info("AGGREGATE STATISTICS")
        logger.info("=" * 50)

        if self.aggregate_stats is None:
            self.get_aggregate_stats()

        agg = self.aggregate_stats
        logger.info("Number of volumes:           %d", agg['num_volumes'])
        logger.info("Volumes with tumor:          %d (%.1f%%)", agg['num_with_tumor'],
                    agg['tumor_proportion']*100)
        logger.info("")
        logger.info("Mean shape (D,H,W):          %s", agg['shape_mean'])
        logger.info("Median shape (D,H,W):        %s", agg['shape_median'])
        logger.info("Shape std (D,H,W):           %s", agg['shape_std'])
        logger.info("")
        logger.info("Mean spacing (mm) (X,Y,Z):   %s", agg['spacing_mean'])
        logger.info("Spacing std (mm) (X,Y,Z):    %s", agg['spacing_std'])
        logger.info("")
        logger.info("Orientation distribution:")
        for orient, count in agg['orientation_distribution'].items():
            logger.info("  %s: %d volumes (%.1f%%)", orient, count, count/agg['num_volumes']*100)
        logger.info("")
        logger.info("Liver span (slices):         mean=%.1f, std=%.1f",
                    agg['liver_span_mean'], agg['liver_span_std'])
        logger.info("Tumor span (slices):         mean=%.1f, std=%.1f",
                    agg['tumor_span_mean'], agg['tumor_span_std'])
        logger.info("")
        logger.info("Liver voxel ratio:           mean=%.3f%%, std=%.3f%%",
                    agg['liver_ratio_mean']*100, agg['liver_ratio_std']*100)
        logger.info("Tumor voxel ratio:           mean=%.4f%%, std=%.4f%%",
                    agg['tumor_ratio_mean']*100, agg['tumor_ratio_std']*100)
        logger.info("")
        logger.info("CT intensity (mean range):   %d to %d", agg['ct_min_mean'], agg['ct_max_mean'])
        logger.info("=" * 50)

    def export_csv(self, output_path: str) -> None:
        '''
        Export per-case rows to a CSV file for thesis analysis.
        
        Parameters
        ----------
        output_path : str
            Path to the output CSV file.
        '''
        if not self.per_case_rows:
            raise ValueError("No data analyzed. Call analyze_all() first.")

        # Flatten some fields for CSV
        csv_rows = []
        for r in self.per_case_rows:
            csv_row = {
                'case_index': r['case_index'],
                'case_name': r['case_name'],
                'image_path': r['image_path'],
                'label_path': r['label_path'],
                'image_depth': r['image_shape'][0],
                'image_height': r['image_shape'][1],
                'image_width': r['image_shape'][2],
                'label_depth': r['label_shape'][0],
                'label_height': r['label_shape'][1],
                'label_width': r['label_shape'][2],
                'ct_min': r['ct_min'],
                'ct_max': r['ct_max'],
                'spacing_x': r['spacing_x'],
                'spacing_y': r['spacing_y'],
                'spacing_z': r['spacing_z'],
                'affine_R': r['affine_codes'][0],
                'affine_A': r['affine_codes'][1],
                'affine_S': r['affine_codes'][2],
                'unique_labels': ';'.join(map(str, r['unique_labels'])),
                'liver_first': r['liver_first'] if r['liver_first'] is not None else '',
                'liver_last': r['liver_last'] if r['liver_last'] is not None else '',
                'tumor_first': r['tumor_first'] if r['tumor_first'] is not None else '',
                'tumor_last': r['tumor_last'] if r['tumor_last'] is not None else '',
                'liver_voxels': r['liver_voxels'],
                'tumor_voxels': r['tumor_voxels'],
                'liver_ratio': r['liver_ratio'],
                'tumor_ratio': r['tumor_ratio'],
                'has_tumor': r['has_tumor']
            }
            csv_rows.append(csv_row)

        # Write CSV
        fieldnames = list(csv_rows[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"CSV exported to: {output_path}")

    def export_aggregate_csv(self, output_path: str) -> None:
        '''
        Export aggregate statistics to a CSV file.
        
        Parameters
        ----------
        output_path : str
            Path to the output CSV file.
        '''
        if self.aggregate_stats is None:
            self.get_aggregate_stats()

        agg = self.aggregate_stats

        # Convert nested structures to strings
        row = {
            'num_volumes': agg['num_volumes'],
            'num_with_tumor': agg['num_with_tumor'],
            'tumor_proportion': agg['tumor_proportion'],
            'shape_mean_D': agg['shape_mean'][0],
            'shape_mean_H': agg['shape_mean'][1],
            'shape_mean_W': agg['shape_mean'][2],
            'shape_median_D': agg['shape_median'][0],
            'shape_median_H': agg['shape_median'][1],
            'shape_median_W': agg['shape_median'][2],
            'shape_std_D': agg['shape_std'][0],
            'shape_std_H': agg['shape_std'][1],
            'shape_std_W': agg['shape_std'][2],
            'spacing_mean_x': agg['spacing_mean'][0],
            'spacing_mean_y': agg['spacing_mean'][1],
            'spacing_mean_z': agg['spacing_mean'][2],
            'spacing_std_x': agg['spacing_std'][0],
            'spacing_std_y': agg['spacing_std'][1],
            'spacing_std_z': agg['spacing_std'][2],
            'orientation_distribution': str(agg['orientation_distribution']),
            'liver_span_mean': agg['liver_span_mean'],
            'liver_span_std': agg['liver_span_std'],
            'tumor_span_mean': agg['tumor_span_mean'],
            'tumor_span_std': agg['tumor_span_std'],
            'liver_ratio_mean': agg['liver_ratio_mean'],
            'liver_ratio_std': agg['liver_ratio_std'],
            'tumor_ratio_mean': agg['tumor_ratio_mean'],
            'tumor_ratio_std': agg['tumor_ratio_std'],
            'ct_min_mean': agg['ct_min_mean'],
            'ct_max_mean': agg['ct_max_mean']
        }

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        logger.info("Aggregate stats exported to: %s", output_path)


def analyse_dataset(datasources: List[Dict[str, str]], 
                         output_csv: Optional[str] = None,
                         output_agg_csv: Optional[str] = None,
                         verbose: bool = True) -> DatasetSummary:
    '''
    Convenience function to run a complete LiTS dataset analysis.
    
    Parameters
    ----------
    datasources : list of dict
        Paired image/label paths from DataCollector
    output_csv : str, optional
        If provided, export per-case rows to this CSV path
    output_agg_csv : str, optional
        If provided, export aggregate stats to this CSV path
    verbose : bool
        If True, print progress and results
    
    Returns
    -------
    DatasetSummary
        The summary object with per_case_rows and aggregate_stats populated
    '''
    summary = DatasetSummary(datasources)
    summary.analyze_all(verbose=verbose)
    summary.get_aggregate_stats()

    if verbose:
        summary.print_table()

    if output_csv:
        summary.export_csv(output_csv)

    if output_agg_csv:
        summary.export_aggregate_csv(output_agg_csv)

    return summary
