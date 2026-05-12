'''
Module for all the disk loading and parsing logic.
This module is responsible for reading the dataset from disk, discovering and
pairing image and label files, and providing the necessary file paths for further
processing in the pipeline.
'''

import datetime
import glob
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from monai.data import partition_dataset

from idssp.sonk import config
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)


class CustomDataset:
    '''
    Custom dataset class to handle different dataset sources and their specific file
    naming conventions. This class is responsible for discovering and pairing image
    and label files based on the dataset source, and providing the necessary file paths
    for further processing in the pipeline.

    Currently supports the following dataset sources:
    - "LiTS": The Liver Tumor Segmentation Challenge dataset
    '''
    SUPPORTED_SOURCES = ["LiTS"]

    def __init__(self, ds_source: str, files: list[str] = None):
        self.ds_source = ds_source
        self.files: list[str] = files
        '''The list of file paths for the dataset, sorted'''

    def discover_and_pair(self) -> list[dict[str, str]]:
        '''
        Discovers and pairs image and label files based on the dataset source.

        This method is responsible for identifying the image and label files in the dataset
        and pairing them together based on their naming conventions. The specific logic for
        pairing may vary depending on the dataset source.

        Returns
        -------
        paired_files: list of dict[str, str]
            A list of dictionaries, each containing the paths for the image and label files.
            Example:
            [
                {
                    "image": "path/to/image_volume.nii",
                    "label": "path/to/segmentation_volume.nii"
                },
                ...
            ]
        '''

        if self.files is None:
            raise ValueError("Files have not been set for this dataset.")

        if self.ds_source == "LiTS":
            return self.get_lits_paths()

        raise ValueError(f"Dataset source [{self.ds_source}] is not supported. Please choose from {CustomDataset.SUPPORTED_SOURCES}")

    def get_lits_paths(self) -> list[dict[str, str]]:
        '''
        Parses the file paths for the LiTS dataset to separate image and label files.

        The LiTS dataset typically has a specific naming convention where image volumes
        and their corresponding segmentation volumes can be identified and paired.

        Returns
        -------
        paired_files: list of dict[str, str]
            A list of dictionaries, each containing the paths for the image and label files.
            Example:
            [
                {
                    "image": "path/to/image_volume.nii",
                    "label": "path/to/segmentation_volume.nii"
                },
                ...
            ]
        '''
        paired_files = []
        not_volumes = []
        for file in self.files:
            file_path = Path(file)
            if "volume" in file_path.stem:
                label_filename = file_path.name.replace("volume", "segmentation")
                label_path = file_path.parent / label_filename

                if label_path.exists():
                    paired_files.append({
                        "image": str(file_path),
                        "label": str(label_path)
                    })
                else:
                    logger.warning("Label file not found for image %s. "
                                   "Expected label path: %s",
                                   file_path.name, label_filename)
            else:
                not_volumes.append(file)

        # Check for any files that were not identified as volumes
        for paired in paired_files:
            if paired["label"] in not_volumes:
                not_volumes.remove(paired["label"])

        if not_volumes:
            logger.warning("Warning: The following files were not identified as"
                           " image volumes and may be unpaired:")
            for file in not_volumes:
                logger.warning(" - {%s}", file)
        return paired_files

class DataCollector:
    '''
    DataCollector class responsible for the global loading of the dataset.
    This class manages the overall process of reading the dataset from the
    different specified directories.
    '''
    def __init__(self):
        self.d_sets: list[CustomDataset] = []
        '''The plain paths for the dataset'''
        self.datasources: list[dict[str, str]] = []
        '''The paired image and label paths'''

    def read_dir(self, ds_dir: Path, ds_source: str):
        '''
        Reads the directory specified in config.CT_ROOT and lists all files.

        Params
        -----
        `ds_dir`: Path
            The directory to read.
        `ds_source`: str
            The source of the dataset, e.g., "LiTS".

            Currently the following dataset sources are supported:
            - "LiTS": The Liver Tumor Segmentation Challenge dataset
        '''

        if ds_source not in CustomDataset.SUPPORTED_SOURCES:
            raise ValueError(f"Dataset source [{ds_source}] is not supported. "
                             "Please choose from {CustomDataset.SUPPORTED_SOURCES}")
        logger.info("Reading directory: %s", ds_dir)
        if not ds_dir.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {ds_dir}")
        files = sorted(glob.glob(str(ds_dir / "*")))

        if len(files) == 0:
            raise ValueError(f"No files found in the directory: {ds_dir}")

        logger.info("Found %d files in the directory.", len(files))

        if len(files) % 2 != 0:
            logger.warning("Warning:")
            logger.warning("Found an odd number of files (%d) in the directory.", len(files))
            logger.warning("This may indicate that some image-label pairs are incomplete.")


        self.d_sets.append(CustomDataset(ds_source, files))

    def extract_images_and_labels(self) -> list[dict[str, str]]:
        '''
        Extracts image and label file paths from the dataset.

        Returns
        -------
        paired_files: list of dict[str, str]
            A list of dictionaries, each containing the paths for the image and label files.
            Example:
            [
                {
                    "image": "path/to/image_volume.nii",
                    "label": "path/to/segmentation_volume.nii"
                },
                ...
            ]
        '''
        if not self.d_sets:
            raise ValueError("No datasets have been loaded. Please call read_dir() first.")

        for ds in self.d_sets:
            paired_files = ds.discover_and_pair()
            self.datasources.extend(paired_files)

        logger.debug("Extracted %d image-label pairs from the dataset.", len(self.datasources))

    def get_reproducible_split(self, train_ratio: float = 0.8) -> tuple[list, list]:
        """
        Splits the paired data into training and validation sets deterministically.
        
        Returns
        -------
        train_files: list[dict]
        val_files: list[dict]
        """
        if not self.datasources:
            raise ValueError("No data loaded. Call read_dir() first.")

        train_files, val_files = partition_dataset(
            data=self.datasources,
            ratios=[train_ratio, 1.0 - train_ratio],
            shuffle=True,
            seed=config.RANDOM_SEED
        )

        logger.info("Split dataset into %d training and %d validation samples.",
                    len(train_files), len(val_files))

        # Log split for thesis appendix
        log_path = config.SPLIT_DIR / f"{self.d_sets[0].ds_source}_split_seed{config.RANDOM_SEED}.json"
        log_path.write_text(
            json.dumps(
                {
                    "train": [Path(f["image"]).name for f in train_files],
                    "val":   [Path(f["image"]).name for f in val_files],
                    "creation_date": datetime.datetime.now().isoformat()
                },
                indent=2
            ),
            encoding="utf-8"
        )
        logger.info("Split logged to %s", log_path)

        return train_files, val_files

    def _assign_bin(self, row) -> str:
        '''Assigns a tumour burden bin label based on the tumor-to-liver ratio.

        - "none": No tumor present (tumor_to_liver_ratio = 0)
        - "micro": Very low tumor burden (0 < tumor_to_liver_ratio < 0.2%)
        - "low": Low tumor burden (0.2 % ≤ tumor_to_liver_ratio < 1%)
        - "mid": Moderate tumor burden (1% ≤ tumor_to_liver_ratio < 6%)
        - "high": High tumor burden (tumor_to_liver_ratio ≥ 6%)
        '''
        if not row["has_tumor"]:
            return "none"
        r = row["tumor_to_liver_ratio"]
        if r < 0.002:  return "micro"
        if r < 0.01:   return "low"
        if r < 0.06:   return "mid"
        return "high"

    def get_stratified_split(self, train_ratio: float = 0.8) -> tuple[list, list]:
        """
        Splits paired data into training and validation sets with stratification
        by tumour burden bin, ensuring proportional representation of:
        - tumour-negative cases
        - micro / low / mid / high burden cases
        """
        if not self.datasources:
            raise ValueError("No data loaded. Call read_dir() first.")

        # --- Build burden bin lookup from per-case CSV ---

        if not config.PER_CASE_TRAIN_STATS_FILE.exists():
            raise FileNotFoundError(
                f"per_case_summary.csv not found at {config.PER_CASE_TRAIN_STATS_FILE}. "
                "Run analyse_dataset.py first."
            )

        stats = pd.read_csv(config.PER_CASE_TRAIN_STATS_FILE)

        # add classification 
        stats["burden_bin"] = stats.apply(self._assign_bin, axis=1)
        # Map volume filename → burden bin
        bin_lookup = dict(zip(stats["case_name"], stats["burden_bin"]))
        # produces a mapping like {"volume-01.nii": "low", "volume-02.nii": "high", ...}

        # --- Group datasources by bin ---
        bins: dict[str, list] = defaultdict(list)

        for ds in self.datasources:
            fname = Path(ds["image"]).name
            b = bin_lookup.get(fname)
            if b is None:
                raise ValueError(f"No stats entry for {fname}. Check if the filename matches the case_name in the CSV.")
            else:
                bins[b].append(ds)

        # --- Stratified split: partition each bin independently ---
        train_files, val_files = [], []
        bin_order = ["none", "micro", "low", "mid", "high"]

        for bin_name in bin_order:
            group = bins.get(bin_name, [])
            if not group:
                logger.warning("No samples found for bin '%s'. Skipping this bin.", bin_name)
                continue
            t, v = partition_dataset(
                data=group,
                ratios=[train_ratio, 1.0 - train_ratio],
                shuffle=True,
                seed=config.RANDOM_SEED
            )
            train_files.extend(t)
            val_files.extend(v)
            logger.debug(
                "Bin %-5s: %2d total → %2d train / %2d val",
                bin_name, len(group), len(t), len(v)
            )

        logger.info(
            "Stratified split: %d train / %d val (from %d total)",
            len(train_files), len(val_files), len(self.datasources)
        )

        # --- Log split for thesis appendix ---
        log_path = config.SPLIT_DIR / f"{self.d_sets[0].ds_source}_split_seed{config.RANDOM_SEED}.json"
        log_path.write_text(
            json.dumps(
                {
                    "train": [Path(f["image"]).name for f in train_files],
                    "val":   [Path(f["image"]).name for f in val_files],
                    "creation_date": datetime.datetime.now().isoformat(),
                    "stratified_by": "tumor_to_liver_ratio_bin",
                    "bins": {
                        b: [Path(f["image"]).name for f in bins[b]]
                        for b in bin_order if b in bins
                    }
                },
                indent=2
            ),
            encoding="utf-8"
        )
        logger.info("Split logged to %s. Next time you run the script, it will use this split.", log_path)

        return train_files, val_files
