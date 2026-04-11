'''
Module for all the disk loading and parsing logic.
This module is responsible for reading the dataset from disk, discovering and
pairing image and label files, and providing the necessary file paths for further
processing in the pipeline.
'''

import datetime
import glob
import json
from pathlib import Path

from monai.data import partition_dataset

from idssp.sonk import config


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
                    print(f"Warning: Label file not found for image {file_path.name}. Expected label path: {label_filename}")
            else:
                not_volumes.append(file)

        # Check for any files that were not identified as volumes
        for paired in paired_files:
            if paired["label"] in not_volumes:
                not_volumes.remove(paired["label"])

        if not_volumes:
            print("Warning: The following files were not identified as image volumes and may be unpaired:")
            for file in not_volumes:
                print(f" - {file}")
        return paired_files

    def get_images_and_labels(self):
        '''
        Parses the file paths to separate image and label files.

        Returns
        -------
        images: list
            List of file paths for the image volumes.
        labels: list
            List of file paths for the corresponding label volumes.
        '''
        if self.files is None:
            raise ValueError("Files have not been set for this dataset.")
        if self.ds_source == "LiTS":
            return self.get_lits_paths()
        else:
            raise ValueError(f"Dataset source [{self.ds_source}] is not supported. Please choose from {CustomDataset.SUPPORTED_SOURCES}")

class DataCollector:
    '''
    DataCollector class responsible for the global loading of the dataset.
    This class manages the overall process of reading the dataset from the
    different specified directories.
    '''
    def __init__(self):
        self.d_sets = []
        '''The plain paths for the dataset'''
        self.datasources = []
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
            raise ValueError(f"Dataset source [{ds_source}] is not supported. Please choose from {CustomDataset.SUPPORTED_SOURCES}")
        print(f"Reading directory: {ds_dir}")
        if not ds_dir.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {ds_dir}")
        files = sorted(glob.glob(str(ds_dir / "*")))

        if len(files) == 0:
            raise ValueError(f"No files found in the directory: {ds_dir}")

        print(f"Found {len(files)} files in the directory.")

        if len(files) % 2 != 0:
            print("Warning:")
            print(f"Found an odd number of files ({len(files)}) in the directory.")
            print("This may indicate that some image-label pairs are incomplete.")


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
            paired_files = ds.get_images_and_labels()
            self.datasources.extend(paired_files)

        print(f"Extracted {len(self.datasources)} image-label pairs from the dataset.")

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

        print(f"Split dataset into {len(train_files)} training and {len(val_files)} validation samples.")

        # Log split for thesis appendix
        log_path = Path(f"{self.d_sets[0].ds_source}_split_seed{config.RANDOM_SEED}.json")
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
        print(f"Split logged to {log_path}")

        return train_files, val_files
