import os
from pathlib import Path
import glob

class CustomDataset:
    SUPPORTED_SOURCES = ["LiTS"]
    def __init__(self, ds_source: str, files: list = None):
        self.ds_source = ds_source
        self.files = files

    def set_files(self, files: list):
        self.files = files

    def get_lits_paths(self):
        '''
        Parses the file paths for the LiTS dataset to separate image and label files.

        Returns
        -------
        images: list
            List of file paths for the image volumes.
        labels: list
            List of file paths for the corresponding label volumes.
        '''
        if self.files is None:
            raise ValueError("Files have not been set for this dataset.")

        images = []
        labels = []

        for file in self.files:
            if "volume" in file:
                images.append(file)
            elif "segmentation" in file:
                labels.append(file)

        return images, labels

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

class DataLoader:
    def __init__(self):
        self.images = []
        self.labels = []
        self.d_sets = []

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
        files = glob.glob(str(ds_dir / "*"))
        print(f"Found {len(files)} files in the directory.")

        self.d_sets.append(CustomDataset(ds_source, files))

    def extract_images_and_labels(self):
        '''
        Extracts image and label file paths from the dataset.

        Returns
        -------
        `images`: list
            List of file paths for the image volumes.
        `labels`: list
            List of file paths for the corresponding label volumes.
        '''
        if not self.d_sets:
            raise ValueError("No datasets have been loaded. Please call read_dir() first.")

        for ds in self.d_sets:
            images, labels = ds.get_images_and_labels()
            self.images.extend(images)
            self.labels.extend(labels)

        print(f"Extracted {len(self.images)} image files and {len(self.labels)} label files from the dataset.")
