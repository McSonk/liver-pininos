'''
Module for all the disk loading and parsing logic.
This module is responsible for reading the dataset from disk, discovering and
pairing image and label files, and providing the necessary file paths for further
processing in the pipeline.
'''
import glob
import json
from pathlib import Path

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
        self.config = config.get()
        self.d_sets: list[CustomDataset] = []
        '''The plain paths for the dataset'''
        self.datasources: list[dict[str, str]] = []
        '''The paired image and label paths'''

    def read_dir(self, ds_dir: Path, ds_source: str):
        '''
        Reads the directory specified and lists all files.

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
        return self.datasources

    def _load_split(self, file_path: Path) -> tuple[list, list]:
        logger.info("Existing split log found at %s. Loading split...", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        train_names = set(split_data["train"])
        val_names   = set(split_data["val"])

        train_files = [f for f in self.datasources if Path(f["image"]).name in train_names]
        val_files   = [f for f in self.datasources if Path(f["image"]).name in val_names]

        # catch mismatches between JSON and what's actually on disk
        loaded = set(Path(f["image"]).name for f in train_files) | \
                set(Path(f["image"]).name for f in val_files)
        expected = train_names | val_names
        missing_from_disk = expected - loaded
        missing_from_json = set(Path(f["image"]).name for f in self.datasources) - expected

        if missing_from_disk:
            raise FileNotFoundError(
                f"Split JSON references {len(missing_from_disk)} file(s) not found on disk: "
                f"{missing_from_disk}"
            )
        if missing_from_json:
            logger.warning(
                "%d file(s) on disk are not in the split JSON and will be ignored: %s",
                len(missing_from_json), missing_from_json
            )

        return train_files, val_files

    def get_stratified_split(self) -> tuple[list, list]:
        """
        Splits paired data into training and validation sets with stratification
        by tumour burden bin, ensuring proportional representation of:
        - tumour-negative cases
        - micro / low / mid / high burden cases
        """
        if not self.datasources:
            raise ValueError("No data loaded. Call read_dir() first.")

        json_file_name = f"{self.d_sets[0].ds_source}_split_seed{self.config.RANDOM_SEED}.json"
        json_file_path = self.config.SPLIT_DIR / json_file_name
        if not json_file_path.exists():
            raise FileNotFoundError(
                f"Stratified split JSON file not found at {json_file_path}. "
                "Please run the stratification notebook to generate the split and log it."
            )

        train_files, val_files = self._load_split(json_file_path)


        logger.info(
            "Stratified split: %d train / %d val (from %d total)",
            len(train_files), len(val_files), len(self.datasources)
        )

        return train_files, val_files
