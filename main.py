from monai.utils import set_determinism

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.data import DataWrapper
from idssp.sonk.model.training import ModelBuilder
from idssp.sonk.utils.logger import get_logger

# For reproducibility
set_determinism(seed=42)

# Initialize logger
logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Reading directories...")
    loader = DataCollector()
    loader.read_dir(config.CT_ROOT, ds_source='LiTS')
    loader.extract_images_and_labels()
    logger.debug("Splitting data into train and val sets...")
    train_files, val_files = loader.get_reproducible_split()
    logger.debug("Initializing data wrapper and model builder...")
    wrapper = DataWrapper()

    if config.is_limited_env():
        logger.info("Limited environment detected. Using a subset of the data for quick testing.")
        train_files = train_files[:2]  # Use only 2 samples for training
        val_files = val_files[:2]      # Use only 2 samples for validation

    logger.info("%d training files", len(train_files))
    for file in train_files:
        logger.debug(file)

    logger.info("%d validation files", len(val_files))
    for file in val_files:
        logger.debug(file)

    builder = ModelBuilder()
    builder.init_data_loaders(train_files, val_files)

    builder.init_model()
    logger.info("Model initialized. Starting training...")
    builder.train()
