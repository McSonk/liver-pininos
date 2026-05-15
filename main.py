print("[main.py] Importing torch... (This may take a moment)")
import torch
from monai.utils import set_determinism

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.data import DataWrapper
from idssp.sonk.model.training import ModelBuilder
from idssp.sonk.utils.logger import (get_logger,
                                     install_global_exception_handlers,
                                     log_memory_usage, get_active_log_file)
from idssp.sonk.utils.mail import send_training_email

# For reproducibility
set_determinism(seed=42)

# Initialize logger
logger = get_logger(__name__)
# Install global hooks (for logging unhandled exceptions)
install_global_exception_handlers(logger)

if __name__ == "__main__":
    logger.info("Reading directories...")
    loader = DataCollector()
    loader.read_dir(config.CT_ROOT, ds_source='LiTS')
    loader.extract_images_and_labels()
    logger.debug("Done! Some information about the environment:")
    logger.debug("ISO spacing: %s", config.ISO_SPACING)
    logger.debug("Training patch size: %s", config.TRAIN_PATCH_SIZE)
    val_patch_size = getattr(config, "VAL_PATCH_SIZE", None)
    if val_patch_size is not None:
        logger.debug("Validation patch size: %s", val_patch_size)
    logger.debug("Batch size: %d", config.BATCH_SIZE)
    logger.debug("Number of epochs: %d", config.NUM_EPOCHS)
    log_memory_usage(logger, prefix="At program start: ")
    logger.debug("Splitting data into train and val sets...")
    train_files, val_files = loader.get_stratified_split()
    logger.debug("Initializing model builder...")

    if config.is_limited_env(include_vram=False):
        logger.info("Limited environment detected. Using a subset of the data for quick testing.")
        train_files = train_files[:4]  # Use only 4 samples for training
        val_files = val_files[:2]      # Use only 2 samples for validation

    logger.info("%d training files", len(train_files))
    for file in train_files:
        logger.debug(file)

    logger.info("%d validation files", len(val_files))
    for file in val_files:
        logger.debug(file)

    send_training_email(
        subject="[Thesis] Training Pipeline Started",
        body=(
            f"Training has commenced.\n"
            f"Environment: {config.ENV}\n"
            f"Device: {config.DEVICE}\n"
            f"ISO Spacing: {config.ISO_SPACING}\n"
            f"Patch Size: {config.TRAIN_PATCH_SIZE}\n"
            f"Expected Epochs: {config.NUM_EPOCHS}"
        )
    )

    try:
        builder = ModelBuilder()
        builder.init_data_loaders(train_files, val_files)

        builder.init_model()
        logger.info("Model initialized. Starting training...")
        builder.train()
    except Exception as e:
        logger.error("An error occurred during training: %s", e, exc_info=True)
        send_training_email(
            subject="[Thesis] Training Pipeline Failed",
             body=(
                f"Training terminated unexpectedly.\n"
                f"Error: {str(e)}\n\n"
                f"Check the attached log file for stack traces and debugging information."
            ),
            log_file=get_active_log_file()
        )
    else:
        logger.info("Training completed successfully!")
        send_training_email(
            subject="[Thesis] Training Pipeline Completed",
            body="Training has completed successfully!",
            log_file=get_active_log_file()
        )
