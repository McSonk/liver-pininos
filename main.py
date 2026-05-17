print("[main.py] Importing torch... (This may take a moment)")
import torch
from monai.utils import set_determinism

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.training import ModelBuilder
from idssp.sonk.utils.logger import (get_active_log_file, get_logger,
                                     install_global_exception_handlers,
                                     log_memory_usage)
from idssp.sonk.utils.mail import send_training_email
from idssp.sonk.utils.notifications import send_alert

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

    if config.is_limited_env(include_vram=True):
        logger.info("Limited environment detected. Using a subset of the data for quick testing.")
        train_files = train_files[:4]  # Use only 4 samples for training
        val_files = val_files[:2]      # Use only 2 samples for validation

    logger.info("%d training files", len(train_files))
    for file in train_files:
        logger.debug(file)

    logger.info("%d validation files", len(val_files))
    for file in val_files:
        logger.debug(file)

    INIT_TITLE = "[Thesis] Training Pipeline Starting"
    init_body = (
        f"Training has commenced.\n"
        f"Environment: {config.ENV}\n"
        f"Device: {config.DEVICE}\n"
        f"ISO Spacing: {config.ISO_SPACING}\n"
        f"Patch Size: {config.TRAIN_PATCH_SIZE}\n"
        f"Expected Epochs: {config.NUM_EPOCHS}"
    )

    send_training_email(subject=INIT_TITLE, body=init_body)
    send_alert(INIT_TITLE, init_body)

    try:
        builder = ModelBuilder()
        builder.init_data_loaders(train_files, val_files)

        builder.init_model()
        logger.info("Model initialized. Starting training...")
        builder.train()
    except KeyboardInterrupt:
        logger.warning("Training setup interrupted by user (Ctrl+C) before training began.")
        keyboard_title = "[Thesis] Training Interrupted by User"
        keyboard_body = (
            "Training was manually stopped before or during initialization. "
            "No last-epoch checkpoint is guaranteed to have been saved. "
            "See logs for details."
        )
        send_training_email(
            subject=keyboard_title,
            body=keyboard_body,
            log_file=get_active_log_file(),
            wait_for_completion=True,
            timeout=20.0
        )
        send_alert(
            keyboard_title,
            keyboard_body,
            file_path=get_active_log_file(),
            sync=True,
            timeout=20.0)
        raise
    except Exception as e:
        logger.error("An error occurred during training: %s", e)
        error_title = "[Thesis] Training Pipeline Failed"
        error_body = (
            f"Training terminated unexpectedly.\n"
            f"Error: {str(e)}\n\n"
            f"Check the attached log file for stack traces and debugging information."
        )
        send_training_email(
            subject=error_title,
            body=error_body,
            log_file=get_active_log_file(),
            wait_for_completion=True,
            timeout=20.0
        )
        send_alert(
            error_title,
            error_body,
            file_path=get_active_log_file(),
            sync=True,
            timeout=30.0)
        raise

    success_title = "[Thesis] Training Pipeline Completed"
    success_body = (
        "Training has completed successfully!\n"
        "Check the logs for final metrics and details."
    )
    logger.info("Training completed successfully!")
    send_training_email(
        subject=success_title,
        body=success_body,
        log_file=get_active_log_file(),
        wait_for_completion=True,
        timeout=20.0
    )
    send_alert(
        success_title,
        success_body,
        file_path=get_active_log_file(),
        sync=True,
        timeout=30.0)
