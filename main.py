print("[main.py] Importing torch... (This may take a moment)")
import argparse
import logging
import os
import subprocess
from pathlib import Path

import torch
from monai.utils import set_determinism

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.training import ModelBuilder
from idssp.sonk.utils.logger import (get_active_log_file, get_logger,
                                     install_global_exception_handlers,
                                     log_memory_usage)
from idssp.sonk.utils.mail import send_training_email
from idssp.sonk.utils.notifications import send_alert, send_final_alert

# For reproducibility
set_determinism(seed=42)

def _log_gpu_info(cuda_torch_properties, logger: logging.Logger) -> None:
    '''Logs detailed information about the available CUDA devices, including their names,
       total memory, and the GPU that PyTorch is currently using.'''
    pytorch_uuid = getattr(cuda_torch_properties, "uuid", None)
    logger.info(
        "PyTorch sees GPU: %s | UUID: %s",
        cuda_torch_properties.name,
        pytorch_uuid
    )

    if pytorch_uuid is None:
        logger.info(
            "PyTorch CUDA device properties do not expose a UUID on this build; "
            "skipping PCI Bus ID mapping via nvidia-smi."
        )
        return

    # Query nvidia-smi to map UUID -> Physical PCI Bus ID
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to query nvidia-smi: %s", e.stderr)
        return
    except FileNotFoundError:
        logger.warning("nvidia-smi not found. Cannot map GPU UUID to PCI Bus ID.")
        return

    # Find matching GPU
    active_bus_id = None
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[1] == pytorch_uuid:
            active_bus_id = parts[2]
            break

    logger.info("Active physical PCI Bus ID: %s", active_bus_id)

def log_environment_info(config_obj: config.Config, logger: logging.Logger) -> None:
    '''Logs detailed information about the training environment, including PyTorch version,
    CUDA availability and devices, and key configuration parameters.'''
    cuda_properties = None
    logger.info("Model (code) Version: %s", config_obj.VERSION)
    logger.info("Environment Information:")
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        # 1. Get PyTorch device properties (logical index 0 due to CUDA_VISIBLE_DEVICES)
        cuda_properties = torch.cuda.get_device_properties(0)
        logger.info("CUDA device count: %d", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            logger.info("CUDA device %d: %s", i, torch.cuda.get_device_name(i))
        logger.info("Available GPU memory (GB): %d", cuda_properties.total_memory // (1024 ** 3))
    else:
        logger.info("No CUDA devices available.")

    logger.info("Available CPU cores: %s", os.cpu_count())
    logger.info("PyTorch intra-op threads: %d", torch.get_num_threads())
    logger.info("Available CPU memory (GB): %.2f", config_obj.cpu_memory)
    logger.info("Available container memory (GB): %.2f", config_obj.container_memory)

    if cuda_properties is not None:
        _log_gpu_info(cuda_properties, logger)

    logger.info("Device: %s", config_obj.DEVICE)
    logger.info("Batch Size: %d", config_obj.BATCH_SIZE)
    logger.info("RAND_CROP_NUM_SAMPLES: %d (Effective Batch Size: %d)",
                    config_obj.RAND_CROP_NUM_SAMPLES,
                    config_obj.BATCH_SIZE * config_obj.RAND_CROP_NUM_SAMPLES)
    logger.info("Val Batch Size: %d", config_obj.VAL_BATCH_SIZE)
    if config_obj.USE_CACHE_TRAIN_DATASET or config_obj.USE_CACHE_VAL_DATASET:
        logger.info("Cache Num Workers: %d", config_obj.CACHE_NUM_WORKERS)
    logger.info("Data Loader Workers: %d", config_obj.DL_NUM_WORKERS)
    logger.info("Data Root: %s", config_obj.CT_ROOT)
    logger.info("Checkpoint Dir: %s", config_obj.CHECKPOINT_DIR)
    logger.info("Log Dir: %s", config_obj.LOG_DIR)
    logger.info("Persistent Dataset Dir: %s", config_obj.PERSISTENT_DATASET_DIR)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training for tumour segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging purposes",
    )

    parser.add_argument(
        "-fr", "--fast-run",
        action="store_true",
        help="Enable fast run mode with a smaller subset of the data for quick testing",
    )

    parser.add_argument(
        "-r", "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a checkpoint file (.pth) to resume training from",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    verbose = args.verbose
    fast_run = args.fast_run
    resume_path = args.resume

    logger = get_logger(__name__, verbose=verbose)
    # Install global hooks (for logging unhandled exceptions)
    install_global_exception_handlers(logger)

    cfg = config.get()
    log_environment_info(cfg, logger)

    checkpoint_path = None

    if resume_path:
        logger.info("Resume path provided: %s", resume_path)
        if not Path(resume_path).is_file():
            logger.error("Resume checkpoint file not found: %s", resume_path)
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        else:
            logger.info("Checkpoint file found. Will attempt to resume training from this checkpoint.")
            logger.info("Validation will be performed before loading:")
            logger.info("  - Required keys (model_state_dict)")
            logger.info("  - MODEL architecture compatibility (hard-fail on mismatch)")
            logger.info("  - NUM_CLASSES compatibility (hard-fail on mismatch)")
            logger.info("  - Preprocessing settings (warn on mismatch)")
            logger.info("  - Epoch range validation")
            checkpoint_path = Path(resume_path)
    logger.info("Reading directories...")
    loader = DataCollector()
    loader.read_dir(cfg.CT_ROOT, ds_source='LiTS')
    loader.extract_images_and_labels()
    logger.debug("Done! Some information about the environment:")
    logger.debug("ISO spacing: %s", cfg.ISO_SPACING)
    logger.debug("Training patch size: %s", cfg.TRAIN_PATCH_SIZE)
    val_patch_size = getattr(cfg, "VAL_PATCH_SIZE", None)
    if val_patch_size is not None and config.is_limited_env():
        logger.debug("Validation patch size: %s", val_patch_size)
    logger.debug("Batch size: %d", cfg.BATCH_SIZE)
    logger.debug("Number of epochs: %d", cfg.NUM_EPOCHS)
    log_memory_usage(logger, prefix="At program start: ")
    logger.debug("Splitting data into train and val sets...")
    train_files, val_files = loader.get_stratified_split()
    logger.debug("Initializing model builder...")

    if fast_run or config.is_limited_env(include_vram=True):
        logger.info("Limited environment detected. Using a subset of the data for quick testing.")
        train_files = train_files[:2]  # Use only 2 samples for training
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
        f"Environment: {cfg.ENV}\n"
        f"Device: {cfg.DEVICE}\n"
        f"ISO Spacing: {cfg.ISO_SPACING}\n"
        f"Patch Size: {cfg.TRAIN_PATCH_SIZE}\n"
        f"Expected Epochs: {cfg.NUM_EPOCHS}"
    )

    send_training_email(subject=INIT_TITLE, body=init_body)
    send_alert(INIT_TITLE, init_body)

    try:
        cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        builder = ModelBuilder()
        builder.init_data_loaders(train_files, val_files)

        builder.init_model()
        logger.info("Model initialized. Starting training...")
        builder.train(resume_path=checkpoint_path)
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
    send_final_alert(
        success_title,
        success_body
    )
