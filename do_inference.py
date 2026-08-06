#!/usr/bin/env python3
"""
Test-time inference entry point for HCC tumour segmentation thesis.

This script loads a trained model checkpoint, runs full-volume sliding window 
inference on the test dataset, and saves the raw predictions in the original 
scanner space (NIfTI format). It does NOT compute metrics or apply 
morphological post-processing.

Usage:
    python do_inference.py --checkpoint /path/to/best_model.pth
"""
print("[do_inference.py] Importing torch... (This may take a moment)")
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from monai.utils import set_determinism

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.inferer import InferenceEngine
from idssp.sonk.utils.logger import (configure_logging, get_logger,
                                     install_global_exception_handlers)

# For reproducibility
set_determinism(seed=42)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test-time inference for HCC tumour segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-chk",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save the raw NIfTI predictions. "
             "Defaults to <RUN_DIR>/test_predictions."
    )
    return parser.parse_args()

def log_environment_info(config_obj: config.Config, logger: logging.Logger) -> None:
    '''Logs detailed information about the runtime environment, including PyTorch version,
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

def _main(args: argparse.Namespace):
    cfg = config.init(mode=config.Mode.TEST)
    configure_logging(cfg)
    # Initialize logger
    logger = get_logger(__name__)
    # Install global hooks (for logging unhandled exceptions)
    install_global_exception_handlers(logger)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error("Checkpoint file not found: %s", checkpoint)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else cfg.RUN_DIR / "test_predictions"

    logger.info("This is the test-time inference script for tumour segmentation. "
                "It will load the specified model checkpoint, run full-volume inference "
                "on the test dataset, and save the raw predictions in the original "
                "scanner space.")
    logger.info("Using checkpoint: %s", checkpoint)
    logger.info("Predictions will be saved to: %s", output_dir)

    log_environment_info(cfg, logger)

    logger.info("[Inference] Reading directories...")
    loader = DataCollector()
    loader.read_dir(cfg.CT_TEST, ds_source='LiTS')
    test_files = loader.extract_images_and_labels()

    if not test_files:
        logger.error(
            "No valid test image/label pairs were found in %s. "
            "Check the test dataset configuration and contents before running inference.",
            cfg.CT_TEST,
        )
        return 1

    logger.debug("Done! Some information about the environment:")
    logger.debug("ISO spacing: %s", cfg.ISO_SPACING)
    logger.debug("Training patch size: %s", cfg.TRAIN_PATCH_SIZE)

    logger.info("%d test files found.", len(test_files))
    for file in test_files:
        logger.debug(file)

    engine = InferenceEngine(checkpoint)
    engine.load_checkpoint()
    engine.run_inference(test_files, output_dir)

    logger.info("Inference complete. Raw predictions saved to: %s", output_dir)
    return 0


if __name__ == "__main__":
    exit_code = _main(_parse_args())
    sys.exit(exit_code)
