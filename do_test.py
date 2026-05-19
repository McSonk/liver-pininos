#!/usr/bin/env python3
"""
Test-time evaluation entry point for HCC tumour segmentation thesis.

Usage:
    python do_test.py --checkpoint /path/to/best_model.pth
"""
print("[do_test.py] Importing torch... (This may take a moment)")
import torch
import argparse
import sys
from pathlib import Path

from monai.utils import set_determinism

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.validate import TestEvaluator
from idssp.sonk.utils.logger import (get_logger,
                                     install_global_exception_handlers)

# For reproducibility
set_determinism(seed=42)

# Initialize logger
logger = get_logger(__name__)
# Install global hooks (for logging unhandled exceptions)
install_global_exception_handlers(logger)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run test-time evaluation for HCC tumour segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file)",
    )
    return parser.parse_args()

def _main(args: argparse.Namespace):
    cfg = config.init()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error("Checkpoint file not found: %s", checkpoint)
        return 1
    logger.info("[Validation] Reading directories...")
    loader = DataCollector()
    loader.read_dir(cfg.CT_TEST, ds_source='LiTS')
    test_files = loader.extract_images_and_labels()
    if not test_files:
        logger.error(
            "No valid test image/label pairs were found in %s. "
            "Check the test dataset configuration and contents before running evaluation.",
            cfg.CT_TEST,
        )
        return 1
    logger.debug("Done! Some information about the environment:")
    logger.debug("ISO spacing: %s", cfg.ISO_SPACING)
    logger.debug("Training patch size: %s", cfg.TRAIN_PATCH_SIZE)

    logger.info("%d test files", len(test_files))
    for file in test_files:
        logger.debug(file)

    evaluator = TestEvaluator(checkpoint)
    evaluator.load_checkpoint()
    results = evaluator.run_inference(test_files)
    evaluator.generate_report(results)


if __name__ == "__main__":
    exit_code = _main(_parse_args())
    sys.exit(exit_code)
