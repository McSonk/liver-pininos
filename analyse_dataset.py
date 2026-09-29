"""
Standalone script to run LiTS dataset-wide analysis.

This script analyses all paired volumes in the LiTS dataset (train + test sources
combined) and produces:
1. Per-case progress logging
2. A per-case CSV export (metadata, spacing, orientation, CT range, liver/tumour
   extents, voxel counts) sorted lexicographically by case name

Usage
-----
python analyse_dataset.py [--output-csv PATH] [--dummy]

Examples
--------
# Run with the default output path (<STATS_DIR>/train/per_case_summary.csv)
python analyse_dataset.py

# Custom per-case output path
python analyse_dataset.py --output-csv my_per_case.csv

# Dummy mode (analyse only the first 3 volumes, for a quick smoke test)
python analyse_dataset.py --dummy
"""
print("[analyse_dataset.py] Importing torch. This may take a moment...")
import argparse
import logging
from pathlib import Path

import torch

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.data import analyse_dataset
from idssp.sonk.utils.logger import (configure_logging, get_logger,
                                     install_global_exception_handlers)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset-wide analysis for LiTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-csv", "-o",
        type=str,
        default=None,
        help="Path to save the per-case summary CSV. "
             "Defaults to <STATS_DIR>/train/per_case_summary.csv"
    )

    parser.add_argument(
        "--dummy", "-d",
        action="store_true",
        help="Run in dummy mode for testing purposes. This will perform a quick "
             "analysis with limited data."
    )

    return parser.parse_args()

def _analyse_dataset(
        logger: logging.Logger,
        train_datasource: Path,
        test_datasource: Path,
        per_case_csv_path: Path,
        dummy: bool = False,
    ):
    # Load and pair data
    logger.info("Discovering and pairing image-label volumes...")
    collector = DataCollector()
    collector.read_dir(train_datasource, ds_source='LiTS')
    collector.read_dir(test_datasource, ds_source='LiTS')
    collector.extract_images_and_labels()

    if dummy:
        # In dummy mode, restrict analysis to the first 3 images for quick testing
        logger.info("Dummy mode enabled. Limiting analysis to the first 3 volumes.")
        collector.datasources = collector.datasources[:3]

    logger.info("Found %d paired volumes.\n", len(collector.datasources))

    # Run analysis
    analyse_dataset(collector.datasources, per_case_csv_path)


def main():
    '''
    Main function to execute dataset analysis.
     - Parses command-line arguments
     - Loads and pairs LiTS data (train + test sources)
     - Runs the per-case analysis
     - Exports a single per-case CSV (default <STATS_DIR>/train/per_case_summary.csv)
     - Logs progress through the configured console/file loggers
    '''
    # Parse CLI arguments (was previously defined but never called)
    args = _parse_args()

    cfg = config.init()
    configure_logging(cfg)

    logger = get_logger(__name__)
    install_global_exception_handlers(logger)

    if cfg.STATS_DIR is None:
        logger.error("STATS_DIR is not configured. Please set the 'STATS_DIR' "
                     "environment variable to enable statistics export.")
        return

    logger.debug("=" * 80)
    logger.info("LiTS Dataset-Wide Analysis")
    logger.debug("=" * 80)
    logger.info("LiTS Train Set: %s\n", cfg.CT_ROOT)
    logger.info("LiTS Test Set: %s\n", cfg.CT_TEST)

    logger.info("Creating output directories if they don't exist...")
    per_case_csv_path_train = cfg.STATS_DIR / "train" / "per_case_summary.csv"
    aggregate_csv_path_train = cfg.STATS_DIR / "train" / "aggregate_stats.csv"

    per_case_csv_path_test = cfg.STATS_DIR / "test" / "per_case_summary.csv"
    aggregate_csv_path_test = cfg.STATS_DIR / "test" / "aggregate_stats.csv"

    # Override default per-case CSV path if provided via CLI
    if args.output_csv:
        per_case_csv_path_train = Path(args.output_csv)

    per_case_csv_path_train.parent.mkdir(parents=True, exist_ok=True)
    aggregate_csv_path_train.parent.mkdir(parents=True, exist_ok=True)
    per_case_csv_path_test.parent.mkdir(parents=True, exist_ok=True)
    aggregate_csv_path_test.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting analysis of LiTS Set...")
    _analyse_dataset(
        logger,
        cfg.CT_ROOT,
        cfg.CT_TEST,
        per_case_csv_path_train,
        dummy=args.dummy,
    )

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()