"""
Standalone script to run LiTS dataset-wide analysis.

This script analyses all paired volumes in the LiTS dataset, producing:
1. A terminal table with per-case metadata
2. Aggregate statistics (shapes, spacing, orientations, tumor prevalence, etc.)
3. CSV exports for thesis analysis

Usage
-----
python analyze_lits_dataset.py [--output-csv PATH] [--output-agg-csv PATH] [--no-verbose]

Examples
--------
# Run with default output files
python analyze_lits_dataset.py

# Custom output paths
python analyze_lits_dataset.py --output-csv my_per_case.csv --output-agg-csv my_stats.csv

# Quiet mode (only CSV export)
python analyze_lits_dataset.py --no-verbose --output-csv data.csv
"""
print("[analyse_dataset.py] Importing torch. This may take a moment...")
import logging

import torch
import argparse

from pathlib import Path

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.data import analyse_dataset
from idssp.sonk.utils.logger import (configure_logging, get_logger,
                                     install_global_exception_handlers)

def _analyse_dataset(
        logger: logging.Logger,
        datasource: Path,
        per_case_csv_path: Path,
        aggregate_csv_path: Path,
        verbose: bool):
    # Load and pair data
    logger.info("Discovering and pairing image-label volumes...")
    collector = DataCollector()
    collector.read_dir(datasource, ds_source='LiTS')
    collector.extract_images_and_labels()

    logger.info("Found %d paired volumes.\n", len(collector.datasources))

    # Run analysis
    analyse_dataset(
        datasources=collector.datasources,
        output_csv=str(per_case_csv_path),
        output_agg_csv=str(aggregate_csv_path),
        verbose=verbose
    )


def main():
    '''
    Main function to execute dataset analysis.
     - Parses command-line arguments
     - Loads and pairs LiTS data
     - Runs analysis and outputs results
     - Exports CSV files for further use
     - Provides terminal output unless --no-verbose is set
    '''

    cfg = config.init()
    configure_logging(cfg)

    logger = get_logger(__name__)
    install_global_exception_handlers(logger)

    if cfg.STATS_DIR is None:
        logger.error("STATS_DIR is not configured. Please set the 'STATS_DIR' "
                     "environment variable to enable statistics export.")
        return

    parser = argparse.ArgumentParser(
        description="Analyse LiTS dataset and produce summary statistics"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Suppress terminal output (useful for batch processing)"
    )

    args = parser.parse_args()

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

    per_case_csv_path_train.parent.mkdir(parents=True, exist_ok=True)
    aggregate_csv_path_train.parent.mkdir(parents=True, exist_ok=True)
    per_case_csv_path_test.parent.mkdir(parents=True, exist_ok=True)
    aggregate_csv_path_test.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting analysis of LiTS Train Set...")
    _analyse_dataset(
        logger,
        cfg.CT_ROOT,
        per_case_csv_path_train,
        aggregate_csv_path_train,
        not args.no_verbose)

    logger.info("Starting analysis of LiTS Test Set...")
    _analyse_dataset(
        logger,
        cfg.CT_TEST,
        per_case_csv_path_test,
        aggregate_csv_path_test,
        not args.no_verbose)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
