"""
Standalone script to run LiTS dataset-wide analysis.

This script analyzes all paired volumes in the LiTS dataset, producing:
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

import argparse

from idssp.sonk import config
from idssp.sonk.disk.loader import DataCollector
from idssp.sonk.model.data import analyse_dataset
from idssp.sonk.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    '''
    Main function to execute dataset analysis.
     - Parses command-line arguments
     - Loads and pairs LiTS data
     - Runs analysis and outputs results
     - Exports CSV files for further use
     - Provides terminal output unless --no-verbose is set
    '''

    if config.STATS_DIR is None:
        logger.error("STATS_DIR is not configured. Please set the 'STATS_DIR' "
                     "environment variable to enable statistics export.")
        return

    per_case_csv_path = config.PER_CASE_TRAIN_STATS_FILE
    aggregate_csv_path = config.STATS_DIR / "aggregate_stats.csv"

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
    logger.info("Data root: %s\n", config.CT_ROOT)

    # Load and pair data
    logger.info("Discovering and pairing image-label volumes...")
    collector = DataCollector()
    collector.read_dir(config.CT_ROOT, ds_source='LiTS')
    collector.extract_images_and_labels()

    logger.info("Found %d paired volumes.\n", len(collector.datasources))

    # Run analysis
    analyse_dataset(
        datasources=collector.datasources,
        output_csv=per_case_csv_path,
        output_agg_csv=aggregate_csv_path,
        verbose=not args.no_verbose
    )

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
