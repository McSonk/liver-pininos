#!/usr/bin/env python3
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
from idssp.sonk.model.data import analyze_lits_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LiTS dataset and produce summary statistics"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="lits_per_case_summary.csv",
        help="Path for per-case CSV export (default: lits_per_case_summary.csv)"
    )
    parser.add_argument(
        "--output-agg-csv",
        type=str,
        default="lits_aggregate_stats.csv",
        help="Path for aggregate stats CSV export (default: lits_aggregate_stats.csv)"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Suppress terminal output (useful for batch processing)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LiTS Dataset-Wide Analysis")
    print("=" * 80)
    print(f"Data root: {config.CT_ROOT}")
    print()
    
    # Load and pair data
    print("Discovering and pairing image-label volumes...")
    collector = DataCollector()
    collector.read_dir(config.CT_ROOT, ds_source='LiTS')
    collector.extract_images_and_labels()
    
    print(f"Found {len(collector.datasources)} paired volumes.")
    print()
    
    # Run analysis
    analyze_lits_dataset(
        datasources=collector.datasources,
        output_csv=args.output_csv,
        output_agg_csv=args.output_agg_csv,
        verbose=not args.no_verbose
    )
    
    print()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
