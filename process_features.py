"""
Feature engineering script to process raw data into ML-ready features.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.features.feature_engineering import FeatureEngineer
from src.data_collection.data_loader import DataLoader
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def process_historical_data(input_file: str = None, output_file: str = None):
    """
    Process historical data through feature engineering.
    
    Args:
        input_file: Input CSV file (default: latest historical data)
        output_file: Output CSV file (default: auto-generated name)
    """
    print("Feature Engineering - Processing Historical Data")
    
    if input_file is None:
        # Find the latest historical data file
        historical_files = list(RAW_DATA_DIR.glob("warriors_historical_*.csv"))
        if not historical_files:
            print(f"Error: No historical data files found in {RAW_DATA_DIR}")
            return
        input_file = max(historical_files, key=lambda p: p.stat().st_mtime)
        print(f"Using input file: {input_file.name}")
    else:
        input_file = Path(input_file)
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            return
    
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} games")
    print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    # Initialize feature engineer
    print("\nInitializing feature engineer...")
    engineer = FeatureEngineer()
    
    # Engineer features
    print("Engineering features...")
    print("  - Basic features (win/loss, percentages, derived stats)")
    print("  - Rolling statistics (5, 10, 20 game windows)")
    print("  - Contextual features (home/away, rest days, season phase)")
    print("  - Streak features (win/loss streaks)")
    
    processed_df = engineer.engineer_features(df)
    
    print(f"\nFeature engineering complete!")
    print(f"Original columns: {len(df.columns)}")
    print(f"Processed columns: {len(processed_df.columns)}")
    print(f"New features added: {len(processed_df.columns) - len(df.columns)}")
    
    # Determine output file
    if output_file is None:
        # Generate output filename based on input
        input_name = input_file.stem
        output_file = PROCESSED_DATA_DIR / f"{input_name}_processed.csv"
    else:
        output_file = Path(output_file)
    
    print(f"\nSaving processed data to {output_file}...")
    DataLoader.save_processed_data(processed_df, output_file.name)
    
    print("Feature Engineering Summary")
    print(f"Total games processed: {len(processed_df)}")
    print(f"Total features: {len(processed_df.columns)}")
    
    # Show sample of new features
    new_features = [col for col in processed_df.columns if col not in df.columns]
    print(f"\nNew features created: {len(new_features)}")
    if new_features:
        print("Sample features:")
        for feat in new_features[:10]:
            print(f"  - {feat}")
        if len(new_features) > 10:
            print(f"  ... and {len(new_features) - 10} more")
    
    # Check for missing values
    missing_pct = processed_df.isnull().sum() / len(processed_df) * 100
    high_missing = missing_pct[missing_pct > 10]
    if len(high_missing) > 0:
        print(f"\nWarning: {len(high_missing)} features have >10% missing values")
    
    return processed_df


def process_current_season_data(input_file: str = None, output_file: str = None):
    """
    Process current season data through feature engineering.
    
    Args:
        input_file: Input CSV file (default: latest current season data)
        output_file: Output CSV file (default: auto-generated name)
    """
    from src.utils.config import CURRENT_SEASON_DIR
    
    print("Feature Engineering - Processing Current Season Data")
    
    if input_file is None:
        # Find the latest current season data file
        season_files = list(CURRENT_SEASON_DIR.glob("warriors_current_season_*.csv"))
        if not season_files:
            print(f"Error: No current season data files found in {CURRENT_SEASON_DIR}")
            return
        input_file = max(season_files, key=lambda p: p.stat().st_mtime)
        print(f"Using input file: {input_file.name}")
    else:
        input_file = Path(input_file)
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            return
    
    # Load raw data
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} games")
    
    # Initialize feature engineer
    print("\nInitializing feature engineer...")
    engineer = FeatureEngineer()
    
    # Engineer features
    print("Engineering features...")
    processed_df = engineer.engineer_features(df)
    
    print(f"\nFeature engineering complete!")
    print(f"Original columns: {len(df.columns)}")
    print(f"Processed columns: {len(processed_df.columns)}")
    
    # Determine output file
    if output_file is None:
        input_name = input_file.stem
        output_file = PROCESSED_DATA_DIR / f"{input_name}_processed.csv"
    else:
        output_file = Path(output_file)
    
    print(f"\nSaving processed data to {output_file}...")
    DataLoader.save_processed_data(processed_df, output_file.name)
    
    print(f"\nProcessed {len(processed_df)} games with {len(processed_df.columns)} features")
    
    return processed_df


def main():
    parser = argparse.ArgumentParser(description='Process raw data into ML-ready features')
    parser.add_argument(
        '--historical',
        action='store_true',
        help='Process historical data'
    )
    parser.add_argument(
        '--current',
        action='store_true',
        help='Process current season data'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process both historical and current season data'
    )
    
    args = parser.parse_args()
    
    # If no arguments, process historical data
    if not args.historical and not args.current and not args.all:
        args.historical = True
    
    if args.all or args.historical:
        process_historical_data(args.input, args.output)
        print("\n")
    
    if args.all or args.current:
        process_current_season_data(args.input, args.output)
    
    print("\nFeature engineering complete!")


if __name__ == "__main__":
    main()

