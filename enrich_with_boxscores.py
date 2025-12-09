"""
Script to enrich game log data with opponent points from boxscore data.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.data_collection.nba_api_client import NBAApiClient
from src.data_collection.data_loader import DataLoader
from src.utils.config import RAW_DATA_DIR


def enrich_historical_data(input_file: str = None, output_file: str = None, delay: float = 0.6, force: bool = False):
    """
    Enrich historical data with opponent points from boxscore data.
    
    Args:
        input_file: Input CSV file (default: latest historical data)
        output_file: Output CSV file (default: overwrite input file)
        delay: Delay between API calls in seconds
    """
    print("Enriching Historical Data with Boxscore Information")
    
    if input_file is None:
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
    
    # Check if already enriched
    if 'OPP_PTS' in df.columns and df['OPP_PTS'].notna().sum() > 0 and not force:
        print(f"\nData already has opponent points for {df['OPP_PTS'].notna().sum()} games")
        print("Use --force flag to re-fetch.")
        return df
    
    print("\nInitializing NBA API client...")
    client = NBAApiClient(delay=delay)
    
    # Enrich with opponent points
    print("\nFetching boxscore data...")
    print("Note: This will take some time for 793 games due to rate limiting.")
    enriched_df = client.enrich_game_log_with_opponent_points(df, show_progress=True, force=force)
    
    # Determine output file
    if output_file is None:
        output_file = input_file  # Overwrite input file
    else:
        output_file = Path(output_file)
    
    print(f"\nSaving enriched data to {output_file}...")
    enriched_df.to_csv(output_file, index=False)
    
    print("Enrichment Summary")
    print(f"Total games: {len(enriched_df)}")
    print(f"Games with opponent points: {enriched_df['OPP_PTS'].notna().sum()}")
    print(f"Games with estimated points: {enriched_df['OPP_PTS'].isna().sum()}")
    
    if 'POINT_DIFF' in enriched_df.columns:
        print(f"\nPoint differential statistics:")
        print(f"  Mean: {enriched_df['POINT_DIFF'].mean():.2f}")
        print(f"  Std: {enriched_df['POINT_DIFF'].std():.2f}")
        print(f"  Min: {enriched_df['POINT_DIFF'].min():.2f}")
        print(f"  Max: {enriched_df['POINT_DIFF'].max():.2f}")
    
    print(f"\nData saved to: {output_file}")
    
    return enriched_df


def main():
    parser = argparse.ArgumentParser(description='Enrich game log data with opponent points from boxscores')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input CSV file path (default: latest historical data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: overwrite input file)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.6,
        help='Delay between API calls in seconds (default: 0.6)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-fetch even if opponent points already exist'
    )
    
    args = parser.parse_args()
    
    enrich_historical_data(args.input, args.output, args.delay, args.force)
    
    print("\nEnrichment complete!")


if __name__ == "__main__":
    main()

