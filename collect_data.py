"""
Data collection script for Warriors game data.
Run this script to fetch historical and current season data.
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.nba_api_client import NBAApiClient
from src.data_collection.data_loader import DataLoader
from src.utils.config import CURRENT_SEASON, RAW_DATA_DIR


def collect_historical_data(start_season: str = "2015-16", end_season: str = "2024-25"):
    """
    Collect historical Warriors game data.
    
    Args:
        start_season: Starting season (e.g., "2015-16")
        end_season: Ending season (e.g., "2024-25")
    """
    print("Collecting Historical Warriors Game Data")
    print(f"Team: Golden State Warriors")
    print(f"Seasons: {start_season} to {end_season}")
    print()
    
    # Initialize client
    client = NBAApiClient()
    print(f"Team: {client.get_team_name()}")
    print()
    
    # Fetch historical data
    historical_data = client.get_historical_seasons(
        start_season=start_season,
        end_season=end_season,
        season_type="Regular Season"
    )
    
    if historical_data.empty:
        print("No data collected. Please check your connection and try again.")
        return
    
    # Save raw data
    filename = f"warriors_historical_{start_season.replace('-', '_')}_to_{end_season.replace('-', '_')}.csv"
    DataLoader.save_raw_data(historical_data, filename)
    
    print(f"\nData saved to: {RAW_DATA_DIR / filename}")
    print(f"Total games collected: {len(historical_data)}")
    print(f"Date range: {historical_data['GAME_DATE'].min()} to {historical_data['GAME_DATE'].max()}")
    
    # Display summary statistics
    if 'WL' in historical_data.columns:
        wins = (historical_data['WL'] == 'W').sum()
        losses = (historical_data['WL'] == 'L').sum()
        win_pct = wins / len(historical_data) * 100
        print(f"\nWin-Loss Record: {wins}-{losses} ({win_pct:.1f}%)")
    
    return historical_data


def collect_current_season_data(season: str = None):
    """
    Collect current season Warriors game data.
    
    Args:
        season: Season string (e.g., "2025-26"). If None, uses config default.
    """
    if season is None:
        season = CURRENT_SEASON
    
    print("Collecting Current Season Warriors Game Data")
    print(f"Season: {season}")
    print()
    
    # Initialize client
    client = NBAApiClient()
    print(f"Team: {client.get_team_name()}")
    print()
    
    # Fetch current season data
    current_data = client.get_current_season_games(season=season)
    
    if current_data.empty:
        print(f"No data collected for {season}. Season may not have started yet.")
        return
    
    # Save current season data
    filename = f"warriors_current_season_{season.replace('-', '_')}.csv"
    DataLoader.save_current_season_data(current_data, filename)
    
    print(f"\nData saved to: {Path('data/current_season') / filename}")
    print(f"Total games collected: {len(current_data)}")
    
    if 'GAME_DATE' in current_data.columns:
        print(f"Date range: {current_data['GAME_DATE'].min()} to {current_data['GAME_DATE'].max()}")
    
    # Display summary statistics
    if 'WL' in current_data.columns:
        wins = (current_data['WL'] == 'W').sum()
        losses = (current_data['WL'] == 'L').sum()
        if wins + losses > 0:
            win_pct = wins / (wins + losses) * 100
            print(f"\nCurrent Record: {wins}-{losses} ({win_pct:.1f}%)")
    
    return current_data


def main():
    parser = argparse.ArgumentParser(description='Collect Warriors game data from NBA API')
    parser.add_argument(
        '--historical',
        action='store_true',
        help='Collect historical data (2015-16 to 2024-25)'
    )
    parser.add_argument(
        '--current',
        action='store_true',
        help='Collect current season data'
    )
    parser.add_argument(
        '--start-season',
        type=str,
        default='2015-16',
        help='Starting season for historical data (default: 2015-16)'
    )
    parser.add_argument(
        '--end-season',
        type=str,
        default='2024-25',
        help='Ending season for historical data (default: 2024-25)'
    )
    parser.add_argument(
        '--season',
        type=str,
        default=None,
        help='Current season to collect (default: 2025-26)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Collect both historical and current season data'
    )
    
    args = parser.parse_args()
    
    # If no arguments, collect both
    if not args.historical and not args.current and not args.all:
        args.all = True
    
    if args.all or args.historical:
        collect_historical_data(
            start_season=args.start_season,
            end_season=args.end_season
        )
        print("\n")
    
    if args.all or args.current:
        collect_current_season_data(season=args.season)
    
    print("\nData collection complete!")


if __name__ == "__main__":
    main()

