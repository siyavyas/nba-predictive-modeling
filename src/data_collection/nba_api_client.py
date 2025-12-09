"""
NBA API client for fetching game data and statistics.
"""

from nba_api.stats.endpoints import teamgamelog, boxscoretraditionalv3
from nba_api.stats.static import teams
import pandas as pd
import numpy as np
import time
from typing import Optional, List, Dict
from pathlib import Path
from tqdm import tqdm

from src.utils.config import WARRIORS_TEAM_ID


class NBAApiClient:
    def __init__(self, team_id: int = WARRIORS_TEAM_ID, delay: float = 0.6):
        """
        Initialize NBA API client.
        
        Args:
            team_id: NBA team ID (default: Warriors)
            delay: Delay between API calls in seconds (default: 0.6 to avoid rate limits)
        """
        self.team_id = team_id
        self.delay = delay
        self.team_info = self._get_team_info()
    
    def _get_team_info(self) -> Dict:
        teams_dict = teams.get_teams()
        team = [t for t in teams_dict if t['id'] == self.team_id][0]
        return team
    
    def get_team_game_log(self, season: str, season_type: str = "Regular Season") -> pd.DataFrame:
        """
        Get team game log for a specific season.
        
        Args:
            season: Season in format "YYYY-YY" (e.g., "2024-25")
            season_type: "Regular Season" or "Playoffs"
        
        Returns:
            DataFrame with game log data
        """
        try:
            # NBA API uses different parameter names
            game_log = teamgamelog.TeamGameLog(
                team_id=str(self.team_id),
                season=season,
                season_type_all_star=season_type
            )
            df = game_log.get_data_frames()[0]
            
            # Standardize column names if needed
            if 'GAME_DATE' in df.columns:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y', errors='coerce')
            
            time.sleep(self.delay)
            return df
        except Exception as e:
            print(f"Error fetching game log for {season} ({season_type}): {e}")
            time.sleep(self.delay)
            return pd.DataFrame()
    
    def get_historical_seasons(
        self, 
        start_season: str = "2015-16", 
        end_season: str = "2024-25",
        season_type: str = "Regular Season",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Get game logs for multiple seasons.
        
        Args:
            start_season: Starting season
            end_season: Ending season
            season_type: "Regular Season" or "Playoffs"
            show_progress: Whether to show progress bar
        
        Returns:
            Combined DataFrame with all seasons
        """
        all_games = []
        seasons = self._generate_season_range(start_season, end_season)
        
        iterator = tqdm(seasons, desc="Fetching historical data") if show_progress else seasons
        
        for season in iterator:
            if show_progress:
                iterator.set_description(f"Fetching {season}")
            df = self.get_team_game_log(season, season_type=season_type)
            if not df.empty:
                df['SEASON'] = season
                df['SEASON_TYPE'] = season_type
                all_games.append(df)
        
        if all_games:
            combined_df = pd.concat(all_games, ignore_index=True)
            # Sort by date
            if 'GAME_DATE' in combined_df.columns:
                combined_df = combined_df.sort_values('GAME_DATE').reset_index(drop=True)
            print(f"\nSuccessfully fetched {len(combined_df)} games from {len(seasons)} seasons")
            return combined_df
        return pd.DataFrame()
    
    def get_current_season_games(self, season: str = "2025-26") -> pd.DataFrame:
        """
        Get current season games.
        
        Args:
            season: Current season string
        
        Returns:
            DataFrame with current season games
        """
        return self.get_team_game_log(season)
    
    def _generate_season_range(self, start: str, end: str) -> List[str]:
        """Generate list of season strings between start and end."""
        start_year = int(start.split("-")[0])
        end_year = int(end.split("-")[0])
        
        seasons = []
        for year in range(start_year, end_year + 1):
            next_year = str(year + 1)[-2:]
            seasons.append(f"{year}-{next_year}")
        
        return seasons
    
    def get_game_boxscore(self, game_id: str) -> Optional[Dict]:
        """
        Get boxscore data for a specific game to extract opponent points.
        
        Args:
            game_id: NBA game ID (can be int or string)
        
        Returns:
            Dictionary with team points and opponent points, or None if error
        """
        try:
            # Convert game_id to string and ensure proper format (10 digits with leading zeros)
            game_id_str = str(int(game_id)).zfill(10)
            if not game_id_str.startswith('00'):
                game_id_str = '00' + game_id_str
            
            try:
                boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id_str)
                data_frames = boxscore.get_data_frames()
                # DataFrame 2 has team totals (2 teams), DataFrame 1 has starters/bench breakdown
                team_stats = data_frames[2] if len(data_frames) > 2 else data_frames[1]
            except:
                # Fall back to V2 for older games
                from nba_api.stats.endpoints import boxscoretraditionalv2
                boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id_str)
                data_frames = boxscore.get_data_frames()
                team_stats = data_frames[-1] if len(data_frames) > 1 else data_frames[0]
            
            if team_stats.empty or len(team_stats) < 2:
                return None
            
            # Handle different column name formats (V2 uses UPPER_CASE, V3 uses camelCase)
            if 'TEAM_ID' in team_stats.columns:
                # V2 format
                team_id_col = 'TEAM_ID'
                points_col = 'PTS'
            else:
                # V3 format
                team_id_col = 'teamId'
                points_col = 'points'
            
            # Get points for both teams
            team_points = {}
            for _, row in team_stats.iterrows():
                team_id = int(row[team_id_col])
                points = int(row[points_col])
                team_points[team_id] = points
            
            # Find Warriors and opponent points
            warriors_points = team_points.get(self.team_id)
            if warriors_points is None:
                return None
            
            # Opponent points is the other team's points
            opponent_points = [pts for tid, pts in team_points.items() if tid != self.team_id]
            if not opponent_points:
                return None
            
            opponent_points = opponent_points[0]
            
            time.sleep(self.delay)  # Rate limiting
            
            return {
                'GAME_ID': game_id,
                'PTS': warriors_points,
                'OPP_PTS': opponent_points,
                'POINT_DIFF': warriors_points - opponent_points
            }
        except Exception as e:
            # Silently fail for individual games to avoid cluttering output
            time.sleep(self.delay)
            return None
    
    def enrich_game_log_with_opponent_points(self, df: pd.DataFrame, show_progress: bool = True, force: bool = False) -> pd.DataFrame:
        """
        Enrich game log DataFrame with opponent points from boxscore data.
        
        Args:
            df: Game log DataFrame with Game_ID column
            show_progress: Whether to show progress bar
            force: Force re-fetch even if OPP_PTS already exists
        
        Returns:
            DataFrame with OPP_PTS and updated POINT_DIFF columns
        """
        df = df.copy()
        
        # Check if we already have opponent points
        if 'OPP_PTS' in df.columns and df['OPP_PTS'].notna().any() and not force:
            print("Opponent points already present in data.")
            return df
        
        # If forcing, remove existing OPP_PTS and POINT_DIFF columns
        if force and 'OPP_PTS' in df.columns:
            df = df.drop(columns=['OPP_PTS', 'POINT_DIFF'], errors='ignore')
        
        print(f"Fetching boxscore data for {len(df)} games...")
        print("This may take a while due to rate limiting...")
        
        opponent_points = []
        point_diffs = []
        failed_games = []
        
        iterator = tqdm(df.iterrows(), total=len(df), desc="Fetching boxscores") if show_progress else df.iterrows()
        
        for idx, row in iterator:
            game_id = str(row['Game_ID'])
            boxscore_data = self.get_game_boxscore(game_id)
            
            if boxscore_data:
                opponent_points.append(boxscore_data['OPP_PTS'])
                point_diffs.append(boxscore_data['POINT_DIFF'])
            else:
                opponent_points.append(None)
                point_diffs.append(None)
                failed_games.append(game_id)
        
        # Add opponent points to dataframe
        df['OPP_PTS'] = opponent_points
        df['POINT_DIFF'] = point_diffs
        
        # Fill missing values with estimated values based on win/loss
        if 'WL' in df.columns:
            missing_mask = df['OPP_PTS'].isna()
            if missing_mask.any():
                print(f"\nWarning: Could not fetch {missing_mask.sum()} games. Using estimates.")
                df.loc[missing_mask, 'POINT_DIFF'] = np.where(
                    df.loc[missing_mask, 'WL'] == 'W', 8, -8
                )
                # Estimate opponent points for missing games
                df.loc[missing_mask, 'OPP_PTS'] = df.loc[missing_mask, 'PTS'] - df.loc[missing_mask, 'POINT_DIFF']
        
        success_count = df['OPP_PTS'].notna().sum()
        print(f"\nSuccessfully fetched {success_count}/{len(df)} games ({success_count/len(df)*100:.1f}%)")
        
        return df
    
    def get_team_info(self) -> Dict:
        return self.team_info
    
    def get_team_name(self) -> str:
        return self.team_info.get('full_name', 'Unknown')

