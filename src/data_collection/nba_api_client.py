"""
NBA API client for fetching game data and statistics.
"""

from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
import pandas as pd
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
    
    def get_opponent_stats(self, game_date: str, opponent_team_id: int) -> Optional[Dict]:
        """
        Get opponent team statistics for a specific game.
        This is a placeholder for future implementation.
        
        Args:
            game_date: Date of the game
            opponent_team_id: Opponent team ID
        
        Returns:
            Dictionary with opponent stats (to be implemented)
        """
        # TODO: Implement opponent stats retrieval
        return None
    
    def get_team_info(self) -> Dict:
        return self.team_info
    
    def get_team_name(self) -> str:
        return self.team_info.get('full_name', 'Unknown')

