"""
NBA API client for fetching game data and statistics.
"""

from nba_api.stats.endpoints import teamgamelog, boxscoretraditionalv3, scheduleleaguev2
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
    
    def get_team_schedule(
        self, 
        season: str = None,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Get team schedule including upcoming games.
        
        Args:
            season: Season string (e.g., "2025-26"). If None, uses current season.
            season_type: "Regular Season" or "Playoffs"
        
        Returns:
            DataFrame with team schedule including upcoming games
        """
        try:
            # If season not provided, try to infer from current date
            if season is None:
                from datetime import datetime
                current_year = datetime.now().year
                
                if datetime.now().month >= 10:
                    season = f"{current_year}-{str(current_year + 1)[-2:]}"
                else:
                    season = f"{current_year - 1}-{str(current_year)[-2:]}"
            
            # Use ScheduleLeagueV2 to get full league schedule, then filter for GSW
            # Note: ScheduleLeagueV2 uses different parameter names
            schedule = scheduleleaguev2.ScheduleLeagueV2(
                season=season,
                league_id='00'  # NBA league ID
            )
            df = schedule.get_data_frames()[0]
            
            if df.empty:
                return pd.DataFrame()
            
            # Filter for GSW's games using the correct column names
            team_abbrev = self.team_info.get('abbreviation', 'GSW')
            team_name = self.team_info.get('full_name', 'Golden State Warriors')
            
            # Filter for games where GSW is either home or away
            # ScheduleLeagueV2 uses: homeTeam_teamId, awayTeam_teamId, homeTeam_teamName, awayTeam_teamName
            team_games = df[
                (df['homeTeam_teamId'] == self.team_id) |
                (df['awayTeam_teamId'] == self.team_id) |
                (df['homeTeam_teamName'].str.contains('Warriors', case=False, na=False)) |
                (df['awayTeam_teamName'].str.contains('Warriors', case=False, na=False))
            ].copy()
            
            # Standardize column names
            if 'gameDate' in team_games.columns:
                team_games['GAME_DATE'] = pd.to_datetime(team_games['gameDate'], errors='coerce')
            elif 'gameDateEst' in team_games.columns:
                team_games['GAME_DATE'] = pd.to_datetime(team_games['gameDateEst'], errors='coerce')
            elif 'GAME_DATE' in team_games.columns:
                team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'], errors='coerce')
            
            # Create MATCHUP column
            if 'MATCHUP' not in team_games.columns:
                team_games['MATCHUP'] = team_games.apply(
                    lambda row: f"{team_abbrev} vs. {row.get('awayTeam_teamTricode', row.get('awayTeam_teamName', 'Unknown'))}" 
                    if row.get('homeTeam_teamId') == self.team_id or 'Warriors' in str(row.get('homeTeam_teamName', ''))
                    else f"{team_abbrev} @ {row.get('homeTeam_teamTricode', row.get('homeTeam_teamName', 'Unknown'))}",
                    axis=1
                )
            
            # Add season info
            team_games['SEASON'] = season
            team_games['SEASON_TYPE'] = season_type
            
            # Add Team_ID and Game_ID if available
            if 'gameId' in team_games.columns:
                team_games['Game_ID'] = team_games['gameId']
            elif 'GAME_ID' in team_games.columns:
                team_games['Game_ID'] = team_games['GAME_ID']
            team_games['Team_ID'] = self.team_id
            
            time.sleep(self.delay)
            return team_games
        except Exception as e:
            print(f"Error fetching team schedule for {season} ({season_type}): {e}")
            import traceback
            traceback.print_exc()
            time.sleep(self.delay)
            return pd.DataFrame()
    
    def get_upcoming_games(
        self,
        season: str = None,
        days_ahead: int = 14,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Get upcoming games for the team.
        
        Args:
            season: Season string. If None, uses current season.
            days_ahead: Number of days ahead to look for games
            season_type: "Regular Season" or "Playoffs"
        
        Returns:
            DataFrame with upcoming games (formatted like game log for compatibility)
        """
        schedule = self.get_team_schedule(season, season_type)
        
        if schedule.empty:
            return pd.DataFrame()
        
        # Convert date column
        if 'GAME_DATE' in schedule.columns:
            schedule['GAME_DATE'] = pd.to_datetime(schedule['GAME_DATE'])
        elif 'GAME_DATE_EST' in schedule.columns:
            schedule['GAME_DATE'] = pd.to_datetime(schedule['GAME_DATE_EST'])
        
        # Get today's date
        from datetime import datetime, timedelta
        today = datetime.now().date()
        future_date = today + timedelta(days=days_ahead)
        
        # Filter for upcoming games (future dates or no result)
        upcoming = schedule[
            (schedule['GAME_DATE'].dt.date >= today) &
            (schedule['GAME_DATE'].dt.date <= future_date)
        ].copy()
        
        # Also include games without results if WL column exists
        if 'WL' in schedule.columns:
            no_result = schedule[
                schedule['WL'].isna() | 
                (schedule['WL'] == '') |
                (schedule['WL'].astype(str).str.strip() == '')
            ]
            if not no_result.empty:
                upcoming = pd.concat([upcoming, no_result]).drop_duplicates(
                    subset=['GAME_ID'] if 'GAME_ID' in schedule.columns else ['GAME_DATE'],
                    keep='first'
                )
        
        # Format matchup if needed - schedule API might have different column names
        if 'MATCHUP' not in upcoming.columns:
            # Try different possible column name combinations
            if 'VISITOR_TEAM_NAME' in upcoming.columns and 'HOME_TEAM_NAME' in upcoming.columns:
                team_abbrev = self.team_info.get('abbreviation', 'GSW')
                upcoming['MATCHUP'] = upcoming.apply(
                    lambda row: f"{team_abbrev} vs. {row['HOME_TEAM_NAME']}" 
                    if str(row.get('HOME_TEAM_NAME', '')).find('Warriors') >= 0 or str(row.get('HOME_TEAM_NAME', '')).find(team_abbrev) >= 0
                    else f"{team_abbrev} @ {row['VISITOR_TEAM_NAME']}" 
                    if str(row.get('VISITOR_TEAM_NAME', '')).find('Warriors') >= 0 or str(row.get('VISITOR_TEAM_NAME', '')).find(team_abbrev) >= 0
                    else f"{row.get('VISITOR_TEAM_NAME', 'Unknown')} @ {row.get('HOME_TEAM_NAME', 'Unknown')}",
                    axis=1
                )
            elif 'MATCHUP' in schedule.columns:
                # Copy from original if it exists
                upcoming['MATCHUP'] = schedule.loc[upcoming.index, 'MATCHUP'] if len(upcoming) > 0 else ''
        
        # Ensure we have required columns for compatibility with game log format
        # Add missing columns with default values for upcoming games
        required_cols = ['Team_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'SEASON', 'SEASON_TYPE']
        for col in required_cols:
            if col not in upcoming.columns:
                if col == 'Team_ID':
                    upcoming['Team_ID'] = self.team_id
                elif col == 'Game_ID':
                    # Generate placeholder game IDs if not available
                    upcoming['Game_ID'] = upcoming.index.astype(str)
                elif col in ['SEASON', 'SEASON_TYPE']:
                    upcoming[col] = season if col == 'SEASON' else season_type
                else:
                    upcoming[col] = None
        
        return upcoming.sort_values('GAME_DATE').reset_index(drop=True)

