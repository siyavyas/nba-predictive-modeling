"""
Feature engineering for NBA game predictions.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from src.utils.config import ROLLING_WINDOWS
from src.utils.helpers import calculate_rest_days, is_back_to_back, get_season_phase


class FeatureEngineer:
    def __init__(self, rolling_windows: List[int] = None):
        """
        Initialize feature engineer.
        
        Args:
            rolling_windows: List of window sizes for rolling averages
        """
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to engineer all features.
        
        Args:
            df: Raw game log DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Sort by date
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        df = self._add_basic_features(df)
        df = self._add_rolling_features(df)
        df = self._add_contextual_features(df)
        df = self._add_streak_features(df)
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Win/Loss binary
        if 'WL' in df.columns:
            df['WIN'] = (df['WL'] == 'W').astype(int)
        
        # Point differential
        if 'PTS' in df.columns and 'OPP_PTS' in df.columns:
            df['POINT_DIFF'] = df['PTS'] - df['OPP_PTS']
        elif 'PTS' in df.columns and 'PTS_OPP' in df.columns:
            df['POINT_DIFF'] = df['PTS'] - df['PTS_OPP']
        elif 'PLUS_MINUS' in df.columns:
            # Plus/minus is point differential, but need to verify sign
            # For team game log, PLUS_MINUS should be point differential
            df['POINT_DIFF'] = df['PLUS_MINUS']
        elif 'PTS' in df.columns and 'WL' in df.columns:
            # If we have win/loss but no point diff, we'll need to estimate
            df['POINT_DIFF'] = np.where(df['WL'] == 'W', 8, -8)  # Conservative estimate
        
        # Field goal percentage (recalculate to ensure consistency)
        if 'FGM' in df.columns and 'FGA' in df.columns:
            df['FG_PCT'] = df['FGM'] / df['FGA'].replace(0, np.nan)
        
        # Three-point percentage
        if 'FG3M' in df.columns and 'FG3A' in df.columns:
            df['FG3_PCT'] = df['FG3M'] / df['FG3A'].replace(0, np.nan)
        
        # Free throw percentage
        if 'FTM' in df.columns and 'FTA' in df.columns:
            df['FT_PCT'] = df['FTM'] / df['FTA'].replace(0, np.nan)
        
        # True shooting percentage (approximate)
        if 'PTS' in df.columns and 'FGA' in df.columns and 'FTA' in df.columns:
            df['TSA'] = df['FGA'] + 0.44 * df['FTA']  # True shot attempts
            df['TS_PCT'] = df['PTS'] / (2 * df['TSA'].replace(0, np.nan))
        
        # Effective field goal percentage
        if 'FGM' in df.columns and 'FG3M' in df.columns and 'FGA' in df.columns:
            df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, np.nan)
        
        # Assist to turnover ratio
        if 'AST' in df.columns and 'TOV' in df.columns:
            df['AST_TOV_RATIO'] = df['AST'] / df['TOV'].replace(0, np.nan)
        
        # Rebound rate (offensive rebounds / total rebounds)
        if 'OREB' in df.columns and 'REB' in df.columns:
            df['OREB_PCT'] = df['OREB'] / df['REB'].replace(0, np.nan)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        key_features = [
            'PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 
            'STL', 'BLK', 'PF', 'PLUS_MINUS', 'POINT_DIFF',
            'TS_PCT', 'EFG_PCT', 'AST_TOV_RATIO'
        ]
        
        # Only use features that exist in the dataframe
        available_features = [col for col in key_features if col in df.columns]
        
        for window in self.rolling_windows:
            for col in available_features:
                if col not in ['WIN', 'POINT_DIFF']:  # Exclude target variables from rolling
                    rolling_col = f'{col}_ROLLING_{window}'
                    df[rolling_col] = df[col].rolling(
                        window=window, min_periods=1
                    ).mean().shift(1)  # Shift to avoid data leakage
        
        return df
    
    def _add_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Home/Away
        if 'MATCHUP' in df.columns:
            df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
        
        # Rest days
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df['REST_DAYS'] = calculate_rest_days(
                df['GAME_DATE'],
                df['GAME_DATE'].shift(1)
            )
            df['IS_BACK_TO_BACK'] = is_back_to_back(df['REST_DAYS']).astype(int)
        
        # Season phase
        if 'GAME_DATE' in df.columns:
            df['SEASON_PHASE'] = get_season_phase(df['GAME_DATE'])
            df['SEASON_PHASE_EARLY'] = (df['SEASON_PHASE'] == 'early').astype(int)
            df['SEASON_PHASE_MID'] = (df['SEASON_PHASE'] == 'mid').astype(int)
            df['SEASON_PHASE_LATE'] = (df['SEASON_PHASE'] == 'late').astype(int)
        
        return df
    
    def _add_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'WIN' in df.columns:
            # Win streak (current streak of wins)
            win_groups = (df['WIN'] != df['WIN'].shift()).cumsum()
            df['WIN_STREAK'] = (df['WIN'] == 1).groupby(win_groups).cumsum()
            
            # Loss streak (current streak of losses)
            df['LOSS_STREAK'] = (df['WIN'] == 0).groupby(win_groups).cumsum()
            
            # Shift streaks to avoid data leakage (use previous game's streak)
            df['WIN_STREAK'] = df['WIN_STREAK'].shift(1).fillna(0)
            df['LOSS_STREAK'] = df['LOSS_STREAK'].shift(1).fillna(0)
        
        return df

