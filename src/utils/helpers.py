"""
Helper utility functions for the NBA Predictive Modeling project.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def load_data(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)


def calculate_rest_days(game_date: pd.Series, previous_game_date: pd.Series) -> pd.Series:
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    if isinstance(previous_game_date, str):
        previous_game_date = pd.to_datetime(previous_game_date)
    
    rest_days = (game_date - previous_game_date).dt.days - 1
    return rest_days.fillna(0)


def is_back_to_back(rest_days: pd.Series) -> pd.Series:
    return rest_days == 0


def get_season_phase(game_date: pd.Series, season_start: str = "10-15") -> pd.Series:
    """
    Determine season phase: early (Oct-Dec), mid (Jan-Feb), late (Mar-Apr).
    
    Args:
        game_date: Series of game dates
        season_start: Month-day of season start (default: October 15)
    """
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    
    game_date = pd.to_datetime(game_date)
    month = game_date.dt.month
    
    phase = pd.Series("mid", index=game_date.index)
    phase[month.isin([10, 11, 12])] = "early"
    phase[month.isin([3, 4])] = "late"
    
    return phase


def format_date(date_str: str) -> str:
    if isinstance(date_str, str):
        date = pd.to_datetime(date_str)
        return date.strftime("%Y-%m-%d")
    return str(date_str)

