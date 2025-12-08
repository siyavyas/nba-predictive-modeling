"""
Data loading utilities for processed and raw data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CURRENT_SEASON_DIR


class DataLoader:   
    @staticmethod
    def load_raw_data(filename: str) -> pd.DataFrame:
        filepath = RAW_DATA_DIR / filename
        if filepath.exists():
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
    
    @staticmethod
    def load_processed_data(filename: str) -> pd.DataFrame:
        filepath = PROCESSED_DATA_DIR / filename
        if filepath.exists():
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
    
    @staticmethod
    def load_current_season_data(filename: str) -> pd.DataFrame:
        filepath = CURRENT_SEASON_DIR / filename
        if filepath.exists():
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Current season data file not found: {filepath}")
    
    @staticmethod
    def save_raw_data(df: pd.DataFrame, filename: str) -> None:
        filepath = RAW_DATA_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved raw data to {filepath}")
    
    @staticmethod
    def save_processed_data(df: pd.DataFrame, filename: str) -> None:
        filepath = PROCESSED_DATA_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved processed data to {filepath}")
    
    @staticmethod
    def save_current_season_data(df: pd.DataFrame, filename: str) -> None:
        filepath = CURRENT_SEASON_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved current season data to {filepath}")

