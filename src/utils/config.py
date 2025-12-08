"""
Configuration settings for the NBA Predictive Modeling project.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CURRENT_SEASON_DIR = DATA_DIR / "current_season"

# Predictions directory
PREDICTIONS_DIR = PROJECT_ROOT / "predictions" / "upcoming_games"

# Warriors team ID (from NBA API)
WARRIORS_TEAM_ID = 1610612744

# Current season
CURRENT_SEASON = "2025-26"

# Model parameters
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_VERSION = "v1.0"

# Feature engineering parameters
ROLLING_WINDOWS = [5, 10, 20]  # Games for rolling averages
MIN_GAMES_FOR_TRAINING = 20  # Minimum games needed to train model

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CURRENT_SEASON_DIR, PREDICTIONS_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

