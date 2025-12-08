"""
Feature definitions and feature lists for model training.
"""

from typing import List


# Base features from raw data
BASE_FEATURES = [
    'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
    'TOV', 'STL', 'BLK', 'PF', 'PLUS_MINUS'
]

# Engineered features
ENGINEERED_FEATURES = [
    'POINT_DIFF', 'IS_HOME', 'REST_DAYS', 'IS_BACK_TO_BACK',
    'SEASON_PHASE_EARLY', 'SEASON_PHASE_MID', 'SEASON_PHASE_LATE',
    'WIN_STREAK', 'LOSS_STREAK'
]

# Rolling feature prefixes (will be expanded with window sizes)
ROLLING_FEATURE_PREFIXES = [
    'PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST',
    'TOV', 'STL', 'BLK', 'PLUS_MINUS'
]


def get_feature_list(include_rolling: bool = True, rolling_windows: List[int] = [5, 10, 20]) -> List[str]:
    """
    Get complete list of features for model training.
    
    Args:
        include_rolling: Whether to include rolling features
        rolling_windows: List of window sizes for rolling features
    
    Returns:
        List of feature names
    """
    features = ENGINEERED_FEATURES.copy()
    
    if include_rolling:
        for prefix in ROLLING_FEATURE_PREFIXES:
            for window in rolling_windows:
                features.append(f'{prefix}_ROLLING_{window}')
    
    return features


def get_target_variables() -> List[str]:
    return ['WIN', 'POINT_DIFF']

