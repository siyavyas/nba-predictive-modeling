"""
Prediction pipeline for upcoming NBA games.

1. Loads trained models (classification and regression)
2. Fetches upcoming games or uses current season data
3. Engineers features for upcoming games using historical data
4. Makes predictions (win probability, point differential)
5. Saves predictions in readable format
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from src.models import RandomForestModel, XGBoostModel, LightGBMModel
from src.models.base_model import BaseModel
from src.data_collection.nba_api_client import NBAApiClient
from src.data_collection.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_definitions import get_feature_list
from src.utils.config import (
    PROCESSED_DATA_DIR, CURRENT_SEASON_DIR, PREDICTIONS_DIR, 
    MODEL_DIR, CURRENT_SEASON, ROLLING_WINDOWS
)


def find_best_model(model_dir: Path, model_type: str, task_type: str) -> Optional[str]:
    """
    Find the best model file based on naming convention.
    Priority: tuned + feature selection > tuned > baseline
    
    Args:
        model_dir: Directory containing model files
        model_type: 'random_forest', 'xgboost', or 'lightgbm'
        task_type: 'classification' or 'regression'
    
    Returns:
        Filename of best model or None
    """
    model_files = list(model_dir.glob(f"{model_type}_*_{task_type}_*.joblib"))
    
    if not model_files:
        return None
    
    # Priority order: fs_selectkbest > fs_importance > tuned > baseline
    priority_keywords = ['fs_selectkbest', 'fs_importance', 'tuned', 'baseline']
    
    for keyword in priority_keywords:
        for model_file in model_files:
            if keyword in model_file.stem:
                return model_file.name
    
    # If no priority match, return the first one
    return model_files[0].name


def load_model(model_path: Path) -> BaseModel:
    model_data = joblib.load(model_path)
    
    # Determine model class from model name
    model_name = model_data.get('model_name', '')
    if 'random_forest' in model_name:
        model_class = RandomForestModel
    elif 'xgboost' in model_name:
        model_class = XGBoostModel
    elif 'lightgbm' in model_name:
        model_class = LightGBMModel
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Create model instance
    model = model_class(
        model_name=model_data['model_name'],
        model_type=model_data['model_type']
    )
    
    model.model = model_data['model']
    model.feature_names = model_data['feature_names']
    model.is_trained = model_data['is_trained']
    
    return model


def get_upcoming_games(
    current_season_data: pd.DataFrame,
    days_ahead: int = 14
) -> pd.DataFrame:
    """
    Extract upcoming games from current season data.
    
    Args:
        current_season_data: Current season game log
        days_ahead: Number of days ahead to look for games
    
    Returns:
        DataFrame with upcoming games
    """
    if current_season_data.empty:
        return pd.DataFrame()
    
    # Convert date column
    if 'GAME_DATE' in current_season_data.columns:
        current_season_data['GAME_DATE'] = pd.to_datetime(current_season_data['GAME_DATE'])
    
    # Get today's date
    today = datetime.now().date()
    future_date = today + timedelta(days=days_ahead)
    
    # First, try to find games without results (these are likely upcoming)
    upcoming = pd.DataFrame()
    if 'WL' in current_season_data.columns:
        no_result = current_season_data[
            current_season_data['WL'].isna() | 
            (current_season_data['WL'] == '') |
            (current_season_data['WL'].astype(str).str.strip() == '')
        ].copy()
        if not no_result.empty:
            upcoming = no_result
    
    # Also include games with future dates
    future_games = current_season_data[
        (current_season_data['GAME_DATE'].dt.date >= today) &
        (current_season_data['GAME_DATE'].dt.date <= future_date)
    ].copy()
    
    # Combine and remove duplicates
    if not future_games.empty:
        if upcoming.empty:
            upcoming = future_games
        else:
            upcoming = pd.concat([upcoming, future_games]).drop_duplicates(
                subset=['Game_ID'] if 'Game_ID' in current_season_data.columns else ['GAME_DATE', 'MATCHUP'],
                keep='first'
            )
    
    # If still no upcoming games, get the last few games (might be recent without results)
    if upcoming.empty and len(current_season_data) > 0:
        # Get last 5 games and check if any don't have results
        last_games = current_season_data.tail(5).copy()
        if 'WL' in last_games.columns:
            no_result_last = last_games[
                last_games['WL'].isna() | 
                (last_games['WL'] == '') |
                (last_games['WL'].astype(str).str.strip() == '')
            ]
            if not no_result_last.empty:
                upcoming = no_result_last
                print(f"Note: Using recent games without results as upcoming games")
    
    if upcoming.empty:
        return pd.DataFrame()
    
    if 'GAME_DATE' in upcoming.columns:
        return upcoming.sort_values('GAME_DATE').reset_index(drop=True)
    else:
        return upcoming.reset_index(drop=True)


def engineer_features_for_upcoming_games(
    upcoming_games: pd.DataFrame,
    historical_data: pd.DataFrame,
    feature_engineer: FeatureEngineer
) -> pd.DataFrame:
    """
    Engineer features for upcoming games using historical data.
    
    Args:
        upcoming_games: DataFrame with upcoming games
        historical_data: Historical processed data for rolling features
        feature_engineer: FeatureEngineer instance
    
    Returns:
        DataFrame with engineered features for upcoming games
    """
    if upcoming_games.empty:
        return pd.DataFrame()
    
    # First, engineer features on historical data to get rolling stats
    historical_processed = feature_engineer.engineer_features(historical_data.copy())
    
    # Get the last row of historical data for rolling features
    # This will be used as the "previous game" for upcoming games
    if historical_processed.empty:
        print("Warning: No historical data available for feature engineering")
        return upcoming_games.copy()
    
    # Create a copy of upcoming games
    upcoming_with_features = upcoming_games.copy()
    
    # Add basic contextual features that don't depend on game results
    if 'MATCHUP' in upcoming_with_features.columns:
        upcoming_with_features['IS_HOME'] = upcoming_with_features['MATCHUP'].str.contains('vs.').astype(int)
    
    # Add rest days (calculate from last historical game)
    if 'GAME_DATE' in upcoming_with_features.columns and 'GAME_DATE' in historical_processed.columns:
        upcoming_with_features['GAME_DATE'] = pd.to_datetime(upcoming_with_features['GAME_DATE'])
        last_historical_date = pd.to_datetime(historical_processed['GAME_DATE'].max())
        
        from src.utils.helpers import calculate_rest_days, is_back_to_back, get_season_phase
        
        # Calculate rest days for first upcoming game
        first_upcoming_date = pd.to_datetime(upcoming_with_features['GAME_DATE'].iloc[0])
        rest_days = (first_upcoming_date - last_historical_date).days - 1
        
        upcoming_with_features['REST_DAYS'] = rest_days
        upcoming_with_features['IS_BACK_TO_BACK'] = is_back_to_back(pd.Series([rest_days])).iloc[0]
        
        # For subsequent games, calculate from previous upcoming game
        for i in range(1, len(upcoming_with_features)):
            prev_date = pd.to_datetime(upcoming_with_features['GAME_DATE'].iloc[i-1])
            curr_date = pd.to_datetime(upcoming_with_features['GAME_DATE'].iloc[i])
            rest_days = (curr_date - prev_date).days - 1
            upcoming_with_features.loc[upcoming_with_features.index[i], 'REST_DAYS'] = rest_days
            upcoming_with_features.loc[upcoming_with_features.index[i], 'IS_BACK_TO_BACK'] = is_back_to_back(pd.Series([rest_days])).iloc[0]
        
        # Season phase
        upcoming_with_features['SEASON_PHASE'] = get_season_phase(upcoming_with_features['GAME_DATE'])
        upcoming_with_features['SEASON_PHASE_EARLY'] = (upcoming_with_features['SEASON_PHASE'] == 'early').astype(int)
        upcoming_with_features['SEASON_PHASE_MID'] = (upcoming_with_features['SEASON_PHASE'] == 'mid').astype(int)
        upcoming_with_features['SEASON_PHASE_LATE'] = (upcoming_with_features['SEASON_PHASE'] == 'late').astype(int)
    
    rolling_features = [col for col in historical_processed.columns if '_ROLLING_' in col]
    
    if rolling_features and len(historical_processed) > 0:
        last_historical_row = historical_processed.iloc[-1]
        for feature in rolling_features:
            if feature in last_historical_row:
                # Use the last historical rolling value for all upcoming games
                upcoming_with_features[feature] = last_historical_row[feature]
    
    # Get win/loss streaks from last historical game
    if 'WIN_STREAK' in historical_processed.columns and len(historical_processed) > 0:
        last_historical_row = historical_processed.iloc[-1]
        upcoming_with_features['WIN_STREAK'] = last_historical_row.get('WIN_STREAK', 0)
        upcoming_with_features['LOSS_STREAK'] = last_historical_row.get('LOSS_STREAK', 0)
    
    return upcoming_with_features


def prepare_features_for_prediction(
    upcoming_games: pd.DataFrame,
    model: BaseModel
) -> pd.DataFrame:
    """
    Prepare features for prediction, ensuring all model features are present.
    
    Args:
        upcoming_games: DataFrame with engineered features
        model: Trained model with feature_names
    
    Returns:
        DataFrame with features ready for prediction
    """
    if upcoming_games.empty:
        return pd.DataFrame()
    
    required_features = model.feature_names
    
    X = pd.DataFrame(index=upcoming_games.index)
    
    for feature in required_features:
        if feature in upcoming_games.columns:
            X[feature] = upcoming_games[feature]
        else:
            # Fill missing features with 0 or mean (depending on feature type)
            X[feature] = 0
    
    X = X.fillna(X.mean())
    X = X.fillna(0)  # Fill any remaining NaN with 0
    
    # Ensure feature order matches model
    X = X[required_features]
    
    return X


def make_predictions(
    model: BaseModel,
    X: pd.DataFrame
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Make predictions using the model.
    
    Args:
        model: Trained model
        X: Feature matrix
    
    Returns:
        Tuple of (predictions, probabilities if available)
    """
    predictions = model.predict(X)
    
    probabilities = None
    if model.model_type == "classification":
        try:
            probabilities = model.predict_proba(X)
        except:
            pass
    
    return predictions, probabilities


def format_predictions(
    upcoming_games: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    model_type: str
) -> pd.DataFrame:
    """
    Format predictions into a readable DataFrame.
    
    Args:
        upcoming_games: Original upcoming games DataFrame
        predictions: Model predictions
        probabilities: Prediction probabilities (if available)
        model_type: 'classification' or 'regression'
    
    Returns:
        Formatted predictions DataFrame
    """
    results = upcoming_games[['GAME_DATE', 'MATCHUP']].copy() if 'MATCHUP' in upcoming_games.columns else upcoming_games.copy()
    
    if model_type == "classification":
        results['Predicted_Outcome'] = ['Win' if p == 1 else 'Loss' for p in predictions]
        if probabilities is not None:
            results['Win_Probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.flatten()
            results['Loss_Probability'] = probabilities[:, 0] if probabilities.shape[1] > 1 else 1 - probabilities.flatten()
        else:
            results['Win_Probability'] = np.where(predictions == 1, 0.75, 0.25)  # Estimate
            results['Loss_Probability'] = 1 - results['Win_Probability']
    else:  # regression
        results['Predicted_Point_Diff'] = predictions
        results['Predicted_Outcome'] = ['Win' if p > 0 else 'Loss' for p in predictions]
        results['Predicted_Margin'] = np.abs(predictions)
    
    return results


def save_predictions(
    predictions_df: pd.DataFrame,
    model_name: str,
    output_format: str = 'csv'
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format == 'csv':
        filename = f"predictions_{model_name}_{timestamp}.csv"
        filepath = PREDICTIONS_DIR / filename
        predictions_df.to_csv(filepath, index=False)
        print(f"\nPredictions saved to: {filepath}")
    
    # Also save a human-readable summary
    summary_filename = f"predictions_summary_{timestamp}.txt"
    summary_path = PREDICTIONS_DIR / summary_filename
    
    with open(summary_path, 'w') as f:
        f.write("\nNBA GAME PREDICTIONS - GOLDEN STATE WARRIORS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Games: {len(predictions_df)}\n")
        f.write("-"*80 + "\n\n")
        
        for idx, row in predictions_df.iterrows():
            f.write(f"Game {idx + 1}:\n")
            if 'GAME_DATE' in row:
                f.write(f"  Date: {row['GAME_DATE']}\n")
            if 'MATCHUP' in row:
                f.write(f"  Matchup: {row['MATCHUP']}\n")
            if 'Predicted_Outcome' in row:
                f.write(f"  Prediction: {row['Predicted_Outcome']}\n")
            if 'Win_Probability' in row:
                f.write(f"  Win Probability: {row['Win_Probability']:.1%}\n")
            if 'Predicted_Point_Diff' in row:
                f.write(f"  Predicted Point Differential: {row['Predicted_Point_Diff']:+.1f}\n")
            f.write("\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Make predictions for upcoming NBA games')
    parser.add_argument(
        '--model',
        type=str,
        choices=['random_forest', 'xgboost', 'lightgbm', 'best'],
        default='best',
        help='Model to use for predictions (default: best available)'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'regression', 'both'],
        default='both',
        help='Prediction task: classification (win/loss), regression (point diff), or both'
    )
    parser.add_argument(
        '--days-ahead',
        type=int,
        default=14,
        help='Number of days ahead to predict (default: 14)'
    )
    parser.add_argument(
        '--historical-data',
        type=str,
        default=None,
        help='Path to historical processed data (default: latest processed file)'
    )
    parser.add_argument(
        '--current-season-data',
        type=str,
        default=None,
        help='Path to current season data (default: latest current season file)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Demo mode: use last game as test case (for demonstration when no upcoming games)'
    )
    
    args = parser.parse_args()
    
    print("NBA GAME PREDICTION PIPELINE")
    
    if args.historical_data:
        historical_path = Path(args.historical_data)
    else:
        processed_files = list(PROCESSED_DATA_DIR.glob("*_processed.csv"))
        if not processed_files:
            print(f"Error: No processed data files found in {PROCESSED_DATA_DIR}")
            return
        historical_path = max(processed_files, key=lambda p: p.stat().st_mtime)
    
    print(f"\nLoading historical data from: {historical_path.name}")
    historical_data = pd.read_csv(historical_path)
    print(f"Loaded {len(historical_data)} historical games")
    
    # Initialize NBA API client for fetching upcoming games
    client = NBAApiClient()
    
    print(f"\nFetching upcoming games from NBA API (next {args.days_ahead} days)...")
    upcoming_games = client.get_upcoming_games(season=CURRENT_SEASON, days_ahead=args.days_ahead)
    
    # If no upcoming games from API, try current season data file
    if upcoming_games.empty:
        print("No upcoming games found in API schedule. Checking current season data file...")
        if args.current_season_data:
            current_season_path = Path(args.current_season_data)
        else:
            season_files = list(CURRENT_SEASON_DIR.glob("warriors_current_season_*.csv"))
            if season_files:
                current_season_path = max(season_files, key=lambda p: p.stat().st_mtime)
            else:
                current_season_path = None
        
        if current_season_path and current_season_path.exists():
            print(f"Loading current season data from: {current_season_path.name}")
            current_season_data = pd.read_csv(current_season_path)
            print(f"Loaded {len(current_season_data)} current season games")
            upcoming_games = get_upcoming_games(current_season_data, days_ahead=args.days_ahead)
    
    # Demo mode: use last game as test case
    if upcoming_games.empty and args.demo:
        print("No upcoming games found. Using demo mode with last game...")
        if not current_season_data.empty:
            upcoming_games = current_season_data.tail(1).copy()
            print(f"Demo: Using last game ({upcoming_games['GAME_DATE'].iloc[0] if 'GAME_DATE' in upcoming_games.columns else 'N/A'}) as test case")
        else:
            print("Error: No current season data available for demo mode.")
            return
    
    if upcoming_games.empty:
        print("No upcoming games found in the specified time range.")
        print("Tip: Use --demo flag to test with the last game, or ensure current season data includes future games.")
        return
    
    print(f"Found {len(upcoming_games)} upcoming games")
    
    # Engineer features
    print("\nEngineering features for upcoming games...")
    feature_engineer = FeatureEngineer()
    upcoming_with_features = engineer_features_for_upcoming_games(
        upcoming_games, historical_data, feature_engineer
    )
    
    # Determine which models to use
    model_types = []
    if args.task in ['classification', 'both']:
        model_types.append(('classification', 'Win/Loss'))
    if args.task in ['regression', 'both']:
        model_types.append(('regression', 'Point Differential'))
    
    all_predictions = []
    
    for task_type, task_name in model_types:
        print(f"\n{'='*80}")
        print(f"MAKING {task_name.upper()} PREDICTIONS")
        print(f"{'='*80}")
        
        if args.model == 'best':
            model_name = 'xgboost'  # XGBoost performed best in optimization
        else:
            model_name = args.model
        
        model_filename = find_best_model(MODEL_DIR, model_name, task_type)
        
        if not model_filename:
            print(f"Warning: No {task_type} model found for {model_name}. Trying other models...")
            # Try other models
            for alt_model in ['xgboost', 'random_forest', 'lightgbm']:
                if alt_model != model_name:
                    model_filename = find_best_model(MODEL_DIR, alt_model, task_type)
                    if model_filename:
                        model_name = alt_model
                        break
        
        if not model_filename:
            print(f"Error: No {task_type} model found. Skipping {task_name} predictions.")
            continue
        
        print(f"Loading model: {model_filename}")
        model_path = MODEL_DIR / model_filename
        model = load_model(model_path)
        print(f"Model loaded: {model.model_name}")
        print(f"Features: {len(model.feature_names)}")
        
        # Prepare features
        X = prepare_features_for_prediction(upcoming_with_features, model)
        
        if X.empty:
            print("Error: Could not prepare features. Skipping predictions.")
            continue
        
        # Make predictions
        print("Making predictions...")
        predictions, probabilities = make_predictions(model, X)
        
        # Format predictions
        formatted = format_predictions(upcoming_with_features, predictions, probabilities, task_type)
        formatted['Model'] = model.model_name
        formatted['Task'] = task_name
        
        all_predictions.append(formatted)
        
        # Display predictions
        print(f"\nPredictions ({task_name}):")
        print("-" * 80)
        for idx, row in formatted.iterrows():
            print(f"\nGame {idx + 1}:")
            if 'GAME_DATE' in row:
                print(f"  Date: {row['GAME_DATE']}")
            if 'MATCHUP' in row:
                print(f"  Matchup: {row['MATCHUP']}")
            if 'Predicted_Outcome' in row:
                print(f"  Prediction: {row['Predicted_Outcome']}")
            if 'Win_Probability' in row:
                print(f"  Win Probability: {row['Win_Probability']:.1%}")
            if 'Predicted_Point_Diff' in row:
                print(f"  Predicted Point Differential: {row['Predicted_Point_Diff']:+.1f}")
    
    # Combine and save all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Save predictions
        model_identifier = f"{model_name}_{task_type}" if args.model != 'best' else "best"
        save_predictions(combined_predictions, model_identifier)
        
        print("\nPREDICTION PIPELINE COMPLETE!")
        print(f"\nTotal predictions made: {len(combined_predictions)}")
        print(f"Predictions saved to: {PREDICTIONS_DIR}")
    else:
        print("\nNo predictions were made. Please check model availability.")


if __name__ == "__main__":
    main()