"""
Model training script for NBA game predictions.

This script trains multiple ML models (Random Forest, XGBoost, LightGBM) 
for both classification (win/loss) and regression (point differential) tasks.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from src.models import RandomForestModel, XGBoostModel, LightGBMModel, ModelTrainer
from src.data_collection.data_loader import DataLoader
from src.features.feature_definitions import get_feature_list, get_target_variables
from src.utils.config import PROCESSED_DATA_DIR, MODEL_DIR, ROLLING_WINDOWS


def get_available_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of available features from the dataframe.
    
    Args:
        df: Processed dataframe
        
    Returns:
        List of available feature column names
    """
    # Get all potential features
    potential_features = get_feature_list(include_rolling=True, rolling_windows=ROLLING_WINDOWS)
    
    # Filter to only features that exist in the dataframe
    available_features = [f for f in potential_features if f in df.columns]
    
    # Also include any rolling features that might have been created
    rolling_features = [col for col in df.columns if any(
        f'_ROLLING_{w}' in col for w in ROLLING_WINDOWS
    )]
    
    # Combine and remove duplicates
    all_features = list(set(available_features + rolling_features))
    
    # Exclude target variables and metadata columns
    exclude_cols = [
        'WIN', 'POINT_DIFF', 'WL', 'W', 'L', 'W_PCT',
        'GAME_DATE', 'MATCHUP', 'SEASON', 'SEASON_TYPE',
        'Team_ID', 'Game_ID', 'MIN', 'OPP_PTS', 'PTS_OPP'
    ]
    
    feature_cols = [f for f in all_features if f not in exclude_cols]
    
    return feature_cols


def train_classification_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_suffix: str = ""
) -> Dict[str, Dict]:
    """
    Train classification models for win/loss prediction.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        model_suffix: Suffix to add to model names
        
    Returns:
        Dictionary of model results
    """
    print("\nTRAINING CLASSIFICATION MODELS (Win/Loss Prediction)")
    
    models = {
        'random_forest': RandomForestModel(
            model_name=f"random_forest_classifier{model_suffix}",
            model_type="classification",
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        'xgboost': XGBoostModel(
            model_name=f"xgboost_classifier{model_suffix}",
            model_type="classification",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        ),
        'lightgbm': LightGBMModel(
            model_name=f"lightgbm_classifier{model_suffix}",
            model_type="classification",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- Training {model_name.upper()} ---")
        trainer = ModelTrainer(model)
        
        trainer.train_model(X_train, y_train)
        metrics = trainer.evaluate_model(X_test, y_test, verbose=True)
        
        feature_importance = model.get_feature_importance()
        if feature_importance:
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            print("\nTop 10 Most Important Features:")
            for feat, importance in top_features:
                print(f"  {feat}: {importance:.4f}")
        
        model.save()
        
        results[model_name] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    return results


def train_regression_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_suffix: str = ""
) -> Dict[str, Dict]:
    """
    Train regression models for point differential prediction.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        model_suffix: Suffix to add to model names
        
    Returns:
        Dictionary of model results
    """
    print("TRAINING REGRESSION MODELS (Point Differential Prediction)")
    
    models = {
        'random_forest': RandomForestModel(
            model_name=f"random_forest_regressor{model_suffix}",
            model_type="regression",
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        'xgboost': XGBoostModel(
            model_name=f"xgboost_regressor{model_suffix}",
            model_type="regression",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        ),
        'lightgbm': LightGBMModel(
            model_name=f"lightgbm_regressor{model_suffix}",
            model_type="regression",
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- Training {model_name.upper()} ---")
        trainer = ModelTrainer(model)
        
        trainer.train_model(X_train, y_train)
        metrics = trainer.evaluate_model(X_test, y_test, verbose=True)
        
        feature_importance = model.get_feature_importance()
        if feature_importance:
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            print("\nTop 10 Most Important Features:")
            for feat, importance in top_features:
                print(f"  {feat}: {importance:.4f}")
        
        model.save()
        
        results[model_name] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train ML models for NBA game predictions')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input processed CSV file (default: latest processed file)'
    )
    parser.add_argument(
        '--target',
        type=str,
        choices=['win', 'point_diff', 'both'],
        default='both',
        help='Target variable to predict: win (classification), point_diff (regression), or both'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("NBA PREDICTIVE MODELING - MODEL TRAINING")
    
    # Load data
    if args.input is None:
        # Find the latest processed data file
        processed_files = list(PROCESSED_DATA_DIR.glob("*_processed.csv"))
        if not processed_files:
            print(f"Error: No processed data files found in {PROCESSED_DATA_DIR}")
            return
        input_file = max(processed_files, key=lambda p: p.stat().st_mtime)
        print(f"\nUsing input file: {input_file.name}")
    else:
        input_file = Path(args.input)
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            return
    
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} games")
    
    # Convert date column if present
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    # Get available features
    print("\nIdentifying available features...")
    feature_cols = get_available_features(df)
    print(f"Found {len(feature_cols)} features")
    
    # Check for missing values in features
    missing_counts = df[feature_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\nWarning: {missing_counts.sum()} missing values in features")
        print("Features with most missing values:")
        print(missing_counts[missing_counts > 0].sort_values(ascending=False).head(10))
    
    # Prepare model suffix
    model_suffix = f"_{input_file.stem.replace('_processed', '')}"
    
    all_results = {}
    
    # Train classification models (Win/Loss)
    if args.target in ['win', 'both']:
        if 'WIN' not in df.columns:
            print("\nWarning: 'WIN' column not found. Skipping classification models.")
        else:
            print("\nPreparing data for classification...")
            trainer_temp = ModelTrainer(RandomForestModel(model_type="classification"))
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = trainer_temp.prepare_data(
                df,
                feature_cols,
                'WIN',
                test_size=args.test_size,
                random_state=args.random_state
            )
            
            print(f"Training set: {len(X_train_clf)} games")
            print(f"Test set: {len(X_test_clf)} games")
            
            clf_results = train_classification_models(
                X_train_clf, X_test_clf, y_train_clf, y_test_clf, model_suffix
            )
            all_results['classification'] = clf_results
    
    # Train regression models (Point Differential)
    if args.target in ['point_diff', 'both']:
        if 'POINT_DIFF' not in df.columns:
            print("\nWarning: 'POINT_DIFF' column not found. Skipping regression models.")
        else:
            print("\nPreparing data for regression...")
            trainer_temp = ModelTrainer(RandomForestModel(model_type="regression"))
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = trainer_temp.prepare_data(
                df,
                feature_cols,
                'POINT_DIFF',
                test_size=args.test_size,
                random_state=args.random_state
            )
            
            print(f"Training set: {len(X_train_reg)} games")
            print(f"Test set: {len(X_test_reg)} games")
            
            reg_results = train_regression_models(
                X_train_reg, X_test_reg, y_train_reg, y_test_reg, model_suffix
            )
            all_results['regression'] = reg_results
    
    print("\nTRAINING SUMMARY")
    
    if 'classification' in all_results:
        print("\nClassification Models (Win/Loss):")
        for model_name, result in all_results['classification'].items():
            accuracy = result['metrics']['accuracy']
            print(f"  {model_name.upper()}: Accuracy = {accuracy:.4f}")
    
    if 'regression' in all_results:
        print("\nRegression Models (Point Differential):")
        for model_name, result in all_results['regression'].items():
            rmse = result['metrics']['rmse']
            r2 = result['metrics']['r2']
            print(f"  {model_name.upper()}: RMSE = {rmse:.4f}, R² = {r2:.4f}")
    
    print(f"\nModels saved to: {MODEL_DIR}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()