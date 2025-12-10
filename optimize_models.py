"""
Model optimization script with hyperparameter tuning and feature selection.

1. Trains baseline models with default parameters
2. Performs hyperparameter tuning using GridSearchCV
3. Implements feature selection (importance-based and SelectKBest)
4. Compares performance before and after optimization
5. Uses cross-validation for robust evaluation
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import (
    GridSearchCV, 
    cross_val_score, 
    train_test_split,
    TimeSeriesSplit
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    mean_squared_error, 
    r2_score,
    confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent))

from src.models import RandomForestModel, XGBoostModel, LightGBMModel, ModelTrainer
from src.models.base_model import BaseModel
from src.features.feature_definitions import get_feature_list
from src.utils.config import PROCESSED_DATA_DIR, MODEL_DIR, ROLLING_WINDOWS


def get_available_features(df: pd.DataFrame) -> List[str]:
    potential_features = get_feature_list(include_rolling=True, rolling_windows=ROLLING_WINDOWS)
    available_features = [f for f in potential_features if f in df.columns]
    rolling_features = [col for col in df.columns if any(
        f'_ROLLING_{w}' in col for w in ROLLING_WINDOWS
    )]
    all_features = list(set(available_features + rolling_features))
    exclude_cols = [
        'WIN', 'POINT_DIFF', 'WL', 'W', 'L', 'W_PCT',
        'GAME_DATE', 'MATCHUP', 'SEASON', 'SEASON_TYPE',
        'Team_ID', 'Game_ID', 'MIN', 'OPP_PTS', 'PTS_OPP'
    ]
    feature_cols = [f for f in all_features if f not in exclude_cols]
    return feature_cols


def get_hyperparameter_grids(model_name: str, model_type: str) -> Dict:
    if model_name == 'random_forest':
        if model_type == 'classification':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:  # regression
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    
    elif model_name == 'xgboost':
        if model_type == 'classification':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:  # regression
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
    
    elif model_name == 'lightgbm':
        if model_type == 'classification':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        else:  # regression
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
    
    return {}


def select_features_by_importance(
    model,
    feature_names: List[str],
    top_k: int = 20
) -> List[str]:
    feature_importance = model.get_feature_importance()
    if not feature_importance:
        return feature_names
    
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    selected_features = [feat for feat, _ in sorted_features[:top_k]]
    return selected_features


def select_features_selectkbest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: List[str],
    k: int = 20,
    model_type: str = 'classification'
) -> List[str]:
    if model_type == 'classification':
        selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_names)))
    else:
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_names)))
    
    selector.fit(X_train, y_train)
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    return selected_features


def train_baseline_model(
    model_class,
    model_name: str,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5
) -> Dict:
    print(f"\n--- Baseline {model_name.upper()} ---")
    
    # Create baseline model with default parameters
    if model_name == 'random_forest':
        model = model_class(
            model_name=f"{model_name}_baseline_{model_type}",
            model_type=model_type,
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
    elif model_name == 'xgboost':
        model = model_class(
            model_name=f"{model_name}_baseline_{model_type}",
            model_type=model_type,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    else:  # lightgbm
        model = model_class(
            model_name=f"{model_name}_baseline_{model_type}",
            model_type=model_type,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    
    trainer = ModelTrainer(model)
    trainer.train_model(X_train, y_train)
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_model(X_test, y_test, verbose=False)
    
    # Cross-validation
    scoring = 'accuracy' if model_type == 'classification' else 'r2'
    cv_scores = trainer.cross_validate(X_train, y_train, cv=cv, scoring=scoring)
    
    results = {
        'model': model,
        'test_metrics': test_metrics,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    if model_type == 'classification':
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    else:
        print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test R²: {test_metrics['r2']:.4f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results


def tune_hyperparameters(
    model_class,
    model_name: str,
    model_type: str,
    param_grid: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    n_jobs: int = -1
) -> Tuple[BaseModel, Dict]:
    print(f"\n--- Tuning {model_name.upper()} Hyperparameters ---")
    
    if model_name == 'random_forest':
        base_model = model_class(
            model_name=f"{model_name}_tuned_{model_type}",
            model_type=model_type
        )
    elif model_name == 'xgboost':
        base_model = model_class(
            model_name=f"{model_name}_tuned_{model_type}",
            model_type=model_type
        )
    else:  # lightgbm
        base_model = model_class(
            model_name=f"{model_name}_tuned_{model_type}",
            model_type=model_type
        )
    
    # Use TimeSeriesSplit for time-series data
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Scoring metric
    scoring = 'accuracy' if model_type == 'classification' else 'r2'
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model.model,
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    print(f"  Searching over {len(param_grid)} parameter combinations...")
    grid_search.fit(X_train, y_train)
    
    # Update model with best parameters
    best_params = grid_search.best_params_
    print(f"  Best parameters: {best_params}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    
    # Create model with best parameters
    # Filter out None values and ensure proper parameter passing
    filtered_params = {k: v for k, v in best_params.items() if v is not None}
    best_model = model_class(
        model_name=f"{model_name}_tuned_{model_type}",
        model_type=model_type,
        **filtered_params
    )
    
    # Train with best parameters
    trainer = ModelTrainer(best_model)
    trainer.train_model(X_train, y_train)
    best_model.feature_names = X_train.columns.tolist()
    
    return best_model, {
        'best_params': best_params,
        'best_cv_score': grid_search.best_score_,
        'grid_search': grid_search
    }


def evaluate_with_feature_selection(
    model_class,
    model_name: str,
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    selection_method: str = 'importance',
    k: int = 20,
    best_params: Optional[Dict] = None,
    cv: int = 5
) -> Dict:
    print(f"\n--- {model_name.upper()} with Feature Selection ({selection_method}, k={k}) ---")
    

    if selection_method == 'importance':
        temp_model = model_class(
            model_name="temp",
            model_type=model_type,
            n_estimators=50  # Quick training for feature selection
        )
        temp_trainer = ModelTrainer(temp_model)
        temp_trainer.train_model(X_train, y_train)
        selected_features = select_features_by_importance(temp_model, feature_names, k)
    else:  # selectkbest
        selected_features = select_features_selectkbest(
            X_train, y_train, feature_names, k, model_type
        )
    
    print(f"  Selected {len(selected_features)} features")
    
    # Filter data
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    if best_params:
        # Filter out None values
        filtered_params = {k: v for k, v in best_params.items() if v is not None}
        model = model_class(
            model_name=f"{model_name}_fs_{selection_method}_{model_type}",
            model_type=model_type,
            **filtered_params
        )
    else:
        # Use default parameters
        if model_name == 'random_forest':
            model = model_class(
                model_name=f"{model_name}_fs_{selection_method}_{model_type}",
                model_type=model_type,
                n_estimators=100,
                max_depth=10
            )
        elif model_name == 'xgboost':
            model = model_class(
                model_name=f"{model_name}_fs_{selection_method}_{model_type}",
                model_type=model_type,
                n_estimators=100,
                max_depth=6
            )
        else:  # lightgbm
            model = model_class(
                model_name=f"{model_name}_fs_{selection_method}_{model_type}",
                model_type=model_type,
                n_estimators=100,
                max_depth=6
            )
    
    trainer = ModelTrainer(model)
    trainer.train_model(X_train_selected, y_train)
    
    test_metrics = trainer.evaluate_model(X_test_selected, y_test, verbose=False)
    
    # Cross-validation
    scoring = 'accuracy' if model_type == 'classification' else 'r2'
    cv_scores = trainer.cross_validate(X_train_selected, y_train, cv=cv, scoring=scoring)
    
    results = {
        'model': model,
        'selected_features': selected_features,
        'test_metrics': test_metrics,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }
    
    if model_type == 'classification':
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    else:
        print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test R²: {test_metrics['r2']:.4f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results


def optimize_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    model_type: str,
    cv: int = 5,
    feature_selection_k: int = 20
) -> Dict:
    results = {}
    
    model_classes = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel
    }
    
    for model_name, model_class in model_classes.items():
        print(f"\n{'='*60}")
        print(f"OPTIMIZING {model_name.upper()}")
        print(f"{'='*60}")
        
        model_results = {}
        
        # Baseline model
        baseline = train_baseline_model(
            model_class, model_name, model_type,
            X_train, y_train, X_test, y_test, cv
        )
        model_results['baseline'] = baseline
        
        # Hyperparameter tuning
        param_grid = get_hyperparameter_grids(model_name, model_type)
        tuned_model, tuning_info = tune_hyperparameters(
            model_class, model_name, model_type, param_grid,
            X_train, y_train, cv
        )
        
        # Evaluate tuned model
        trainer = ModelTrainer(tuned_model)
        tuned_test_metrics = trainer.evaluate_model(X_test, y_test, verbose=False)
        tuned_cv_scores = trainer.cross_validate(X_train, y_train, cv=cv, 
                                                 scoring='accuracy' if model_type == 'classification' else 'r2')
        
        model_results['tuned'] = {
            'model': tuned_model,
            'test_metrics': tuned_test_metrics,
            'cv_mean': tuned_cv_scores.mean(),
            'cv_std': tuned_cv_scores.std(),
            'cv_scores': tuned_cv_scores,
            'best_params': tuning_info['best_params'],
            'best_cv_score': tuning_info['best_cv_score']
        }
        
        print(f"\n  Tuned Model Performance:")
        if model_type == 'classification':
            print(f"    Test Accuracy: {tuned_test_metrics['accuracy']:.4f}")
            print(f"    CV Accuracy: {tuned_cv_scores.mean():.4f} (+/- {tuned_cv_scores.std() * 2:.4f})")
        else:
            print(f"    Test RMSE: {tuned_test_metrics['rmse']:.4f}")
            print(f"    Test R²: {tuned_test_metrics['r2']:.4f}")
            print(f"    CV R²: {tuned_cv_scores.mean():.4f} (+/- {tuned_cv_scores.std() * 2:.4f})")
        
        # Feature selection with importance
        fs_importance = evaluate_with_feature_selection(
            model_class, model_name, model_type,
            X_train, y_train, X_test, y_test, feature_names,
            'importance', feature_selection_k, tuning_info['best_params'], cv
        )
        model_results['fs_importance'] = fs_importance
        
        # Feature selection with SelectKBest
        fs_selectkbest = evaluate_with_feature_selection(
            model_class, model_name, model_type,
            X_train, y_train, X_test, y_test, feature_names,
            'selectkbest', feature_selection_k, tuning_info['best_params'], cv
        )
        model_results['fs_selectkbest'] = fs_selectkbest
        
        results[model_name] = model_results
    
    return results


def print_comparison_summary(results: Dict, model_type: str):
    print("PERFORMANCE COMPARISON SUMMARY")
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        
        if model_type == 'classification':
            print(f"{'Method':<25} {'Test Accuracy':<15} {'CV Accuracy':<20}")
            print("-" * 60)
            
            baseline = model_results['baseline']
            print(f"{'Baseline':<25} {baseline['test_metrics']['accuracy']:<15.4f} "
                  f"{baseline['cv_mean']:.4f} (+/- {baseline['cv_std']*2:.4f})")
            
            tuned = model_results['tuned']
            print(f"{'Tuned (HP only)':<25} {tuned['test_metrics']['accuracy']:<15.4f} "
                  f"{tuned['cv_mean']:.4f} (+/- {tuned['cv_std']*2:.4f})")
            
            fs_imp = model_results['fs_importance']
            print(f"{'Tuned + FS (Importance)':<25} {fs_imp['test_metrics']['accuracy']:<15.4f} "
                  f"{fs_imp['cv_mean']:.4f} (+/- {fs_imp['cv_std']*2:.4f})")
            
            fs_skb = model_results['fs_selectkbest']
            print(f"{'Tuned + FS (SelectKBest)':<25} {fs_skb['test_metrics']['accuracy']:<15.4f} "
                  f"{fs_skb['cv_mean']:.4f} (+/- {fs_skb['cv_std']*2:.4f})")
            
            # Improvement
            improvement = tuned['test_metrics']['accuracy'] - baseline['test_metrics']['accuracy']
            print(f"\n  Improvement from tuning: {improvement:+.4f} ({improvement/baseline['test_metrics']['accuracy']*100:+.2f}%)")
            
        else:  # regression
            print(f"{'Method':<25} {'Test RMSE':<15} {'Test R²':<15} {'CV R²':<20}")
            
            baseline = model_results['baseline']
            print(f"{'Baseline':<25} {baseline['test_metrics']['rmse']:<15.4f} "
                  f"{baseline['test_metrics']['r2']:<15.4f} "
                  f"{baseline['cv_mean']:.4f} (+/- {baseline['cv_std']*2:.4f})")
            
            tuned = model_results['tuned']
            print(f"{'Tuned (HP only)':<25} {tuned['test_metrics']['rmse']:<15.4f} "
                  f"{tuned['test_metrics']['r2']:<15.4f} "
                  f"{tuned['cv_mean']:.4f} (+/- {tuned['cv_std']*2:.4f})")
            
            fs_imp = model_results['fs_importance']
            print(f"{'Tuned + FS (Importance)':<25} {fs_imp['test_metrics']['rmse']:<15.4f} "
                  f"{fs_imp['test_metrics']['r2']:<15.4f} "
                  f"{fs_imp['cv_mean']:.4f} (+/- {fs_imp['cv_std']*2:.4f})")
            
            fs_skb = model_results['fs_selectkbest']
            print(f"{'Tuned + FS (SelectKBest)':<25} {fs_skb['test_metrics']['rmse']:<15.4f} "
                  f"{fs_skb['test_metrics']['r2']:<15.4f} "
                  f"{fs_skb['cv_mean']:.4f} (+/- {fs_skb['cv_std']*2:.4f})")
            
            # Improvement
            improvement_rmse = baseline['test_metrics']['rmse'] - tuned['test_metrics']['rmse']
            improvement_r2 = tuned['test_metrics']['r2'] - baseline['test_metrics']['r2']
            print(f"\n  Improvement from tuning:")
            print(f"    RMSE: {improvement_rmse:+.4f} ({improvement_rmse/baseline['test_metrics']['rmse']*100:+.2f}%)")
            print(f"    R²: {improvement_r2:+.4f} ({improvement_r2/baseline['test_metrics']['r2']*100:+.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Optimize ML models with hyperparameter tuning and feature selection')
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
        help='Target variable to optimize: win (classification), point_diff (regression), or both'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--feature-k',
        type=int,
        default=20,
        help='Number of features to select (default: 20)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    if args.input is None:
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
    
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    # Get features
    print("\nIdentifying available features...")
    feature_cols = get_available_features(df)
    print(f"Found {len(feature_cols)} features")
    
    all_results = {}
    
    # Optimize classification models
    if args.target in ['win', 'both']:
        if 'WIN' not in df.columns:
            print("\nWarning: 'WIN' column not found. Skipping classification models.")
        else:
            print("\nOPTIMIZING CLASSIFICATION MODELS (Win/Loss Prediction)")
            
            trainer_temp = ModelTrainer(RandomForestModel(model_type="classification"))
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = trainer_temp.prepare_data(
                df, feature_cols, 'WIN', test_size=args.test_size, random_state=args.random_state
            )
            
            print(f"\nTraining set: {len(X_train_clf)} games")
            print(f"Test set: {len(X_test_clf)} games")
            
            clf_results = optimize_models(
                X_train_clf, X_test_clf, y_train_clf, y_test_clf,
                feature_cols, 'classification', args.cv, args.feature_k
            )
            all_results['classification'] = clf_results
            
            print_comparison_summary(clf_results, 'classification')
    
    # Optimize regression models
    if args.target in ['point_diff', 'both']:
        if 'POINT_DIFF' not in df.columns:
            print("\nWarning: 'POINT_DIFF' column not found. Skipping regression models.")
        else:
            print("\nOPTIMIZING REGRESSION MODELS (Point Differential Prediction)")
            
            trainer_temp = ModelTrainer(RandomForestModel(model_type="regression"))
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = trainer_temp.prepare_data(
                df, feature_cols, 'POINT_DIFF', test_size=args.test_size, random_state=args.random_state
            )
            
            print(f"\nTraining set: {len(X_train_reg)} games")
            print(f"Test set: {len(X_test_reg)} games")
            
            reg_results = optimize_models(
                X_train_reg, X_test_reg, y_train_reg, y_test_reg,
                feature_cols, 'regression', args.cv, args.feature_k
            )
            all_results['regression'] = reg_results
            
            print_comparison_summary(reg_results, 'regression')
    
    print("\nSAVING OPTIMIZED MODELS")
    
    for task_type, task_results in all_results.items():
        for model_name, model_results in task_results.items():
            best_method = None
            best_score = -np.inf if task_type == 'classification' else np.inf
        
            for method_name in ['tuned', 'fs_importance', 'fs_selectkbest']:
                if method_name in model_results:
                    if task_type == 'classification':
                        score = model_results[method_name]['test_metrics']['accuracy']
                        if score > best_score:
                            best_score = score
                            best_method = method_name
                    else:
                        score = model_results[method_name]['test_metrics']['rmse']
                        if score < best_score:
                            best_score = score
                            best_method = method_name
            
            if best_method:
                best_model = model_results[best_method]['model']
                best_model.save()
                print(f"\nSaved best {model_name} ({task_type}): {best_method}")
                if best_method in ['fs_importance', 'fs_selectkbest']:
                    print(f"  Selected features: {len(model_results[best_method]['selected_features'])}")
    
    print("OPTIMIZATION COMPLETE!")


if __name__ == "__main__":
    main()