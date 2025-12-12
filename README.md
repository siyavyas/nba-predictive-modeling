# NBA Predictive Modeling - Golden State Warriors

A machine learning project for predicting Golden State Warriors game outcomes using feature engineering and incremental learning.

## Project Overview

This project aims to:
1. Analyze historical Warriors game data using feature engineering
2. Build ML models to predict game outcomes
3. Implement an incremental learning system that updates with each new game
4. Make predictions for upcoming games using current season data

## Project Structure

```
nba-predictive-modeling/
├── data/
│   ├── raw/              # Raw game data from NBA API
│   ├── processed/        # Cleaned and engineered features
│   └── current_season/   # 2025-26 season data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_incremental_learning.ipynb
├── src/
│   ├── data_collection/
│   │   ├── nba_api_client.py    # NBA API wrapper
│   │   └── data_loader.py       # Load historical data
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── feature_definitions.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── model_trainer.py
│   │   ├── random_forest_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   └── incremental_updater.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── predictions/
│   └── upcoming_games/   # Predictions output
├── requirements.txt
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify NBA API access:**
   The project uses the `nba-api` Python library, which is free and doesn't require API keys.

## Usage

### 1. Data Collection

```python
from src.data_collection.nba_api_client import NBAApiClient

client = NBAApiClient()
historical_data = client.get_historical_seasons("2015-16", "2024-25")
```

### 2. Feature Engineering

```python
from src.features.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features_df = engineer.engineer_features(raw_data)
```

### 3. Model Training

**Using the training script:**
```bash
# Train both classification and regression models
python train_models.py --target both

# Train only classification models (win/loss)
python train_models.py --target win

# Train only regression models (point differential)
python train_models.py --target point_diff

# Specify custom input file and test size
python train_models.py --input data/processed/your_file.csv --test-size 0.25
```

**Using the optimization script (with hyperparameter tuning and feature selection):**
```bash
# Optimize both classification and regression models
python optimize_models.py --target both

# Optimize with custom parameters
python optimize_models.py --target both --cv 5 --feature-k 20

# Optimize only classification models
python optimize_models.py --target win --cv 3
```

**Using the Python API:**
```python
from src.models import RandomForestModel, XGBoostModel, LightGBMModel, ModelTrainer
from src.data_collection.data_loader import DataLoader

# Load processed data
df = DataLoader.load_processed_data("warriors_historical_2015_16_to_2024_25_processed.csv")

# Initialize model
model = RandomForestModel(model_type="classification")

# Train model
trainer = ModelTrainer(model)
X_train, X_test, y_train, y_test = trainer.prepare_data(df, feature_cols, 'WIN')
trainer.train_model(X_train, y_train)
metrics = trainer.evaluate_model(X_test, y_test)

# Save model
model.save()
```

### 4. Making Predictions

**Using the prediction script:**
```bash
# Make predictions for upcoming games (classification and regression)
python predict_games.py --task both

# Make only win/loss predictions
python predict_games.py --task classification

# Make only point differential predictions
python predict_games.py --task regression

# Demo mode (use last game as test case)
python predict_games.py --demo --task classification

# Specify custom parameters
python predict_games.py --model xgboost --days-ahead 7 --task both
```

**Using the Python API:**
```python
from src.models.base_model import BaseModel
import joblib

# Load a trained model
model = joblib.load('models/xgboost_fs_selectkbest_classification_v1.0.joblib')
model = model['model']  # Extract the model from saved data

# Prepare features for upcoming game
# (Use feature engineering pipeline)
predictions = model.predict(X_features)
probabilities = model.predict_proba(X_features)
```

### 5. Incremental Updates

```python
from src.models.incremental_updater import IncrementalUpdater

updater = IncrementalUpdater(model)
updater.update_model(historical_data, new_games, feature_cols, target_col)
```

## Features

- **Rolling Statistics**: Last 5, 10, 20 game averages
- **Contextual Features**: Home/away, rest days, back-to-back games
- **Streak Features**: Win/loss streaks
- **Season Phase**: Early, mid, late season indicators

## Model Types

### Available Models
- **Random Forest**: Ensemble tree-based model
- **XGBoost**: Gradient boosting model
- **LightGBM**: Light gradient boosting model

### Prediction Tasks
- **Classification**: Win/Loss prediction
- **Regression**: Point differential prediction

All models support both classification and regression tasks and are automatically saved to the `models/` directory after training.

## Model Training Results

The training script trains multiple models and provides evaluation metrics:

- **Classification Models**: Accuracy, precision, recall, F1-score
- **Regression Models**: RMSE, R² score
- **Feature Importance**: Top 10 most important features for each model

Models are saved with versioning in the `models/` directory and can be loaded for predictions.

## Model Optimization

The `optimize_models.py` script provides comprehensive model optimization:

1. **Baseline Models**: Trains models with default hyperparameters
2. **Hyperparameter Tuning**: Uses GridSearchCV with cross-validation to find optimal parameters
3. **Feature Selection**: Implements two methods:
   - Feature importance-based selection
   - SelectKBest statistical selection
4. **Performance Comparison**: Compares baseline vs. tuned vs. tuned+feature-selected models
5. **Cross-Validation**: Uses TimeSeriesSplit for robust evaluation (respects temporal order)

The optimization script automatically saves the best-performing model for each algorithm.

## Prediction Pipeline

The `predict_games.py` script provides a complete prediction pipeline:

1. **Schedule Fetching**: Automatically fetches upcoming games from NBA API schedule
2. **Model Loading**: Automatically finds and loads the best trained models
3. **Feature Engineering**: Engineers features for upcoming games using historical data
4. **Predictions**: Makes win/loss and point differential predictions
5. **Output**: Saves predictions in CSV and human-readable text format

**Key Features:**
- **NBA Schedule API Integration**: Automatically fetches upcoming games from NBA API
- Automatically uses optimized models (tuned + feature selection)
- Handles upcoming games without results
- Uses historical data to calculate rolling features
- Provides win probabilities and point differential predictions
- Falls back to current season data file if API unavailable
- Demo mode for testing with recent games

**Output Files:**
- `predictions_<model>_<timestamp>.csv`: Detailed predictions in CSV format
- `predictions_summary_<timestamp>.txt`: Human-readable summary report

**Example Output:**
```
Game 1:
  Date: 2025-12-12
  Matchup: GSW vs. MIN
  Prediction: Win
  Win Probability: 51.9%
```

## Next Steps

1. Enhance the feature engineering to add more variation to the predictions -- Regression predictions lack diversity due to limited feature variation for upcoming games. To improve this, we could add opponent strength features or other contextual information.
2. Implement incremental learning pipeline
3. Set up automated updates for current season
4. Create prediction accuracy tracking system

## Notes

- The NBA API may have rate limits; use delays between requests
- Models are saved in the `models/` directory
- Predictions are saved in `predictions/upcoming_games/`

## License

MIT