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

```python
from src.models.model_trainer import ModelTrainer
# Import your specific model implementation

trainer = ModelTrainer(model)
trainer.train_model(X_train, y_train)
```

### 4. Incremental Updates

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

- Classification: Win/Loss prediction
- Regression: Point differential prediction

## Next Steps

1. Collect historical Warriors data
2. Perform exploratory data analysis
3. Engineer and select features
4. Train initial models
5. Implement incremental learning pipeline
6. Set up automated updates for current season

## Notes

- The NBA API may have rate limits; use delays between requests
- Models are saved in the `models/` directory
- Predictions are saved in `predictions/upcoming_games/`

## License

MIT