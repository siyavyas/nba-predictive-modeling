# Data Collection Guide

## Quick Start

### Collect Historical Data (2015-16 to 2023-24)
```bash
python3 collect_data.py --historical
```

### Collect Current Season Data (2025-26)
```bash
python3 collect_data.py --current
```

### Collect Both
```bash
python3 collect_data.py --all
```

### Custom Season Range
```bash
python3 collect_data.py --historical --start-season 2020-21 --end-season 2023-24
```

## Data Structure

The collected data includes the following columns:

### Game Information
- `Game_ID`: Unique game identifier
- `GAME_DATE`: Date of the game
- `MATCHUP`: Opponent (e.g., "GSW vs. LAL" or "GSW @ LAL")
- `SEASON`: Season string (e.g., "2023-24")
- `SEASON_TYPE`: "Regular Season" or "Playoffs"

### Game Outcome
- `WL`: Win/Loss ("W" or "L")
- `W`: Cumulative wins
- `L`: Cumulative losses
- `W_PCT`: Win percentage

### Team Statistics
- `PTS`: Points scored
- `FGM`, `FGA`, `FG_PCT`: Field goals made, attempted, percentage
- `FG3M`, `FG3A`, `FG3_PCT`: Three-pointers made, attempted, percentage
- `FTM`, `FTA`, `FT_PCT`: Free throws made, attempted, percentage
- `REB`, `OREB`, `DREB`: Total, offensive, defensive rebounds
- `AST`: Assists
- `STL`: Steals
- `BLK`: Blocks
- `TOV`: Turnovers
- `PF`: Personal fouls
- `MIN`: Minutes played
- `PLUS_MINUS`: Plus/minus (if available)

## Data Storage

- **Historical Data**: `data/raw/warriors_historical_YYYY_YY_to_YYYY_YY.csv`
- **Current Season**: `data/current_season/warriors_current_season_YYYY_YY.csv`

## Notes

1. **Rate Limiting**: The script includes a 0.6 second delay between API calls to avoid rate limits.

2. **Opponent Points**: The NBA API team game log doesn't directly provide opponent points. This will need to be fetched from boxscore data in a future enhancement.

3. **Data Updates**: For current season, run the collection script regularly to get the latest games.

4. **Error Handling**: If a season fails to fetch, the script will continue with other seasons and report errors.

## Next Steps
1. Run feature engineering to create derived features
2. Explore the data in a Jupyter notebook
3. Train initial models
4. Set up incremental learning pipeline