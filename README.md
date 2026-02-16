# NBA Rookie Progression Predictor

Predict an NBA player’s **sophomore-season per-game stats** (Points, Rebounds, Assists) from their **rookie-season per-game stats** using a TensorFlow/Keras neural network.

This repo contains:
- A data pipeline that pulls NBA career stats via `nba_api` and exports CSV datasets
- A training notebook that builds and trains a neural network model
- A prediction notebook that loads the trained model and provides an interactive CLI-style prompt for predictions
- Simple statistical confidence interval estimation based on model residuals

---

## What this predicts

**Inputs (rookie season features)**  
The model trains on a tested, idealized subset of rookie-year features:

- `GP_r` (games played)
- `MIN_r` (minutes per game)
- `FG_PCT_r` (field goal %)
- `REB_r` (rebounds per game)
- `AST_r` (assists per game)
- `PTS_r` (points per game)
- `TOV_r` (turnovers per game)

**Outputs (sophomore season targets)**
- `PTS_s` (predicted points per game)
- `REB_s` (predicted rebounds per game)
- `AST_s` (predicted assists per game)

---

## Repo structure

- `training.csv`  
  Training dataset containing rookie + sophomore stats (generated via `nba_api`).

- `prediction.csv`  
  Dataset containing rookie stats for active players (used for interactive predictions).

- `data_processing.ipynb`  
  Pulls player career stats using `nba_api` and exports `training.csv` and `prediction.csv`.

- `network.ipynb`  
  Trains the neural network model and saves it to disk.

- `predictions.ipynb`  
  Loads the trained model and allows interactive predictions by player name.

- `stat_prediction_model.h5` / `stat_prediction_model.keras`  
  Saved trained model artifacts.

- `blog_figures/`  
  Images used for writeups/visualizations.

---