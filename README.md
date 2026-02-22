# Nitor Energy — Day-Ahead Electricity Price Forecasting

**Team:** Cartesian Trading — Sebastian Poshteh & Robert Sølyst Lildholdt

## Overview

20-model seed-diversity ensemble for day-ahead electricity price forecasting across multiple European markets. Final predictions are produced via per-market SLSQP weight optimisation on a held-out 2024 validation set.

## Model Architecture

| Component | Description |
|---|---|
| **Per-market LightGBM** | Deep (127 leaves) + regularised (63 leaves) blend, per market |
| **Per-market XGBoost** | Depth-7, per market |
| **Global XGBoost depth=7** | 7 seed variants (42, 123, 456, 789, 999, 314, 2024) |
| **Global XGBoost depth=8** | 5 seed variants (42, 123, 456, 789, 999) |
| **Global LightGBM** | num_leaves=127 and num_leaves=255 |
| **Global LightGBM DART** | 4 seed variants (42, 123, 456, 789) |

**Total: 20 models** ensembled with per-market SLSQP-optimised weights (100 random Dirichlet starts).

## Feature Engineering

- **Calendar:** hour, day-of-week, month, cyclic encodings, peak/off-peak flags
- **Weather:** solar irradiance, wind speed/direction, temperature, pressure, cloud cover, precipitation
- **Supply-demand balance:** load vs. renewable forecasts, penetration ratios
- **Target encoding:** per (market, hour, day-of-week) with leakage-safe exclusion windows
- **1-year lags:** 364–367 day offsets plus monthly aggregated lags
- **Cross-market:** per-timestamp aggregates across all markets
- **Intraday:** 3-hour rolling statistics within (date, market)
- **Regime clustering:** KMeans on daily balance/weather features (fitted on training split only)

## Data Leakage Safeguards

All feature pipelines are designed to prevent data leakage:

- Target encoding excludes Sep–Nov 2023 and Sep–Nov 2024 holdout windows
- Lag features source from historical targets only (364–367 days back)
- Winsorisation bounds and regime clustering fitted on training split only
- Cross-market and intraday features computed independently per dataframe

See the docstring at the top of `solution_final.py` for a full audit.

## Validation Strategy

- **Holdout 1:** Sep–Nov 2023
- **Holdout 2:** Sep–Nov 2024
- SLSQP weights optimised on Holdout 2; Holdout 1 used for out-of-sample monitoring
- All models retrained on the full training set before final test prediction

## Reproducibility

All random seeds are fixed. Running the script twice on the same machine produces identical output.

## Usage

### Setup

```bash
pip install -r requirements.txt
```

### Run

Place the competition data files in the project root:
- `train.csv`
- `test_for_participants.csv`
- `sample_submission.csv`

Then run:

```bash
python solution_final.py
```

Output: `submission_v34.csv`

## Project Structure

```
├── solution_final.py      # Full pipeline: features, training, ensemble, prediction
├── submission_v34.csv     # Final submission file
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```
