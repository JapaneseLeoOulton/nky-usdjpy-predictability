# Forecasting Pipeline Design

## Target and Horizon
- Primary target: next-day log return of NKY (`r_nky_t1`).
- Parallel classification target: `sign(r_nky_t1)`.
- All features at time t use information available no later than NKY close on t.

## Data Scope & Splits
- Load merged market data from `data/processed/merged.csv`.
- Enforce NKY trading calendar as master index; forward fill USDJPY, SPX, VIX to prior NKY close.
- Training/validation range: 2023-01-01 to 2024-12-31.
- Locked test range: 2025-01-01 to 2025-12-31.
- Rolling CV: 4 expanding folds. Each fold trains on all data up to the fold start (minimum 120 trading days) and validates on the following ~3 months of data. Fold start anchors: 2023-07-01, 2024-01-01, 2024-04-01, 2024-07-01 (respecting available sample).

## Data Hygiene
- Drop duplicate dates; ensure strictly increasing index.
- Impute forward the control assets to match NKY dates; never backward fill future info.
- Remove rows with missing NKY after alignment.
- Winsorize return-based columns at configurable quantiles (default 1%/99%) using training data statistics only.
- Track both raw and winsorized experiments.

## Features (<=60)
1. **NKY returns**: log return `r_nky_t`, lags 1–5, 5/20 day rolling mean and std (lagged by 1), 14-day RSI, rolling z-score (5,20).
2. **USDJPY FX**: log return `r_fx_t`, lags 1–5, rolling std (5,20), rolling z-score (5,20), momentum over 3/5/10 day horizons.
3. **SPX controls**: log return lags 1–5.
4. **VIX controls**: difference lags 1–5, level lag1 and rolling z-score.
5. **Optional TOPIX**: auto-detected if column present; include return lags 1–5 and 20-day z-score.
6. **Interactions**: e.g., `r_fx_lag1 * vix_lag1`, `nky_vol_5 * abs(r_fx_lag1)`, `high_vix_dummy * r_nky_lag1`.
7. **Regime flags**: high/low VIX indicator (>=75th percentile), month-of-year one-hot, Friday dummy, BoJ/FOMC week placeholders (toggle via calendar table if provided).
8. **Targets**: shift NKY return by -1 for regression target; classification target via numpy sign.

## Models & Baselines
- Baselines: controls-only (OLS/Ridge on SPX/VIX features), AR(1) using NKY returns, random walk (predict 0).
- Regularised linear: Ridge, Lasso, Elastic Net via sklearn pipeline with StandardScaler.
- Tree/Boosting: RandomForestRegressor (shallow), XGBoostRegressor (if installed, fallback to HistGradientBoosting), LightGBM optional hook.
- Probabilistic: GradientBoostingRegressor with quantile loss at 0.1/0.5/0.9.
- Directional classifier: LogisticRegression and XGBClassifier (or HistGradientBoostingClassifier fallback).
- Hyperparameter grids kept compact; tuned using rolling CV.

## CV & Hyperparam Selection
- Custom splitter (expanding window) to honour time ordering.
- Optimise OOS RMSE; track MAE, R^2, hit-rate. Store per-fold metrics.
- For classifiers: optimise Brier score with AUC and calibration diagnostics.
- Select best model by median CV RMSE; keep top-3 for potential ensemble averaging (simple mean of predictions, equal weights).

## Locked Test Workflow (2025)
1. Refit tuned models on full 2023-2024 data.
2. Generate recursive day-ahead predictions for 2025, updating features sequentially.
3. Save outputs: point predictions, quantile bands, classification probability.

## Evaluation & Reporting
- Compute 2025 RMSE, MAE, OOS R^2 vs each baseline. Run Clark-West and Diebold-Mariano vs controls-only.
- Directional accuracy with Wilson score intervals.
- Residual diagnostics: autocorrelation (Ljung-Box), residual vs fitted plot data.
- Economic strategy: thresholded daily strategy with 10% vol targeting, 20 bps round-trip cost, leverage cap 1.5x. Track P&L, drawdown, turnover, Sharpe, hit-rate.

## Visuals & Artifacts
- Plots saved to `outputs/plots/2025/`: cumulative returns (actual vs predicted), scatter with 45° line, prediction bands, equity curves, monthly metrics bar chart.
- Tabular outputs: `outputs/models/cv_metrics.csv`, `outputs/models/final_predictions_2025.csv`, `outputs/models/test_metrics_2025.json`.
- Research log updated in `outputs/research_log.md` with data sources, preprocessing summary, experiments, and decisions.
- Model card stored under `outputs/model_card.md` documenting limitations and governance notes.

## Robustness & Ablations
- Automated reruns for feature subsets: NKY-only, FX-only, controls-only.
- Lag depth variants (1–3 vs 1–10) with summary metrics per configuration.
- Cost sensitivity (10 bps, 25 bps) and holding rule (daily vs Friday-only) toggles.
- Quarterly 2025 stability breakdown.

## Implementation Notes
- Use sklearn pipelines with `ColumnTransformer` to avoid leakage during scaling.
- Custom transformer for winsorisation and interaction features.
- Reuse HAC/Clark-West utilities from `cleandatacreation.py` where relevant.
- Ensure reproducibility with fixed random seeds and persisted configuration JSON.
