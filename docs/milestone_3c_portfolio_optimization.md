# Milestone 3C — Institutional Portfolio Optimization

Milestone 3C does not add another alpha model. It takes the already-frozen Ridge,
Random Forest, XGBoost, and LightGBM monthly predictions and asks a different
question:

> Can the same signals be translated into a more institutional portfolio by
> explicitly balancing expected alpha, covariance risk, concentration, and
> turnover?

For each signal date the optimizer solves a long-only, fully-invested problem of
roughly the form

`maximize alpha' w - lambda * w' Sigma w - gamma * ||w - w_pretrade||_1`

subject to an individual position cap and a candidate turnover cap.

## Point-in-time safeguards

- The covariance matrix is Ledoit-Wolf shrinkage estimated only from daily returns
  observed on or before the signal date.
- Previous target weights are drifted through realized returns to the current
  signal close before turnover is measured.
- Alpha model specifications are not retrained or retuned in 3C.
- Optimizer hyperparameters are selected on 2020–2022 validation only.
- 2023–2025 remains an already-inspected historical evaluation period, not a
  pristine holdout.

## Validation search

The default compact candidate set varies risk aversion, turnover penalty, and
turnover cap. Position cap, covariance lookback, and prediction scaling are held
fixed. Candidate selection uses net validation Sharpe after the existing
transaction-cost/slippage engine; lower turnover breaks exact Sharpe ties.

## Outputs

`artifacts/milestone3c/` contains:

- `optimizer_validation_search.csv`
- validation-selected target/diagnostic files per alpha model
- evaluation target and optimizer diagnostics per alpha model
- `optimizer_diagnostics.csv`
- `optimizer_concentration_summary.csv`
- `portfolio/performance_metrics.csv`
- `portfolio/yearly_performance_metrics.csv`
- optimized daily backtests per alpha model
- validation efficiency, concentration, risk/turnover, and equity-curve plots
- `summary.json`

## Interpretation

A lower-turnover optimized portfolio is not automatically evidence of better
alpha. 3C separates signal quality from portfolio construction quality. The
optimizer should be judged on whether it improves the conversion of a frozen
signal into risk-adjusted, diversified, cost-aware returns.
