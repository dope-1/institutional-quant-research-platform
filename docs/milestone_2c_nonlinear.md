# Milestone 2C — Nonlinear Model Comparison

Milestone 2C asks whether nonlinear tree ensembles add persistent cross-sectional predictive value beyond the frozen linear Ridge benchmark from Milestone 2B.

## Models

The milestone evaluates three model families:

- Random Forest
- XGBoost
- LightGBM

All models use the monthly cross-sectional rank dataset created in Milestone 2A. No raw-price prediction target is introduced.

## Selection policy

For each model family:

1. A small, predeclared set of feature families is evaluated on the 2020–2022 validation period.
2. A small, predeclared hyperparameter set is evaluated using expanding walk-forward predictions.
3. The winning feature family and hyperparameters are selected by mean validation Spearman Rank IC.
4. A portfolio implementation is selected on validation Sharpe, with lower turnover as a tie-breaker.
5. All choices are frozen before evaluating 2023–2025.

The label-availability rule remains strict: a training row is eligible only when its `label_end_date` is strictly earlier than the signal date.

## Important evaluation disclosure

The 2023–2025 period has already been inspected in Milestones 2A and 2B. Milestone 2C does not tune on that period, but it should therefore be described as a **frozen historical evaluation period**, not as a pristine never-before-seen holdout.

A later post-study/live holdout is preferable for a final external confirmation.

## Outputs

Milestone 2C writes:

- validation configuration-search tables and predictions for each model
- validation portfolio-selection tables
- test predictions, Rank IC, decile returns, top-minus-bottom spreads, and yearly IC
- pre-test native tree feature importances
- nonlinear portfolio backtests
- a master nonlinear-vs-Ridge-vs-baseline comparison table
- bootstrap robustness summaries and comparison plots

All outputs are written beneath `artifacts/milestone2c/`.
