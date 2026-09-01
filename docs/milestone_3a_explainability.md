# Milestone 3A — Explainability and Interaction Analysis

Milestone 3A explains the validation-selected nonlinear models from Milestone 2C without changing their features, hyperparameters, or portfolio rules.

## Design

For Random Forest, XGBoost, and LightGBM, the command fits one **pre-test reference model** using the selected 2C specification and only observations whose 21-day labels matured before the first 2023 signal. This reference model is used only for structural explanation; the expanding walk-forward results from 2C remain the performance evidence.

The analysis includes:

- TreeSHAP global feature importance.
- Spearman-Rank-IC permutation importance.
- Year-by-year SHAP and permutation-importance stability.
- TreeSHAP pairwise interaction strength.
- SHAP dependence plots for predeclared momentum × liquidity and momentum × volatility pairs.
- Quartile interaction tables containing mean SHAP contribution, prediction, target rank, and realised 21-day return.

## Interpretation discipline

SHAP values describe how a fitted model maps inputs to predictions. They are not causal effects. Permutation importance is evaluated retrospectively against realised target ranks and therefore measures predictive dependence, not economic causality. Milestone 3A performs no model selection and must not be used to retroactively change the 2C specifications based on 2023–2025 results.
