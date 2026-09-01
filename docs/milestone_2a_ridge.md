# Milestone 2A — Cross-Sectional Ridge Research

Milestone 2A is the first machine-learning experiment. It deliberately uses a
regularised linear model before tree ensembles or neural networks so that any
incremental predictive value is easy to audit and defend.

## Research question

Can a regularised linear model improve cross-sectional equity return ranking and
risk-adjusted portfolio performance relative to the hardened equal-weight and
momentum baselines after transaction costs?

## Dataset design

Milestone 2A consumes the Milestone 1.2 `research_panel.parquet` and purged
train/validation/test partitions. It does not redownload market data.

Only *true* calendar month-end trading dates from the full panel are sampled.
This prevents the last surviving date of a purged partition from being mistaken
for a month end.

The initial feature set is stock-specific:

- 21/63/126/252-day momentum
- 6-1 and 12-1 momentum
- 21/63-day annualised volatility
- log dollar volume
- 21-day median dollar volume
- 21-day Amihud-style illiquidity

Market-wide macro/factor levels are preserved in the research panel but are not
used directly in this first cross-sectional model because they have no same-day
variation across stocks. Later milestones can use them through interactions or
regime conditioning.

## Robust cross-sectional transforms

For every month and feature, the model uses centered percentile ranks:

`rank(x_i,t) / N_t - 0.5`

The forward 21-trading-day return target is transformed the same way for model
fitting. This aligns the regression objective with ranking rather than exact
return-point prediction and bounds the numerical leverage of extreme observations
such as corporate-action-related price discontinuities.

Raw forward returns are retained for economic evaluation.

## Walk-forward protocol

Ridge alpha is selected only from the validation period using expanding monthly
walk-forward predictions. For every prediction date `t`, a row may enter model
training only when its `label_end_date` is strictly before `t`. Earlier validation
or test months join the expanding training history only after their labels have
fully matured.

After alpha selection, the test period is evaluated with the selected alpha and
an expanding history. Test performance is not used to select alpha.

## Evaluation

Milestone 2A writes:

- validation and test monthly Spearman Rank IC
- validation alpha-selection table
- test decile forward-return analysis
- pre-test Ridge coefficients
- a top-quintile long-only Ridge portfolio
- equal-weight and momentum portfolios on exactly the same signal dates
- drift-aware, transaction-cost-adjusted test metrics and equity curves

The main outputs are under `artifacts/milestone2a/`.
