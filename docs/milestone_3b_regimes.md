# Milestone 3B — Point-in-Time Market Regime Analysis

## Purpose

Milestone 3B asks whether the already-frozen ranking signals and portfolios behave differently across market environments. It is **diagnostic only**. No regime label is allowed to choose a model, change a hyperparameter, switch a portfolio, or alter position sizes.

The 2023–2025 period has already been inspected in earlier milestones. Regime results are therefore descriptive historical evidence, not a new pristine holdout.

## Point-in-time regime definitions

Level regimes for VIX, the 10-year Treasury yield, and Fed funds are classified as low / medium / high using rolling historical tercile thresholds. Thresholds at date `t` are estimated only from observations before `t`:

```text
prior observations only
        ↓
rolling lower / upper tercile thresholds
        ↓
current value at signal date
        ↓
low / medium / high
```

Defaults:

- rolling threshold window: 504 trading days;
- minimum history: 252 trading days;
- lower / upper thresholds: 1/3 and 2/3 quantiles.

Other regimes use fixed trailing rules:

- **Rate direction:** 63-trading-day change in the 10Y Treasury yield. Rising if above +0.10 percentage points, falling below -0.10 pp, otherwise flat.
- **Yield curve:** `10Y - 2Y`. Inverted below 0, flat from 0 to 0.25 pp, normal above 0.25 pp.
- **Market trend:** sign of the trailing 63-trading-day compounded Fama–French `Mkt-RF` return.

These rules use information available at the signal close. They do not use future macro observations.

## Ranking diagnostics

For Ridge, Random Forest, XGBoost, and LightGBM, Milestone 3B joins each monthly Rank IC and top-minus-bottom spread to the regime known on that signal date. It reports:

- mean and median metric;
- standard deviation;
- positive fraction;
- conventional t-statistic;
- non-parametric bootstrap interval;
- number of regime observations;
- a small-sample flag for cells with fewer than six signal months.

Because many model × regime cells are inspected, these statistics are descriptive and are **not corrected for multiple testing**.

## Portfolio diagnostics

The frozen daily portfolios are converted to signal-to-next-signal holding-period returns. A signal on `t` owns the regime at `t`, while portfolio returns begin on the following trading day. Each period is therefore approximately:

```text
(signal_t, signal_t+1]
```

for the next monthly signal date.

Strategies analysed:

- equal weight;
- momentum top quintile;
- frozen Ridge portfolio;
- Random Forest;
- XGBoost;
- LightGBM.

Conditional portfolio summaries include average holding-period return, annualised-from-monthly Sharpe, positive-period fraction, and mean turnover per holding period. Conditional max drawdown is intentionally omitted because concatenating non-contiguous regime episodes would be economically misleading.

## Outputs

`artifacts/milestone3b/` contains:

```text
summary.json
signal_date_regimes.csv
regime_counts.csv
daily_regime_history.parquet
rank_ic_by_regime.csv
top_bottom_spread_by_regime.csv
rank_ic_regime_dispersion.csv
portfolio_holding_period_returns.csv
portfolio_metrics_by_regime.csv
plots/
    regime_inputs_timeline.png
    rank_ic_<regime>.png
    portfolio_sharpe_<regime>.png
```

## Interpretation rule

Regime results can motivate hypotheses for future validation, but they must not be used retrospectively to build a rule such as “use Random Forest only when VIX is high.” Any such adaptive strategy would require a separately specified and genuinely future evaluation.
