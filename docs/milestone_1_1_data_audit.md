# Milestone 1.1 — Data audit and baseline sanity checks

Milestone 1.1 exists to stop bad data from becoming apparently sophisticated alpha.

Before fitting any machine-learning model, the research panel is audited for:

1. **Ticker history coverage** — first/last observation, missing dates inside each security's own history, and coverage of the global research window.
2. **Cross-sectional coverage** — how many securities are available each day and how many have a complete feature vector.
3. **Feature diagnostics** — missingness, infinities, central tendency, dispersion, and tail quantiles.
4. **Target correctness** — `forward_return_21d` is independently recomputed from prices and compared row-by-row with the stored target.
5. **Duplicate observations** — duplicate `(date, ticker)` keys fail the audit.
6. **Extreme-return review flags** — large adjusted daily returns are surfaced for inspection rather than silently deleted.
7. **Baseline stability** — equal-weight and momentum results are broken down year by year instead of judged only by one full-period Sharpe ratio.

## Outputs

After `quant-research audit-milestone1` or a full `quant-research run-milestone1`, the audit directory contains:

```text
artifacts/milestone1/audit/
├── data_quality_summary.json
├── ticker_coverage.csv
├── cross_sectional_coverage.csv
├── feature_diagnostics.csv
├── annual_strategy_metrics.csv
├── cross_sectional_coverage.png
└── annual_strategy_returns.png
```

## Interpreting missingness

Missing values are not automatically errors. A 252-day momentum feature must be missing during a security's warm-up period. A newly listed company will also have no earlier history. The important question is whether missingness is expected and understood, or unexplained.

## Survivorship bias

The current 60-name universe is static. A company that is in the universe today is implicitly treated as though it were eligible historically whenever price data exist. Therefore Milestone 1.x results are engineering and methodology validation, not a claim of unbiased historical alpha.

## What must be true before Milestone 2

The project should not start model tuning until:

- duplicate `(date, ticker)` rows are zero;
- independently recomputed labels exactly match stored labels;
- no feature contains infinite values;
- cross-sectional feature coverage is understood;
- extreme return flags have been inspected;
- baseline performance is examined by subperiod/year rather than only in aggregate.
