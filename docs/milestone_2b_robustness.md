# Milestone 2B — Signal Robustness and Portfolio Efficiency

Milestone 2B strengthens the Ridge benchmark before any nonlinear model is introduced.
It deliberately keeps the 2023–2025 test period out of model and portfolio selection.

## Validation-only decisions

The command first compares six predeclared feature families on the validation period:

- momentum only
- volatility only
- liquidity only
- momentum + volatility
- momentum + liquidity
- all stock-specific features

Each family tunes Ridge alpha on validation walk-forward mean Spearman Rank IC. The
feature family with the highest validation mean Rank IC is frozen together with its
validation-selected alpha.

The selected signal is then converted into five predeclared long-only portfolio variants:

- equal-weight top 10%
- equal-weight top 20%
- equal-weight top 30%
- buffered top 20% (hold names until they leave the top 30%)
- linearly rank-weighted top 20%

The variant with the highest validation Sharpe is frozen. Lower annualised turnover is
used only as a tie-breaker.

Only after these decisions are frozen does the command run the test period.

## Robustness diagnostics

The selected Ridge model produces:

- monthly Rank IC and yearly Rank IC summaries
- top-minus-bottom realised 21-day spread
- bootstrap confidence intervals for mean IC and mean spread
- conventional mean t-statistics
- walk-forward coefficient histories
- coefficient sign-consistency and dispersion
- test-period portfolio metrics vs equal weight and momentum
- year-by-year test portfolio metrics

The bootstrap is deterministic from the configured seed and resamples monthly
observations. It is a robustness diagnostic, not a claim that monthly observations are
perfectly independent.

## Turnover buffering

The buffered portfolio retains an existing holding while it remains inside a wider exit
rank band. For the default top-20% portfolio with a 30% exit band, a stock can drift
outside the exact top quintile without forcing an immediate sell. Vacancies are filled
from the highest-ranked available names. This targets the specific high-turnover weakness
found in Milestone 2A without changing the predictive model.

## Command

```bash
quant-research run-milestone2b
```

It requires Milestone 2A's monthly files under `data/processed/ml/` and writes only to
`artifacts/milestone2b/`.
