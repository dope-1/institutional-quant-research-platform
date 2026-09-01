# Research design: Milestone 1

## Research question

Can cross-sectional ranking signals improve risk-adjusted equity portfolio performance relative to simple baselines after transaction costs, without look-ahead leakage?

## Unit of observation

Daily `(date, ticker)` panel rows.

## Feature timing

Features at date `t` use only data available at or before the close of `t`. Portfolio targets generated from those features become active on `t+1`, enforced by the backtest engine.

## Target

`forward_return_21d(t) = close(t+21) / close(t) - 1`

The target is deliberately future-looking and must never appear in the model feature matrix. It exists only for supervised learning in later milestones.

## Baselines

1. Equal-weight long-only portfolio.
2. Cross-sectional top-quintile 12-1 momentum portfolio.

Both rebalance monthly and include transaction cost + slippage assumptions.

## Known limitations

1. The initial static universe is survivorship-biased and is used to validate engineering, not to claim historical alpha.
2. Free market data are not institutional-grade licensed data.
3. FRED macro series can be revised; a future milestone should use vintage/release-aware macro data where relevant.
4. Corporate-action handling depends on the selected vendor's adjusted-price methodology.
5. No borrow constraints, capacity model, market impact model, sector neutrality, or optimizer is included yet.
