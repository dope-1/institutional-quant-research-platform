# Milestone 3D — Genuine 2026 Holdout

Milestone 3D freezes the research choices selected before 2026 and evaluates them on incremental 2026 market data.

Key integrity rules:

- The historical `data/raw/market_ohlcv.parquet` snapshot is read-only for 3D.
- Only 2026-onward Yahoo market data are downloaded into `data/holdout2026/raw/`.
- Ridge choices come from the frozen Milestone 2B summary.
- Random Forest, XGBoost, and LightGBM choices come from the frozen Milestone 2C summary.
- Optimizer settings come from the frozen Milestone 3C summary.
- Monthly refitting is allowed only because expanding matured-label retraining was already part of the frozen methodology.
- Rank IC is reported only for signals whose 21-trading-day forward label is fully observed.
- The current, incomplete calendar month is never treated as a month-end signal.
- Portfolio evaluation is activated by the 31-Dec-2025 signal so January 2026 is included.

Default reproducible cutoff:

```bash
quant-research run-milestone3d --as-of 2026-09-01
```

Later, the exact same frozen methodology can be extended without retuning:

```bash
quant-research run-milestone3d --as-of 2026-10-01
```

Once 2026 results have been inspected, modifying the methodology because of those results contaminates 2026 as a holdout. Any changed methodology should be evaluated on a new future period.


### As-of cutoff semantics
`--as-of YYYY-MM-DD` is an **exclusive** market-data cutoff. Daily OHLCV rows dated exactly on the as-of date are excluded, so an in-progress Yahoo session can never contaminate the prospective holdout. For `--as-of 2026-09-01`, the latest eligible completed daily bar is therefore 2026-08-31.
