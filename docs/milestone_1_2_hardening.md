# Milestone 1.2 — Pre-ML Methodology Hardening

Milestone 1.2 hardens the real-data research pipeline before any ML model is fitted.

## Compatibility fixes retained

The repository includes regression coverage for issues found during the first real run:

1. **Adjusted OHLC floating-point noise** — genuine OHLC violations still fail, but machine-precision differences around `1e-14` are tolerated.
2. **Kenneth French `PeriodIndex`** — factor dates are converted with `to_timestamp()` when required.
3. **Pandas datetime precision** — merge keys are normalized to timezone-naive `datetime64[ns]` before macro/factor merges.

## Macro point-in-time handling

FRED daily series can contain an explicit missing observation on an equity trading day. The macro table is now forward-filled *within the macro source table* and then backward-as-of merged into equities. Therefore a market row can receive only the last known macro value; a future observation is never used.

## Drift-aware backtest

Targets formed at close `t` become active for the close-`t` to close-`t+1` return. Once active, portfolio weights drift with asset returns until the next rebalance. Turnover is measured against the drifted pre-trade portfolio.

This removes the previous implicit assumption that target weights were restored every day without charging the associated turnover.

## Purged chronological splits

For a `21` trading-observation forward target, late training rows can depend on validation-period prices. Milestone 1.2 computes each row's label-end date and retains it only if the label remains inside the row's own partition.

Purged split parquet files are written to:

```text
data/processed/splits/train.parquet
data/processed/splits/validation.parquet
data/processed/splits/test.parquet
```

`data/processed/split_summary.json` records unpurged rows and the number removed by the purge.

## Explicit anomaly files

The audit now additionally writes:

```text
artifacts/milestone1/audit/extreme_return_rows.csv
artifacts/milestone1/audit/zero_volume_rows.csv
artifacts/milestone1/audit/missing_macro_dates.csv
```

These are review outputs, not automatic deletion rules.

## Reuse existing raw snapshots

If Milestone 1.1 already downloaded the three raw snapshots, run:

```bash
quant-research run-milestone1 --reuse-raw
```

This validates and reuses the existing Parquet snapshots, rebuilds features and labels, creates the purged splits, reruns the hardened baselines, and rewrites the audit outputs without contacting Yahoo/FRED/Kenneth French again.
