# Data policy

The repository intentionally does **not** commit downloaded market data.

- `data/raw/` contains immutable provider snapshots plus `manifest.json` hashes.
- `data/processed/` contains engineered research panels and split metadata.
- Regenerate data with `quant-research run-milestone1`.

## Source policy for Milestone 1

- **Market OHLCV:** Yahoo Finance accessed through `yfinance`, used only as a convenient public research/education source. This is not a substitute for licensed institutional market data.
- **Macro:** Federal Reserve Economic Data (FRED).
- **Factors:** Kenneth French Data Library.

For institutional-grade production research, replace the market-data adapter with a licensed, point-in-time source (for example CRSP, Bloomberg, Refinitiv, Polygon, or another source your organisation is authorised to use).

## Reproducibility caveat

Data providers can revise history. The pipeline therefore writes local parquet snapshots and SHA-256 hashes. Re-running on another date may produce a different provider snapshot; the manifest makes that visible instead of silently pretending the data are immutable.
