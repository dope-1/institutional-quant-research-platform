# Upgrade from Milestone 2C to Milestone 3A

Copy the contents of this upgrade into the existing project root and replace matching source/config files. The upgrade intentionally contains no `data/` or `artifacts/` directories.

Milestone 3A adds SHAP as an ML dependency, so with the existing `.venv` activated run:

```powershell
pip install -e ".[dev,ml]"
pytest
quant-research run-milestone3a
```

You do not need to recreate `.venv`, redownload market data, or rerun Milestones 1–2C if their outputs are still present.

Milestone 3A reads `data/processed/ml/*.parquet` and `artifacts/milestone2c/summary.json`, and writes only to `artifacts/milestone3a/`.
