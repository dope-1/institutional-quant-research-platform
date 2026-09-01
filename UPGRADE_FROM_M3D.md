# Upgrade from Milestone 3D to 4A

Copy this upgrade over the existing repository. Keep `.venv/`, `data/`, and
`artifacts/`; Milestone 4A reads the frozen derived artifacts and never retrains a
model.

Install the API dependencies and verify the upgrade:

```powershell
python -m pip install -e ".[dev,ml]"
pytest
quant-research export-web
quant-research serve-api
```

The static export is written to `web/public/data/`. The API is available at
`http://127.0.0.1:8000`, with OpenAPI documentation at
`http://127.0.0.1:8000/docs`.

Only whitelisted derived summaries are exposed. Raw OHLCV snapshots, online
predictions, and ticker-level target weights are not part of the public API or
web export.
