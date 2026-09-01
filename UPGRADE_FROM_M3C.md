# Upgrade from Milestone 3C to 3D

Copy the contents of this upgrade over the existing project. Keep `.venv/`, `data/`, and `artifacts/`.

No new Python dependency is introduced.

Run:

```powershell
pytest
quant-research run-milestone3d --as-of 2026-09-01
```

Milestone 3D creates only new paths:

- `data/holdout2026/`
- `artifacts/milestone3d/`

It does not overwrite the frozen pre-2026 raw market snapshot.

For the first 3D run, do **not** use `--reuse-holdout-raw`; it must download the incremental 2026 snapshot. On later reruns with the same cutoff you may reuse it. If the stored snapshot is stale for a newer cutoff, the command refuses reuse and asks for a refresh.


### As-of cutoff semantics
`--as-of YYYY-MM-DD` is an **exclusive** market-data cutoff. Daily OHLCV rows dated exactly on the as-of date are excluded, so an in-progress Yahoo session can never contaminate the prospective holdout. For `--as-of 2026-09-01`, the latest eligible completed daily bar is therefore 2026-08-31.
