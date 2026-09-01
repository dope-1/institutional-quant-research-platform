# Upgrade Milestone 2A to Milestone 2B

This is a drop-in code/config upgrade. It intentionally contains **no `data/` folder and
no `artifacts/` folder**.

Copy the contents into the root of your existing project and allow matching files to be
replaced. Keep your existing:

- `.venv/`
- `data/`
- `artifacts/`

No new Python dependency is required. Your existing editable installation can be reused.

Verify the upgrade:

```powershell
pytest
```

Expected result: **28 passed**.

Do not rerun Milestone 1.2 or Milestone 2A if they already completed. Run:

```powershell
quant-research run-milestone2b
```

The command reads the existing `research_panel.parquet` and the monthly ML partitions
created by Milestone 2A. It writes new outputs only under `artifacts/milestone2b/`.
