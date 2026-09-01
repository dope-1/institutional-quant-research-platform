# Upgrade Milestone 1.2 to Milestone 2A

This upgrade package intentionally contains **no `data/` folder and no
`artifacts/` folder**. Your existing downloaded data, processed Milestone 1.2
files, reports, and `.venv` should stay in place.

Copy the contents of this upgrade package into the root of your existing project:

`E:\Projects\institutional-quant-research-platform\`

Allow Windows to replace matching code/config/documentation files.

Do not delete:

- `.venv/`
- `data/`
- `artifacts/`

No new Python dependency is required, so the existing editable environment can
be reused. If the terminal is not active, run:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then verify:

```powershell
pytest
```

Expected result: **22 passed**.

Milestone 1.2 does not need to be rerun if it already completed successfully.
Run the new ML milestone directly:

```powershell
quant-research run-milestone2a
```

The command reads:

- `data/processed/research_panel.parquet`
- `data/processed/splits/train.parquet`
- `data/processed/splits/validation.parquet`
- `data/processed/splits/test.parquet`

and writes new files only under:

- `data/processed/ml/`
- `artifacts/milestone2a/`
