# Upgrade from Milestone 2B to Milestone 2C

This package is a drop-in code/config upgrade for an existing Milestone 2B repository.

## Preserve these existing directories

Do not replace or delete:

- `.venv/`
- `data/`
- `artifacts/`

Milestone 2C depends on the existing Milestone 2A monthly parquet datasets and Milestone 2B frozen Ridge/baseline artifacts.

## Copy/replace

Copy the contents of this upgrade package into the existing project root and allow Windows to replace files with the same names.

The important changed/new files are:

- `src/quant_platform/cli.py`
- `src/quant_platform/ml/nonlinear.py`
- `src/quant_platform/ml/reports.py`
- `src/quant_platform/ml/__init__.py`
- `tests/test_ml_nonlinear.py`
- `configs/research.yaml`
- `docs/milestone_2c_nonlinear.md`
- `README.md`
- `Makefile`
- `pyproject.toml`

The `ml` optional dependency group already contains XGBoost and LightGBM, so if the environment was originally installed with `pip install -e ".[dev,ml]"`, no dependency reinstall is normally required.

## Verify

```powershell
pytest
```

Expected combined suite:

```text
34 passed
```

## Run

```powershell
quant-research run-milestone2c
```

Milestone 2C reads existing data/artifacts and writes new outputs only under:

```text
artifacts/milestone2c/
```
