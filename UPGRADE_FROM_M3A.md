# Upgrade from Milestone 3A to Milestone 3B

This package is a drop-in code/config upgrade. It contains no `data/` or `artifacts/` directories.

## Copy

Copy the contents of this upgrade into your existing project root and replace matching files.

Keep your existing:

```text
.venv/
data/
artifacts/
```

Replace/update:

```text
src/
tests/
configs/
docs/
README.md
Makefile
pyproject.toml
```

No new Python dependency is introduced beyond Milestone 3A, so the existing environment should work immediately.

## Verify

```powershell
pytest
```

Expected after this upgrade:

```text
42 passed
```

## Run

Milestone 3B requires the existing Milestone 2B, 2C and 3A outputs. It does not retrain or retune those models.

```powershell
quant-research run-milestone3b
```

Outputs are written only to:

```text
artifacts/milestone3b/
```

## Methodology warning

This milestone is diagnostic. Do not use the best-looking regime cell to create a retrospective switching strategy. The 2023–2025 period is already-inspected historical evidence and the many regime comparisons create substantial multiple-testing risk.
