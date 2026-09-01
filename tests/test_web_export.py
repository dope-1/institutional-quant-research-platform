from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_platform.api.artifacts import (
    REQUIRED_CSV,
    WEB_FILENAMES,
    ArtifactRepository,
    ArtifactValidationError,
)

ROOT = Path(__file__).resolve().parents[1]


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def test_export_web_writes_exact_stable_public_dataset(tmp_path: Path) -> None:
    repository = ArtifactRepository(ROOT / "artifacts")
    first = repository.export_all(tmp_path)
    snapshot = {path.name: path.read_bytes() for path in first}
    second = repository.export_all(tmp_path)

    assert {path.name for path in first} == set(WEB_FILENAMES.values())
    assert {path.name: path.read_bytes() for path in second} == snapshot
    for path in first:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite)
        assert payload["schema_version"] == "1.0.0"
        assert payload["provenance"]["methodology_freeze_date"] == "2026-09-01"


def test_holdout_export_preserves_negative_prospective_rank_ic(tmp_path: Path) -> None:
    ArtifactRepository(ROOT / "artifacts").export_all(tmp_path)
    payload = json.loads((tmp_path / "holdout.json").read_text(encoding="utf-8"))
    means = {row["model"]: row["mean_rank_ic"] for row in payload["rank_ic_summary"]}
    assert set(means) == {"ridge", "random_forest", "xgboost", "lightgbm"}
    assert all(value < 0 for value in means.values())
    assert "negative" in payload["interpretation"].lower()


def test_artifact_validation_reports_schema_mismatches(tmp_path: Path) -> None:
    path = tmp_path / "milestone1" / "performance_metrics.csv"
    path.parent.mkdir(parents=True)
    path.write_text("strategy,sharpe\nequal_weight,1.0\n", encoding="utf-8")
    repository = ArtifactRepository(tmp_path)
    with pytest.raises(ArtifactValidationError) as exc_info:
        repository.validate()
    message = str(exc_info.value)
    assert "performance_metrics.csv is missing columns" in message
    assert set(REQUIRED_CSV["milestone1/performance_metrics.csv"]) - {"strategy", "sharpe"}
