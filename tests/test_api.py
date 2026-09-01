from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from quant_platform.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = [
    "/api/v1/overview",
    "/api/v1/models",
    "/api/v1/performance",
    "/api/v1/risk",
    "/api/v1/rank-ic",
    "/api/v1/explainability",
    "/api/v1/regimes",
    "/api/v1/optimization",
    "/api/v1/holdout",
    "/api/v1/methodology",
]


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def test_health_validates_frozen_artifacts() -> None:
    client = TestClient(create_app(ROOT / "artifacts"))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["methodology_freeze_date"] == "2026-09-01"
    assert response.json()["validated_artifacts"] >= 20


def test_environment_can_set_artifact_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QUANT_ARTIFACT_ROOT", str(tmp_path))
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 503
    assert str(tmp_path) in str(client.app.state.artifact_repository.root)


def test_all_research_endpoints_have_stable_validated_responses() -> None:
    client = TestClient(create_app(ROOT / "artifacts"))
    for endpoint in ENDPOINTS:
        first = client.get(endpoint)
        second = client.get(endpoint)
        assert first.status_code == 200, (endpoint, first.text)
        assert first.content == second.content
        assert first.json()["schema_version"] == "1.0.0"


def test_openapi_documents_the_complete_v1_boundary() -> None:
    client = TestClient(create_app(ROOT / "artifacts"))
    schema = client.get("/openapi.json").json()
    assert client.get("/docs").status_code == 200
    assert set(ENDPOINTS).issubset(schema["paths"])
    for endpoint in ENDPOINTS:
        assert "responses" in schema["paths"][endpoint]["get"]


def test_public_boundary_does_not_expose_row_level_market_or_position_data() -> None:
    client = TestClient(create_app(ROOT / "artifacts"))
    forbidden = {"open", "high", "low", "close", "volume", "ticker", "target_weight", "prediction"}
    exposed: set[str] = set()
    for endpoint in ENDPOINTS:
        exposed |= _keys(client.get(endpoint).json())
    assert forbidden.isdisjoint(exposed)


def test_missing_artifacts_return_a_structured_503(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/v1/overview")
    assert response.status_code == 503
    assert response.json()["code"] == "artifact_validation_failed"
    assert "Required artifact is missing" in response.json()["detail"]
