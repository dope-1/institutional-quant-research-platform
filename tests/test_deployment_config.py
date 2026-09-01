from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_render_blueprint_deploys_read_only_api() -> None:
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    service = blueprint["services"][0]
    assert service["type"] == "web"
    assert service["runtime"] == "docker"
    assert service["plan"] == "free"
    assert service["healthCheckPath"] == "/health"
    assert service["autoDeployTrigger"] == "checksPass"
    assert {item["key"]: item["value"] for item in service["envVars"]}[
        "QUANT_ARTIFACT_ROOT"
    ] == "/app/artifacts"


def test_vercel_configuration_keeps_nextjs_framework_detection() -> None:
    config = json.loads((ROOT / "web" / "vercel.json").read_text(encoding="utf-8"))
    assert config["framework"] == "nextjs"


def test_ci_and_manual_smoke_workflows_are_versioned() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    smoke = (ROOT / ".github" / "workflows" / "deployment-smoke.yml").read_text(
        encoding="utf-8"
    )
    assert "pytest" in ci
    assert "ruff check" in ci
    assert "npm run lint" in ci
    assert "npm run build" in ci
    assert "docker compose build" in ci
    assert "workflow_dispatch" in smoke
    assert "smoke_deployment.py" in smoke
