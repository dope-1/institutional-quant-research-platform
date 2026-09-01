from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_starts_api_and_dashboard_on_declared_ports() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {"api", "web"}
    assert services["api"]["ports"] == ["8000:8000"]
    assert services["web"]["ports"] == ["3000:3000"]
    assert services["api"]["healthcheck"]["test"]
    assert services["web"]["healthcheck"]["test"]


def test_api_container_serves_frozen_artifacts_instead_of_running_research() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    assert "quant-research serve-api" in dockerfile
    assert "QUANT_ARTIFACT_ROOT=/app/artifacts" in dockerfile
    assert "${PORT:-8000}" in dockerfile
    assert "pip install --no-cache-dir ." in dockerfile
    assert "pip install --no-cache-dir -e" not in dockerfile
    assert "run-milestone" not in dockerfile
    assert compose["services"]["api"]["volumes"][0]["read_only"] is True
    assert compose["services"]["api"]["environment"]["QUANT_ARTIFACT_ROOT"] == "/app/artifacts"


def test_dashboard_container_uses_standalone_production_output() -> None:
    dockerfile = (ROOT / "web" / "Dockerfile").read_text(encoding="utf-8")
    next_config = (ROOT / "web" / "next.config.ts").read_text(encoding="utf-8")

    assert 'CMD ["node", "server.js"]' in dockerfile
    assert '/app/.next/standalone' in dockerfile
    assert 'output: "standalone"' in next_config
    assert "process.env.VERCEL" in next_config
