from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from quant_platform import __version__
from quant_platform.api.artifacts import ArtifactRepository, ArtifactValidationError
from quant_platform.api.routers import (
    explainability,
    holdout,
    methodology,
    models,
    optimization,
    overview,
    performance,
    regimes,
    risk,
)
from quant_platform.api.schemas import ErrorResponse, HealthResponse

ROOT = Path(__file__).resolve().parents[3]


def create_app(artifact_root: Path | str | None = None) -> FastAPI:
    application = FastAPI(
        title="Institutional Quant Research API",
        version=__version__,
        description=(
            "Read-only API for frozen, derived research artifacts. It does not train models "
            "or expose raw market data."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    if artifact_root is None:
        artifact_root = os.environ.get("QUANT_ARTIFACT_ROOT", ROOT / "artifacts")
    application.state.artifact_repository = ArtifactRepository(artifact_root)

    @application.exception_handler(ArtifactValidationError)
    async def artifact_error_handler(
        _request: Request, exc: ArtifactValidationError
    ) -> JSONResponse:
        payload = ErrorResponse(code="artifact_validation_failed", detail=str(exc))
        return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        validated = application.state.artifact_repository.validate()
        holdout_summary = application.state.artifact_repository._json(
            "milestone3d/summary.json", {"latest_market_date"}
        )
        return HealthResponse(
            status="ok",
            service="institutional-quant-research-api",
            package_version=__version__,
            methodology_freeze_date="2026-09-01",
            latest_market_date=holdout_summary["latest_market_date"],
            validated_artifacts=len(validated),
        )

    api = "/api/v1"
    for router in (
        overview.router,
        models.router,
        performance.router,
        risk.router,
        explainability.router,
        regimes.router,
        optimization.router,
        holdout.router,
        methodology.router,
    ):
        application.include_router(router, prefix=api)

    return application


app = create_app()
