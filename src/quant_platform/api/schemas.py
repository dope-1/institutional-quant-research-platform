from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(StrictModel):
    methodology_freeze_date: str
    artifacts: list[str]
    artifact_sha256: dict[str, str]


class HealthResponse(StrictModel):
    status: Literal["ok"]
    service: str
    package_version: str
    schema_version: str = SCHEMA_VERSION
    methodology_freeze_date: str
    latest_market_date: str
    validated_artifacts: int = Field(ge=1)


class ErrorResponse(StrictModel):
    status: Literal["error"] = "error"
    code: str
    detail: str


class OverviewResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    project: dict[str, Any]
    data_quality: dict[str, Any]
    evaluation_periods: list[dict[str, Any]]
    key_findings: list[str]
    disclosures: list[str]
    provenance: Provenance


class StrategiesResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    strategies: list[dict[str, Any]]
    period_metrics: list[dict[str, Any]]
    provenance: Provenance


class ModelsResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    models: list[dict[str, Any]]
    selection_policy: list[str]
    holdout_freeze_confirmation: dict[str, Any]
    provenance: Provenance


class PerformanceResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    periods: list[dict[str, Any]]
    yearly_metrics: list[dict[str, Any]]
    disclosures: list[str]
    provenance: Provenance


class RiskResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    periods: list[dict[str, Any]]
    methodology: dict[str, Any]
    disclosures: list[str]
    provenance: Provenance


class RankICResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    historical_evaluation: dict[str, Any]
    prospective_holdout: dict[str, Any]
    disclosures: list[str]
    provenance: Provenance


class ExplainabilityResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    policy: str
    global_importance: list[dict[str, Any]]
    yearly_importance: list[dict[str, Any]]
    interactions: list[dict[str, Any]]
    model_summaries: dict[str, Any]
    disclosures: list[str]
    provenance: Provenance


class RegimesResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    methodology: dict[str, Any]
    counts: list[dict[str, Any]]
    rank_ic: list[dict[str, Any]]
    portfolio_metrics: list[dict[str, Any]]
    dispersion: list[dict[str, Any]]
    disclosures: list[str]
    provenance: Provenance


class OptimizationResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    objective: str
    constraints: dict[str, Any]
    selected_candidates: dict[str, Any]
    historical_metrics: list[dict[str, Any]]
    concentration: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    disclosures: list[str]
    provenance: Provenance


class HoldoutResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    status: str
    window: dict[str, Any]
    rank_ic_summary: list[dict[str, Any]]
    rank_ic_by_month: list[dict[str, Any]]
    portfolio_metrics: list[dict[str, Any]]
    methodology: dict[str, Any]
    integrity_warning: str
    interpretation: str
    provenance: Provenance


class MethodologyResponse(StrictModel):
    schema_version: str = SCHEMA_VERSION
    research_question: str
    universe: dict[str, Any]
    target: dict[str, Any]
    features: list[str]
    chronology_safeguards: list[str]
    purged_splits: list[dict[str, Any]]
    frozen_research_rules: list[str]
    data_publication_policy: list[str]
    provenance: Provenance
