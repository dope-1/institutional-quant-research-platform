from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.api.schemas import (
    ExplainabilityResponse,
    HoldoutResponse,
    MethodologyResponse,
    ModelsResponse,
    OptimizationResponse,
    OverviewResponse,
    PerformanceResponse,
    Provenance,
    RankICResponse,
    RegimesResponse,
    RiskResponse,
    StrategiesResponse,
)

FREEZE_DATE = "2026-09-01"
WEB_FILENAMES = {
    "overview": "overview.json",
    "strategies": "strategies.json",
    "performance": "performance.json",
    "rank_ic": "rank_ic.json",
    "models": "models.json",
    "shap": "shap.json",
    "regimes": "regimes.json",
    "optimization": "optimization.json",
    "holdout": "holdout.json",
    "methodology": "methodology.json",
}


class ArtifactValidationError(RuntimeError):
    """Raised when a required public research artifact is missing or malformed."""


REQUIRED_JSON: dict[str, set[str]] = {
    "milestone1/audit/data_quality_summary.json": {"rows", "tickers", "audit_pass"},
    "milestone2b/summary.json": {
        "selected_alpha",
        "selected_rank_features",
        "selected_portfolio_variant",
    },
    "milestone2c/summary.json": {"selected_models", "selection_policy"},
    "milestone3a/summary.json": {"models", "policy"},
    "milestone3b/summary.json": {"regime_methodology", "policy"},
    "milestone3c/summary.json": {
        "optimizer",
        "selected_optimizer_candidates",
        "selection_policy",
    },
    "milestone3d/summary.json": {
        "latest_market_date",
        "rank_ic_summary",
        "portfolio_metrics",
        "holdout_integrity_warning",
    },
    "milestone3d/frozen_specifications.json": {
        "ridge",
        "nonlinear_models",
        "optimizer_candidates",
    },
}

METRIC_COLUMNS = {
    "strategy",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "var_95_daily",
    "cvar_95_daily",
    "annualized_turnover",
}

REQUIRED_CSV: dict[str, set[str]] = {
    "milestone1/performance_metrics.csv": METRIC_COLUMNS,
    "milestone2b/test_rank_ic.csv": {"date", "rank_ic", "n_assets"},
    "milestone2c/model_comparison.csv": {"model", "model_class", "sharpe"},
    "milestone2c/portfolio/yearly_performance_metrics.csv": {
        "strategy",
        "year",
        "total_return",
        "sharpe",
    },
    "milestone2c/test/random_forest_rank_ic.csv": {"date", "rank_ic", "n_assets"},
    "milestone2c/test/xgboost_rank_ic.csv": {"date", "rank_ic", "n_assets"},
    "milestone2c/test/lightgbm_rank_ic.csv": {"date", "rank_ic", "n_assets"},
    "milestone3a/shap_global_importance.csv": {
        "model",
        "feature",
        "normalized_mean_abs_shap",
    },
    "milestone3a/shap_importance_by_year.csv": {
        "model",
        "year",
        "feature",
        "normalized_importance",
    },
    "milestone3a/shap_interaction_importance.csv": {
        "model",
        "feature_a",
        "feature_b",
        "mean_abs_interaction",
    },
    "milestone3b/regime_counts.csv": {"regime_dimension", "regime", "signal_months"},
    "milestone3b/rank_ic_by_regime.csv": {
        "model",
        "regime_dimension",
        "regime",
        "mean",
        "observations",
    },
    "milestone3b/portfolio_metrics_by_regime.csv": {
        "strategy",
        "regime_dimension",
        "regime",
        "conditional_sharpe",
    },
    "milestone3b/rank_ic_regime_dispersion.csv": {
        "model",
        "regime_dimension",
        "rank_ic_dispersion",
    },
    "milestone3c/portfolio/performance_metrics.csv": METRIC_COLUMNS,
    "milestone3c/portfolio/yearly_performance_metrics.csv": {
        "strategy",
        "year",
        "total_return",
        "sharpe",
    },
    "milestone3c/optimizer_concentration_summary.csv": {
        "model",
        "candidate",
        "mean_effective_n",
        "mean_max_weight",
    },
    "milestone3c/optimizer_diagnostics.csv": {
        "date",
        "model",
        "candidate",
        "solver_success",
        "effective_n",
    },
    "milestone3d/holdout_rank_ic.csv": {"date", "rank_ic", "n_assets", "model"},
    "milestone3d/holdout_rank_ic_summary.csv": {
        "model",
        "mean_rank_ic",
        "months",
    },
    "milestone3d/portfolio/performance_metrics.csv": METRIC_COLUMNS,
}


STRATEGY_CATALOG = [
    {
        "id": "equal_weight",
        "name": "Equal weight",
        "kind": "baseline",
        "description": "Long-only equal allocation across the eligible universe.",
    },
    {
        "id": "momentum_top_quintile",
        "name": "Momentum top quintile",
        "kind": "baseline",
        "description": "Long-only top quintile ranked by frozen 12-1 momentum.",
    },
    {
        "id": "ridge_heuristic",
        "name": "Ridge heuristic",
        "kind": "linear_ml",
        "description": "Validation-frozen Ridge ranking with the selected top-10% portfolio.",
    },
    {
        "id": "random_forest_heuristic",
        "name": "Random Forest heuristic",
        "kind": "nonlinear_ml",
        "description": "Frozen Random Forest ranking with validation-selected rank weighting.",
    },
    {
        "id": "xgboost_heuristic",
        "name": "XGBoost heuristic",
        "kind": "nonlinear_ml",
        "description": "Frozen XGBoost ranking with a buffered long-only portfolio.",
    },
    {
        "id": "lightgbm_heuristic",
        "name": "LightGBM heuristic",
        "kind": "nonlinear_ml",
        "description": "Frozen LightGBM ranking with a buffered long-only portfolio.",
    },
    *[
        {
            "id": f"{model}_optimized",
            "name": f"{model.replace('_', ' ').title()} optimized",
            "kind": "optimized_ml",
            "description": (
                "Frozen alpha translated through the validation-selected constrained optimizer."
            ),
        }
        for model in ("ridge", "random_forest", "xgboost", "lightgbm")
    ],
]


class ArtifactRepository:
    """Whitelisted, read-only access to derived public research artifacts."""

    def __init__(self, artifact_root: Path | str) -> None:
        self.root = Path(artifact_root).resolve()

    def _path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if self.root != path and self.root not in path.parents:
            raise ArtifactValidationError(f"Artifact path escapes configured root: {relative}")
        return path

    def _json(self, relative: str, required_keys: Iterable[str] = ()) -> dict[str, Any]:
        path = self._path(relative)
        if not path.is_file():
            raise ArtifactValidationError(f"Required artifact is missing: {relative}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(f"Invalid JSON artifact {relative}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ArtifactValidationError(f"JSON artifact must contain an object: {relative}")
        missing = sorted(set(required_keys) - set(payload))
        if missing:
            raise ArtifactValidationError(
                f"JSON artifact {relative} is missing keys: {', '.join(missing)}"
            )
        return self._clean(payload)

    def _csv(self, relative: str, required_columns: Iterable[str] = ()) -> pd.DataFrame:
        path = self._path(relative)
        if not path.is_file():
            raise ArtifactValidationError(f"Required artifact is missing: {relative}")
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ArtifactValidationError(f"Invalid CSV artifact {relative}: {exc}") from exc
        missing = sorted(set(required_columns) - set(frame.columns))
        if missing:
            raise ArtifactValidationError(
                f"CSV artifact {relative} is missing columns: {', '.join(missing)}"
            )
        return frame

    @staticmethod
    def _clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): ArtifactRepository._clean(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [ArtifactRepository._clean(item) for item in value]
        if isinstance(value, pd.Timestamp | np.datetime64):
            return str(pd.Timestamp(value).date())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if pd.isna(value) if not isinstance(value, str | bytes) else False:
            return None
        return value

    def _records(
        self,
        relative: str,
        required_columns: Iterable[str],
        *,
        sort_by: list[str],
    ) -> list[dict[str, Any]]:
        frame = self._csv(relative, required_columns)
        available = [column for column in sort_by if column in frame.columns]
        if available:
            frame = frame.sort_values(available, kind="stable", na_position="last")
        return self._clean(frame.to_dict(orient="records"))

    def validate(self) -> list[str]:
        errors: list[str] = []
        for relative, keys in REQUIRED_JSON.items():
            try:
                self._json(relative, keys)
            except ArtifactValidationError as exc:
                errors.append(str(exc))
        for relative, columns in REQUIRED_CSV.items():
            try:
                self._csv(relative, columns)
            except ArtifactValidationError as exc:
                errors.append(str(exc))
        if errors:
            raise ArtifactValidationError("; ".join(errors))
        return sorted([*REQUIRED_JSON, *REQUIRED_CSV])

    def _provenance(self, relatives: Iterable[str]) -> Provenance:
        paths = sorted(set(relatives))
        hashes: dict[str, str] = {}
        for relative in paths:
            path = self._path(relative)
            if not path.is_file():
                raise ArtifactValidationError(f"Required artifact is missing: {relative}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[relative] = digest
        return Provenance(
            methodology_freeze_date=FREEZE_DATE,
            artifacts=paths,
            artifact_sha256=hashes,
        )

    def overview(self) -> OverviewResponse:
        paths = [
            "milestone1/audit/data_quality_summary.json",
            "milestone2c/summary.json",
            "milestone3c/summary.json",
            "milestone3d/summary.json",
        ]
        quality = self._json(paths[0], REQUIRED_JSON[paths[0]])
        m2c = self._json(paths[1], REQUIRED_JSON[paths[1]])
        m3c = self._json(paths[2], REQUIRED_JSON[paths[2]])
        holdout = self._json(paths[3], REQUIRED_JSON[paths[3]])
        return OverviewResponse(
            project={
                "name": "institutional-quant-research-platform",
                "research_status": "FROZEN",
                "methodology_freeze_date": FREEZE_DATE,
                "latest_market_date": holdout["latest_market_date"],
                "latest_milestone": "4A",
                "universe_size": quality["tickers"],
            },
            data_quality=quality,
            evaluation_periods=[
                {
                    "id": "historical_baseline",
                    "label": "Historical baseline research",
                    "start": quality["start_date"],
                    "end": quality["end_date"],
                    "status": "historical",
                },
                {
                    "id": "historical_evaluation",
                    "label": "Already-inspected historical evaluation",
                    "start": "2023-01-03",
                    "end": "2025-12-31",
                    "status": "already_inspected",
                },
                {
                    "id": "prospective_holdout",
                    "label": "Genuine prospective holdout",
                    "start": holdout["holdout_start"],
                    "end": holdout["latest_market_date"],
                    "status": holdout["status"],
                },
            ],
            key_findings=[
                "Momentum remains a strong conventional benchmark.",
                (
                    "Random Forest had the most persistent positive historical nonlinear "
                    "Rank IC, but its confidence interval included zero."
                ),
                (
                    "Turnover-aware optimization improved Random Forest historical "
                    "risk-adjusted efficiency and implementability."
                ),
                (
                    "All learned models produced negative mean Rank IC in the first seven "
                    "matured 2026 holdout months."
                ),
                (
                    "The early prospective result is negative evidence, not a final "
                    "statistical verdict."
                ),
            ],
            disclosures=[
                m2c["test_reuse_disclosure"],
                m3c["historical_evaluation_disclosure"],
                holdout["holdout_integrity_warning"],
                (
                    "The static equity universe is survivorship-biased and is not "
                    "point-in-time index membership."
                ),
            ],
            provenance=self._provenance(paths),
        )

    def strategies(self) -> StrategiesResponse:
        paths = [
            "milestone2c/model_comparison.csv",
            "milestone3c/portfolio/performance_metrics.csv",
            "milestone3d/portfolio/performance_metrics.csv",
        ]
        rows: list[dict[str, Any]] = []
        for period, path in zip(
            ("historical_2023_2025", "optimized_2023_2025", "holdout_2026"),
            paths,
            strict=True,
        ):
            key = "model" if path.endswith("model_comparison.csv") else "strategy"
            records = self._records(path, REQUIRED_CSV[path], sort_by=[key])
            rows.extend({"period": period, **record} for record in records)
        return StrategiesResponse(
            strategies=self._clean(STRATEGY_CATALOG),
            period_metrics=rows,
            provenance=self._provenance(paths),
        )

    def models(self) -> ModelsResponse:
        paths = [
            "milestone2b/summary.json",
            "milestone2c/summary.json",
            "milestone3d/frozen_specifications.json",
        ]
        ridge = self._json(paths[0], REQUIRED_JSON[paths[0]])
        nonlinear = self._json(paths[1], REQUIRED_JSON[paths[1]])
        frozen = self._json(paths[2], REQUIRED_JSON[paths[2]])
        models: list[dict[str, Any]] = [
            {
                "id": "ridge",
                "family": "linear_ml",
                "feature_group": ridge["selected_feature_group"],
                "rank_features": ridge["selected_rank_features"],
                "hyperparameters": {"alpha": ridge["selected_alpha"]},
                "portfolio_variant": ridge["selected_portfolio_variant"],
            }
        ]
        for name in sorted(nonlinear["selected_models"]):
            spec = nonlinear["selected_models"][name]
            models.append(
                {
                    "id": name,
                    "family": "nonlinear_ml",
                    "feature_group": spec["feature_group"],
                    "rank_features": spec["rank_features"],
                    "candidate": spec["candidate"],
                    "hyperparameters": spec["params"],
                    "portfolio_variant": spec["portfolio_variant"],
                }
            )
        return ModelsResponse(
            models=models,
            selection_policy=[ridge["selection_policy"], nonlinear["selection_policy"]],
            holdout_freeze_confirmation={
                "matches_milestone3d": frozen,
                "retuned_on_2026": False,
            },
            provenance=self._provenance(paths),
        )

    def performance(self) -> PerformanceResponse:
        paths = [
            "milestone1/performance_metrics.csv",
            "milestone2c/model_comparison.csv",
            "milestone3c/portfolio/performance_metrics.csv",
            "milestone3d/portfolio/performance_metrics.csv",
            "milestone2c/portfolio/yearly_performance_metrics.csv",
            "milestone3c/portfolio/yearly_performance_metrics.csv",
        ]
        periods = [
            {
                "id": "baseline_2012_2025",
                "label": "Historical baselines, 2012-2025",
                "status": "historical",
                "metrics": self._records(paths[0], REQUIRED_CSV[paths[0]], sort_by=["strategy"]),
            },
            {
                "id": "historical_2023_2025",
                "label": "Already-inspected historical evaluation, 2023-2025",
                "status": "already_inspected",
                "metrics": self._records(paths[1], REQUIRED_CSV[paths[1]], sort_by=["model"]),
            },
            {
                "id": "optimized_2023_2025",
                "label": "Optimized historical evaluation, 2023-2025",
                "status": "already_inspected",
                "metrics": self._records(paths[2], REQUIRED_CSV[paths[2]], sort_by=["strategy"]),
            },
            {
                "id": "holdout_2026",
                "label": "Prospective partial-year holdout, 2026",
                "status": "genuine_holdout_partial_year",
                "metrics": self._records(paths[3], REQUIRED_CSV[paths[3]], sort_by=["strategy"]),
            },
        ]
        yearly: list[dict[str, Any]] = []
        for source, path in (("heuristic", paths[4]), ("optimized", paths[5])):
            records = self._records(path, REQUIRED_CSV[path], sort_by=["strategy", "year"])
            yearly.extend({"source": source, **record} for record in records)
        return PerformanceResponse(
            periods=periods,
            yearly_metrics=yearly,
            disclosures=[
                (
                    "All performance is net of the configured transaction-cost and "
                    "slippage assumptions."
                ),
                "2023-2025 is already-inspected historical evidence, not a pristine holdout.",
                "2026 annualized metrics are provisional because the sample is a partial year.",
                (
                    "Portfolio returns and Rank IC measure different properties and must "
                    "be interpreted separately."
                ),
            ],
            provenance=self._provenance(paths),
        )

    def risk(self) -> RiskResponse:
        performance = self.performance()
        risk_fields = {
            "strategy",
            "model",
            "annualized_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "var_95_daily",
            "cvar_95_daily",
            "annualized_turnover",
            "information_ratio",
        }
        periods: list[dict[str, Any]] = []
        for period in performance.periods:
            metrics = [
                {key: value for key, value in row.items() if key in risk_fields}
                for row in period["metrics"]
            ]
            periods.append(
                {key: value for key, value in period.items() if key != "metrics"}
                | {"metrics": metrics}
            )
        return RiskResponse(
            periods=periods,
            methodology={
                "var": "Historical daily 95% Value at Risk magnitude.",
                "cvar": "Historical daily 95% Conditional Value at Risk magnitude.",
                "turnover": "Measured against naturally drifted pre-trade weights.",
                "execution": "Signals execute with a one-trading-day lag.",
            },
            disclosures=performance.disclosures,
            provenance=performance.provenance,
        )

    def rank_ic(self) -> RankICResponse:
        paths = [
            "milestone2b/test_rank_ic.csv",
            "milestone2c/test/random_forest_rank_ic.csv",
            "milestone2c/test/xgboost_rank_ic.csv",
            "milestone2c/test/lightgbm_rank_ic.csv",
            "milestone2b/summary.json",
            "milestone2c/summary.json",
            "milestone3d/holdout_rank_ic.csv",
            "milestone3d/holdout_rank_ic_summary.csv",
            "milestone3d/summary.json",
        ]
        monthly: list[dict[str, Any]] = []
        for model, path in zip(
            ("ridge", "random_forest", "xgboost", "lightgbm"), paths[:4], strict=True
        ):
            records = self._records(path, REQUIRED_CSV[path], sort_by=["date"])
            monthly.extend({"model": model, **record} for record in records)
        m2b = self._json(paths[4], REQUIRED_JSON[paths[4]])
        m2c = self._json(paths[5], REQUIRED_JSON[paths[5]])
        historical_summary = [{"model": "ridge", **m2b["test_statistical_robustness"]["rank_ic"]}]
        for model in ("random_forest", "xgboost", "lightgbm"):
            historical_summary.append(
                {"model": model, **m2c["selected_models"][model]["test_statistics"]["rank_ic"]}
            )
        holdout = self._json(paths[8], REQUIRED_JSON[paths[8]])
        return RankICResponse(
            historical_evaluation={
                "period": "2023-2025",
                "status": "already_inspected",
                "summary": historical_summary,
                "monthly": sorted(monthly, key=lambda row: (row["model"], row["date"])),
            },
            prospective_holdout={
                "period": "2026 partial year",
                "status": holdout["status"],
                "summary": self._records(paths[7], REQUIRED_CSV[paths[7]], sort_by=["model"]),
                "monthly": self._records(
                    paths[6], REQUIRED_CSV[paths[6]], sort_by=["model", "date"]
                ),
                "latest_matured_signal": holdout["latest_matured_rank_signal"],
            },
            disclosures=[
                (
                    "Rank IC is the monthly Spearman correlation between cross-sectional "
                    "predictions and forward returns."
                ),
                "Only 2026 signals with fully matured 21-trading-day labels are included.",
                (
                    "Negative 2026 mean Rank IC is shown for every learned model without "
                    "cherry-picking."
                ),
            ],
            provenance=self._provenance(paths),
        )

    def explainability(self) -> ExplainabilityResponse:
        paths = [
            "milestone3a/summary.json",
            "milestone3a/shap_global_importance.csv",
            "milestone3a/shap_importance_by_year.csv",
            "milestone3a/shap_interaction_importance.csv",
        ]
        summary = self._json(paths[0], REQUIRED_JSON[paths[0]])
        return ExplainabilityResponse(
            policy=summary["policy"],
            global_importance=self._records(
                paths[1], REQUIRED_CSV[paths[1]], sort_by=["model", "normalized_mean_abs_shap"]
            ),
            yearly_importance=self._records(
                paths[2], REQUIRED_CSV[paths[2]], sort_by=["model", "year", "feature"]
            ),
            interactions=self._records(
                paths[3], REQUIRED_CSV[paths[3]], sort_by=["model", "mean_abs_interaction"]
            ),
            model_summaries=summary["models"],
            disclosures=[
                summary["reference_model_disclosure"],
                "SHAP and permutation importance are explanatory rather than causal.",
                "Milestone 3A did not select features or retune frozen models.",
            ],
            provenance=self._provenance(paths),
        )

    def regimes(self) -> RegimesResponse:
        paths = [
            "milestone3b/summary.json",
            "milestone3b/regime_counts.csv",
            "milestone3b/rank_ic_by_regime.csv",
            "milestone3b/portfolio_metrics_by_regime.csv",
            "milestone3b/rank_ic_regime_dispersion.csv",
        ]
        summary = self._json(paths[0], REQUIRED_JSON[paths[0]])
        return RegimesResponse(
            methodology=summary["regime_methodology"],
            counts=self._records(
                paths[1], REQUIRED_CSV[paths[1]], sort_by=["regime_dimension", "regime"]
            ),
            rank_ic=self._records(
                paths[2], REQUIRED_CSV[paths[2]], sort_by=["model", "regime_dimension", "regime"]
            ),
            portfolio_metrics=self._records(
                paths[3], REQUIRED_CSV[paths[3]], sort_by=["strategy", "regime_dimension", "regime"]
            ),
            dispersion=self._records(
                paths[4], REQUIRED_CSV[paths[4]], sort_by=["model", "regime_dimension"]
            ),
            disclosures=[
                summary["policy"],
                summary["small_sample_rule"],
                summary["multiple_comparisons_warning"],
                summary["conditional_portfolio_metric_disclosure"],
            ],
            provenance=self._provenance(paths),
        )

    def optimization(self) -> OptimizationResponse:
        paths = [
            "milestone3c/summary.json",
            "milestone3c/portfolio/performance_metrics.csv",
            "milestone3c/optimizer_concentration_summary.csv",
            "milestone3c/optimizer_diagnostics.csv",
        ]
        summary = self._json(paths[0], REQUIRED_JSON[paths[0]])
        optimizer = summary["optimizer"]
        metrics = self._records(paths[1], REQUIRED_CSV[paths[1]], sort_by=["strategy"])
        optimized_metrics = [row for row in metrics if str(row["strategy"]).endswith("_optimized")]
        return OptimizationResponse(
            objective=optimizer["objective"],
            constraints=optimizer["constraints"] | {"covariance": optimizer["covariance"]},
            selected_candidates=summary["selected_optimizer_candidates"],
            historical_metrics=optimized_metrics,
            concentration=self._records(paths[2], REQUIRED_CSV[paths[2]], sort_by=["model"]),
            diagnostics=self._records(paths[3], REQUIRED_CSV[paths[3]], sort_by=["model", "date"]),
            disclosures=[
                summary["selection_policy"],
                summary["historical_evaluation_disclosure"],
                summary["constraint_disclosure"],
            ],
            provenance=self._provenance(paths),
        )

    def holdout(self) -> HoldoutResponse:
        paths = [
            "milestone3d/summary.json",
            "milestone3d/frozen_specifications.json",
            "milestone3d/holdout_rank_ic.csv",
            "milestone3d/holdout_rank_ic_summary.csv",
            "milestone3d/portfolio/performance_metrics.csv",
        ]
        summary = self._json(paths[0], REQUIRED_JSON[paths[0]])
        self._json(paths[1], REQUIRED_JSON[paths[1]])
        return HoldoutResponse(
            status=summary["status"],
            window={
                "holdout_start": summary["holdout_start"],
                "requested_as_of_exclusive": summary["requested_as_of"],
                "latest_market_date": summary["latest_market_date"],
                "activation_signal": summary["activation_signal"],
                "completed_signal_dates": summary["completed_signal_dates"],
                "latest_matured_rank_signal": summary["latest_matured_rank_signal"],
                "matured_rank_months_by_model": summary["matured_rank_months_by_model"],
                "missing_universe_tickers": summary["missing_universe_tickers_in_holdout"],
            },
            rank_ic_summary=self._records(paths[3], REQUIRED_CSV[paths[3]], sort_by=["model"]),
            rank_ic_by_month=self._records(
                paths[2], REQUIRED_CSV[paths[2]], sort_by=["model", "date"]
            ),
            portfolio_metrics=self._records(paths[4], REQUIRED_CSV[paths[4]], sort_by=["strategy"]),
            methodology=summary["methodology"],
            integrity_warning=summary["holdout_integrity_warning"],
            interpretation=(
                "All four learned models have negative mean Rank IC over the first seven matured "
                "prospective months. This is negative early evidence, not a final "
                "statistical verdict."
            ),
            provenance=self._provenance(paths),
        )

    def methodology(self) -> MethodologyResponse:
        paths = [
            "milestone1/audit/data_quality_summary.json",
            "milestone2b/summary.json",
            "milestone2c/summary.json",
            "milestone3b/summary.json",
            "milestone3c/summary.json",
            "milestone3d/summary.json",
        ]
        quality = self._json(paths[0], REQUIRED_JSON[paths[0]])
        m2b = self._json(paths[1], REQUIRED_JSON[paths[1]])
        m2c = self._json(paths[2], REQUIRED_JSON[paths[2]])
        holdout = self._json(paths[5], REQUIRED_JSON[paths[5]])
        features = sorted(
            {
                feature.removeprefix("rank__")
                for feature in [
                    *m2b["selected_rank_features"],
                    *m2c["selected_models"]["random_forest"]["rank_features"],
                ]
            }
        )
        return MethodologyResponse(
            research_question=m2c["research_question"],
            universe={
                "asset_class": "Liquid US equities",
                "tickers": quality["tickers"],
                "membership": "Static prototype universe",
                "survivorship_bias": True,
                "historical_start": quality["start_date"],
                "historical_end": quality["end_date"],
            },
            target={
                "name": "forward_return_21d",
                "definition": "P(t+21) / P(t) - 1",
                "model_transform": "Centered cross-sectional percentile rank",
                "maturity_rule": "label_end_date must be strictly before the signal date",
            },
            features=features,
            chronology_safeguards=[
                "Future-price mutation invariance test",
                "One-trading-day execution lag",
                "Self-financing weight drift between rebalances",
                "Turnover measured against drifted pre-trade weights",
                "Purged partition label boundaries",
                "Strict matured-label expanding walk-forward training",
                "True calendar month-end signals",
                "Backward-only point-in-time macro joins",
                "Trailing-history-only regime thresholds and covariance",
                "Exclusive holdout as-of market-data cutoff",
            ],
            purged_splits=[
                {
                    "name": "train",
                    "start": "2012-01-03",
                    "end": "2019-11-29",
                    "rows": 117142,
                    "purged_rows_removed": 1260,
                },
                {
                    "name": "validation",
                    "start": "2020-01-02",
                    "end": "2022-11-30",
                    "rows": 44100,
                    "purged_rows_removed": 1260,
                },
                {
                    "name": "historical_evaluation",
                    "start": "2023-01-03",
                    "end": "2025-12-01",
                    "rows": 43860,
                    "purged_rows_removed": 1260,
                },
            ],
            frozen_research_rules=[
                (
                    "Do not change features, model hyperparameters, portfolio rules, "
                    "optimizer settings, or regime definitions because of 2026 results."
                ),
                (
                    "Monthly refitting is allowed only under the predeclared expanding "
                    "matured-label rule."
                ),
                "Only fully matured 2026 labels are eligible for Rank IC.",
                (
                    "Future data extend the prospective record and do not trigger "
                    "retrospective tuning."
                ),
            ],
            data_publication_policy=[
                "The API and web export expose derived research summaries only.",
                "Raw Yahoo OHLCV is never exposed by the API or web export.",
                (
                    "Public repositories should exclude data/raw/*.parquet and "
                    "data/holdout2026/raw/*.parquet."
                ),
                holdout["holdout_integrity_warning"],
            ],
            provenance=self._provenance(paths),
        )

    def export_all(self, output_dir: Path | str) -> list[Path]:
        self.validate()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        payloads = {
            "overview": self.overview(),
            "strategies": self.strategies(),
            "performance": self.performance(),
            "rank_ic": self.rank_ic(),
            "models": self.models(),
            "shap": self.explainability(),
            "regimes": self.regimes(),
            "optimization": self.optimization(),
            "holdout": self.holdout(),
            "methodology": self.methodology(),
        }
        written: list[Path] = []
        for name, filename in WEB_FILENAMES.items():
            path = destination / filename
            temporary = path.with_suffix(path.suffix + ".tmp")
            content = (
                json.dumps(
                    payloads[name].model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + os.linesep
            )
            temporary.write_text(content, encoding="utf-8", newline="")
            temporary.replace(path)
            written.append(path)
        return written
