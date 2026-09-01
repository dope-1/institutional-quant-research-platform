from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class ExplainabilityResult:
    shap_values: pd.DataFrame
    shap_importance: pd.DataFrame
    permutation_importance: pd.DataFrame
    yearly_shap_importance: pd.DataFrame
    yearly_permutation_importance: pd.DataFrame
    interaction_importance: pd.DataFrame


def _require_shap():
    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'Milestone 3A requires SHAP. Re-run pip install -e ".[dev,ml]" after the upgrade.'
        ) from exc
    return shap


def spearman_rank_scorer(estimator, x, y) -> float:
    pred = np.asarray(estimator.predict(x), dtype=float)
    actual = np.asarray(y, dtype=float)
    if len(pred) < 3 or np.nanstd(pred) == 0 or np.nanstd(actual) == 0:
        return 0.0
    value = pd.Series(pred).corr(pd.Series(actual), method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def tree_shap_values(model, x: pd.DataFrame) -> np.ndarray:
    shap = _require_shap()
    explainer = shap.TreeExplainer(model)
    explanation = explainer(x)
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape != x.shape:
        raise ValueError(f"Unexpected SHAP value shape {values.shape}; expected {x.shape}.")
    return values


def shap_long_table(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    values: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    if values.shape != (len(frame), len(feature_columns)):
        raise ValueError("SHAP matrix does not match rows/features.")
    identity = frame[[c for c in ["date", "ticker", "target_rank", "forward_return_21d"] if c in frame]].copy()
    identity["row_id"] = np.arange(len(identity))
    value_df = pd.DataFrame(values, columns=feature_columns)
    feature_df = frame[feature_columns].reset_index(drop=True)
    rows = []
    for feature in feature_columns:
        part = identity.copy()
        part["model"] = model_name
        part["feature"] = feature
        part["feature_value"] = feature_df[feature].to_numpy()
        part["shap_value"] = value_df[feature].to_numpy()
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize_shap_importance(shap_long: pd.DataFrame) -> pd.DataFrame:
    result = (
        shap_long.groupby(["model", "feature"], as_index=False)
        .agg(
            mean_abs_shap=("shap_value", lambda s: float(np.mean(np.abs(s)))),
            mean_shap=("shap_value", "mean"),
            shap_std=("shap_value", "std"),
            positive_fraction=("shap_value", lambda s: float((s > 0).mean())),
            observations=("shap_value", "size"),
        )
    )
    result["normalized_mean_abs_shap"] = result.groupby("model")["mean_abs_shap"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0.0
    )
    return result.sort_values(["model", "mean_abs_shap"], ascending=[True, False]).reset_index(drop=True)


def summarize_yearly_shap_importance(shap_long: pd.DataFrame) -> pd.DataFrame:
    work = shap_long.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["year"] = work["date"].dt.year
    result = (
        work.groupby(["model", "year", "feature"], as_index=False)
        .agg(mean_abs_shap=("shap_value", lambda s: float(np.mean(np.abs(s)))), observations=("shap_value", "size"))
    )
    result["normalized_importance"] = result.groupby(["model", "year"])["mean_abs_shap"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0.0
    )
    return result


def permutation_importance_table(
    model,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    work = frame.dropna(subset=[target_column, *feature_columns]).copy()
    result = permutation_importance(
        model,
        work[feature_columns],
        work[target_column],
        scoring=spearman_rank_scorer,
        n_repeats=int(repeats),
        random_state=int(seed),
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def yearly_permutation_importance(
    model,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    rows = []
    for year, part in work.groupby(work["date"].dt.year):
        table = permutation_importance_table(
            model,
            part,
            feature_columns=feature_columns,
            target_column=target_column,
            model_name=model_name,
            repeats=repeats,
            seed=seed + int(year),
        )
        table["year"] = int(year)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def shap_interaction_importance(
    model,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_name: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    shap = _require_shap()
    x = frame[feature_columns].dropna().copy()
    if len(x) > int(max_rows):
        x = x.sample(int(max_rows), random_state=int(seed))
    explainer = shap.TreeExplainer(model)
    try:
        values = np.asarray(explainer.shap_interaction_values(x), dtype=float)
    except Exception as exc:  # pragma: no cover - library/model dependent fallback
        return pd.DataFrame(
            [{"model": model_name, "feature_a": "UNAVAILABLE", "feature_b": type(exc).__name__, "mean_abs_interaction": np.nan}]
        )
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3:
        return pd.DataFrame(
            [{"model": model_name, "feature_a": "UNAVAILABLE", "feature_b": f"shape={values.shape}", "mean_abs_interaction": np.nan}]
        )
    rows = []
    for i, feature_a in enumerate(feature_columns):
        for j in range(i + 1, len(feature_columns)):
            rows.append(
                {
                    "model": model_name,
                    "feature_a": feature_a,
                    "feature_b": feature_columns[j],
                    "mean_abs_interaction": float(np.mean(np.abs(values[:, i, j]))),
                }
            )
    return pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False, ignore_index=True)


def interaction_grid(
    explanation_wide: pd.DataFrame,
    *,
    feature_x: str,
    feature_y: str,
    shap_x: str,
) -> pd.DataFrame:
    work = explanation_wide.dropna(subset=[feature_x, feature_y, shap_x]).copy()
    bins = [-0.500001, -0.25, 0.0, 0.25, 0.500001]
    labels = ["Q1_low", "Q2", "Q3", "Q4_high"]
    work["x_bin"] = pd.cut(work[feature_x], bins=bins, labels=labels, include_lowest=True)
    work["y_bin"] = pd.cut(work[feature_y], bins=bins, labels=labels, include_lowest=True)
    agg: dict[str, tuple[str, str]] = {
        "mean_shap_x": (shap_x, "mean"),
        "observations": (shap_x, "size"),
    }
    if "prediction" in work:
        agg["mean_prediction"] = ("prediction", "mean")
    if "target_rank" in work:
        agg["mean_target_rank"] = ("target_rank", "mean")
    if "forward_return_21d" in work:
        agg["mean_forward_return_21d"] = ("forward_return_21d", "mean")
    return work.groupby(["x_bin", "y_bin"], observed=True).agg(**agg).reset_index()
