from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.ml.evaluation import ic_summary
from quant_platform.ml.walk_forward import select_ridge_alpha


@dataclass(frozen=True)
class AblationSelectionResult:
    selected_group: str
    selected_alpha: float
    table: pd.DataFrame
    validation_predictions: pd.DataFrame


def bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Deterministic non-parametric bootstrap interval for a sample mean."""
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(draws, tail)),
        "ci_high": float(np.quantile(draws, 1.0 - tail)),
    }


def mean_t_stat(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(pd.Series(values).dropna(), dtype=float)
    if arr.size < 2:
        return float("nan")
    std = float(arr.std(ddof=1))
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(arr.mean() / (std / math.sqrt(arr.size)))


def top_bottom_spread_by_date(
    deciles: pd.DataFrame,
    *,
    bottom_bucket: int = 1,
    top_bucket: int = 10,
) -> pd.DataFrame:
    """Compute top-minus-bottom realised forward-return spread by signal date."""
    if deciles.empty:
        return pd.DataFrame(columns=["date", "top_return", "bottom_return", "spread"])
    pivot = deciles.pivot_table(
        index="date", columns="bucket", values="mean_forward_return", aggfunc="mean"
    )
    if bottom_bucket not in pivot.columns or top_bucket not in pivot.columns:
        return pd.DataFrame(columns=["date", "top_return", "bottom_return", "spread"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(pivot.index),
            "top_return": pivot[top_bucket].to_numpy(dtype=float),
            "bottom_return": pivot[bottom_bucket].to_numpy(dtype=float),
        }
    )
    out["spread"] = out["top_return"] - out["bottom_return"]
    return out.reset_index(drop=True)


def statistical_summary(
    ic: pd.DataFrame,
    spread: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    ic_values = pd.to_numeric(ic.get("rank_ic", pd.Series(dtype=float)), errors="coerce").dropna()
    spread_values = pd.to_numeric(
        spread.get("spread", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    ic_boot = bootstrap_mean_ci(
        ic_values, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed
    )
    spread_boot = bootstrap_mean_ci(
        spread_values, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed + 1
    )
    return {
        "rank_ic": {
            **ic_summary(ic),
            "t_stat": mean_t_stat(ic_values),
            "bootstrap_ci_low": ic_boot["ci_low"],
            "bootstrap_ci_high": ic_boot["ci_high"],
            "confidence": confidence,
        },
        "top_bottom_spread": {
            "mean": float(spread_values.mean()) if len(spread_values) else float("nan"),
            "median": float(spread_values.median()) if len(spread_values) else float("nan"),
            "positive_fraction": float((spread_values > 0).mean()) if len(spread_values) else float("nan"),
            "months": int(len(spread_values)),
            "t_stat": mean_t_stat(spread_values),
            "bootstrap_ci_low": spread_boot["ci_low"],
            "bootstrap_ci_high": spread_boot["ci_high"],
            "confidence": confidence,
        },
    }


def yearly_ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    if ic.empty:
        return pd.DataFrame(
            columns=["year", "mean_rank_ic", "median_rank_ic", "positive_fraction", "months"]
        )
    frame = ic.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, object]] = []
    for year, group in frame.groupby(frame["date"].dt.year, sort=True):
        values = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        rows.append(
            {
                "year": int(year),
                "mean_rank_ic": float(values.mean()),
                "median_rank_ic": float(values.median()),
                "positive_fraction": float((values > 0).mean()),
                "months": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def select_feature_group_and_alpha(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_groups: dict[str, list[str]],
    target_column: str,
    actual_return_column: str,
    alpha_grid: list[float],
    min_training_rows: int,
    min_assets_for_ic: int,
) -> AblationSelectionResult:
    """Select feature group and alpha using validation mean Rank IC only."""
    if not feature_groups:
        raise ValueError("feature_groups cannot be empty.")

    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, float], pd.DataFrame] = {}
    for group_name, features in feature_groups.items():
        if not features:
            raise ValueError(f"Feature group '{group_name}' is empty.")
        result = select_ridge_alpha(
            train,
            validation,
            feature_columns=features,
            target_column=target_column,
            actual_return_column=actual_return_column,
            alpha_grid=alpha_grid,
            min_training_rows=min_training_rows,
            min_assets_for_ic=min_assets_for_ic,
        )
        best = result.table.loc[result.table["alpha"] == result.best_alpha].iloc[0]
        rows.append(
            {
                "feature_group": group_name,
                "n_features": int(len(features)),
                "selected_alpha": float(result.best_alpha),
                "validation_mean_rank_ic": float(best["mean_rank_ic"]),
                "validation_median_rank_ic": float(best["median_rank_ic"]),
                "validation_rank_ic_ir": float(best["rank_ic_ir"]),
                "validation_positive_rank_ic_fraction": float(best["positive_rank_ic_fraction"]),
                "validation_months": int(best["months"]),
            }
        )
        predictions[(group_name, result.best_alpha)] = result.predictions

    table = pd.DataFrame(rows).sort_values(
        ["validation_mean_rank_ic", "validation_rank_ic_ir", "feature_group"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    if table.empty or not np.isfinite(table.iloc[0]["validation_mean_rank_ic"]):
        raise ValueError("No feature group produced a valid validation Rank IC.")
    selected_group = str(table.iloc[0]["feature_group"])
    selected_alpha = float(table.iloc[0]["selected_alpha"])
    return AblationSelectionResult(
        selected_group=selected_group,
        selected_alpha=selected_alpha,
        table=table,
        validation_predictions=predictions[(selected_group, selected_alpha)],
    )
