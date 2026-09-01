from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.ml.dataset import rank_feature_name


@dataclass(frozen=True)
class FrozenHoldoutSpec:
    ridge_features: list[str]
    ridge_alpha: float
    ridge_portfolio_variant: dict[str, Any]
    nonlinear_models: dict[str, dict[str, Any]]
    optimizer_candidates: dict[str, dict[str, Any]]
    primary_optimizer_alpha_model: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_holdout_spec(
    m2b_summary_path: Path,
    m2c_summary_path: Path,
    m3c_summary_path: Path,
) -> FrozenHoldoutSpec:
    """Load already-selected research choices without performing any new selection."""
    m2b = load_json(m2b_summary_path)
    m2c = load_json(m2c_summary_path)
    m3c = load_json(m3c_summary_path)
    if str(m2b.get("milestone")) != "2B":
        raise ValueError("Expected a Milestone 2B summary.")
    if str(m2c.get("milestone")) != "2C":
        raise ValueError("Expected a Milestone 2C summary.")
    if str(m3c.get("milestone")) != "3C":
        raise ValueError("Expected a Milestone 3C summary.")

    nonlinear: dict[str, dict[str, Any]] = {}
    for model_name in ("random_forest", "xgboost", "lightgbm"):
        raw = m2c["selected_models"][model_name]
        nonlinear[model_name] = {
            "feature_group": str(raw["feature_group"]),
            "rank_features": [str(x) for x in raw["rank_features"]],
            "candidate": str(raw["candidate"]),
            "params": dict(raw["params"]),
            "portfolio_variant": dict(raw["portfolio_variant"]),
        }

    return FrozenHoldoutSpec(
        ridge_features=[str(x) for x in m2b["selected_rank_features"]],
        ridge_alpha=float(m2b["selected_alpha"]),
        ridge_portfolio_variant=dict(m2b["selected_portfolio_variant"]),
        nonlinear_models=nonlinear,
        optimizer_candidates={
            str(name): dict(value)
            for name, value in m3c["selected_optimizer_candidates"].items()
        },
        primary_optimizer_alpha_model=str(m3c["primary_validation_selected_alpha_model"]),
    )



def truncate_market_before_as_of(
    market: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Apply a deterministic end-of-day-safe holdout cutoff.

    ``--as-of`` is treated as an exclusive information cutoff: only completed
    daily bars strictly before the supplied date are eligible. This prevents an
    in-progress Yahoo daily bar on the as-of date from entering validation,
    features, or holdout performance, and makes reruns reproducible regardless
    of the wall-clock time at which they are executed.
    """
    frame = market.copy()
    if "date" not in frame.columns:
        raise ValueError("Market data missing date column.")
    cutoff = pd.Timestamp(as_of_date).tz_localize(None).normalize()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    frame = frame.loc[frame["date"] < cutoff].copy()
    if frame.empty:
        raise ValueError(
            f"No completed market observations are available strictly before as-of {cutoff.date()}."
        )
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)

def combine_incremental_market(
    historical: pd.DataFrame,
    incremental: pd.DataFrame,
    *,
    holdout_start: str | pd.Timestamp,
) -> pd.DataFrame:
    """Append holdout data without permitting revisions to the frozen history."""
    cutoff = pd.Timestamp(holdout_start)
    old = historical.copy()
    new = incremental.copy()
    for frame in (old, new):
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).astype("datetime64[ns]")
        frame["ticker"] = frame["ticker"].astype(str)
    old = old.loc[old["date"] < cutoff].copy()
    new = new.loc[new["date"] >= cutoff].copy()
    if new.empty:
        raise ValueError("Incremental holdout market data contains no observations on/after holdout_start.")
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).drop_duplicates(
        ["date", "ticker"], keep="last"
    )
    if combined.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker rows remain after incremental merge.")
    return combined.reset_index(drop=True)


def completed_month_end_dates(
    panel: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return last trading dates only for calendar months that are fully complete.

    The current calendar month is deliberately excluded even if the latest downloaded
    observation happens to be the last observation available today. This prevents a
    partial month such as 1-Sep-2026 from being mislabeled as September month-end.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"]).dropna().unique()).sort_values()
    if dates.empty:
        return dates
    current_period = pd.Timestamp(as_of_date).to_period("M")
    frame = pd.DataFrame({"date": dates})
    frame["period"] = frame["date"].dt.to_period("M")
    frame = frame.loc[frame["period"] < current_period]
    if frame.empty:
        return pd.DatetimeIndex([])
    ends = frame.groupby("period", sort=True)["date"].max()
    return pd.DatetimeIndex(ends.to_numpy(dtype="datetime64[ns]"))


def build_monthly_online_partition(
    panel: pd.DataFrame,
    *,
    month_ends: pd.DatetimeIndex,
    feature_columns: list[str],
    target_column: str,
    min_assets: int,
) -> pd.DataFrame:
    """Build monthly ranked features while retaining not-yet-matured holdout labels.

    Unlike the historical ML partition builder, a missing future target does not remove
    a row: the row can still be predicted in real time. ``target_rank`` is populated
    only where the target is observable.
    """
    required = {"date", "ticker", "label_end_date", target_column, *feature_columns}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {sorted(missing)}")
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).astype("datetime64[ns]")
    work["label_end_date"] = pd.to_datetime(work["label_end_date"]).astype("datetime64[ns]")
    work = work.loc[work["date"].isin(pd.DatetimeIndex(month_ends))].copy()

    rank_names = [rank_feature_name(x) for x in feature_columns]
    output: list[pd.DataFrame] = []
    for date, snapshot in work.groupby("date", sort=True):
        eligible = snapshot.dropna(subset=feature_columns).drop_duplicates("ticker").copy()
        numeric = eligible[feature_columns].apply(pd.to_numeric, errors="coerce")
        eligible = eligible.loc[np.isfinite(numeric).all(axis=1)].copy()
        if len(eligible) < int(min_assets):
            continue
        for feature in feature_columns:
            eligible[rank_feature_name(feature)] = (
                pd.to_numeric(eligible[feature], errors="coerce").rank(method="average", pct=True)
                - 0.5
            )
        eligible["target_rank"] = np.nan
        target_mask = pd.to_numeric(eligible[target_column], errors="coerce").notna()
        if int(target_mask.sum()) >= int(min_assets):
            eligible.loc[target_mask, "target_rank"] = (
                pd.to_numeric(eligible.loc[target_mask, target_column], errors="coerce")
                .rank(method="average", pct=True)
                - 0.5
            )
        eligible["cross_section_size"] = int(len(eligible))
        output.append(eligible)

    keep = [
        "date",
        "ticker",
        "label_end_date",
        "cross_section_size",
        target_column,
        "target_rank",
        *feature_columns,
        *rank_names,
    ]
    if not output:
        return pd.DataFrame(columns=keep)
    return (
        pd.concat(output, ignore_index=True)[keep]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def matured_holdout_predictions(
    predictions: pd.DataFrame,
    *,
    holdout_start: str | pd.Timestamp,
    target_column: str = "forward_return_21d",
) -> pd.DataFrame:
    """Return only genuine holdout rows whose forward label is observable."""
    start = pd.Timestamp(holdout_start)
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    if target_column not in frame.columns:
        raise ValueError(f"Predictions missing target column: {target_column}")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    return frame.loc[(frame["date"] >= start) & target.notna()].copy()


def frozen_spec_to_dict(spec: FrozenHoldoutSpec) -> dict[str, Any]:
    return {
        "ridge": {
            "rank_features": spec.ridge_features,
            "alpha": spec.ridge_alpha,
            "portfolio_variant": spec.ridge_portfolio_variant,
        },
        "nonlinear_models": spec.nonlinear_models,
        "optimizer_candidates": spec.optimizer_candidates,
        "primary_optimizer_alpha_model": spec.primary_optimizer_alpha_model,
    }
