from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def rank_feature_name(feature: str) -> str:
    return f"rank__{feature}"


def add_label_end_dates(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Attach the exact future trading date used to finish each forward label."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    required = {"date", "ticker"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {sorted(missing)}")

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    out = out.sort_values(["ticker", "date"])
    out["label_end_date"] = out.groupby("ticker", sort=False)["date"].shift(-horizon)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def global_month_end_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """Return actual last trading dates of each calendar month in the full panel."""
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"]).dropna().unique()).sort_values()
    if dates.empty:
        return dates
    frame = pd.DataFrame({"date": dates})
    month_ends = frame.groupby(frame["date"].dt.to_period("M"))["date"].max()
    return pd.DatetimeIndex(month_ends.to_numpy(dtype="datetime64[ns]"))


def _rank_centered(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True) - 0.5


def build_monthly_ml_partition(
    partition: pd.DataFrame,
    *,
    true_month_ends: Iterable[pd.Timestamp],
    feature_columns: list[str],
    target_column: str,
    min_assets: int = 20,
) -> pd.DataFrame:
    """Create monthly, cross-sectionally ranked ML observations.

    Only true month-end trading dates from the *full* research calendar are
    eligible. This avoids treating an artificial purged-split cutoff as a month
    end. Features and the target are converted to centered percentile ranks so
    extreme corporate-action/data observations have bounded influence on Ridge.
    """
    if min_assets < 2:
        raise ValueError("min_assets must be at least 2.")
    required = {
        "date",
        "ticker",
        "close",
        "label_end_date",
        target_column,
        *feature_columns,
    }
    missing = required - set(partition.columns)
    if missing:
        raise ValueError(f"Partition missing columns: {sorted(missing)}")

    work = partition.copy()
    work["date"] = pd.to_datetime(work["date"]).astype("datetime64[ns]")
    work["label_end_date"] = pd.to_datetime(work["label_end_date"]).astype("datetime64[ns]")
    month_ends = pd.DatetimeIndex(pd.to_datetime(list(true_month_ends))).astype("datetime64[ns]")
    work = work.loc[work["date"].isin(month_ends)].copy()

    rows: list[pd.DataFrame] = []
    required_values = ["close", target_column, *feature_columns]
    for date, snapshot in work.groupby("date", sort=True):
        eligible = snapshot.dropna(subset=required_values).drop_duplicates("ticker").copy()
        finite_mask = np.isfinite(
            eligible[[target_column, *feature_columns]].apply(pd.to_numeric, errors="coerce")
        ).all(axis=1)
        eligible = eligible.loc[finite_mask]
        if len(eligible) < min_assets:
            continue

        for feature in feature_columns:
            eligible[rank_feature_name(feature)] = _rank_centered(eligible[feature])
        eligible["target_rank"] = _rank_centered(eligible[target_column])
        eligible["cross_section_size"] = int(len(eligible))
        rows.append(eligible)

    rank_columns = [rank_feature_name(feature) for feature in feature_columns]
    keep = [
        "date",
        "ticker",
        "label_end_date",
        "cross_section_size",
        target_column,
        "target_rank",
        *feature_columns,
        *rank_columns,
    ]
    if not rows:
        return pd.DataFrame(columns=keep)
    result = pd.concat(rows, ignore_index=True)
    return result[keep].sort_values(["date", "ticker"]).reset_index(drop=True)
