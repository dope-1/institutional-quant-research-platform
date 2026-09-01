from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _spearman_without_scipy(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    xr = pair["x"].rank(method="average")
    yr = pair["y"].rank(method="average")
    return float(xr.corr(yr, method="pearson"))


def rank_ic_by_date(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    actual_column: str = "forward_return_21d",
    min_assets: int = 10,
) -> pd.DataFrame:
    """Calculate cross-sectional Spearman Rank IC for each signal date."""
    required = {"date", prediction_column, actual_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for date, snapshot in predictions.groupby("date", sort=True):
        usable = snapshot[[prediction_column, actual_column]].dropna()
        if len(usable) < min_assets:
            continue
        rows.append(
            {
                "date": pd.Timestamp(date),
                "rank_ic": _spearman_without_scipy(
                    usable[prediction_column], usable[actual_column]
                ),
                "n_assets": int(len(usable)),
            }
        )
    return pd.DataFrame(rows, columns=["date", "rank_ic", "n_assets"])


def ic_summary(ic: pd.DataFrame, periods_per_year: int = 12) -> dict[str, float]:
    values = pd.to_numeric(ic.get("rank_ic", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return {
            "mean_rank_ic": float("nan"),
            "median_rank_ic": float("nan"),
            "rank_ic_std": float("nan"),
            "rank_ic_ir": float("nan"),
            "positive_rank_ic_fraction": float("nan"),
            "months": 0.0,
        }
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    mean = float(values.mean())
    ir = (
        mean / std * math.sqrt(periods_per_year)
        if np.isfinite(std) and std > 0
        else float("nan")
    )
    return {
        "mean_rank_ic": mean,
        "median_rank_ic": float(values.median()),
        "rank_ic_std": std,
        "rank_ic_ir": float(ir),
        "positive_rank_ic_fraction": float((values > 0).mean()),
        "months": float(len(values)),
    }


def decile_forward_returns(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    actual_column: str = "forward_return_21d",
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Return mean realised forward return by prediction bucket and month."""
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2.")
    rows: list[dict[str, object]] = []
    for date, snapshot in predictions.groupby("date", sort=True):
        usable = snapshot[[prediction_column, actual_column]].dropna().copy()
        if len(usable) < n_buckets:
            continue
        order = usable[prediction_column].rank(method="first", pct=True)
        bucket = np.ceil(order * n_buckets).astype(int).clip(1, n_buckets)
        usable["bucket"] = bucket
        for bucket_id, group in usable.groupby("bucket", sort=True):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "bucket": int(bucket_id),
                    "mean_forward_return": float(group[actual_column].mean()),
                    "n_assets": int(len(group)),
                }
            )
    return pd.DataFrame(
        rows, columns=["date", "bucket", "mean_forward_return", "n_assets"]
    )
