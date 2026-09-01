from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_target_weights(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    top_fraction: float = 0.20,
    min_assets: int = 10,
) -> pd.DataFrame:
    """Convert monthly prediction rankings into equal-weight long-only targets."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    rows: list[dict[str, object]] = []
    for date, snapshot in predictions.groupby("date", sort=True):
        ranked = snapshot.dropna(subset=[prediction_column]).drop_duplicates("ticker")
        if len(ranked) < min_assets:
            continue
        n = max(1, int(np.ceil(len(ranked) * top_fraction)))
        selected = ranked.nlargest(n, prediction_column)["ticker"]
        for ticker in selected:
            rows.append(
                {"date": pd.Timestamp(date), "ticker": ticker, "target_weight": 1.0 / n}
            )
    return pd.DataFrame(rows, columns=["date", "ticker", "target_weight"])


def targets_on_dates(
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    strategy: str,
    score_column: str = "momentum_12_1",
    top_fraction: float = 0.20,
    min_assets: int = 10,
) -> pd.DataFrame:
    """Build benchmark targets on exactly the same signal dates as an ML strategy."""
    signal_dates = pd.DatetimeIndex(pd.to_datetime(dates)).astype("datetime64[ns]")
    rows: list[dict[str, object]] = []
    for date in signal_dates:
        snapshot = panel.loc[pd.to_datetime(panel["date"]) == date].copy()
        if strategy == "equal_weight":
            eligible = snapshot.dropna(subset=["close"]).drop_duplicates("ticker")
            if len(eligible) < min_assets:
                continue
            selected = eligible["ticker"]
        elif strategy == "momentum":
            eligible = snapshot.dropna(subset=["close", score_column]).drop_duplicates("ticker")
            if len(eligible) < min_assets:
                continue
            n = max(1, int(np.ceil(len(eligible) * top_fraction)))
            selected = eligible.nlargest(n, score_column)["ticker"]
        else:
            raise ValueError(f"Unknown benchmark strategy: {strategy}")

        weight = 1.0 / len(selected)
        for ticker in selected:
            rows.append({"date": date, "ticker": ticker, "target_weight": weight})
    return pd.DataFrame(rows, columns=["date", "ticker", "target_weight"])


def buffered_prediction_target_weights(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    top_fraction: float = 0.20,
    exit_fraction: float = 0.30,
    min_assets: int = 10,
) -> pd.DataFrame:
    """Equal-weight top portfolio with a rank buffer to reduce boundary churn.

    Existing holdings remain eligible while they stay inside ``exit_fraction``;
    vacancies are filled with the highest-ranked names. The target holding count
    remains approximately ``top_fraction`` of the cross-section.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    if not top_fraction <= exit_fraction <= 1:
        raise ValueError("exit_fraction must be >= top_fraction and <= 1.")
    rows: list[dict[str, object]] = []
    previous: set[str] = set()
    for date, snapshot in predictions.groupby("date", sort=True):
        ranked = (
            snapshot.dropna(subset=[prediction_column])
            .drop_duplicates("ticker")
            .sort_values(prediction_column, ascending=False)
        )
        if len(ranked) < min_assets:
            continue
        target_n = max(1, int(np.ceil(len(ranked) * top_fraction)))
        exit_n = max(target_n, int(np.ceil(len(ranked) * exit_fraction)))
        exit_names = set(ranked.head(exit_n)["ticker"].astype(str))
        retained = [t for t in ranked["ticker"].astype(str) if t in previous and t in exit_names]
        selected = retained[:target_n]
        for ticker in ranked["ticker"].astype(str):
            if len(selected) >= target_n:
                break
            if ticker not in selected:
                selected.append(ticker)
        previous = set(selected)
        weight = 1.0 / len(selected)
        for ticker in selected:
            rows.append({"date": pd.Timestamp(date), "ticker": ticker, "target_weight": weight})
    return pd.DataFrame(rows, columns=["date", "ticker", "target_weight"])


def rank_weighted_prediction_target_weights(
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    top_fraction: float = 0.20,
    min_assets: int = 10,
) -> pd.DataFrame:
    """Long-only top-ranked portfolio with deterministic linear rank weights."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    rows: list[dict[str, object]] = []
    for date, snapshot in predictions.groupby("date", sort=True):
        ranked = (
            snapshot.dropna(subset=[prediction_column])
            .drop_duplicates("ticker")
            .sort_values(prediction_column, ascending=False)
        )
        if len(ranked) < min_assets:
            continue
        n = max(1, int(np.ceil(len(ranked) * top_fraction)))
        selected = ranked.head(n).copy()
        raw = np.arange(n, 0, -1, dtype=float)
        weights = raw / raw.sum()
        for ticker, weight in zip(selected["ticker"].astype(str), weights, strict=True):
            rows.append(
                {"date": pd.Timestamp(date), "ticker": ticker, "target_weight": float(weight)}
            )
    return pd.DataFrame(rows, columns=["date", "ticker", "target_weight"])
