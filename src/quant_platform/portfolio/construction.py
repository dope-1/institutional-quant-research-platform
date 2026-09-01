from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_rebalance_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(df["date"].unique())))
    if dates.empty:
        return dates
    return pd.Series(dates, index=dates).groupby(dates.to_period("M")).max().array


def equal_weight_targets(snapshot: pd.DataFrame, min_assets: int = 10) -> pd.Series:
    eligible = snapshot.loc[snapshot["close"].notna(), "ticker"].drop_duplicates()
    if len(eligible) < min_assets:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(eligible), index=eligible, dtype=float)


def top_fraction_long_only_targets(
    snapshot: pd.DataFrame,
    score_col: str,
    top_fraction: float = 0.20,
    min_assets: int = 10,
) -> pd.Series:
    ranked = snapshot.dropna(subset=[score_col, "close"]).drop_duplicates("ticker")
    if len(ranked) < min_assets:
        return pd.Series(dtype=float)
    n = max(1, int(np.ceil(len(ranked) * top_fraction)))
    selected = ranked.nlargest(n, score_col)["ticker"]
    return pd.Series(1.0 / n, index=selected, dtype=float)


def build_target_weight_table(
    df: pd.DataFrame,
    strategy: str,
    score_col: str = "momentum_12_1",
    top_fraction: float = 0.20,
    min_assets: int = 10,
) -> pd.DataFrame:
    dates = monthly_rebalance_dates(df)
    rows: list[dict[str, object]] = []
    for date in dates:
        snapshot = df.loc[df["date"] == date]
        if strategy == "equal_weight":
            weights = equal_weight_targets(snapshot, min_assets=min_assets)
        elif strategy == "momentum":
            weights = top_fraction_long_only_targets(
                snapshot,
                score_col=score_col,
                top_fraction=top_fraction,
                min_assets=min_assets,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        for ticker, weight in weights.items():
            rows.append({"date": pd.Timestamp(date), "ticker": ticker, "target_weight": weight})
    return pd.DataFrame(rows, columns=["date", "ticker", "target_weight"])
