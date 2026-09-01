from __future__ import annotations

import pandas as pd


def _normalise_dates(series: pd.Series) -> pd.Series:
    """Normalise merge keys to timezone-naive datetime64[ns]."""
    converted = pd.to_datetime(series)
    if getattr(converted.dt, "tz", None) is not None:
        converted = converted.dt.tz_localize(None)
    return converted.astype("datetime64[ns]")


def merge_macro_asof(market: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Merge the latest *available* macro observation without looking forward.

    Some FRED daily series contain explicit NaNs on equity trading days when the
    underlying Treasury/bond market is closed. Forward-filling the macro table
    before a backward as-of merge carries only the last known observation and
    never imports a future value.
    """
    left = market.copy()
    left["date"] = _normalise_dates(left["date"])
    left = left.sort_values("date")

    right = macro.copy().sort_index()
    right = right.apply(pd.to_numeric, errors="coerce").ffill()
    right = right.reset_index()
    right["date"] = _normalise_dates(right["date"])
    right = right.sort_values("date")

    return pd.merge_asof(left, right, on="date", direction="backward")


def merge_factors_same_day(market: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Merge same-day factor returns using an explicitly matched datetime dtype."""
    left = market.copy()
    left["date"] = _normalise_dates(left["date"])

    right = factors.copy().reset_index()
    right["date"] = _normalise_dates(right["date"])
    return left.merge(right, on="date", how="left", validate="many_to_one")
