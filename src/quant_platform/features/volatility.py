from __future__ import annotations

import numpy as np
import pandas as pd


def add_volatility_features(
    df: pd.DataFrame,
    windows: list[int],
    annualization_days: int = 252,
) -> pd.DataFrame:
    out = df.copy()
    if "return_1d" not in out:
        out["return_1d"] = out.groupby("ticker", sort=False)["close"].pct_change(fill_method=None)

    grouped = out.groupby("ticker", sort=False)["return_1d"]
    for window in windows:
        out[f"volatility_{window}d"] = grouped.transform(
            lambda s: s.rolling(window, min_periods=window).std(ddof=1)
        ) * np.sqrt(annualization_days)
    return out
