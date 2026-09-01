from __future__ import annotations

import pandas as pd


def add_momentum_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    close = out.groupby("ticker", sort=False)["close"]
    for window in windows:
        out[f"momentum_{window}d"] = close.pct_change(periods=window, fill_method=None)

    # Classic 12-1 style signal: ignore the most recent ~1 month.
    out["momentum_12_1"] = close.shift(21) / close.shift(252) - 1.0
    out["momentum_6_1"] = close.shift(21) / close.shift(126) - 1.0
    return out
