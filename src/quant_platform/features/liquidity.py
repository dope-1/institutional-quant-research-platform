from __future__ import annotations

import numpy as np
import pandas as pd


def add_liquidity_features(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    out = df.copy()
    out["dollar_volume"] = out["close"] * out["volume"]
    out["log_dollar_volume"] = np.log1p(out["dollar_volume"])

    out[f"dollar_volume_median_{window}d"] = out.groupby("ticker", sort=False)[
        "dollar_volume"
    ].transform(lambda s: s.rolling(window, min_periods=window).median())

    # Amihud-like illiquidity proxy: |return| / dollar volume in millions.
    denom = out["dollar_volume"].replace(0, np.nan) / 1_000_000.0
    out["amihud_raw"] = out["return_1d"].abs() / denom
    out[f"amihud_{window}d"] = out.groupby("ticker", sort=False)["amihud_raw"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    return out
