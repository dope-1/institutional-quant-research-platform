import numpy as np
import pandas as pd
import pandas.testing as pdt

from quant_platform.data.preprocessing import clean_market_data
from quant_platform.features.liquidity import add_liquidity_features
from quant_platform.features.momentum import add_momentum_features
from quant_platform.features.volatility import add_volatility_features


def make_data(n=360):
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.cumprod(1 + np.linspace(-0.001, 0.0015, n))
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAA",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 2_000_000,
        }
    )


def build_features(df):
    out = clean_market_data(df)
    out = add_momentum_features(out, [21, 63, 126, 252])
    out = add_volatility_features(out, [21, 63])
    out = add_liquidity_features(out, 21)
    return out


def test_future_price_mutation_does_not_change_past_features():
    original = make_data()
    cutoff = original.loc[280, "date"]
    mutated = original.copy()
    mutated.loc[mutated["date"] > cutoff, "close"] *= 7.0
    mutated.loc[mutated["date"] > cutoff, "open"] *= 7.0
    mutated.loc[mutated["date"] > cutoff, "high"] *= 7.0
    mutated.loc[mutated["date"] > cutoff, "low"] *= 7.0

    a = build_features(original)
    b = build_features(mutated)
    cols = [
        "momentum_21d",
        "momentum_63d",
        "momentum_12_1",
        "volatility_21d",
        "dollar_volume_median_21d",
        "amihud_21d",
    ]
    pdt.assert_frame_equal(
        a.loc[a["date"] <= cutoff, cols].reset_index(drop=True),
        b.loc[b["date"] <= cutoff, cols].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
