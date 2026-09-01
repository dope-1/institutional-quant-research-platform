import numpy as np
import pandas as pd

from quant_platform.data.preprocessing import clean_market_data
from quant_platform.features.liquidity import add_liquidity_features
from quant_platform.features.momentum import add_momentum_features
from quant_platform.features.volatility import add_volatility_features


def sample_panel(n=320):
    dates = pd.bdate_range("2020-01-01", periods=n)
    frames = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        close = 100 + i * 10 + np.arange(n) * 0.05
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000 + i,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_feature_columns_exist():
    df = clean_market_data(sample_panel())
    df = add_momentum_features(df, [21, 63, 126, 252])
    df = add_volatility_features(df, [21, 63])
    df = add_liquidity_features(df, 21)
    expected = {"momentum_12_1", "volatility_21d", "amihud_21d", "dollar_volume"}
    assert expected.issubset(df.columns)
