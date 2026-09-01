import pandas as pd

from quant_platform.features.macro import merge_factors_same_day, merge_macro_asof


def test_macro_merge_matches_datetime_precision_and_carries_last_known_value():
    market_dates = pd.Series(pd.date_range("2024-01-01", periods=3)).astype("datetime64[s]")
    market = pd.DataFrame(
        {
            "date": market_dates,
            "ticker": ["AAA"] * 3,
            "close": [100.0, 101.0, 102.0],
        }
    )
    macro_index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3)).astype(
        "datetime64[us]"
    )
    macro = pd.DataFrame({"treasury_10y": [4.0, None, 4.2]}, index=macro_index)
    macro.index.name = "date"

    merged = merge_macro_asof(market, macro)
    assert str(merged["date"].dtype) == "datetime64[ns]"
    assert merged.loc[1, "treasury_10y"] == 4.0
    assert merged.loc[2, "treasury_10y"] == 4.2


def test_factor_merge_matches_datetime_precision():
    market = pd.DataFrame(
        {
            "date": pd.Series(pd.date_range("2024-01-01", periods=2)).astype("datetime64[s]"),
            "ticker": ["AAA", "AAA"],
        }
    )
    factors = pd.DataFrame(
        {"mkt_rf": [0.01, 0.02]},
        index=pd.DatetimeIndex(pd.date_range("2024-01-01", periods=2)).astype("datetime64[us]"),
    )
    factors.index.name = "date"
    merged = merge_factors_same_day(market, factors)
    assert list(merged["mkt_rf"]) == [0.01, 0.02]
