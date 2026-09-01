import pandas as pd
import pytest

from quant_platform.data.validation import validate_market_data


def test_duplicate_rows_rejected():
    row = {
        "date": pd.Timestamp("2024-01-02"),
        "ticker": "AAA",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1000,
    }
    df = pd.DataFrame([row, row])
    with pytest.raises(ValueError, match="duplicate"):
        validate_market_data(df)


def test_machine_precision_ohlc_rounding_is_accepted():
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "AAA",
                "open": 60.0,
                "high": 61.102081,
                "low": 59.5,
                "close": 61.102081 + 7e-15,
                "volume": 1000,
            }
        ]
    )
    result = validate_market_data(df)
    assert result.rows == 1


def test_genuine_ohlc_range_violation_is_rejected():
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "ticker": "AAA",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 110.0,
                "volume": 1000,
            }
        ]
    )
    with pytest.raises(ValueError, match="genuinely inconsistent"):
        validate_market_data(df)
