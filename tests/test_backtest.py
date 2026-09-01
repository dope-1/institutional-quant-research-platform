import pandas as pd
import pytest

from quant_platform.backtest.engine import run_target_weight_backtest


def test_signal_weights_are_lagged_one_day():
    dates = pd.bdate_range("2024-01-01", periods=4)
    market = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * 4,
            "return_1d": [0.0, 0.10, 0.20, 0.30],
        }
    )
    targets = pd.DataFrame(
        {"date": [dates[0]], "ticker": ["AAA"], "target_weight": [1.0]}
    )
    result = run_target_weight_backtest(market, targets, 0.0, 0.0)
    # Day 0 signal is not active on day 0; it becomes active on day 1.
    assert result.daily.loc[dates[0], "gross_return"] == 0.0
    assert result.daily.loc[dates[1], "gross_return"] == 0.10


def test_transaction_cost_is_applied_after_rebalance():
    dates = pd.bdate_range("2024-01-01", periods=3)
    market = pd.DataFrame(
        {"date": dates, "ticker": ["AAA"] * 3, "return_1d": [0.0, 0.0, 0.0]}
    )
    targets = pd.DataFrame(
        {"date": [dates[0]], "ticker": ["AAA"], "target_weight": [1.0]}
    )
    result = run_target_weight_backtest(market, targets, 10.0, 0.0)
    assert result.daily.loc[dates[1], "trading_cost"] == 0.001


def test_weights_drift_between_rebalances():
    dates = pd.bdate_range("2024-01-01", periods=3)
    market = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "ticker": ["AAA", "BBB"] * 3,
            "return_1d": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    targets = pd.DataFrame(
        {
            "date": [dates[0], dates[0]],
            "ticker": ["AAA", "BBB"],
            "target_weight": [0.5, 0.5],
        }
    )
    result = run_target_weight_backtest(market, targets, 0.0, 0.0)
    assert result.weights.loc[dates[1], "AAA"] == 0.5
    assert result.weights.loc[dates[2], "AAA"] == pytest.approx(2.0 / 3.0)
    assert result.weights.loc[dates[2], "BBB"] == pytest.approx(1.0 / 3.0)


def test_rebalance_turnover_uses_drifted_pretrade_weights():
    dates = pd.bdate_range("2024-01-01", periods=3)
    market = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "ticker": ["AAA", "BBB"] * 3,
            "return_1d": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    targets = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "target_weight": [0.5, 0.5, 0.5, 0.5],
        }
    )
    result = run_target_weight_backtest(market, targets, 0.0, 0.0)
    assert result.daily.loc[dates[2], "turnover"] == pytest.approx(1.0 / 3.0)
