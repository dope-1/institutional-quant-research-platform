from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.ml.regime import (
    build_point_in_time_regimes,
    daily_macro_panel,
    holding_period_returns,
    metric_summary_by_regime,
    portfolio_summary_by_regime,
    regimes_on_signal_dates,
)


def _daily(n: int = 700) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "date": dates,
            "vix": 12 + 20 * x,
            "treasury_10y": 1.0 + 3.0 * x,
            "treasury_2y": 0.8 + 3.2 * x,
            "fed_funds": 0.25 + 4.0 * x,
            "mkt_rf": 0.0005 + 0.0002 * np.sin(np.arange(n) / 10),
        }
    )


def test_quantile_regime_is_point_in_time_and_future_changes_do_not_rewrite_history():
    daily = _daily()
    first = build_point_in_time_regimes(daily, quantile_lookback_days=300, quantile_min_history_days=100)
    changed = daily.copy()
    changed.loc[changed.index > 500, "vix"] = 500.0
    second = build_point_in_time_regimes(changed, quantile_lookback_days=300, quantile_min_history_days=100)
    pd.testing.assert_series_equal(
        first.loc[:500, "vix_regime"].reset_index(drop=True),
        second.loc[:500, "vix_regime"].reset_index(drop=True),
    )


def test_rate_direction_curve_and_market_trend_labels_are_created():
    regimes = build_point_in_time_regimes(_daily(), quantile_lookback_days=300, quantile_min_history_days=100)
    assert {"rising", "falling", "flat"} & set(regimes["rate_direction"].dropna())
    assert set(regimes["yield_curve_regime"].dropna()) <= {"inverted", "flat", "normal"}
    assert set(regimes["market_trend_regime"].dropna()) <= {"positive", "negative"}


def test_daily_macro_panel_requires_cross_sectional_consistency():
    daily = _daily(5)
    panel = pd.concat(
        [daily.assign(ticker="A"), daily.assign(ticker="B")], ignore_index=True
    )
    collapsed = daily_macro_panel(panel)
    assert len(collapsed) == 5
    panel.loc[(panel["date"] == panel["date"].min()) & (panel["ticker"] == "B"), "vix"] = 99
    try:
        daily_macro_panel(panel)
    except ValueError as exc:
        assert "varies across tickers" in str(exc)
    else:
        raise AssertionError("Expected inconsistent market-wide data to fail.")


def test_metric_summary_joins_signal_regimes_and_marks_small_samples():
    daily = build_point_in_time_regimes(_daily(), quantile_lookback_days=300, quantile_min_history_days=100)
    signals = regimes_on_signal_dates(daily, daily.loc[300:320:5, "date"])
    ic = pd.DataFrame({"date": signals["date"], "rank_ic": np.linspace(-0.1, 0.2, len(signals))})
    table = metric_summary_by_regime(
        {"rf": ic}, signals, value_column="rank_ic", regime_columns=["vix_regime"], n_bootstrap=200
    )
    assert not table.empty
    assert set(table["model"]) == {"rf"}
    assert table["observations"].sum() == len(ic)


def test_holding_period_returns_assign_signal_regime_to_following_returns():
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "vix_regime": ["low", "high"],
        }
    )
    dates = pd.bdate_range("2024-01-31", "2024-03-15")
    daily = pd.DataFrame({"net_return": 0.001, "turnover": 0.0}, index=dates)
    periods = holding_period_returns({"strategy": daily}, signals, regime_columns=["vix_regime"])
    assert len(periods) == 2
    assert periods.iloc[0]["vix_regime"] == "low"
    expected_days = int(((dates > pd.Timestamp("2024-01-31")) & (dates <= pd.Timestamp("2024-02-29"))).sum())
    assert periods.iloc[0]["trading_days"] == expected_days
    summary = portfolio_summary_by_regime(periods, regime_columns=["vix_regime"])
    assert set(summary["regime"]) == {"low", "high"}


def test_historical_quantile_regime_tied_thresholds_use_medium_for_equality():
    # A step-like policy rate can have identical rolling 1/3 and 2/3 quantiles.
    # Equality must not be overwritten into the high bucket.
    from quant_platform.ml.regime import _historical_quantile_regime

    values = pd.Series([5.0] * 30 + [4.0, 5.0, 6.0], dtype=float)
    labels, lower, upper = _historical_quantile_regime(
        values,
        lookback_days=30,
        min_history_days=20,
        lower_quantile=1 / 3,
        upper_quantile=2 / 3,
    )
    assert lower.iloc[30] == 5.0
    assert upper.iloc[30] == 5.0
    assert labels.iloc[30] == "low"
    assert labels.iloc[31] == "medium"
    assert labels.iloc[32] == "high"
