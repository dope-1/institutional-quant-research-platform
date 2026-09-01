from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.ml.optimizer import (
    build_optimized_targets,
    drift_weights_through_returns,
    optimize_long_only_portfolio,
    trailing_shrinkage_covariance,
)


def _market(days: int = 320, tickers: tuple[str, ...] = ("A", "B", "C", "D")) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=days)
    rows = []
    for j, ticker in enumerate(tickers):
        returns = rng.normal(0.0002 + 0.00005 * j, 0.01 + 0.001 * j, size=days)
        for date, ret in zip(dates, returns, strict=True):
            rows.append({"date": date, "ticker": ticker, "return_1d": float(ret)})
    return pd.DataFrame(rows)


def test_trailing_covariance_is_invariant_to_future_return_changes():
    market = _market()
    cutoff = pd.Timestamp("2020-12-31")
    cov1, n1 = trailing_shrinkage_covariance(
        market,
        signal_date=cutoff,
        tickers=["A", "B", "C", "D"],
        lookback_days=120,
        min_history_days=60,
    )
    changed = market.copy()
    changed.loc[pd.to_datetime(changed["date"]) > cutoff, "return_1d"] *= 25.0
    cov2, n2 = trailing_shrinkage_covariance(
        changed,
        signal_date=cutoff,
        tickers=["A", "B", "C", "D"],
        lookback_days=120,
        min_history_days=60,
    )
    assert n1 == n2 == 120
    np.testing.assert_allclose(cov1.to_numpy(), cov2.to_numpy(), atol=1e-12, rtol=1e-12)


def test_optimizer_respects_long_only_full_investment_and_position_cap():
    names = ["A", "B", "C", "D"]
    alpha = pd.Series([0.10, 0.03, 0.00, -0.02], index=names)
    covariance = pd.DataFrame(np.eye(4) * 0.04, index=names, columns=names)
    previous = pd.Series(0.25, index=names)
    result = optimize_long_only_portfolio(
        alpha,
        covariance,
        previous,
        risk_aversion=0.5,
        turnover_penalty=0.0,
        max_weight=0.40,
        turnover_cap=1.0,
    )
    assert result.solver_success
    assert abs(result.weights.sum() - 1.0) < 1e-8
    assert (result.weights >= -1e-10).all()
    assert result.weights.max() <= 0.4000001
    assert result.weights["A"] > result.weights["D"]


def test_optimizer_enforces_turnover_cap_when_feasible():
    names = ["A", "B", "C", "D"]
    alpha = pd.Series([0.10, 0.05, -0.02, -0.04], index=names)
    covariance = pd.DataFrame(np.eye(4) * 0.03, index=names, columns=names)
    previous = pd.Series([0.25, 0.25, 0.25, 0.25], index=names)
    result = optimize_long_only_portfolio(
        alpha,
        covariance,
        previous,
        risk_aversion=1.0,
        turnover_penalty=0.0,
        max_weight=0.50,
        turnover_cap=0.20,
    )
    assert not result.turnover_cap_relaxed
    assert result.turnover <= 0.200001


def test_first_investment_relaxes_only_to_minimum_feasible_turnover():
    names = ["A", "B", "C", "D"]
    alpha = pd.Series([0.1, 0.05, 0.0, -0.05], index=names)
    covariance = pd.DataFrame(np.eye(4) * 0.03, index=names, columns=names)
    previous = pd.Series(0.0, index=names)
    result = optimize_long_only_portfolio(
        alpha,
        covariance,
        previous,
        risk_aversion=2.0,
        turnover_penalty=0.02,
        max_weight=0.40,
        turnover_cap=0.40,
    )
    assert result.turnover_cap_relaxed
    assert abs(result.min_feasible_turnover - 1.0) < 1e-8
    assert result.effective_turnover_cap is not None
    assert result.effective_turnover_cap >= 1.0
    assert abs(result.weights.sum() - 1.0) < 1e-8


def test_drift_weights_matches_cumulative_holding_values():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    returns = pd.DataFrame({"A": [0.10, 0.10], "B": [0.0, 0.0]})
    drifted = drift_weights_through_returns(weights, returns)
    expected_a_value = 0.5 * 1.1 * 1.1
    expected_b_value = 0.5
    assert np.isclose(drifted["A"], expected_a_value / (expected_a_value + expected_b_value))
    assert np.isclose(drifted.sum(), 1.0)


def test_build_optimized_targets_produces_monthly_constraints_and_diagnostics():
    market = _market(days=320)
    dates = pd.DatetimeIndex(sorted(market["date"].unique()))
    signal_dates = [dates[200], dates[220], dates[240], dates[260]]
    rows = []
    for k, date in enumerate(signal_dates):
        for j, ticker in enumerate(["A", "B", "C", "D"]):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "prediction": float(j + 0.1 * k),
                }
            )
    predictions = pd.DataFrame(rows)
    targets, diagnostics = build_optimized_targets(
        predictions,
        market,
        alpha_scale_annual=0.10,
        covariance_lookback_days=120,
        covariance_min_history_days=60,
        annualization_days=252,
        risk_aversion=4.0,
        turnover_penalty=0.02,
        max_weight=0.40,
        turnover_cap=0.60,
        min_assets=4,
    )
    assert len(diagnostics) == len(signal_dates)
    sums = targets.groupby("date")["target_weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-7)
    assert targets["target_weight"].max() <= 0.400001
    assert bool(diagnostics.iloc[0]["turnover_cap_relaxed"])
    assert not diagnostics.iloc[1:]["turnover_cap_relaxed"].any()
