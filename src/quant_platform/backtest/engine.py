from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_platform.portfolio.costs import one_way_cost_rate


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    weights: pd.DataFrame


def run_target_weight_backtest(
    market: pd.DataFrame,
    targets: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> BacktestResult:
    """Run a self-financing-style long-only target-weight backtest.

    A target formed at close ``t`` is first tradable for the return from close
    ``t`` to close ``t+1``. Between rebalances the portfolio is *not* reset to its
    old target every day: weights drift naturally with asset returns. Turnover at
    the next rebalance is therefore measured against the drifted pre-trade
    portfolio, not against the previous target vector.

    Transaction costs are deducted from portfolio return on the activation day.
    Weight drift itself is computed from gross asset returns; this keeps the
    accounting transparent while avoiding an implicit daily rebalancing subsidy.
    """
    if targets.empty:
        raise ValueError("Target-weight table is empty.")
    required_market = {"date", "ticker", "return_1d"}
    missing_market = required_market - set(market.columns)
    if missing_market:
        raise ValueError(f"Market data missing columns: {sorted(missing_market)}")
    required_targets = {"date", "ticker", "target_weight"}
    missing_targets = required_targets - set(targets.columns)
    if missing_targets:
        raise ValueError(f"Target table missing columns: {sorted(missing_targets)}")

    returns = market.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    returns.index = pd.DatetimeIndex(pd.to_datetime(returns.index)).astype("datetime64[ns]")
    all_dates = returns.index
    all_tickers = returns.columns

    target_w = targets.copy()
    target_w["date"] = pd.to_datetime(target_w["date"]).astype("datetime64[ns]")
    target_w = target_w.pivot(index="date", columns="ticker", values="target_weight")
    target_w = target_w.reindex(columns=all_tickers)

    # Map each close-t signal to the next available trading date. This enforces
    # the one-trading-day execution lag even around weekends/holidays.
    activation_targets: dict[pd.Timestamp, np.ndarray] = {}
    for signal_date, row in target_w.iterrows():
        pos = int(all_dates.searchsorted(pd.Timestamp(signal_date), side="right"))
        if pos >= len(all_dates):
            continue
        activation_date = pd.Timestamp(all_dates[pos])
        activation_targets[activation_date] = row.fillna(0.0).to_numpy(dtype=float)

    current_weights = np.zeros(len(all_tickers), dtype=float)
    active_rows: list[np.ndarray] = []
    daily_rows: list[dict[str, float]] = []
    cost_rate = one_way_cost_rate(transaction_cost_bps, slippage_bps)

    for date in all_dates:
        pretrade = current_weights.copy()
        turnover = 0.0
        if pd.Timestamp(date) in activation_targets:
            active = activation_targets[pd.Timestamp(date)].copy()
            turnover = float(np.abs(active - pretrade).sum())
        else:
            active = pretrade

        day_returns = (
            pd.to_numeric(returns.loc[date], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        gross_return = float(np.dot(active, day_returns))
        trading_cost = float(turnover * cost_rate)
        net_return = gross_return - trading_cost
        if net_return <= -1.0:
            raise ValueError(f"Portfolio return <= -100% on {pd.Timestamp(date).date()}.")

        active_rows.append(active.copy())
        daily_rows.append(
            {
                "gross_return": gross_return,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "net_return": net_return,
            }
        )

        # Drift risky-asset weights using gross holdings values. Cash is the
        # residual 1-sum(weights), which supports the pre-first-rebalance period.
        end_risky_values = active * (1.0 + day_returns)
        cash_weight = 1.0 - float(active.sum())
        gross_growth = cash_weight + float(end_risky_values.sum())
        if gross_growth <= 0.0:
            raise ValueError(f"Non-positive gross portfolio value on {pd.Timestamp(date).date()}.")
        current_weights = end_risky_values / gross_growth

    daily = pd.DataFrame(daily_rows, index=all_dates)
    daily.index.name = "date"
    daily["equity_curve"] = (1.0 + daily["net_return"]).cumprod()

    weights = pd.DataFrame(active_rows, index=all_dates, columns=all_tickers)
    weights.index.name = "date"
    return BacktestResult(daily=daily, weights=weights)
