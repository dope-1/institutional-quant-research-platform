from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.portfolio.risk import historical_cvar, historical_var


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min()) if len(drawdown) else float("nan")


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    annualization_days: int = 252,
) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        raise ValueError("Cannot evaluate an empty return series.")

    n = len(r)
    total_return = float((1.0 + r).prod() - 1.0)
    annualized_return = float((1.0 + total_return) ** (annualization_days / n) - 1.0)
    annualized_vol = float(r.std(ddof=1) * np.sqrt(annualization_days))
    r_std = r.std(ddof=1)
    sharpe = (
        float(r.mean() / r_std * np.sqrt(annualization_days))
        if r_std > 0
        else float("nan")
    )

    downside = r[r < 0]
    downside_std = downside.std(ddof=1)
    sortino = (
        float(r.mean() / downside_std * np.sqrt(annualization_days))
        if pd.notna(downside_std) and downside_std > 0
        else float("nan")
    )

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown(r),
        "var_95_daily": historical_var(r, 0.95),
        "cvar_95_daily": historical_cvar(r, 0.95),
    }
    if turnover is not None:
        aligned_turnover = turnover.reindex(r.index).fillna(0.0)
        metrics["annualized_turnover"] = float(
            aligned_turnover.mean() * annualization_days
        )

    if benchmark_returns is not None:
        aligned = pd.concat(
            [r.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
        ).dropna()
        active = aligned["strategy"] - aligned["benchmark"]
        te = active.std(ddof=1)
        metrics["information_ratio"] = (
            float(active.mean() / te * np.sqrt(annualization_days))
            if te > 0
            else float("nan")
        )
    return metrics
