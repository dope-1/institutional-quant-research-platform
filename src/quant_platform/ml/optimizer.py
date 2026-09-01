from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class OptimizedPortfolio:
    weights: pd.Series
    expected_alpha: float
    ex_ante_volatility: float
    turnover: float
    min_feasible_turnover: float
    effective_turnover_cap: float | None
    turnover_cap_relaxed: bool
    effective_n: float
    solver_success: bool
    solver_message: str


def _normalise_prediction_alpha(
    snapshot: pd.DataFrame,
    *,
    prediction_column: str,
    alpha_scale_annual: float,
) -> pd.Series:
    frame = snapshot.dropna(subset=[prediction_column]).drop_duplicates("ticker").copy()
    if frame.empty:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[prediction_column], errors="coerce")
    frame = frame.loc[values.notna()].copy()
    values = pd.to_numeric(frame[prediction_column], errors="coerce")
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        scores = pd.Series(0.0, index=frame.index)
    else:
        scores = (values - float(values.mean())) / std
    alpha = pd.Series(
        scores.to_numpy(dtype=float) * float(alpha_scale_annual),
        index=frame["ticker"].astype(str).to_numpy(),
        dtype=float,
    )
    return alpha.sort_index()


def trailing_shrinkage_covariance(
    market: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
    tickers: list[str],
    lookback_days: int = 252,
    min_history_days: int = 126,
    annualization_days: int = 252,
) -> tuple[pd.DataFrame, int]:
    """Estimate a point-in-time annualized Ledoit-Wolf covariance matrix.

    Only return observations on or before ``signal_date`` are used. Missing values
    inside the trailing window are filled with the per-asset trailing mean after
    requiring a minimum amount of observed history.
    """
    if lookback_days < min_history_days or min_history_days < 20:
        raise ValueError("lookback_days must be >= min_history_days >= 20.")
    required = {"date", "ticker", "return_1d"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"Market data missing columns: {sorted(missing)}")
    names = list(dict.fromkeys(str(t) for t in tickers))
    if len(names) < 2:
        raise ValueError("At least two assets are required for covariance estimation.")

    frame = market.loc[
        (pd.to_datetime(market["date"]) <= pd.Timestamp(signal_date))
        & market["ticker"].astype(str).isin(names),
        ["date", "ticker", "return_1d"],
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    wide = frame.pivot(index="date", columns="ticker", values="return_1d").sort_index()
    wide = wide.reindex(columns=names).tail(int(lookback_days))
    counts = wide.notna().sum()
    insufficient = counts[counts < int(min_history_days)]
    if not insufficient.empty:
        raise ValueError(
            "Insufficient covariance history for: "
            + ", ".join(f"{ticker}({int(count)})" for ticker, count in insufficient.items())
        )
    means = wide.mean(axis=0)
    matrix = wide.fillna(means).fillna(0.0)
    if len(matrix) < int(min_history_days):
        raise ValueError(
            f"Only {len(matrix)} trailing observations available before {pd.Timestamp(signal_date).date()}."
        )
    lw = LedoitWolf().fit(matrix.to_numpy(dtype=float))
    covariance = lw.covariance_ * float(annualization_days)
    cov = pd.DataFrame(covariance, index=names, columns=names)
    return cov, int(len(matrix))


def drift_weights_through_returns(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> pd.Series:
    """Drift close-of-signal target weights through subsequent gross asset returns."""
    if weights.empty:
        return weights.copy()
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).astype(float)
    if returns.empty:
        return w
    frame = returns.reindex(columns=w.index).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    growth = (1.0 + frame).prod(axis=0)
    end_values = w * growth
    cash = 1.0 - float(w.sum())
    total = cash + float(end_values.sum())
    if total <= 0:
        raise ValueError("Non-positive portfolio value while drifting optimizer weights.")
    return end_values / total


def _minimum_turnover_solution(
    previous: np.ndarray,
    *,
    max_weight: float,
) -> tuple[np.ndarray, float]:
    n = len(previous)
    # x = [w_1..w_n, u_1..u_n], minimise sum(u)
    c = np.concatenate([np.zeros(n), np.ones(n)])
    a_eq = np.zeros((1, 2 * n))
    a_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # w - u <= previous; -w - u <= -previous
    a_ub = np.zeros((2 * n, 2 * n))
    b_ub = np.zeros(2 * n)
    for i in range(n):
        a_ub[2 * i, i] = 1.0
        a_ub[2 * i, n + i] = -1.0
        b_ub[2 * i] = previous[i]
        a_ub[2 * i + 1, i] = -1.0
        a_ub[2 * i + 1, n + i] = -1.0
        b_ub[2 * i + 1] = -previous[i]
    bounds = [(0.0, float(max_weight))] * n + [(0.0, 2.0)] * n
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise ValueError(f"Unable to find a feasible long-only portfolio: {result.message}")
    return result.x[:n], float(result.fun)


def optimize_long_only_portfolio(
    alpha: pd.Series,
    covariance: pd.DataFrame,
    previous_weights: pd.Series,
    *,
    risk_aversion: float,
    turnover_penalty: float,
    max_weight: float,
    turnover_cap: float | None,
    tolerance: float = 1e-8,
) -> OptimizedPortfolio:
    """Solve a long-only mean-variance portfolio with explicit L1 turnover control.

    The convex objective is::

        risk_aversion * w' Sigma w - alpha' w + turnover_penalty * sum(u)

    with ``u >= |w - w_prev|``, full investment, a position cap, and an optional
    turnover cap. SLSQP is used after a linear-program feasibility warm start.
    """
    if risk_aversion < 0 or turnover_penalty < 0:
        raise ValueError("risk_aversion and turnover_penalty must be non-negative.")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1].")
    names = list(alpha.index.astype(str))
    n = len(names)
    if n < 2:
        raise ValueError("At least two assets are required for optimization.")
    if max_weight * n < 1.0 - 1e-10:
        raise ValueError("max_weight is infeasible for the number of eligible assets.")

    a = pd.to_numeric(alpha.reindex(names), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cov = covariance.reindex(index=names, columns=names).to_numpy(dtype=float)
    cov = 0.5 * (cov + cov.T)
    # Guard against tiny numerical negative eigenvalues.
    eigen_min = float(np.linalg.eigvalsh(cov).min())
    if eigen_min < 0:
        cov = cov + np.eye(n) * (-eigen_min + 1e-10)

    prev = pd.to_numeric(previous_weights.reindex(names), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    feasible_w, min_turnover = _minimum_turnover_solution(prev, max_weight=max_weight)
    cap_relaxed = False
    effective_cap = None if turnover_cap is None else float(turnover_cap)
    if effective_cap is not None and effective_cap < min_turnover - tolerance:
        effective_cap = min_turnover + 1e-7
        cap_relaxed = True

    u0 = np.abs(feasible_w - prev)
    x0 = np.concatenate([feasible_w, u0])

    def objective(x: np.ndarray) -> float:
        w = x[:n]
        u = x[n:]
        return float(
            float(risk_aversion) * (w @ cov @ w)
            - (a @ w)
            + float(turnover_penalty) * u.sum()
        )

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda x: float(x[:n].sum() - 1.0)},
        {"type": "ineq", "fun": lambda x: x[n:] - (x[:n] - prev)},
        {"type": "ineq", "fun": lambda x: x[n:] + (x[:n] - prev)},
    ]
    if effective_cap is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda x, cap=effective_cap: float(cap - x[n:].sum())}
        )

    bounds = [(0.0, float(max_weight))] * n + [(0.0, 2.0)] * n
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )
    if not result.success:
        raise ValueError(f"Portfolio optimizer failed: {result.message}")
    w = np.asarray(result.x[:n], dtype=float)
    # Clean numerical dust and renormalize without violating the cap materially.
    w[np.abs(w) < 1e-12] = 0.0
    w = w / w.sum()
    turnover = float(np.abs(w - prev).sum())
    expected_alpha = float(a @ w)
    variance = float(w @ cov @ w)
    ex_ante_vol = float(np.sqrt(max(variance, 0.0)))
    effective_n = float(1.0 / np.sum(np.square(w)))
    return OptimizedPortfolio(
        weights=pd.Series(w, index=names, dtype=float),
        expected_alpha=expected_alpha,
        ex_ante_volatility=ex_ante_vol,
        turnover=turnover,
        min_feasible_turnover=min_turnover,
        effective_turnover_cap=effective_cap,
        turnover_cap_relaxed=cap_relaxed,
        effective_n=effective_n,
        solver_success=bool(result.success),
        solver_message=str(result.message),
    )


def build_optimized_targets(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    alpha_scale_annual: float = 0.10,
    covariance_lookback_days: int = 252,
    covariance_min_history_days: int = 126,
    annualization_days: int = 252,
    risk_aversion: float = 4.0,
    turnover_penalty: float = 0.02,
    max_weight: float = 0.10,
    turnover_cap: float | None = 0.60,
    min_assets: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert monthly alpha predictions into drift-aware optimized target weights."""
    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"]).astype("datetime64[ns]")
    pred["ticker"] = pred["ticker"].astype(str)
    market_frame = market.copy()
    market_frame["date"] = pd.to_datetime(market_frame["date"]).astype("datetime64[ns]")
    market_frame["ticker"] = market_frame["ticker"].astype(str)
    returns = market_frame.pivot(index="date", columns="ticker", values="return_1d").sort_index()

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    previous_target = pd.Series(dtype=float)
    previous_signal: pd.Timestamp | None = None

    for date, snapshot in pred.groupby("date", sort=True):
        date = pd.Timestamp(date)
        alpha = _normalise_prediction_alpha(
            snapshot,
            prediction_column=prediction_column,
            alpha_scale_annual=alpha_scale_annual,
        )
        if len(alpha) < int(min_assets):
            continue
        if previous_signal is None:
            pretrade_all = pd.Series(0.0, index=returns.columns, dtype=float)
        else:
            window = returns.loc[(returns.index > previous_signal) & (returns.index <= date)]
            start = previous_target.reindex(returns.columns).fillna(0.0)
            pretrade_all = drift_weights_through_returns(start, window)

        # Include any still-held name even if a prediction is temporarily missing,
        # so liquidation/rebalancing turnover is accounted for rather than silently
        # dropping a position from the optimizer state. Missing alpha is neutral (0).
        held = list(pretrade_all[pretrade_all.abs() > 1e-12].index.astype(str))
        tickers = list(dict.fromkeys([*alpha.index.astype(str), *held]))
        alpha = alpha.reindex(tickers).fillna(0.0)
        pretrade = pretrade_all.reindex(tickers).fillna(0.0)

        covariance, covariance_observations = trailing_shrinkage_covariance(
            market_frame,
            signal_date=date,
            tickers=tickers,
            lookback_days=covariance_lookback_days,
            min_history_days=covariance_min_history_days,
            annualization_days=annualization_days,
        )
        result = optimize_long_only_portfolio(
            alpha,
            covariance,
            pretrade,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            max_weight=max_weight,
            turnover_cap=turnover_cap,
        )
        for ticker, weight in result.weights.items():
            if weight > 1e-12:
                rows.append(
                    {"date": date, "ticker": str(ticker), "target_weight": float(weight)}
                )
        diagnostics.append(
            {
                "date": date,
                "assets": int(len(result.weights)),
                "positions": int((result.weights > 1e-8).sum()),
                "expected_alpha": result.expected_alpha,
                "ex_ante_volatility": result.ex_ante_volatility,
                "optimizer_turnover": result.turnover,
                "min_feasible_turnover": result.min_feasible_turnover,
                "configured_turnover_cap": turnover_cap,
                "effective_turnover_cap": result.effective_turnover_cap,
                "turnover_cap_relaxed": result.turnover_cap_relaxed,
                "max_weight_realized": float(result.weights.max()),
                "effective_n": result.effective_n,
                "covariance_observations": covariance_observations,
                "solver_success": result.solver_success,
                "solver_message": result.solver_message,
            }
        )
        previous_target = result.weights.reindex(returns.columns).fillna(0.0)
        previous_signal = date

    target_table = pd.DataFrame(rows, columns=["date", "ticker", "target_weight"])
    diagnostic_table = pd.DataFrame(diagnostics)
    return target_table, diagnostic_table
