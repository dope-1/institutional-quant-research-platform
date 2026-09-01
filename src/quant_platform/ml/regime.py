from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_platform.ml.robustness import bootstrap_mean_ci, mean_t_stat

DEFAULT_REGIME_COLUMNS = [
    "vix_regime",
    "treasury_10y_regime",
    "fed_funds_regime",
    "rate_direction",
    "yield_curve_regime",
    "market_trend_regime",
]


def daily_macro_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the equity panel to one point-in-time macro/factor row per date.

    Macro/factor columns are market-wide, so taking the first row per date is
    appropriate after verifying that any available values are identical across
    the cross-section.
    """
    required = {"date"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    candidates = [
        c
        for c in [
            "treasury_10y",
            "treasury_2y",
            "vix",
            "fed_funds",
            "mkt_rf",
            "smb",
            "hml",
            "rf",
        ]
        if c in frame.columns
    ]
    if not candidates:
        raise ValueError("Panel does not contain any macro/factor columns for regime analysis.")

    for col in candidates:
        varying = frame.groupby("date", sort=False)[col].nunique(dropna=True)
        if (varying > 1).any():
            bad_date = varying[varying > 1].index[0]
            raise ValueError(
                f"Market-wide column '{col}' varies across tickers on {pd.Timestamp(bad_date).date()}."
            )

    daily = (
        frame[["date", *candidates]]
        .sort_values("date")
        .groupby("date", as_index=False, sort=True)
        .first()
    )
    return daily


def _historical_quantile_regime(
    values: pd.Series,
    *,
    lookback_days: int,
    min_history_days: int,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Classify a series using trailing thresholds available before each date."""
    if not 0 < lower_quantile < upper_quantile < 1:
        raise ValueError("Quantiles must satisfy 0 < lower < upper < 1.")
    if lookback_days < min_history_days or min_history_days < 20:
        raise ValueError("lookback_days must be >= min_history_days >= 20.")

    numeric = pd.to_numeric(values, errors="coerce")
    history = numeric.shift(1)
    lower = history.rolling(lookback_days, min_periods=min_history_days).quantile(lower_quantile)
    upper = history.rolling(lookback_days, min_periods=min_history_days).quantile(upper_quantile)

    labels = pd.Series(pd.NA, index=values.index, dtype="string")
    valid = numeric.notna() & lower.notna() & upper.notna()

    # Rolling quantiles can coincide for step-like macro series (especially the
    # Fed funds rate). The old <= / >= rules made a value equal to both tied
    # thresholds end up in ``high`` because that assignment happened last. That
    # creates a spurious two-bucket regime. When thresholds tie, equality is the
    # economically neutral middle state; values strictly below/above remain low/high.
    tied = valid & np.isclose(lower, upper, rtol=0.0, atol=1e-12)
    labels.loc[tied & (numeric < lower)] = "low"
    labels.loc[tied & np.isclose(numeric, lower, rtol=0.0, atol=1e-12)] = "medium"
    labels.loc[tied & (numeric > upper)] = "high"

    distinct = valid & ~tied
    labels.loc[distinct & (numeric <= lower)] = "low"
    labels.loc[distinct & (numeric > lower) & (numeric < upper)] = "medium"
    labels.loc[distinct & (numeric >= upper)] = "high"
    return labels, lower, upper


def build_point_in_time_regimes(
    daily: pd.DataFrame,
    *,
    quantile_lookback_days: int = 504,
    quantile_min_history_days: int = 252,
    lower_quantile: float = 1 / 3,
    upper_quantile: float = 2 / 3,
    rate_direction_window_days: int = 63,
    rate_flat_threshold_pp: float = 0.10,
    yield_curve_flat_upper_pp: float = 0.25,
    market_trend_window_days: int = 63,
) -> pd.DataFrame:
    """Create diagnostic market-regime labels using only information known by each date.

    Quantile-based level regimes use thresholds estimated from *prior* observations.
    Rate direction, yield-curve shape and trailing market trend use fixed rules and
    trailing data only. No future observations enter a date's classification.
    """
    frame = daily.copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    if frame["date"].duplicated().any():
        raise ValueError("Regime input must have exactly one row per date.")

    for col in ["vix", "treasury_10y", "treasury_2y", "fed_funds", "mkt_rf"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    for source, out_name in [
        ("vix", "vix_regime"),
        ("treasury_10y", "treasury_10y_regime"),
        ("fed_funds", "fed_funds_regime"),
    ]:
        if source in frame.columns:
            regime, lower, upper = _historical_quantile_regime(
                frame[source],
                lookback_days=quantile_lookback_days,
                min_history_days=quantile_min_history_days,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
            frame[out_name] = regime
            frame[f"{out_name}_lower_threshold"] = lower
            frame[f"{out_name}_upper_threshold"] = upper

    if "treasury_10y" in frame.columns:
        delta = frame["treasury_10y"] - frame["treasury_10y"].shift(rate_direction_window_days)
        frame["treasury_10y_change"] = delta
        direction = pd.Series(pd.NA, index=frame.index, dtype="string")
        direction.loc[delta > rate_flat_threshold_pp] = "rising"
        direction.loc[delta < -rate_flat_threshold_pp] = "falling"
        direction.loc[delta.abs() <= rate_flat_threshold_pp] = "flat"
        frame["rate_direction"] = direction

    if {"treasury_10y", "treasury_2y"}.issubset(frame.columns):
        spread = frame["treasury_10y"] - frame["treasury_2y"]
        frame["yield_curve_spread"] = spread
        curve = pd.Series(pd.NA, index=frame.index, dtype="string")
        curve.loc[spread < 0.0] = "inverted"
        curve.loc[(spread >= 0.0) & (spread <= yield_curve_flat_upper_pp)] = "flat"
        curve.loc[spread > yield_curve_flat_upper_pp] = "normal"
        frame["yield_curve_regime"] = curve

    if "mkt_rf" in frame.columns:
        # Fama-French Mkt-RF is stored as a decimal daily return. A trailing
        # compounded excess-market return is known at the signal close.
        market_return = (1.0 + frame["mkt_rf"].fillna(0.0)).rolling(
            market_trend_window_days, min_periods=market_trend_window_days
        ).apply(np.prod, raw=True) - 1.0
        frame["market_trailing_return"] = market_return
        trend = pd.Series(pd.NA, index=frame.index, dtype="string")
        trend.loc[market_return >= 0.0] = "positive"
        trend.loc[market_return < 0.0] = "negative"
        frame["market_trend_regime"] = trend

    return frame


def regimes_on_signal_dates(
    regime_daily: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return exact-date regime labels for the supplied equity signal dates."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).astype("datetime64[ns]")
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    lookup = regime_daily.copy()
    lookup["date"] = pd.to_datetime(lookup["date"]).astype("datetime64[ns]")
    out = pd.DataFrame({"date": dates}).merge(lookup, on="date", how="left", validate="one_to_one")
    return out


def _summary_values(values: pd.Series, *, n_bootstrap: int, confidence: float, seed: int) -> dict[str, object]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "positive_fraction": float("nan"),
            "observations": 0,
            "t_stat": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
        }
    boot = bootstrap_mean_ci(
        clean,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else float("nan"),
        "positive_fraction": float((clean > 0).mean()),
        "observations": int(len(clean)),
        "t_stat": mean_t_stat(clean),
        "bootstrap_ci_low": float(boot["ci_low"]),
        "bootstrap_ci_high": float(boot["ci_high"]),
    }


def metric_summary_by_regime(
    metric_frames: dict[str, pd.DataFrame],
    signal_regimes: pd.DataFrame,
    *,
    value_column: str,
    regime_columns: list[str] | None = None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarise a monthly model metric (Rank IC or spread) by point-in-time regime."""
    regime_columns = regime_columns or [c for c in DEFAULT_REGIME_COLUMNS if c in signal_regimes.columns]
    rows: list[dict[str, object]] = []
    for model_index, (model, metric) in enumerate(metric_frames.items()):
        frame = metric.copy()
        if frame.empty or value_column not in frame.columns:
            continue
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
        joined = frame.merge(signal_regimes[["date", *regime_columns]], on="date", how="left", validate="many_to_one")
        for regime_index, regime_col in enumerate(regime_columns):
            for label, group in joined.dropna(subset=[regime_col]).groupby(regime_col, sort=True):
                stats = _summary_values(
                    group[value_column],
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    seed=seed + 100 * model_index + regime_index,
                )
                rows.append(
                    {
                        "model": model,
                        "regime_dimension": regime_col,
                        "regime": str(label),
                        **stats,
                        "small_sample": bool(stats["observations"] < 6),
                    }
                )
    return pd.DataFrame(rows)


def holding_period_returns(
    daily_by_strategy: dict[str, pd.DataFrame],
    signal_regimes: pd.DataFrame,
    *,
    regime_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compound daily strategy returns over each signal-to-next-signal holding period.

    A signal made at date t applies from the following trading day. Therefore a
    period is (signal_t, signal_{t+1}] and the final period runs to the end of the
    available daily backtest. Each holding period is tagged by the regime known on
    signal_t.
    """
    regime_columns = regime_columns or [c for c in DEFAULT_REGIME_COLUMNS if c in signal_regimes.columns]
    signals = signal_regimes[["date", *regime_columns]].copy().sort_values("date").reset_index(drop=True)
    signals["date"] = pd.to_datetime(signals["date"]).astype("datetime64[ns]")
    rows: list[dict[str, object]] = []
    for strategy, daily in daily_by_strategy.items():
        frame = daily.copy()
        if "date" in frame.columns:
            frame = frame.set_index("date")
        frame.index = pd.to_datetime(frame.index).astype("datetime64[ns]")
        frame = frame.sort_index()
        if "net_return" not in frame.columns:
            raise ValueError(f"Daily strategy '{strategy}' is missing net_return.")
        for i, signal in signals.iterrows():
            start = pd.Timestamp(signal["date"])
            end = pd.Timestamp(signals.iloc[i + 1]["date"]) if i + 1 < len(signals) else pd.Timestamp(frame.index.max())
            window = frame.loc[(frame.index > start) & (frame.index <= end)]
            if window.empty:
                continue
            net = pd.to_numeric(window["net_return"], errors="coerce").dropna()
            if net.empty:
                continue
            row: dict[str, object] = {
                "strategy": strategy,
                "signal_date": start,
                "period_end": pd.Timestamp(window.index.max()),
                "holding_return": float((1.0 + net).prod() - 1.0),
                "trading_days": int(len(net)),
                "turnover": float(pd.to_numeric(window.get("turnover", 0.0), errors="coerce").fillna(0.0).sum())
                if "turnover" in window.columns
                else 0.0,
            }
            for col in regime_columns:
                row[col] = signal[col]
            rows.append(row)
    return pd.DataFrame(rows)


def portfolio_summary_by_regime(
    holding_periods: pd.DataFrame,
    *,
    regime_columns: list[str] | None = None,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Summarise non-contiguous conditional holding-period returns by regime.

    Sharpe is calculated from monthly-ish holding-period returns and annualised
    with sqrt(12). We intentionally do not report conditional max drawdown because
    concatenating non-contiguous regime episodes would be economically misleading.
    """
    if holding_periods.empty:
        return pd.DataFrame()
    regime_columns = regime_columns or [c for c in DEFAULT_REGIME_COLUMNS if c in holding_periods.columns]
    rows: list[dict[str, object]] = []
    for strategy, strategy_frame in holding_periods.groupby("strategy", sort=True):
        for regime_col in regime_columns:
            for label, group in strategy_frame.dropna(subset=[regime_col]).groupby(regime_col, sort=True):
                returns = pd.to_numeric(group["holding_return"], errors="coerce").dropna()
                if returns.empty:
                    continue
                std = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
                sharpe = (
                    float(returns.mean() / std * math.sqrt(periods_per_year))
                    if std and np.isfinite(std) and std > 0
                    else float("nan")
                )
                rows.append(
                    {
                        "strategy": strategy,
                        "regime_dimension": regime_col,
                        "regime": str(label),
                        "periods": int(len(returns)),
                        "mean_holding_return": float(returns.mean()),
                        "median_holding_return": float(returns.median()),
                        "holding_return_std": std,
                        "conditional_sharpe": sharpe,
                        "positive_fraction": float((returns > 0).mean()),
                        "mean_turnover_per_period": float(pd.to_numeric(group["turnover"], errors="coerce").mean()),
                        "annualized_return_approx": float((1.0 + returns.mean()) ** periods_per_year - 1.0)
                        if returns.mean() > -1.0
                        else float("nan"),
                        "small_sample": bool(len(returns) < 6),
                    }
                )
    return pd.DataFrame(rows)


def regime_counts(signal_regimes: pd.DataFrame, regime_columns: list[str] | None = None) -> pd.DataFrame:
    regime_columns = regime_columns or [c for c in DEFAULT_REGIME_COLUMNS if c in signal_regimes.columns]
    rows: list[dict[str, object]] = []
    for col in regime_columns:
        counts = signal_regimes[col].value_counts(dropna=False)
        for label, count in counts.items():
            rows.append(
                {
                    "regime_dimension": col,
                    "regime": "missing" if pd.isna(label) else str(label),
                    "signal_months": int(count),
                }
            )
    return pd.DataFrame(rows)
