from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.evaluation.metrics import max_drawdown

RAW_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "return_1d",
    "dollar_volume",
    "amihud_raw",
}


@dataclass(frozen=True)
class AuditResult:
    summary: dict[str, object]
    ticker_coverage: pd.DataFrame
    cross_sectional_coverage: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    annual_strategy_metrics: pd.DataFrame
    extreme_return_rows: pd.DataFrame
    zero_volume_rows: pd.DataFrame
    missing_macro_dates: pd.DataFrame


def infer_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Infer model-input candidates while explicitly excluding raw fields and targets."""
    cols: list[str] = []
    for col in panel.columns:
        if col in RAW_COLUMNS or col.startswith("forward_return_"):
            continue
        if pd.api.types.is_numeric_dtype(panel[col]):
            cols.append(col)
    return cols


def ticker_coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    if dates.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for ticker, part in panel.groupby("ticker", sort=True):
        ticker_dates = pd.DatetimeIndex(sorted(pd.to_datetime(part["date"]).unique()))
        first = ticker_dates.min()
        last = ticker_dates.max()
        in_span = dates[(dates >= first) & (dates <= last)]
        rows.append(
            {
                "ticker": ticker,
                "first_date": first,
                "last_date": last,
                "observations": int(len(ticker_dates)),
                "expected_dates_within_own_span": int(len(in_span)),
                "within_span_coverage_ratio": (
                    float(len(ticker_dates) / len(in_span)) if len(in_span) else np.nan
                ),
                "global_history_coverage_ratio": float(len(ticker_dates) / len(dates)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["global_history_coverage_ratio", "ticker"], ascending=[True, True]
    )


def cross_sectional_coverage_report(
    panel: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
) -> pd.DataFrame:
    work = panel[["date", "ticker", *feature_columns, target_col]].copy()
    work["all_features_available"] = work[feature_columns].notna().all(axis=1)
    work["target_available"] = work[target_col].notna()
    out = (
        work.groupby("date", sort=True)
        .agg(
            assets=("ticker", "nunique"),
            complete_feature_rows=("all_features_available", "sum"),
            target_rows=("target_available", "sum"),
        )
        .reset_index()
    )
    out["complete_feature_fraction"] = out["complete_feature_rows"] / out["assets"]
    out["target_fraction"] = out["target_rows"] / out["assets"]
    return out


def feature_diagnostics_report(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = max(len(panel), 1)
    for col in columns:
        series = pd.to_numeric(panel[col], errors="coerce")
        values = series.to_numpy(dtype=float, na_value=np.nan)
        finite_mask = np.isfinite(values)
        finite = series.loc[finite_mask]
        quantiles = finite.quantile([0.01, 0.50, 0.99]) if not finite.empty else pd.Series(dtype=float)
        rows.append(
            {
                "column": col,
                "rows": int(len(series)),
                "missing_rows": int(series.isna().sum()),
                "missing_fraction": float(series.isna().sum() / n),
                "infinite_rows": int(np.isinf(values).sum()),
                "mean": float(finite.mean()) if not finite.empty else np.nan,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                "p01": float(quantiles.get(0.01, np.nan)),
                "median": float(quantiles.get(0.50, np.nan)),
                "p99": float(quantiles.get(0.99, np.nan)),
                "min": float(finite.min()) if not finite.empty else np.nan,
                "max": float(finite.max()) if not finite.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def label_alignment_mismatches(panel: pd.DataFrame, horizon: int, atol: float = 1e-12) -> int:
    target_col = f"forward_return_{horizon}d"
    if target_col not in panel:
        raise ValueError(f"Expected target column {target_col!r}.")
    ordered = panel.sort_values(["ticker", "date"]).copy()
    expected = ordered.groupby("ticker", sort=False)["close"].shift(-horizon) / ordered["close"] - 1.0
    actual = pd.to_numeric(ordered[target_col], errors="coerce")
    comparable = expected.notna() & actual.notna()
    mismatch = ~np.isclose(
        expected.loc[comparable].to_numpy(),
        actual.loc[comparable].to_numpy(),
        atol=atol,
        rtol=1e-10,
    )
    missing_mismatch = expected.isna() ^ actual.isna()
    return int(mismatch.sum() + missing_mismatch.sum())


def annual_strategy_report(strategy_daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, frame in strategy_daily.items():
        if frame.empty or "net_return" not in frame:
            continue
        data = frame.copy()
        data.index = pd.to_datetime(data.index)
        for year, part in data.groupby(data.index.year):
            r = pd.to_numeric(part["net_return"], errors="coerce").dropna()
            if r.empty:
                continue
            vol = r.std(ddof=1)
            rows.append(
                {
                    "strategy": strategy,
                    "year": int(year),
                    "trading_days": int(len(r)),
                    "net_return": float((1.0 + r).prod() - 1.0),
                    "annualized_volatility": float(vol * np.sqrt(252)) if pd.notna(vol) else np.nan,
                    "sharpe": (
                        float(r.mean() / vol * np.sqrt(252))
                        if pd.notna(vol) and vol > 0
                        else np.nan
                    ),
                    "max_drawdown": max_drawdown(r),
                    "average_daily_turnover": (
                        float(part["turnover"].mean()) if "turnover" in part else np.nan
                    ),
                    "trading_cost": (
                        float(part["trading_cost"].sum()) if "trading_cost" in part else np.nan
                    ),
                }
            )
    columns = [
        "strategy",
        "year",
        "trading_days",
        "net_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "average_daily_turnover",
        "trading_cost",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["strategy", "year"]).reset_index(drop=True)


def extreme_return_rows_report(panel: pd.DataFrame, threshold: float) -> pd.DataFrame:
    ret = pd.to_numeric(panel["return_1d"], errors="coerce")
    cols = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume", "return_1d"] if c in panel]
    out = panel.loc[ret.abs() > threshold, cols].copy()
    if not out.empty:
        out["abs_return_1d"] = pd.to_numeric(out["return_1d"], errors="coerce").abs()
        out = out.sort_values("abs_return_1d", ascending=False)
    return out.reset_index(drop=True)


def zero_volume_rows_report(panel: pd.DataFrame) -> pd.DataFrame:
    if "volume" not in panel:
        return pd.DataFrame(columns=["date", "ticker", "volume"])
    volume = pd.to_numeric(panel["volume"], errors="coerce")
    cols = [c for c in ["date", "ticker", "close", "volume", "dollar_volume"] if c in panel]
    return panel.loc[volume <= 0, cols].copy().sort_values(["date", "ticker"]).reset_index(drop=True)


def missing_macro_dates_report(panel: pd.DataFrame, macro_columns: list[str]) -> pd.DataFrame:
    available = [c for c in macro_columns if c in panel.columns]
    if not available:
        return pd.DataFrame(columns=["date", "missing_columns", "affected_rows"])
    rows: list[dict[str, object]] = []
    for date, part in panel.groupby("date", sort=True):
        missing = [c for c in available if part[c].isna().all()]
        if missing:
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "missing_columns": ",".join(missing),
                    "affected_rows": int(len(part)),
                }
            )
    return pd.DataFrame(rows, columns=["date", "missing_columns", "affected_rows"])


def run_research_audit(
    panel: pd.DataFrame,
    horizon: int,
    strategy_daily: dict[str, pd.DataFrame] | None = None,
    extreme_return_threshold: float = 0.25,
    macro_columns: list[str] | None = None,
) -> AuditResult:
    required = {"date", "ticker", "close", "return_1d", f"forward_return_{horizon}d"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Research panel missing required audit columns: {sorted(missing)}")
    if panel.empty:
        raise ValueError("Cannot audit an empty research panel.")

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).astype("datetime64[ns]")
    target_col = f"forward_return_{horizon}d"
    features = infer_feature_columns(work)
    coverage = ticker_coverage_report(work)
    cross_section = cross_sectional_coverage_report(work, features, target_col)
    diagnostics = feature_diagnostics_report(work, [*features, target_col])
    annual = annual_strategy_report(strategy_daily or {})
    extreme_rows = extreme_return_rows_report(work, extreme_return_threshold)
    zero_volume = zero_volume_rows_report(work)
    missing_macro = missing_macro_dates_report(work, macro_columns or [])

    duplicate_rows = int(work.duplicated(["date", "ticker"]).sum())
    label_mismatches = label_alignment_mismatches(work, horizon)
    nonfinite_feature_rows = int(
        diagnostics.loc[diagnostics["column"].isin(features), "infinite_rows"].sum()
    )

    summary: dict[str, object] = {
        "rows": int(len(work)),
        "tickers": int(work["ticker"].nunique()),
        "start_date": str(work["date"].min().date()),
        "end_date": str(work["date"].max().date()),
        "feature_count": int(len(features)),
        "duplicate_date_ticker_rows": duplicate_rows,
        "label_alignment_mismatches": label_mismatches,
        "nonfinite_feature_values": nonfinite_feature_rows,
        "extreme_abs_daily_return_threshold": float(extreme_return_threshold),
        "extreme_abs_daily_return_rows": int(len(extreme_rows)),
        "zero_or_negative_volume_rows": int(len(zero_volume)),
        "dates_with_all_missing_macro_observations": int(len(missing_macro)),
        "minimum_assets_per_date": int(cross_section["assets"].min()),
        "median_assets_per_date": float(cross_section["assets"].median()),
        "minimum_complete_feature_rows_per_date": int(cross_section["complete_feature_rows"].min()),
        "median_complete_feature_fraction": float(cross_section["complete_feature_fraction"].median()),
        "minimum_ticker_within_span_coverage_ratio": float(coverage["within_span_coverage_ratio"].min()),
        "audit_pass": bool(
            duplicate_rows == 0
            and label_mismatches == 0
            and nonfinite_feature_rows == 0
        ),
        "notes": [
            "Extreme-return and zero-volume rows are review flags, not automatic data errors.",
            "Macro values are carried backward-only from the last available observation; future filling is never used.",
            "Static-universe survivorship bias remains a known Milestone 1 limitation.",
            "Feature warm-up missingness is expected for long lookback features such as 12-1 momentum.",
        ],
    }
    return AuditResult(
        summary,
        coverage,
        cross_section,
        diagnostics,
        annual,
        extreme_rows,
        zero_volume,
        missing_macro,
    )


def write_audit_artifacts(result: AuditResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": out_dir / "data_quality_summary.json",
        "ticker_coverage": out_dir / "ticker_coverage.csv",
        "cross_sectional_coverage": out_dir / "cross_sectional_coverage.csv",
        "feature_diagnostics": out_dir / "feature_diagnostics.csv",
        "annual_strategy_metrics": out_dir / "annual_strategy_metrics.csv",
        "extreme_return_rows": out_dir / "extreme_return_rows.csv",
        "zero_volume_rows": out_dir / "zero_volume_rows.csv",
        "missing_macro_dates": out_dir / "missing_macro_dates.csv",
    }
    paths["summary"].write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    result.ticker_coverage.to_csv(paths["ticker_coverage"], index=False)
    result.cross_sectional_coverage.to_csv(paths["cross_sectional_coverage"], index=False)
    result.feature_diagnostics.to_csv(paths["feature_diagnostics"], index=False)
    result.annual_strategy_metrics.to_csv(paths["annual_strategy_metrics"], index=False)
    result.extreme_return_rows.to_csv(paths["extreme_return_rows"], index=False)
    result.zero_volume_rows.to_csv(paths["zero_volume_rows"], index=False)
    result.missing_macro_dates.to_csv(paths["missing_macro_dates"], index=False)
    return paths
