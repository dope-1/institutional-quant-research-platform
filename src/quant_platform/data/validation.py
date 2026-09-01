from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class ValidationResult:
    rows: int
    tickers: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_close_rows: int


def validate_market_data(
    df: pd.DataFrame,
    *,
    price_rtol: float = 1e-12,
    price_atol: float = 1e-10,
) -> ValidationResult:
    """Validate OHLCV data while tolerating machine-precision rounding noise.

    Adjusted vendor data can produce values that differ by ~1e-14 even when, for
    example, Close and High are mathematically equal. Those are not real range
    violations, so numerical closeness is accepted while genuine inconsistencies
    are still rejected.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing market-data columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Market data is empty.")

    work = df.copy()
    if work.duplicated(["date", "ticker"]).any():
        dupes = int(work.duplicated(["date", "ticker"]).sum())
        raise ValueError(f"Found {dupes} duplicate date/ticker rows.")

    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    nonpositive_prices = (work[["open", "high", "low", "close"]].dropna() <= 0).any(axis=1)
    if nonpositive_prices.any():
        raise ValueError("Found non-positive OHLC prices.")

    upper_reference = work[["open", "close", "low"]].max(axis=1)
    lower_reference = work[["open", "close", "high"]].min(axis=1)

    high_close = np.isclose(
        work["high"].to_numpy(dtype=float),
        upper_reference.to_numpy(dtype=float),
        rtol=price_rtol,
        atol=price_atol,
        equal_nan=True,
    )
    low_close = np.isclose(
        work["low"].to_numpy(dtype=float),
        lower_reference.to_numpy(dtype=float),
        rtol=price_rtol,
        atol=price_atol,
        equal_nan=True,
    )
    bad_high = (work["high"] < upper_reference) & ~high_close
    bad_low = (work["low"] > lower_reference) & ~low_close
    bad_range = (bad_high | bad_low).fillna(False)

    if bad_range.any():
        count = int(bad_range.sum())
        sample = work.loc[
            bad_range,
            ["date", "ticker", "open", "high", "low", "close"],
        ].head(5)
        raise ValueError(
            f"Found {count} genuinely inconsistent OHLC ranges. "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    if (work["volume"].dropna() < 0).any():
        raise ValueError("Found negative volume.")

    dates = pd.to_datetime(work["date"]).astype("datetime64[ns]")
    return ValidationResult(
        rows=len(work),
        tickers=work["ticker"].nunique(),
        start=dates.min(),
        end=dates.max(),
        missing_close_rows=int(work["close"].isna().sum()),
    )
