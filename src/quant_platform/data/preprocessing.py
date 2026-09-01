from __future__ import annotations

import pandas as pd


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative cleaning without inventing missing prices."""
    out = df.copy()
    out["date"] = (
        pd.to_datetime(out["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    )
    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    out = out.dropna(subset=["close"])
    out["volume"] = out["volume"].fillna(0.0)
    out["return_1d"] = out.groupby("ticker", sort=False)["close"].pct_change(fill_method=None)
    return out.reset_index(drop=True)


def add_forward_return_label(df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Add future return target. This column must never be used as an input feature."""
    out = df.copy()
    g = out.groupby("ticker", sort=False)["close"]
    out[f"forward_return_{horizon}d"] = g.shift(-horizon) / out["close"] - 1.0
    return out


def strict_time_split(
    df: pd.DataFrame,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> dict[str, pd.DataFrame]:
    """Simple chronological split retained for diagnostics/backward compatibility."""
    dates = pd.to_datetime(df["date"])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(validation_end)
    test_end_ts = pd.Timestamp(test_end)
    if not train_end_ts < val_end_ts < test_end_ts:
        raise ValueError("Expected train_end < validation_end < test_end.")

    return {
        "train": df.loc[dates <= train_end_ts].copy(),
        "validation": df.loc[(dates > train_end_ts) & (dates <= val_end_ts)].copy(),
        "test": df.loc[(dates > val_end_ts) & (dates <= test_end_ts)].copy(),
    }


def purged_time_split(
    df: pd.DataFrame,
    train_end: str,
    validation_end: str,
    test_end: str,
    horizon: int,
) -> dict[str, pd.DataFrame]:
    """Chronological split whose labels cannot cross into the next partition.

    For each security we calculate the date of the price used to form the
    ``horizon``-day forward-return label. A row is retained only if that label end
    date remains inside its own partition. This prevents late-training labels from
    using validation-period prices, and late-validation labels from using test
    prices.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(validation_end)
    test_end_ts = pd.Timestamp(test_end)
    if not train_end_ts < val_end_ts < test_end_ts:
        raise ValueError("Expected train_end < validation_end < test_end.")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"]).astype("datetime64[ns]")
    work = work.sort_values(["ticker", "date"])
    work["_label_end_date"] = work.groupby("ticker", sort=False)["date"].shift(-horizon)

    dates = work["date"]
    label_end = work["_label_end_date"]
    masks = {
        "train": (dates <= train_end_ts) & (label_end <= train_end_ts),
        "validation": (
            (dates > train_end_ts)
            & (dates <= val_end_ts)
            & (label_end <= val_end_ts)
        ),
        "test": (
            (dates > val_end_ts)
            & (dates <= test_end_ts)
            & (label_end <= test_end_ts)
        ),
    }
    return {
        name: work.loc[mask].drop(columns="_label_end_date").copy()
        for name, mask in masks.items()
    }
