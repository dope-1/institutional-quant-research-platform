from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalise_ticker_for_yahoo(ticker: str) -> str:
    # Yahoo uses BRK-B while some other vendors use BRK.B.
    return ticker


def _datetime_index_ns(index: pd.Index) -> pd.DatetimeIndex:
    """Return a timezone-naive nanosecond-resolution DatetimeIndex."""
    if isinstance(index, pd.PeriodIndex):
        converted = index.to_timestamp()
    else:
        converted = pd.to_datetime(index)
    result = pd.DatetimeIndex(converted)
    if result.tz is not None:
        result = result.tz_localize(None)
    return result.astype("datetime64[ns]")


def download_market_data_yfinance(
    tickers: Iterable[str],
    start: str,
    end: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV data and return a long-format frame.

    Output schema: date, ticker, open, high, low, close, volume.
    The `end` parameter is made inclusive for user-facing configuration.
    """
    tickers = [_normalise_ticker_for_yahoo(t) for t in tickers]
    inclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=inclusive_end,
        auto_adjust=auto_adjust,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Market data download returned no rows.")

    if isinstance(raw.columns, pd.MultiIndex):
        frames: list[pd.DataFrame] = []
        level0 = set(raw.columns.get_level_values(0))
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(level0):
            for ticker in tickers:
                available = [
                    c
                    for c in ["Open", "High", "Low", "Close", "Volume"]
                    if (c, ticker) in raw.columns
                ]
                if len(available) < 5:
                    continue
                part = raw.loc[:, [(c, ticker) for c in available]].copy()
                part.columns = [c.lower() for c in available]
                part["ticker"] = ticker
                part["date"] = part.index
                frames.append(part.reset_index(drop=True))
        else:
            level1 = set(raw.columns.get_level_values(1))
            if {"Open", "High", "Low", "Close", "Volume"}.issubset(level1):
                for ticker in tickers:
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    part = raw[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                    part.columns = [c.lower() for c in part.columns]
                    part["ticker"] = ticker
                    part["date"] = part.index
                    frames.append(part.reset_index(drop=True))
            else:
                raise RuntimeError("Unexpected yfinance column layout.")
        if not frames:
            raise RuntimeError("No ticker frames could be parsed from yfinance output.")
        out = pd.concat(frames, ignore_index=True)
    else:
        ticker = tickers[0]
        out = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        out.columns = OHLCV_COLUMNS
        out["ticker"] = ticker
        out["date"] = out.index
        out = out.reset_index(drop=True)

    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    out = out[["date", "ticker", *OHLCV_COLUMNS]].sort_values(["date", "ticker"])
    return out.reset_index(drop=True)


def download_fred(series_map: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Download public FRED series with friendly column names."""
    df = pdr.get_data_fred(list(series_map), start=start, end=end)
    df = df.rename(columns=series_map)
    df.index = _datetime_index_ns(df.index)
    df.index.name = "date"
    return df.sort_index()


def download_fama_french_daily(dataset: str, start: str, end: str) -> pd.DataFrame:
    """Download the first table from a Kenneth French daily factor dataset.

    pandas-datareader can return the table with a PeriodIndex. Modern pandas
    requires that index to be converted with ``to_timestamp`` rather than passed
    directly to ``pd.to_datetime``.
    """
    ff = pdr.DataReader(dataset, "famafrench", start=start, end=end)
    table = ff[0].copy()
    table.index = _datetime_index_ns(table.index)
    table.index.name = "date"
    table = table.apply(pd.to_numeric, errors="coerce") / 100.0
    table.columns = [str(c).strip().lower().replace("-", "_") for c in table.columns]
    return table.sort_index()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_data_snapshot(df: pd.DataFrame, path: Path) -> dict[str, object]:
    """Persist a parquet snapshot and return deterministic metadata for a manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return {
        "path": str(path),
        "rows": int(len(df)),
        "sha256": sha256_file(path),
    }


def write_manifest(entries: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
