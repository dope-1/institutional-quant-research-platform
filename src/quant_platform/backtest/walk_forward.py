from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def expanding_windows(
    dates: pd.DatetimeIndex,
    min_train_years: int = 5,
    test_months: int = 12,
) -> list[WalkForwardWindow]:
    """Create chronological expanding-train windows for later ML milestones."""
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    if dates.empty:
        return []
    first = dates.min()
    last = dates.max()
    train_end = first + pd.DateOffset(years=min_train_years)
    windows: list[WalkForwardWindow] = []
    while train_end < last:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), last)
        windows.append(WalkForwardWindow(first, train_end, test_start, test_end))
        train_end = test_end
    return windows
