from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, alpha: float = 0.95) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float(-np.quantile(clean, 1.0 - alpha))


def historical_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    cutoff = np.quantile(clean, 1.0 - alpha)
    tail = clean[clean <= cutoff]
    return float(-tail.mean()) if not tail.empty else float("nan")
