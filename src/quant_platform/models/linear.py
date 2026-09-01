from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_model(alpha: float = 1.0) -> Pipeline:
    """Milestone-2 ready baseline; intentionally not fitted in Milestone 1."""
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
