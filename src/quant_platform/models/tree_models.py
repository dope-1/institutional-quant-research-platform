from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def build_random_forest(random_state: int = 42) -> RandomForestRegressor:
    """Milestone-2 ready tree baseline; intentionally conservative defaults."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=random_state,
    )
