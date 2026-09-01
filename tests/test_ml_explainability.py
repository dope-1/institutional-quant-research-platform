from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from quant_platform.ml.explainability import (
    interaction_grid,
    permutation_importance_table,
    shap_long_table,
    summarize_shap_importance,
    summarize_yearly_shap_importance,
    tree_shap_values,
)


def _data(n: int = 120) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(7)
    x1 = np.linspace(-0.5, 0.5, n)
    x2 = rng.uniform(-0.5, 0.5, n)
    y = 0.9 * x1 + 0.05 * rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "ticker": [f"T{i:03d}" for i in range(n)],
            "rank__signal": x1,
            "rank__noise": x2,
            "target_rank": y,
            "forward_return_21d": y / 10,
        }
    )
    return frame, ["rank__signal", "rank__noise"]


def test_tree_shap_values_and_summaries_are_well_formed():
    frame, features = _data()
    model = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=2).fit(
        frame[features], frame["target_rank"]
    )
    values = tree_shap_values(model, frame[features])
    assert values.shape == (len(frame), len(features))
    long = shap_long_table(frame, feature_columns=features, values=values, model_name="rf")
    summary = summarize_shap_importance(long)
    yearly = summarize_yearly_shap_importance(long)
    assert set(summary["feature"]) == set(features)
    assert np.isclose(summary["normalized_mean_abs_shap"].sum(), 1.0)
    assert not yearly.empty


def test_permutation_importance_favours_signal_feature():
    frame, features = _data()
    model = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=3).fit(
        frame[features], frame["target_rank"]
    )
    table = permutation_importance_table(
        model,
        frame,
        feature_columns=features,
        target_column="target_rank",
        model_name="rf",
        repeats=5,
        seed=9,
    )
    assert table.iloc[0]["feature"] == "rank__signal"


def test_interaction_grid_uses_centered_rank_quartiles():
    frame, _ = _data(40)
    frame["shap_signal"] = frame["rank__signal"] * 0.2
    grid = interaction_grid(
        frame,
        feature_x="rank__signal",
        feature_y="rank__noise",
        shap_x="shap_signal",
    )
    assert set(grid["x_bin"].astype(str)) <= {"Q1_low", "Q2", "Q3", "Q4_high"}
    assert grid["observations"].sum() == len(frame)
