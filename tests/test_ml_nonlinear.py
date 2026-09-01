from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.ml.nonlinear import (
    build_nonlinear_model,
    native_feature_importance,
    select_nonlinear_configuration,
    walk_forward_nonlinear_predictions,
)


def _frame(dates: list[str], rows_per_date: int = 30, *, noisy: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    rows: list[dict[str, object]] = []
    for date_idx, date_text in enumerate(dates):
        date = pd.Timestamp(date_text)
        x = np.linspace(-0.5, 0.5, rows_per_date)
        z = rng.normal(0.0, 0.25, rows_per_date)
        for i in range(rows_per_date):
            target = 0.8 * x[i] + (rng.normal(0, 0.02) if noisy else 0.0)
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{i:02d}",
                    "label_end_date": date + pd.Timedelta(days=10),
                    "signal_rank": x[i],
                    "noise_rank": z[i],
                    "target_rank": target,
                    "forward_return_21d": target + 0.01 * date_idx,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("random_forest", {"n_estimators": 8, "max_depth": 2, "min_samples_leaf": 2, "n_jobs": 1}),
        ("xgboost", {"n_estimators": 8, "max_depth": 2, "min_child_weight": 1.0, "n_jobs": 1}),
        ("lightgbm", {"n_estimators": 8, "num_leaves": 5, "min_child_samples": 2, "n_jobs": 1}),
    ],
)
def test_nonlinear_builders_fit_and_predict(model_name: str, params: dict[str, object]) -> None:
    x = pd.DataFrame({"a": [-0.5, -0.2, 0.1, 0.4, 0.5], "b": [0.2, 0.1, 0.0, -0.1, -0.2]})
    y = pd.Series([-0.4, -0.15, 0.08, 0.35, 0.45])
    model = build_nonlinear_model(model_name, params, random_seed=7)
    model.fit(x, y)
    pred = model.predict(x)
    assert len(pred) == len(x)
    assert np.isfinite(pred).all()


def test_walk_forward_nonlinear_predictions_respect_label_maturity() -> None:
    history = _frame(["2020-01-31", "2020-02-29", "2020-03-31"])
    evaluation = _frame(["2020-05-29", "2020-06-30"])
    preds = walk_forward_nonlinear_predictions(
        history,
        evaluation,
        model_name="random_forest",
        feature_columns=["signal_rank"],
        target_column="target_rank",
        params={"n_estimators": 10, "max_depth": 2, "min_samples_leaf": 2, "n_jobs": 1},
        candidate_name="tiny",
        min_training_rows=60,
        random_seed=5,
    )
    assert preds["date"].nunique() == 2
    assert (preds["max_training_label_end"] < preds["date"]).all()
    assert np.isfinite(preds["prediction"]).all()


def test_nonlinear_selection_prefers_validation_signal_feature_group() -> None:
    train = _frame(list(pd.date_range("2019-01-31", periods=8, freq="ME").strftime("%Y-%m-%d")), noisy=True)
    validation = _frame(list(pd.date_range("2020-01-31", periods=4, freq="ME").strftime("%Y-%m-%d")), noisy=True)
    result = select_nonlinear_configuration(
        train,
        validation,
        model_name="random_forest",
        feature_groups={"signal": ["signal_rank"], "noise": ["noise_rank"]},
        candidates=[
            {
                "name": "small",
                "params": {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 2, "n_jobs": 1},
            }
        ],
        target_column="target_rank",
        actual_return_column="forward_return_21d",
        min_training_rows=100,
        min_assets_for_ic=20,
        random_seed=11,
    )
    assert result.selected_group == "signal"
    assert result.table.iloc[0]["feature_group"] == "signal"
    assert result.validation_predictions["date"].nunique() == 4


def test_native_feature_importance_is_normalized() -> None:
    x = pd.DataFrame(
        {
            "signal": np.linspace(-0.5, 0.5, 100),
            "noise": np.random.default_rng(9).normal(size=100),
        }
    )
    y = x["signal"] ** 2 + 0.3 * x["signal"]
    model = build_nonlinear_model(
        "random_forest",
        {"n_estimators": 25, "max_depth": 4, "min_samples_leaf": 2, "n_jobs": 1},
        random_seed=9,
    )
    model.fit(x, y)
    importance = native_feature_importance(model, ["signal", "noise"])
    assert np.isclose(importance["normalized_importance"].sum(), 1.0)
    assert importance.iloc[0]["feature"] == "signal"
