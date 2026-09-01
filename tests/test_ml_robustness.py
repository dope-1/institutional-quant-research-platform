from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.ml.portfolio import (
    buffered_prediction_target_weights,
    rank_weighted_prediction_target_weights,
)
from quant_platform.ml.robustness import (
    bootstrap_mean_ci,
    mean_t_stat,
    statistical_summary,
    top_bottom_spread_by_date,
    yearly_ic_summary,
)
from quant_platform.ml.walk_forward import (
    coefficient_stability_summary,
    walk_forward_ridge_coefficients,
)


def test_bootstrap_mean_ci_is_deterministic_and_contains_mean():
    values = pd.Series([0.01, 0.03, -0.02, 0.04, 0.02, 0.01])
    a = bootstrap_mean_ci(values, n_bootstrap=500, seed=7)
    b = bootstrap_mean_ci(values, n_bootstrap=500, seed=7)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert np.isfinite(mean_t_stat(values))


def test_top_bottom_spread_and_yearly_ic_summary():
    deciles = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-01-31", "2025-01-31", "2025-01-31"]),
            "bucket": [1, 10, 1, 10],
            "mean_forward_return": [0.01, 0.03, -0.01, 0.02],
            "n_assets": [6, 6, 6, 6],
        }
    )
    spread = top_bottom_spread_by_date(deciles)
    assert np.allclose(spread["spread"], [0.02, 0.03])

    ic = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-29", "2025-01-31"]),
            "rank_ic": [0.1, -0.05, 0.2],
            "n_assets": [60, 60, 60],
        }
    )
    yearly = yearly_ic_summary(ic)
    assert list(yearly["year"]) == [2024, 2025]
    summary = statistical_summary(ic, spread, n_bootstrap=500, seed=3)
    assert summary["rank_ic"]["months"] == 3.0
    assert summary["top_bottom_spread"]["months"] == 2


def test_buffered_portfolio_retains_holding_inside_exit_buffer():
    tickers = list("ABCDEFGHIJ")
    first = pd.DataFrame(
        {
            "date": pd.Timestamp("2025-01-31"),
            "ticker": tickers,
            "prediction": np.arange(10, 0, -1, dtype=float),
        }
    )
    # A falls from rank 1 to rank 3. With target top-20% (2 names) and an
    # exit buffer of top-40% (4 names), A should be retained alongside B.
    second_order = ["B", "C", "A", "D", "E", "F", "G", "H", "I", "J"]
    second = pd.DataFrame(
        {
            "date": pd.Timestamp("2025-02-28"),
            "ticker": second_order,
            "prediction": np.arange(10, 0, -1, dtype=float),
        }
    )
    targets = buffered_prediction_target_weights(
        pd.concat([first, second], ignore_index=True),
        top_fraction=0.20,
        exit_fraction=0.40,
        min_assets=5,
    )
    feb = targets.loc[targets["date"] == pd.Timestamp("2025-02-28")]
    assert set(feb["ticker"]) == {"A", "B"}
    assert np.isclose(feb["target_weight"].sum(), 1.0)


def test_rank_weighted_targets_sum_to_one_and_favour_best_rank():
    predictions = pd.DataFrame(
        {
            "date": pd.Timestamp("2025-01-31"),
            "ticker": list("ABCDE"),
            "prediction": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    targets = rank_weighted_prediction_target_weights(
        predictions, top_fraction=0.60, min_assets=5
    )
    assert np.isclose(targets["target_weight"].sum(), 1.0)
    weights = targets.set_index("ticker")["target_weight"]
    assert weights["A"] > weights["B"] > weights["C"]


def _monthly_frame(dates: list[str], rows_per_date: int = 20) -> pd.DataFrame:
    rows = []
    for date_idx, date_text in enumerate(dates):
        date = pd.Timestamp(date_text)
        for i in range(rows_per_date):
            x = (i - rows_per_date / 2) / rows_per_date
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{i:02d}",
                    "label_end_date": date + pd.Timedelta(days=10),
                    "x_rank": x,
                    "target_rank": 0.7 * x + 0.01 * date_idx,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_coefficients_respect_label_maturity():
    history = _monthly_frame(["2020-01-31", "2020-02-29", "2020-03-31"])
    evaluation = _monthly_frame(["2020-05-31", "2020-06-30"])
    coefficients = walk_forward_ridge_coefficients(
        history,
        evaluation,
        feature_columns=["x_rank"],
        target_column="target_rank",
        alpha=0.1,
        min_training_rows=40,
    )
    assert len(coefficients) == 2
    assert (coefficients["max_training_label_end"] < coefficients["date"]).all()
    stability = coefficient_stability_summary(coefficients)
    assert stability.loc[0, "dominant_sign_fraction"] == 1.0


def test_feature_ablation_selects_validation_signal_group():
    from quant_platform.ml.robustness import select_feature_group_and_alpha

    rng = np.random.default_rng(11)
    train_rows = []
    val_rows = []
    for partition, dates in [
        (train_rows, pd.date_range("2019-01-31", periods=8, freq="ME")),
        (val_rows, pd.date_range("2020-01-31", periods=4, freq="ME")),
    ]:
        for date in dates:
            x = np.linspace(-0.5, 0.5, 30)
            z = rng.normal(0, 0.2, 30)
            for i in range(30):
                partition.append(
                    {
                        "date": date,
                        "ticker": f"T{i:02d}",
                        "label_end_date": date + pd.Timedelta(days=10),
                        "signal_rank": x[i],
                        "noise_rank": z[i],
                        "target_rank": x[i],
                        "forward_return_21d": x[i] + rng.normal(0, 0.01),
                    }
                )
    result = select_feature_group_and_alpha(
        pd.DataFrame(train_rows),
        pd.DataFrame(val_rows),
        feature_groups={"signal": ["signal_rank"], "noise": ["noise_rank"]},
        target_column="target_rank",
        actual_return_column="forward_return_21d",
        alpha_grid=[0.01, 1.0],
        min_training_rows=100,
        min_assets_for_ic=20,
    )
    assert result.selected_group == "signal"
    assert result.table.iloc[0]["feature_group"] == "signal"
