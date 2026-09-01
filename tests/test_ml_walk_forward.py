from __future__ import annotations

import pandas as pd

from quant_platform.ml.evaluation import ic_summary, rank_ic_by_date
from quant_platform.ml.portfolio import prediction_target_weights
from quant_platform.ml.walk_forward import walk_forward_ridge_predictions


def test_walk_forward_uses_only_fully_matured_labels() -> None:
    history_rows = []
    for ticker, value in zip(["A", "B", "C"], [-0.3, 0.0, 0.3], strict=True):
        history_rows.append(
            {
                "date": pd.Timestamp("2020-01-31"),
                "ticker": ticker,
                "label_end_date": pd.Timestamp("2020-02-28"),
                "rank__x": value,
                "target_rank": value,
                "forward_return_21d": value,
            }
        )
        history_rows.append(
            {
                "date": pd.Timestamp("2020-02-28"),
                "ticker": ticker,
                # Equal to prediction date: must NOT be usable.
                "label_end_date": pd.Timestamp("2020-03-31"),
                "rank__x": value,
                "target_rank": value,
                "forward_return_21d": value,
            }
        )
    evaluation = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-03-31"),
                "ticker": ticker,
                "label_end_date": pd.Timestamp("2020-04-30"),
                "rank__x": value,
                "target_rank": value,
                "forward_return_21d": value,
            }
            for ticker, value in zip(["A", "B", "C"], [-0.25, 0.0, 0.25], strict=True)
        ]
    )
    preds = walk_forward_ridge_predictions(
        pd.DataFrame(history_rows),
        evaluation,
        feature_columns=["rank__x"],
        alpha=1.0,
        min_training_rows=3,
    )
    assert preds["training_rows"].unique().tolist() == [3]
    assert (preds["max_training_label_end"] < preds["date"]).all()


def test_rank_ic_is_one_for_perfect_ordering() -> None:
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")] * 4,
            "prediction": [1.0, 2.0, 3.0, 4.0],
            "forward_return_21d": [-0.1, 0.0, 0.1, 0.2],
        }
    )
    ic = rank_ic_by_date(predictions, min_assets=4)
    assert ic.loc[0, "rank_ic"] == 1.0
    summary = ic_summary(ic)
    assert summary["mean_rank_ic"] == 1.0
    assert summary["positive_rank_ic_fraction"] == 1.0


def test_prediction_targets_select_only_highest_scores() -> None:
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")] * 5,
            "ticker": ["A", "B", "C", "D", "E"],
            "prediction": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    targets = prediction_target_weights(
        predictions, top_fraction=0.4, min_assets=5
    )
    assert set(targets["ticker"]) == {"A", "B"}
    assert targets["target_weight"].sum() == 1.0
