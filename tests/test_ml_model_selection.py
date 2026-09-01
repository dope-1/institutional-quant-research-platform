from __future__ import annotations

import pandas as pd

from quant_platform.ml.walk_forward import select_ridge_alpha


def _monthly(date: str, label_end: str, scale: float) -> list[dict[str, object]]:
    rows = []
    for i, ticker in enumerate(["A", "B", "C", "D", "E"], start=1):
        x = (i - 3) / 5
        rows.append(
            {
                "date": pd.Timestamp(date),
                "ticker": ticker,
                "label_end_date": pd.Timestamp(label_end),
                "rank__x": x,
                "target_rank": scale * x,
                "forward_return_21d": scale * x,
            }
        )
    return rows


def test_alpha_selection_returns_grid_member_and_validation_predictions() -> None:
    train = pd.DataFrame(
        _monthly("2020-01-31", "2020-02-28", 1.0)
        + _monthly("2020-02-28", "2020-03-31", 1.0)
        + _monthly("2020-03-31", "2020-04-30", 1.0)
    )
    validation = pd.DataFrame(
        _monthly("2020-05-29", "2020-06-30", 1.0)
        + _monthly("2020-06-30", "2020-07-31", 1.0)
    )
    result = select_ridge_alpha(
        train,
        validation,
        feature_columns=["rank__x"],
        target_column="target_rank",
        actual_return_column="forward_return_21d",
        alpha_grid=[0.1, 1.0, 10.0],
        min_training_rows=10,
        min_assets_for_ic=5,
    )
    assert result.best_alpha in {0.1, 1.0, 10.0}
    assert len(result.table) == 3
    assert result.predictions["date"].nunique() == 2
