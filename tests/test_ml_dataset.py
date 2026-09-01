from __future__ import annotations

import pandas as pd

from quant_platform.ml.dataset import (
    build_monthly_ml_partition,
    global_month_end_dates,
    rank_feature_name,
)


def _row(date: str, ticker: str, feature: float, target: float, label_end: str) -> dict:
    return {
        "date": pd.Timestamp(date),
        "ticker": ticker,
        "close": 100.0,
        "label_end_date": pd.Timestamp(label_end),
        "momentum_21d": feature,
        "forward_return_21d": target,
    }


def test_true_month_end_calendar_prevents_false_purged_month_end() -> None:
    full = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-11-28", "2025-12-01", "2025-12-31"]
            )
        }
    )
    month_ends = global_month_end_dates(full)
    assert list(month_ends) == [pd.Timestamp("2025-11-28"), pd.Timestamp("2025-12-31")]

    partition = pd.DataFrame(
        [
            _row("2025-11-28", "A", 0.1, 0.01, "2025-12-20"),
            _row("2025-11-28", "B", 0.2, 0.02, "2025-12-20"),
            _row("2025-12-01", "A", 0.3, 0.03, "2025-12-30"),
            _row("2025-12-01", "B", 0.4, 0.04, "2025-12-30"),
        ]
    )
    result = build_monthly_ml_partition(
        partition,
        true_month_ends=month_ends,
        feature_columns=["momentum_21d"],
        target_column="forward_return_21d",
        min_assets=2,
    )
    assert result["date"].nunique() == 1
    assert result["date"].iloc[0] == pd.Timestamp("2025-11-28")


def test_cross_sectional_features_and_target_are_ranked_and_centered() -> None:
    rows = [
        _row("2020-01-31", "A", 10.0, -0.1, "2020-03-02"),
        _row("2020-01-31", "B", 20.0, 0.0, "2020-03-02"),
        _row("2020-01-31", "C", 30.0, 0.1, "2020-03-02"),
    ]
    result = build_monthly_ml_partition(
        pd.DataFrame(rows),
        true_month_ends=[pd.Timestamp("2020-01-31")],
        feature_columns=["momentum_21d"],
        target_column="forward_return_21d",
        min_assets=3,
    ).set_index("ticker")

    rank_col = rank_feature_name("momentum_21d")
    assert result.loc["A", rank_col] < result.loc["B", rank_col] < result.loc["C", rank_col]
    assert result.loc["A", "target_rank"] < result.loc["B", "target_rank"] < result.loc["C", "target_rank"]
    assert result[rank_col].between(-0.5, 0.5).all()
    assert result["target_rank"].between(-0.5, 0.5).all()
