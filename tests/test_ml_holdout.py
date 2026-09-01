from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.ml.holdout import (
    build_monthly_online_partition,
    combine_incremental_market,
    completed_month_end_dates,
    load_frozen_holdout_spec,
    matured_holdout_predictions,
    truncate_market_before_as_of,
)


def test_incremental_merge_does_not_revise_frozen_history():
    old = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-31", "2026-01-02"]),
            "ticker": ["A", "A"],
            "close": [100.0, 999.0],
        }
    )
    new = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-31", "2026-01-02"]),
            "ticker": ["A", "A"],
            "close": [777.0, 101.0],
        }
    )
    combined = combine_incremental_market(old, new, holdout_start="2026-01-01")
    assert combined.loc[combined["date"] == pd.Timestamp("2025-12-31"), "close"].iloc[0] == 100.0
    assert combined.loc[combined["date"] == pd.Timestamp("2026-01-02"), "close"].iloc[0] == 101.0


def test_completed_month_ends_exclude_partial_current_month():
    dates = pd.to_datetime(
        ["2026-07-30", "2026-07-31", "2026-08-28", "2026-08-31", "2026-09-01"]
    )
    panel = pd.DataFrame({"date": dates, "ticker": "A"})
    ends = completed_month_end_dates(panel, as_of_date="2026-09-01")
    assert pd.Timestamp("2026-07-31") in ends
    assert pd.Timestamp("2026-08-31") in ends
    assert pd.Timestamp("2026-09-01") not in ends


def test_online_partition_retains_unmatured_signal_for_prediction():
    rows = []
    for ticker, f1, target in [("A", 1.0, 0.02), ("B", 2.0, np.nan), ("C", 3.0, np.nan)]:
        rows.append(
            {
                "date": pd.Timestamp("2026-08-31"),
                "ticker": ticker,
                "label_end_date": pd.NaT,
                "forward_return_21d": target,
                "feature": f1,
            }
        )
    panel = pd.DataFrame(rows)
    out = build_monthly_online_partition(
        panel,
        month_ends=pd.DatetimeIndex([pd.Timestamp("2026-08-31")]),
        feature_columns=["feature"],
        target_column="forward_return_21d",
        min_assets=3,
    )
    assert len(out) == 3
    assert out["rank__feature"].notna().all()
    assert out["target_rank"].isna().all()


def test_matured_holdout_predictions_filters_activation_and_unmatured_rows():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-31", "2026-01-30", "2026-08-31"]),
            "ticker": ["A", "A", "A"],
            "prediction": [0.1, 0.2, 0.3],
            "forward_return_21d": [0.01, 0.02, np.nan],
        }
    )
    out = matured_holdout_predictions(frame, holdout_start="2026-01-01")
    assert list(out["date"]) == [pd.Timestamp("2026-01-30")]


def test_frozen_spec_is_loaded_from_prior_summary_files(tmp_path: Path):
    m2b = {
        "milestone": "2B",
        "selected_rank_features": ["rank__momentum_63d"],
        "selected_alpha": 10.0,
        "selected_portfolio_variant": {"name": "top10_equal", "kind": "equal_weight_top", "top_fraction": 0.1},
    }
    selected_models = {}
    for name in ("random_forest", "xgboost", "lightgbm"):
        selected_models[name] = {
            "feature_group": "all_features",
            "rank_features": ["rank__momentum_63d"],
            "candidate": f"{name}_candidate",
            "params": {"depth": 2},
            "portfolio_variant": {"name": "top20_equal", "kind": "equal_weight_top", "top_fraction": 0.2},
        }
    m2c = {"milestone": "2C", "selected_models": selected_models}
    m3c = {
        "milestone": "3C",
        "selected_optimizer_candidates": {
            name: {"name": "conservative", "risk_aversion": 8.0, "turnover_penalty": 0.04, "turnover_cap": 0.5}
            for name in ("ridge", "random_forest", "xgboost", "lightgbm")
        },
        "primary_validation_selected_alpha_model": "ridge",
    }
    paths = []
    for i, obj in enumerate((m2b, m2c, m3c), start=1):
        path = tmp_path / f"m{i}.json"
        path.write_text(json.dumps(obj), encoding="utf-8")
        paths.append(path)
    spec = load_frozen_holdout_spec(*paths)
    assert spec.ridge_alpha == 10.0
    assert spec.nonlinear_models["random_forest"]["feature_group"] == "all_features"
    assert spec.optimizer_candidates["ridge"]["name"] == "conservative"
    assert spec.primary_optimizer_alpha_model == "ridge"


def test_as_of_cutoff_excludes_in_progress_session_before_validation():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-31", "2026-09-01"]),
            "ticker": ["BLK", "BLK"],
            "open": [1120.0, 1142.01],
            "high": [1135.0, 1141.99],  # intentionally impossible on the as-of day
            "low": [1110.0, 1128.70],
            "close": [1130.0, 1137.94],
            "volume": [1000.0, 500.0],
        }
    )
    out = truncate_market_before_as_of(frame, as_of_date="2026-09-01")
    assert list(out["date"].dt.normalize().unique()) == [pd.Timestamp("2026-08-31")]
    assert (out["date"] < pd.Timestamp("2026-09-01")).all()
