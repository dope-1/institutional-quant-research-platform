import numpy as np
import pandas as pd

from quant_platform.data.preprocessing import add_forward_return_label, clean_market_data
from quant_platform.evaluation.audit import (
    infer_feature_columns,
    label_alignment_mismatches,
    run_research_audit,
)
from quant_platform.features.liquidity import add_liquidity_features
from quant_platform.features.momentum import add_momentum_features
from quant_platform.features.volatility import add_volatility_features


def make_panel(n=320):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    for j, ticker in enumerate(["AAA", "BBB", "CCC"]):
        close = 100 * np.cumprod(1 + 0.0002 + 0.0001 * np.sin(np.arange(n) / 13 + j))
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000 + 10_000 * j,
                }
            )
        )
    out = clean_market_data(pd.concat(rows, ignore_index=True))
    out = add_momentum_features(out, [21, 63, 126, 252])
    out = add_volatility_features(out, [21, 63])
    out = add_liquidity_features(out, 21)
    return add_forward_return_label(out, 21)


def test_target_is_excluded_from_inferred_features():
    panel = make_panel()
    features = infer_feature_columns(panel)
    assert "forward_return_21d" not in features
    assert "momentum_12_1" in features


def test_label_alignment_check_detects_corruption():
    panel = make_panel()
    assert label_alignment_mismatches(panel, 21) == 0
    panel.loc[100, "forward_return_21d"] += 0.5
    assert label_alignment_mismatches(panel, 21) == 1


def test_research_audit_passes_well_formed_panel():
    result = run_research_audit(make_panel(), horizon=21)
    assert result.summary["audit_pass"] is True
    assert result.summary["duplicate_date_ticker_rows"] == 0
    assert result.summary["label_alignment_mismatches"] == 0
    assert len(result.ticker_coverage) == 3
    assert not result.feature_diagnostics.empty
