from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from rich.console import Console

from quant_platform.api.artifacts import ArtifactRepository
from quant_platform.backtest.engine import run_target_weight_backtest
from quant_platform.data.loader import (
    download_fama_french_daily,
    download_fred,
    download_market_data_yfinance,
    sha256_file,
    write_data_snapshot,
    write_manifest,
)
from quant_platform.data.preprocessing import (
    add_forward_return_label,
    clean_market_data,
    purged_time_split,
    strict_time_split,
)
from quant_platform.data.validation import validate_market_data
from quant_platform.evaluation.audit import run_research_audit, write_audit_artifacts
from quant_platform.evaluation.audit_plots import (
    write_annual_strategy_returns_plot,
    write_cross_sectional_coverage_plot,
)
from quant_platform.evaluation.metrics import performance_metrics
from quant_platform.evaluation.reports import write_equity_curve_plot, write_metrics_report
from quant_platform.features import (
    add_liquidity_features,
    add_momentum_features,
    add_volatility_features,
    merge_factors_same_day,
    merge_macro_asof,
)
from quant_platform.ml.dataset import (
    add_label_end_dates,
    build_monthly_ml_partition,
    global_month_end_dates,
    rank_feature_name,
)
from quant_platform.ml.evaluation import decile_forward_returns, ic_summary, rank_ic_by_date
from quant_platform.ml.explainability import (
    interaction_grid,
    permutation_importance_table,
    shap_interaction_importance,
    shap_long_table,
    summarize_shap_importance,
    summarize_yearly_shap_importance,
    tree_shap_values,
    yearly_permutation_importance,
)
from quant_platform.ml.explainability_reports import (
    write_global_importance_comparison,
    write_shap_dependence_plot,
    write_top_interactions_plot,
    write_yearly_shap_heatmaps,
)
from quant_platform.ml.holdout import (
    build_monthly_online_partition,
    combine_incremental_market,
    completed_month_end_dates,
    frozen_spec_to_dict,
    load_frozen_holdout_spec,
    matured_holdout_predictions,
    truncate_market_before_as_of,
)
from quant_platform.ml.holdout_reports import (
    write_holdout_equity_curves,
    write_holdout_model_comparison,
    write_holdout_rank_ic,
)
from quant_platform.ml.nonlinear import (
    fit_reference_nonlinear,
    native_feature_importance,
    select_nonlinear_configuration,
    walk_forward_nonlinear_predictions,
)
from quant_platform.ml.optimizer import build_optimized_targets
from quant_platform.ml.optimizer_reports import (
    write_optimizer_concentration,
    write_optimizer_equity_curves,
    write_optimizer_risk_turnover,
    write_optimizer_validation_frontier,
)
from quant_platform.ml.portfolio import (
    buffered_prediction_target_weights,
    prediction_target_weights,
    rank_weighted_prediction_target_weights,
    targets_on_dates,
)
from quant_platform.ml.regime import (
    DEFAULT_REGIME_COLUMNS,
    build_point_in_time_regimes,
    daily_macro_panel,
    holding_period_returns,
    metric_summary_by_regime,
    portfolio_summary_by_regime,
    regime_counts,
    regimes_on_signal_dates,
)
from quant_platform.ml.regime_reports import (
    write_regime_ic_heatmaps,
    write_regime_portfolio_heatmaps,
    write_regime_timeline,
)
from quant_platform.ml.reports import (
    write_ablation_plot,
    write_coefficient_plot,
    write_coefficient_stability_plot,
    write_decile_plot,
    write_json,
    write_model_comparison_equity_curves,
    write_model_rank_ic_comparison,
    write_model_yearly_ic_comparison,
    write_native_importance_plot,
    write_nonlinear_selection_plot,
    write_rank_ic_plot,
    write_test_equity_curves,
    write_turnover_frontier_plot,
    write_yearly_ic_plot,
)
from quant_platform.ml.robustness import (
    select_feature_group_and_alpha,
    statistical_summary,
    top_bottom_spread_by_date,
    yearly_ic_summary,
)
from quant_platform.ml.walk_forward import (
    coefficient_stability_summary,
    fit_reference_ridge,
    select_ridge_alpha,
    walk_forward_ridge_coefficients,
    walk_forward_ridge_predictions,
)
from quant_platform.portfolio.construction import build_target_weight_table

app = typer.Typer(no_args_is_help=True)
console = Console()
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _snapshot_entry(path: Path, rows: int, *, reused: bool) -> dict[str, object]:
    return {
        "path": str(path),
        "rows": int(rows),
        "sha256": sha256_file(path),
        "reused": reused,
    }


def _load_indexed_snapshot(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        raise ValueError(f"Snapshot {path} does not contain a 'date' column.")
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    return frame.set_index("date").sort_index()


def _require_raw(path: Path, label: str) -> None:
    if not path.exists():
        raise typer.BadParameter(
            f"--reuse-raw requested, but {label} snapshot is missing: {path}. "
            "Run once without --reuse-raw to download it."
        )


def _write_milestone1_audit(
    panel: pd.DataFrame,
    horizon: int,
    equal_weight_daily: pd.DataFrame,
    momentum_daily: pd.DataFrame,
    artifacts_dir: Path,
    extreme_return_threshold: float,
    macro_columns: list[str],
) -> None:
    audit_dir = artifacts_dir / "audit"
    result = run_research_audit(
        panel,
        horizon=horizon,
        strategy_daily={
            "equal_weight": equal_weight_daily,
            "momentum_top_quintile": momentum_daily,
        },
        extreme_return_threshold=extreme_return_threshold,
        macro_columns=macro_columns,
    )
    write_audit_artifacts(result, audit_dir)
    write_cross_sectional_coverage_plot(result.cross_sectional_coverage, audit_dir)
    write_annual_strategy_returns_plot(result.annual_strategy_metrics, audit_dir)
    status = "PASS" if result.summary["audit_pass"] else "REVIEW"
    console.print(f"Research audit: [bold]{status}[/bold]. Outputs: {audit_dir}")


@app.command("run-milestone1")
def run_milestone1(
    config_path: Path = typer.Option(ROOT / "configs" / "research.yaml", exists=True),
    reuse_raw: bool = typer.Option(
        False,
        "--reuse-raw",
        help="Reuse existing data/raw parquet snapshots instead of downloading again.",
    ),
) -> None:
    """Build the Milestone 1.2 research panel, purged splits, baselines, and audit."""
    cfg = load_config(config_path)
    research = cfg["research"]
    universe = cfg["universe"]["tickers"]
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    split_dir = processed_dir / "splits"
    artifacts_dir = ROOT / "artifacts" / "milestone1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    market_path = raw_dir / "market_ohlcv.parquet"
    macro_path = raw_dir / "macro_fred.parquet"
    factors_path = raw_dir / "fama_french_factors.parquet"

    console.print("[bold]1/7 Loading market data[/bold]")
    if reuse_raw:
        _require_raw(market_path, "market")
        market = pd.read_parquet(market_path)
        market["date"] = pd.to_datetime(market["date"]).astype("datetime64[ns]")
        console.print(f"Reused raw snapshot: {market_path}")
    else:
        market = download_market_data_yfinance(
            universe,
            research["start_date"],
            research["end_date"],
            auto_adjust=cfg["market_data"].get("auto_adjust", True),
        )
    validation = validate_market_data(market)
    console.print(
        f"Validated {validation.rows:,} rows across {validation.tickers} tickers "
        f"({validation.start.date()} to {validation.end.date()})."
    )

    manifest: dict[str, object] = {
        "config": str(config_path),
        "research_window": {"start": research["start_date"], "end": research["end_date"]},
        "milestone": "1.2",
    }
    if reuse_raw:
        manifest["market"] = _snapshot_entry(market_path, len(market), reused=True)
    else:
        manifest["market"] = write_data_snapshot(market, market_path)

    console.print("[bold]2/7 Loading macro and factor data[/bold]")
    macro = None
    if cfg["macro"].get("enabled", False):
        if reuse_raw:
            _require_raw(macro_path, "macro")
            macro = _load_indexed_snapshot(macro_path)
            manifest["macro"] = _snapshot_entry(macro_path, len(macro), reused=True)
        else:
            macro = download_fred(
                cfg["macro"]["fred_series"], research["start_date"], research["end_date"]
            )
            manifest["macro"] = write_data_snapshot(macro.reset_index(), macro_path)

    factors = None
    if cfg["factors"].get("enabled", False):
        if reuse_raw:
            _require_raw(factors_path, "Fama-French factor")
            factors = _load_indexed_snapshot(factors_path)
            manifest["factors"] = _snapshot_entry(factors_path, len(factors), reused=True)
        else:
            factors = download_fama_french_daily(
                cfg["factors"]["dataset"], research["start_date"], research["end_date"]
            )
            manifest["factors"] = write_data_snapshot(factors.reset_index(), factors_path)

    write_manifest(manifest, raw_dir / "manifest.json")

    console.print("[bold]3/7 Cleaning and engineering point-in-time features[/bold]")
    data = clean_market_data(market)
    data = add_momentum_features(data, cfg["features"]["momentum_windows"])
    data = add_volatility_features(
        data,
        cfg["features"]["volatility_windows"],
        annualization_days=research["annualization_days"],
    )
    data = add_liquidity_features(data, cfg["features"]["liquidity_window"])
    if macro is not None:
        data = merge_macro_asof(data, macro)
    if factors is not None:
        data = merge_factors_same_day(data, factors)
    data = add_forward_return_label(data, research["label_horizon_days"])
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)
    data.to_parquet(processed_dir / "research_panel.parquet", index=False)

    console.print("[bold]4/7 Creating purged chronological splits[/bold]")
    unpurged = strict_time_split(data, **cfg["split"])
    splits = purged_time_split(
        data,
        **cfg["split"],
        horizon=research["label_horizon_days"],
    )
    split_summary: dict[str, object] = {
        "_metadata": {
            "label_horizon_days": int(research["label_horizon_days"]),
            "purge_rule": "A row is retained only when its forward-label end date stays inside its own partition.",
        }
    }
    for name, part in splits.items():
        part = part.sort_values(["date", "ticker"]).reset_index(drop=True)
        part.to_parquet(split_dir / f"{name}.parquet", index=False)
        raw_part = unpurged[name]
        split_summary[name] = {
            "rows": int(len(part)),
            "unpurged_rows": int(len(raw_part)),
            "purged_rows_removed": int(len(raw_part) - len(part)),
            "start": str(part["date"].min().date()) if not part.empty else None,
            "end": str(part["date"].max().date()) if not part.empty else None,
        }
    (processed_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )

    console.print("[bold]5/7 Running drift-aware baseline backtests[/bold]")
    bt_cfg = cfg["backtest"]
    eq_targets = build_target_weight_table(data, "equal_weight", min_assets=bt_cfg["min_assets"])
    mom_targets = build_target_weight_table(
        data,
        "momentum",
        score_col="momentum_12_1",
        top_fraction=bt_cfg["momentum_top_fraction"],
        min_assets=bt_cfg["min_assets"],
    )
    eq = run_target_weight_backtest(
        data, eq_targets, bt_cfg["transaction_cost_bps"], bt_cfg["slippage_bps"]
    )
    mom = run_target_weight_backtest(
        data, mom_targets, bt_cfg["transaction_cost_bps"], bt_cfg["slippage_bps"]
    )

    metrics = {
        "equal_weight": performance_metrics(
            eq.daily["net_return"],
            eq.daily["turnover"],
            annualization_days=research["annualization_days"],
        ),
        "momentum_top_quintile": performance_metrics(
            mom.daily["net_return"],
            mom.daily["turnover"],
            benchmark_returns=eq.daily["net_return"],
            annualization_days=research["annualization_days"],
        ),
    }

    console.print("[bold]6/7 Writing hardened baseline reports[/bold]")
    write_metrics_report(metrics, artifacts_dir)
    write_equity_curve_plot(
        {
            "Equal weight": eq.daily["equity_curve"],
            "Momentum top quintile": mom.daily["equity_curve"],
        },
        artifacts_dir,
    )
    eq.daily.to_parquet(artifacts_dir / "equal_weight_daily.parquet")
    mom.daily.to_parquet(artifacts_dir / "momentum_daily.parquet")

    console.print("[bold]7/7 Auditing anomalies, macro coverage, and baseline stability[/bold]")
    macro_columns = list(cfg.get("macro", {}).get("fred_series", {}).values())
    _write_milestone1_audit(
        data,
        horizon=research["label_horizon_days"],
        equal_weight_daily=eq.daily,
        momentum_daily=mom.daily,
        artifacts_dir=artifacts_dir,
        extreme_return_threshold=cfg.get("audit", {}).get(
            "extreme_abs_daily_return_threshold", 0.25
        ),
        macro_columns=macro_columns,
    )
    console.print(f"[green]Milestone 1.2 complete.[/green] Reports: {artifacts_dir}")


@app.command("audit-milestone1")
def audit_milestone1(
    config_path: Path = typer.Option(ROOT / "configs" / "research.yaml", exists=True),
    panel_path: Path = typer.Option(ROOT / "data" / "processed" / "research_panel.parquet"),
    artifacts_dir: Path = typer.Option(ROOT / "artifacts" / "milestone1"),
) -> None:
    """Re-run the Milestone 1.2 audit without rebuilding the dataset."""
    if not panel_path.exists():
        raise typer.BadParameter(
            f"Research panel not found at {panel_path}. Run 'quant-research run-milestone1' first."
        )
    cfg = load_config(config_path)
    eq_path = artifacts_dir / "equal_weight_daily.parquet"
    mom_path = artifacts_dir / "momentum_daily.parquet"
    if not eq_path.exists() or not mom_path.exists():
        raise typer.BadParameter(
            "Baseline daily result files are missing. Run 'quant-research run-milestone1' first."
        )

    panel = pd.read_parquet(panel_path)
    eq_daily = pd.read_parquet(eq_path)
    mom_daily = pd.read_parquet(mom_path)
    for frame in (eq_daily, mom_daily):
        if "date" in frame.columns:
            frame.set_index("date", inplace=True)
        frame.index = pd.to_datetime(frame.index).astype("datetime64[ns]")

    _write_milestone1_audit(
        panel,
        horizon=cfg["research"]["label_horizon_days"],
        equal_weight_daily=eq_daily,
        momentum_daily=mom_daily,
        artifacts_dir=artifacts_dir,
        extreme_return_threshold=cfg.get("audit", {}).get(
            "extreme_abs_daily_return_threshold", 0.25
        ),
        macro_columns=list(cfg.get("macro", {}).get("fred_series", {}).values()),
    )



def _merge_label_end_dates(split: pd.DataFrame, label_lookup: pd.DataFrame) -> pd.DataFrame:
    out = split.copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    return out.merge(label_lookup, on=["date", "ticker"], how="left", validate="one_to_one")


def _rebased_daily_slice(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sliced = daily.loc[(daily.index >= start) & (daily.index <= end)].copy()
    if sliced.empty:
        raise ValueError(f"No backtest observations between {start.date()} and {end.date()}.")
    sliced["equity_curve"] = (1.0 + sliced["net_return"]).cumprod()
    return sliced


@app.command("run-milestone2a")
def run_milestone2a(
    config_path: Path = typer.Option(ROOT / "configs" / "research.yaml", exists=True),
) -> None:
    """Build monthly cross-sectional Ridge research and an untouched test evaluation."""
    cfg = load_config(config_path)
    research = cfg["research"]
    split_cfg = cfg["split"]
    ml_cfg = cfg.get("ml", {})
    if not ml_cfg:
        raise typer.BadParameter("Missing 'ml' section in research.yaml.")

    panel_path = ROOT / "data" / "processed" / "research_panel.parquet"
    split_dir = ROOT / "data" / "processed" / "splits"
    ml_data_dir = ROOT / "data" / "processed" / "ml"
    artifacts_dir = ROOT / "artifacts" / "milestone2a"
    for path in [panel_path, *(split_dir / f"{name}.parquet" for name in ("train", "validation", "test"))]:
        if not path.exists():
            raise typer.BadParameter(
                f"Required Milestone 1.2 file is missing: {path}. "
                "Run 'quant-research run-milestone1 --reuse-raw' first."
            )
    ml_data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]1/6 Building true month-end cross-sectional ML datasets[/bold]")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    horizon = int(research["label_horizon_days"])
    panel_with_labels = add_label_end_dates(panel, horizon)
    label_lookup = panel_with_labels[["date", "ticker", "label_end_date"]]
    true_month_ends = global_month_end_dates(panel)

    feature_columns = list(ml_cfg["feature_columns"])
    target_column = str(ml_cfg.get("target_column", f"forward_return_{horizon}d"))
    rank_features = [rank_feature_name(feature) for feature in feature_columns]
    monthly: dict[str, pd.DataFrame] = {}
    dataset_summary: dict[str, object] = {
        "feature_columns": feature_columns,
        "rank_feature_columns": rank_features,
        "target_column": target_column,
        "target_transform": "cross-sectional centered percentile rank",
        "signal_frequency": "true calendar month-end trading dates",
    }
    for name in ("train", "validation", "test"):
        split = pd.read_parquet(split_dir / f"{name}.parquet")
        split = _merge_label_end_dates(split, label_lookup)
        part = build_monthly_ml_partition(
            split,
            true_month_ends=true_month_ends,
            feature_columns=feature_columns,
            target_column=target_column,
            min_assets=int(ml_cfg.get("min_assets_per_cross_section", 40)),
        )
        if part.empty:
            raise ValueError(f"Monthly ML partition '{name}' is empty.")
        part.to_parquet(ml_data_dir / f"{name}_monthly.parquet", index=False)
        monthly[name] = part
        dataset_summary[name] = {
            "rows": int(len(part)),
            "months": int(part["date"].nunique()),
            "assets_min": int(part.groupby("date")["ticker"].nunique().min()),
            "assets_median": float(part.groupby("date")["ticker"].nunique().median()),
            "start": str(part["date"].min().date()),
            "end": str(part["date"].max().date()),
        }
    write_json(dataset_summary, ml_data_dir / "dataset_summary.json")

    console.print("[bold]2/6 Selecting Ridge regularisation on validation walk-forward Rank IC[/bold]")
    ridge_cfg = ml_cfg.get("ridge", {})
    selection = select_ridge_alpha(
        monthly["train"],
        monthly["validation"],
        feature_columns=rank_features,
        target_column="target_rank",
        actual_return_column=target_column,
        alpha_grid=[float(x) for x in ridge_cfg.get("alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0])],
        min_training_rows=int(ml_cfg.get("min_training_rows", 1000)),
        min_assets_for_ic=int(ml_cfg.get("min_assets_for_ic", 20)),
    )
    selection.table.to_csv(artifacts_dir / "ridge_model_selection.csv", index=False)
    validation_predictions = selection.predictions
    validation_predictions.to_parquet(
        artifacts_dir / "validation_predictions.parquet", index=False
    )
    console.print(f"Selected Ridge alpha: [bold]{selection.best_alpha:g}[/bold]")

    console.print("[bold]3/6 Running untouched test-period expanding walk-forward predictions[/bold]")
    pretest_history = pd.concat(
        [monthly["train"], monthly["validation"]], ignore_index=True
    ).sort_values(["date", "ticker"])
    test_predictions = walk_forward_ridge_predictions(
        pretest_history,
        monthly["test"],
        feature_columns=rank_features,
        target_column="target_rank",
        alpha=selection.best_alpha,
        min_training_rows=int(ml_cfg.get("min_training_rows", 1000)),
    )
    test_predictions.to_parquet(artifacts_dir / "test_predictions.parquet", index=False)

    validation_ic = rank_ic_by_date(
        validation_predictions,
        actual_column=target_column,
        min_assets=int(ml_cfg.get("min_assets_for_ic", 20)),
    )
    test_ic = rank_ic_by_date(
        test_predictions,
        actual_column=target_column,
        min_assets=int(ml_cfg.get("min_assets_for_ic", 20)),
    )
    validation_ic.to_csv(artifacts_dir / "validation_rank_ic.csv", index=False)
    test_ic.to_csv(artifacts_dir / "test_rank_ic.csv", index=False)

    console.print("[bold]4/6 Evaluating ranking monotonicity and pre-test coefficients[/bold]")
    deciles = decile_forward_returns(
        test_predictions,
        actual_column=target_column,
        n_buckets=int(ml_cfg.get("deciles", 10)),
    )
    deciles.to_csv(artifacts_dir / "test_decile_forward_returns.csv", index=False)

    first_test_date = pd.Timestamp(test_predictions["date"].min())
    reference_model, reference_training = fit_reference_ridge(
        pretest_history,
        before_date=first_test_date,
        feature_columns=rank_features,
        target_column="target_rank",
        alpha=selection.best_alpha,
    )
    coefficients = pd.DataFrame(
        {
            "feature": rank_features,
            "coefficient": reference_model.named_steps["model"].coef_,
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("abs_coefficient", ascending=False)
    coefficients.to_csv(artifacts_dir / "ridge_coefficients_pretest.csv", index=False)

    console.print("[bold]5/6 Backtesting ML vs equal-weight and momentum on identical test signals[/bold]")
    signal_dates = pd.DatetimeIndex(sorted(test_predictions["date"].unique()))
    portfolio_cfg = ml_cfg.get("portfolio", {})
    top_fraction = float(portfolio_cfg.get("top_fraction", 0.20))
    min_assets = int(portfolio_cfg.get("min_assets", 20))
    ml_targets = prediction_target_weights(
        test_predictions,
        top_fraction=top_fraction,
        min_assets=min_assets,
    )
    eq_targets = targets_on_dates(
        panel,
        signal_dates,
        strategy="equal_weight",
        min_assets=min_assets,
    )
    mom_targets = targets_on_dates(
        panel,
        signal_dates,
        strategy="momentum",
        top_fraction=top_fraction,
        min_assets=min_assets,
    )
    bt_cfg = cfg["backtest"]
    ml_bt = run_target_weight_backtest(
        panel, ml_targets, bt_cfg["transaction_cost_bps"], bt_cfg["slippage_bps"]
    )
    eq_bt = run_target_weight_backtest(
        panel, eq_targets, bt_cfg["transaction_cost_bps"], bt_cfg["slippage_bps"]
    )
    mom_bt = run_target_weight_backtest(
        panel, mom_targets, bt_cfg["transaction_cost_bps"], bt_cfg["slippage_bps"]
    )

    market_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    activation_pos = int(market_dates.searchsorted(signal_dates.min(), side="right"))
    if activation_pos >= len(market_dates):
        raise ValueError("No trading date exists after the first test signal.")
    evaluation_start = pd.Timestamp(market_dates[activation_pos])
    evaluation_end = min(pd.Timestamp(split_cfg["test_end"]), pd.Timestamp(market_dates.max()))
    ml_daily = _rebased_daily_slice(ml_bt.daily, evaluation_start, evaluation_end)
    eq_daily = _rebased_daily_slice(eq_bt.daily, evaluation_start, evaluation_end)
    mom_daily = _rebased_daily_slice(mom_bt.daily, evaluation_start, evaluation_end)

    portfolio_metrics = {
        "equal_weight": performance_metrics(
            eq_daily["net_return"],
            eq_daily["turnover"],
            annualization_days=research["annualization_days"],
        ),
        "momentum_top_quintile": performance_metrics(
            mom_daily["net_return"],
            mom_daily["turnover"],
            benchmark_returns=eq_daily["net_return"],
            annualization_days=research["annualization_days"],
        ),
        "ridge_top_quintile": performance_metrics(
            ml_daily["net_return"],
            ml_daily["turnover"],
            benchmark_returns=eq_daily["net_return"],
            annualization_days=research["annualization_days"],
        ),
    }
    write_metrics_report(portfolio_metrics, artifacts_dir / "portfolio")
    ml_daily.to_parquet(artifacts_dir / "portfolio" / "ridge_test_daily.parquet")
    eq_daily.to_parquet(artifacts_dir / "portfolio" / "equal_weight_test_daily.parquet")
    mom_daily.to_parquet(artifacts_dir / "portfolio" / "momentum_test_daily.parquet")

    console.print("[bold]6/6 Writing Milestone 2A research report artifacts[/bold]")
    validation_summary = ic_summary(validation_ic)
    test_summary = ic_summary(test_ic)
    top_bottom = float("nan")
    if not deciles.empty:
        bucket_means = deciles.groupby("bucket")["mean_forward_return"].mean()
        if 1 in bucket_means.index and int(ml_cfg.get("deciles", 10)) in bucket_means.index:
            top_bottom = float(
                bucket_means.loc[int(ml_cfg.get("deciles", 10))] - bucket_means.loc[1]
            )
    summary = {
        "milestone": "2A",
        "research_question": (
            "Can a regularised linear model improve cross-sectional equity return ranking "
            "and test-period portfolio performance relative to hardened baselines?"
        ),
        "best_alpha_selected_on_validation": selection.best_alpha,
        "validation_rank_ic": validation_summary,
        "test_rank_ic": test_summary,
        "test_mean_top_minus_bottom_decile_forward_return": top_bottom,
        "pretest_reference_training_rows": int(len(reference_training)),
        "test_portfolio_window": {
            "start": str(evaluation_start.date()),
            "end": str(evaluation_end.date()),
            "signal_months": int(len(signal_dates)),
        },
        "test_portfolio_metrics": portfolio_metrics,
        "methodology": {
            "features": "stock-specific centered cross-sectional percentile ranks",
            "target": "centered cross-sectional rank of 21-trading-day forward return",
            "validation": "expanding monthly walk-forward; alpha chosen by mean Rank IC",
            "test": "untouched until alpha selection, then expanding monthly walk-forward",
            "label_availability_rule": "training label_end_date must be strictly before signal date",
            "portfolio": "equal-weight long-only top prediction quintile with drift-aware costs",
        },
    }
    write_json(summary, artifacts_dir / "summary.json")
    write_rank_ic_plot(validation_ic, test_ic, artifacts_dir)
    write_decile_plot(deciles, artifacts_dir)
    write_coefficient_plot(coefficients, artifacts_dir)
    write_test_equity_curves(
        {
            "Equal weight": eq_daily["equity_curve"],
            "Momentum top quintile": mom_daily["equity_curve"],
            "Ridge top quintile": ml_daily["equity_curve"],
        },
        artifacts_dir,
    )
    console.print(f"[green]Milestone 2A complete.[/green] Outputs: {artifacts_dir}")



def _targets_for_ml_variant(
    predictions: pd.DataFrame,
    variant: dict[str, object],
    *,
    min_assets: int,
) -> pd.DataFrame:
    kind = str(variant["kind"])
    top_fraction = float(variant.get("top_fraction", 0.20))
    if kind == "equal_weight_top":
        return prediction_target_weights(
            predictions, top_fraction=top_fraction, min_assets=min_assets
        )
    if kind == "buffered_top":
        return buffered_prediction_target_weights(
            predictions,
            top_fraction=top_fraction,
            exit_fraction=float(variant.get("exit_fraction", 0.30)),
            min_assets=min_assets,
        )
    if kind == "rank_weighted_top":
        return rank_weighted_prediction_target_weights(
            predictions, top_fraction=top_fraction, min_assets=min_assets
        )
    raise ValueError(f"Unknown ML portfolio variant kind: {kind}")


def _evaluate_target_table(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    signal_dates: pd.DatetimeIndex,
    evaluation_end: pd.Timestamp,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> pd.DataFrame:
    result = run_target_weight_backtest(
        panel, targets, transaction_cost_bps, slippage_bps
    )
    market_dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    activation_pos = int(market_dates.searchsorted(signal_dates.min(), side="right"))
    if activation_pos >= len(market_dates):
        raise ValueError("No trading date exists after the first portfolio signal.")
    start = pd.Timestamp(market_dates[activation_pos])
    return _rebased_daily_slice(result.daily, start, evaluation_end)


def _yearly_portfolio_metrics(
    daily_by_strategy: dict[str, pd.DataFrame],
    *,
    annualization_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, daily in daily_by_strategy.items():
        frame = daily.copy()
        frame.index = pd.to_datetime(frame.index)
        for year, group in frame.groupby(frame.index.year):
            if len(group) < 2:
                continue
            metrics = performance_metrics(
                group["net_return"],
                group["turnover"],
                annualization_days=annualization_days,
            )
            rows.append({"strategy": strategy, "year": int(year), **metrics})
    return pd.DataFrame(rows)


@app.command("run-milestone2b")
def run_milestone2b(
    config_path: Path = typer.Option(ROOT / "configs" / "research.yaml", exists=True),
) -> None:
    """Validate Ridge signal robustness and select a turnover-aware portfolio using validation only."""
    cfg = load_config(config_path)
    research = cfg["research"]
    split_cfg = cfg["split"]
    ml_cfg = cfg.get("ml", {})
    robustness_cfg = ml_cfg.get("robustness", {})
    if not robustness_cfg:
        raise typer.BadParameter("Missing 'ml.robustness' section in research.yaml.")

    panel_path = ROOT / "data" / "processed" / "research_panel.parquet"
    ml_data_dir = ROOT / "data" / "processed" / "ml"
    artifacts_dir = ROOT / "artifacts" / "milestone2b"
    required = [panel_path, *(ml_data_dir / f"{name}_monthly.parquet" for name in ("train", "validation", "test"))]
    for path in required:
        if not path.exists():
            raise typer.BadParameter(
                f"Required Milestone 2A file is missing: {path}. "
                "Run 'quant-research run-milestone2a' first."
            )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    monthly: dict[str, pd.DataFrame] = {}
    for name in ("train", "validation", "test"):
        part = pd.read_parquet(ml_data_dir / f"{name}_monthly.parquet")
        part["date"] = pd.to_datetime(part["date"]).astype("datetime64[ns]")
        part["label_end_date"] = pd.to_datetime(part["label_end_date"]).astype("datetime64[ns]")
        monthly[name] = part

    target_column = str(ml_cfg.get("target_column", "forward_return_21d"))
    all_raw_features = list(ml_cfg["feature_columns"])
    configured_groups = robustness_cfg.get("feature_groups", {})
    feature_groups: dict[str, list[str]] = {}
    for group_name, raw_features in configured_groups.items():
        missing = sorted(set(raw_features) - set(all_raw_features))
        if missing:
            raise ValueError(f"Feature group '{group_name}' contains unknown features: {missing}")
        feature_groups[str(group_name)] = [rank_feature_name(str(x)) for x in raw_features]

    ridge_cfg = ml_cfg.get("ridge", {})
    alpha_grid = [float(x) for x in ridge_cfg.get("alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0])]
    min_training_rows = int(ml_cfg.get("min_training_rows", 1000))
    min_assets_for_ic = int(ml_cfg.get("min_assets_for_ic", 20))

    console.print("[bold]1/7 Validation-only feature ablation and Ridge alpha selection[/bold]")
    ablation = select_feature_group_and_alpha(
        monthly["train"],
        monthly["validation"],
        feature_groups=feature_groups,
        target_column="target_rank",
        actual_return_column=target_column,
        alpha_grid=alpha_grid,
        min_training_rows=min_training_rows,
        min_assets_for_ic=min_assets_for_ic,
    )
    ablation.table.to_csv(artifacts_dir / "feature_ablation_validation.csv", index=False)
    selected_features = feature_groups[ablation.selected_group]
    validation_predictions = ablation.validation_predictions
    validation_predictions.to_parquet(
        artifacts_dir / "selected_validation_predictions.parquet", index=False
    )
    console.print(
        f"Selected feature group: [bold]{ablation.selected_group}[/bold] | "
        f"alpha: [bold]{ablation.selected_alpha:g}[/bold]"
    )

    console.print("[bold]2/7 Measuring validation coefficient stability[/bold]")
    validation_coefficients = walk_forward_ridge_coefficients(
        monthly["train"],
        monthly["validation"],
        feature_columns=selected_features,
        target_column="target_rank",
        alpha=ablation.selected_alpha,
        min_training_rows=min_training_rows,
    )
    validation_coefficients.to_csv(
        artifacts_dir / "validation_coefficient_history.csv", index=False
    )
    validation_stability = coefficient_stability_summary(validation_coefficients)
    validation_stability.to_csv(
        artifacts_dir / "validation_coefficient_stability.csv", index=False
    )

    console.print("[bold]3/7 Selecting portfolio implementation on validation only[/bold]")
    portfolio_cfg = ml_cfg.get("portfolio", {})
    min_assets = int(portfolio_cfg.get("min_assets", 20))
    variants = list(robustness_cfg.get("portfolio_variants", []))
    if not variants:
        raise ValueError("ml.robustness.portfolio_variants cannot be empty.")
    validation_signal_dates = pd.DatetimeIndex(sorted(validation_predictions["date"].unique()))
    validation_end = min(pd.Timestamp(split_cfg["validation_end"]), pd.Timestamp(panel["date"].max()))
    bt_cfg = cfg["backtest"]
    validation_variant_rows: list[dict[str, object]] = []
    for variant in variants:
        targets = _targets_for_ml_variant(
            validation_predictions, variant, min_assets=min_assets
        )
        daily = _evaluate_target_table(
            panel,
            targets,
            signal_dates=validation_signal_dates,
            evaluation_end=validation_end,
            transaction_cost_bps=float(bt_cfg["transaction_cost_bps"]),
            slippage_bps=float(bt_cfg["slippage_bps"]),
        )
        metrics = performance_metrics(
            daily["net_return"],
            daily["turnover"],
            annualization_days=int(research["annualization_days"]),
        )
        validation_variant_rows.append(
            {"variant": str(variant["name"]), "kind": str(variant["kind"]), **metrics}
        )
    validation_portfolios = pd.DataFrame(validation_variant_rows).sort_values(
        ["sharpe", "annualized_turnover", "variant"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    validation_portfolios.to_csv(
        artifacts_dir / "portfolio_variant_validation.csv", index=False
    )
    if validation_portfolios.empty or not np.isfinite(validation_portfolios.iloc[0]["sharpe"]):
        raise ValueError("No portfolio variant produced a valid validation Sharpe ratio.")
    selected_variant_name = str(validation_portfolios.iloc[0]["variant"])
    selected_variant = next(v for v in variants if str(v["name"]) == selected_variant_name)
    console.print(f"Selected portfolio variant: [bold]{selected_variant_name}[/bold]")

    console.print("[bold]4/7 Freezing specification and running the untouched test period[/bold]")
    pretest_history = pd.concat(
        [monthly["train"], monthly["validation"]], ignore_index=True
    ).sort_values(["date", "ticker"])
    test_predictions = walk_forward_ridge_predictions(
        pretest_history,
        monthly["test"],
        feature_columns=selected_features,
        target_column="target_rank",
        alpha=ablation.selected_alpha,
        min_training_rows=min_training_rows,
    )
    test_predictions.to_parquet(artifacts_dir / "selected_test_predictions.parquet", index=False)
    test_ic = rank_ic_by_date(
        test_predictions,
        actual_column=target_column,
        min_assets=min_assets_for_ic,
    )
    test_ic.to_csv(artifacts_dir / "test_rank_ic.csv", index=False)
    test_deciles = decile_forward_returns(
        test_predictions,
        actual_column=target_column,
        n_buckets=int(ml_cfg.get("deciles", 10)),
    )
    test_deciles.to_csv(artifacts_dir / "test_decile_forward_returns.csv", index=False)
    test_spread = top_bottom_spread_by_date(
        test_deciles, top_bucket=int(ml_cfg.get("deciles", 10))
    )
    test_spread.to_csv(artifacts_dir / "test_top_bottom_spread.csv", index=False)
    test_yearly_ic = yearly_ic_summary(test_ic)
    test_yearly_ic.to_csv(artifacts_dir / "test_rank_ic_by_year.csv", index=False)

    console.print("[bold]5/7 Measuring test-time coefficient stability without future labels[/bold]")
    test_coefficients = walk_forward_ridge_coefficients(
        pretest_history,
        monthly["test"],
        feature_columns=selected_features,
        target_column="target_rank",
        alpha=ablation.selected_alpha,
        min_training_rows=min_training_rows,
    )
    test_coefficients.to_csv(artifacts_dir / "test_coefficient_history.csv", index=False)
    combined_coefficients = pd.concat(
        [validation_coefficients.assign(period="validation"), test_coefficients.assign(period="test")],
        ignore_index=True,
    )
    combined_coefficients.to_csv(artifacts_dir / "coefficient_history.csv", index=False)
    stability = coefficient_stability_summary(combined_coefficients)
    stability.to_csv(artifacts_dir / "coefficient_stability.csv", index=False)

    console.print("[bold]6/7 Backtesting the frozen Ridge portfolio against identical-date baselines[/bold]")
    test_signal_dates = pd.DatetimeIndex(sorted(test_predictions["date"].unique()))
    ridge_targets = _targets_for_ml_variant(
        test_predictions, selected_variant, min_assets=min_assets
    )
    benchmark_top_fraction = float(portfolio_cfg.get("top_fraction", 0.20))
    eq_targets = targets_on_dates(
        panel, test_signal_dates, strategy="equal_weight", min_assets=min_assets
    )
    mom_targets = targets_on_dates(
        panel,
        test_signal_dates,
        strategy="momentum",
        top_fraction=benchmark_top_fraction,
        min_assets=min_assets,
    )
    test_end = min(pd.Timestamp(split_cfg["test_end"]), pd.Timestamp(panel["date"].max()))
    ridge_daily = _evaluate_target_table(
        panel,
        ridge_targets,
        signal_dates=test_signal_dates,
        evaluation_end=test_end,
        transaction_cost_bps=float(bt_cfg["transaction_cost_bps"]),
        slippage_bps=float(bt_cfg["slippage_bps"]),
    )
    eq_daily = _evaluate_target_table(
        panel,
        eq_targets,
        signal_dates=test_signal_dates,
        evaluation_end=test_end,
        transaction_cost_bps=float(bt_cfg["transaction_cost_bps"]),
        slippage_bps=float(bt_cfg["slippage_bps"]),
    )
    mom_daily = _evaluate_target_table(
        panel,
        mom_targets,
        signal_dates=test_signal_dates,
        evaluation_end=test_end,
        transaction_cost_bps=float(bt_cfg["transaction_cost_bps"]),
        slippage_bps=float(bt_cfg["slippage_bps"]),
    )
    test_portfolio_metrics = {
        "equal_weight": performance_metrics(
            eq_daily["net_return"], eq_daily["turnover"],
            annualization_days=int(research["annualization_days"]),
        ),
        "momentum_top_quintile": performance_metrics(
            mom_daily["net_return"], mom_daily["turnover"],
            benchmark_returns=eq_daily["net_return"],
            annualization_days=int(research["annualization_days"]),
        ),
        f"ridge_{selected_variant_name}": performance_metrics(
            ridge_daily["net_return"], ridge_daily["turnover"],
            benchmark_returns=eq_daily["net_return"],
            annualization_days=int(research["annualization_days"]),
        ),
    }
    portfolio_dir = artifacts_dir / "portfolio"
    write_metrics_report(test_portfolio_metrics, portfolio_dir)
    ridge_daily.to_parquet(portfolio_dir / "ridge_selected_test_daily.parquet")
    eq_daily.to_parquet(portfolio_dir / "equal_weight_test_daily.parquet")
    mom_daily.to_parquet(portfolio_dir / "momentum_test_daily.parquet")
    yearly_portfolios = _yearly_portfolio_metrics(
        {
            "ridge_selected": ridge_daily,
            "equal_weight": eq_daily,
            "momentum_top_quintile": mom_daily,
        },
        annualization_days=int(research["annualization_days"]),
    )
    yearly_portfolios.to_csv(portfolio_dir / "yearly_performance_metrics.csv", index=False)

    console.print("[bold]7/7 Writing bootstrap robustness and efficiency reports[/bold]")
    bootstrap_cfg = robustness_cfg.get("bootstrap", {})
    stats = statistical_summary(
        test_ic,
        test_spread,
        n_bootstrap=int(bootstrap_cfg.get("samples", 5000)),
        confidence=float(bootstrap_cfg.get("confidence", 0.95)),
        seed=int(bootstrap_cfg.get("seed", cfg.get("project", {}).get("random_seed", 42))),
    )
    summary = {
        "milestone": "2B",
        "selection_policy": (
            "Feature group and Ridge alpha selected only by validation mean Rank IC; "
            "portfolio variant selected only by validation Sharpe with lower turnover as tie-breaker. "
            "The test period is evaluated only after those choices are frozen."
        ),
        "selected_feature_group": ablation.selected_group,
        "selected_rank_features": selected_features,
        "selected_alpha": ablation.selected_alpha,
        "selected_portfolio_variant": selected_variant,
        "validation_selection": {
            "feature_ablation_leader": ablation.table.iloc[0].to_dict(),
            "portfolio_leader": validation_portfolios.iloc[0].to_dict(),
        },
        "test_statistical_robustness": stats,
        "test_portfolio_metrics": test_portfolio_metrics,
        "test_signal_months": int(test_predictions["date"].nunique()),
        "methodology": {
            "ablation": "validation-only feature-family comparison with per-family validation alpha tuning",
            "coefficient_stability": "same matured-label expanding Ridge fits used at each prediction month",
            "portfolio_selection": "validation-only among predeclared long-only variants",
            "bootstrap": "non-parametric resampling of monthly Rank IC and top-minus-bottom spread",
        },
    }
    write_json(summary, artifacts_dir / "summary.json")
    write_ablation_plot(ablation.table, artifacts_dir)
    write_coefficient_stability_plot(combined_coefficients, artifacts_dir)
    write_turnover_frontier_plot(validation_portfolios, artifacts_dir)
    write_yearly_ic_plot(test_yearly_ic, artifacts_dir)
    write_rank_ic_plot(pd.DataFrame(), test_ic, artifacts_dir)
    write_decile_plot(test_deciles, artifacts_dir)
    write_test_equity_curves(
        {
            "Equal weight": eq_daily["equity_curve"],
            "Momentum top quintile": mom_daily["equity_curve"],
            f"Ridge {selected_variant_name}": ridge_daily["equity_curve"],
        },
        artifacts_dir,
    )
    console.print(f"[green]Milestone 2B complete.[/green] Outputs: {artifacts_dir}")



def _read_backtest_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame = frame.set_index("date")
    frame.index = pd.to_datetime(frame.index).astype("datetime64[ns]")
    return frame.sort_index()


def _select_validation_portfolio_variant(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    variants: list[dict[str, object]],
    *,
    min_assets: int,
    evaluation_end: pd.Timestamp,
    transaction_cost_bps: float,
    slippage_bps: float,
    annualization_days: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    signal_dates = pd.DatetimeIndex(sorted(predictions["date"].unique()))
    rows: list[dict[str, object]] = []
    for variant in variants:
        targets = _targets_for_ml_variant(predictions, variant, min_assets=min_assets)
        daily = _evaluate_target_table(
            panel,
            targets,
            signal_dates=signal_dates,
            evaluation_end=evaluation_end,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        metrics = performance_metrics(
            daily["net_return"], daily["turnover"], annualization_days=annualization_days
        )
        rows.append(
            {
                "variant": str(variant["name"]),
                "kind": str(variant["kind"]),
                **metrics,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["sharpe", "annualized_turnover", "variant"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    if table.empty or not np.isfinite(table.iloc[0]["sharpe"]):
        raise ValueError("No nonlinear portfolio variant produced a valid validation Sharpe ratio.")
    winner_name = str(table.iloc[0]["variant"])
    winner = next(v for v in variants if str(v["name"]) == winner_name)
    return winner, table


@app.command("run-milestone2c")
def run_milestone2c(
    config_path: Path = typer.Option(ROOT / "configs" / "research.yaml", exists=True),
) -> None:
    """Compare validation-selected Random Forest, XGBoost, and LightGBM models."""
    cfg = load_config(config_path)
    research = cfg["research"]
    split_cfg = cfg["split"]
    ml_cfg = cfg.get("ml", {})
    robustness_cfg = ml_cfg.get("robustness", {})
    nonlinear_cfg = ml_cfg.get("nonlinear", {})
    if not nonlinear_cfg:
        raise typer.BadParameter("Missing 'ml.nonlinear' section in research.yaml.")

    panel_path = ROOT / "data" / "processed" / "research_panel.parquet"
    ml_data_dir = ROOT / "data" / "processed" / "ml"
    m2b_dir = ROOT / "artifacts" / "milestone2b"
    artifacts_dir = ROOT / "artifacts" / "milestone2c"
    required = [
        panel_path,
        *(ml_data_dir / f"{name}_monthly.parquet" for name in ("train", "validation", "test")),
        m2b_dir / "summary.json",
        m2b_dir / "portfolio" / "ridge_selected_test_daily.parquet",
        m2b_dir / "portfolio" / "equal_weight_test_daily.parquet",
        m2b_dir / "portfolio" / "momentum_test_daily.parquet",
    ]
    for path in required:
        if not path.exists():
            raise typer.BadParameter(
                f"Required prior-milestone file is missing: {path}. "
                "Run 'quant-research run-milestone2b' first."
            )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = artifacts_dir / "validation"
    test_dir = artifacts_dir / "test"
    portfolio_dir = artifacts_dir / "portfolio"
    for path in (validation_dir, test_dir, portfolio_dir):
        path.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    monthly: dict[str, pd.DataFrame] = {}
    for name in ("train", "validation", "test"):
        part = pd.read_parquet(ml_data_dir / f"{name}_monthly.parquet")
        part["date"] = pd.to_datetime(part["date"]).astype("datetime64[ns]")
        part["label_end_date"] = pd.to_datetime(part["label_end_date"]).astype("datetime64[ns]")
        monthly[name] = part

    target_column = str(ml_cfg.get("target_column", "forward_return_21d"))
    min_training_rows = int(ml_cfg.get("min_training_rows", 1000))
    min_assets_for_ic = int(ml_cfg.get("min_assets_for_ic", 20))
    min_assets = int(ml_cfg.get("portfolio", {}).get("min_assets", 20))
    annualization_days = int(research["annualization_days"])
    random_seed = int(cfg.get("project", {}).get("random_seed", 42))
    bt_cfg = cfg["backtest"]
    transaction_cost_bps = float(bt_cfg["transaction_cost_bps"])
    slippage_bps = float(bt_cfg["slippage_bps"])

    raw_feature_groups = robustness_cfg.get("feature_groups", {})
    group_names = [str(x) for x in nonlinear_cfg.get("feature_group_names", [])]
    if not group_names:
        raise ValueError("ml.nonlinear.feature_group_names cannot be empty.")
    feature_groups: dict[str, list[str]] = {}
    for group_name in group_names:
        if group_name not in raw_feature_groups:
            raise ValueError(f"Unknown nonlinear feature group: {group_name}")
        feature_groups[group_name] = [
            rank_feature_name(str(feature)) for feature in raw_feature_groups[group_name]
        ]

    all_variants = {
        str(v["name"]): v for v in robustness_cfg.get("portfolio_variants", [])
    }
    variant_names = [str(x) for x in nonlinear_cfg.get("portfolio_variant_names", [])]
    variants = [all_variants[name] for name in variant_names if name in all_variants]
    if len(variants) != len(variant_names) or not variants:
        missing = sorted(set(variant_names) - set(all_variants))
        raise ValueError(f"Unknown or missing nonlinear portfolio variants: {missing}")

    family_cfg = nonlinear_cfg.get("model_families", {})
    model_names = ["random_forest", "xgboost", "lightgbm"]
    missing_models = [name for name in model_names if name not in family_cfg]
    if missing_models:
        raise ValueError(f"Missing nonlinear model-family configuration: {missing_models}")

    console.print("[bold]1/8 Validation-only nonlinear feature and hyperparameter selection[/bold]")
    selections = {}
    selected_features: dict[str, list[str]] = {}
    selected_variants: dict[str, dict[str, object]] = {}
    validation_portfolio_tables: dict[str, pd.DataFrame] = {}
    validation_end = min(pd.Timestamp(split_cfg["validation_end"]), pd.Timestamp(panel["date"].max()))

    for model_name in model_names:
        candidates = list(family_cfg[model_name].get("candidates", []))
        selection = select_nonlinear_configuration(
            monthly["train"],
            monthly["validation"],
            model_name=model_name,
            feature_groups=feature_groups,
            candidates=candidates,
            target_column="target_rank",
            actual_return_column=target_column,
            min_training_rows=min_training_rows,
            min_assets_for_ic=min_assets_for_ic,
            random_seed=random_seed,
        )
        selections[model_name] = selection
        selected_features[model_name] = feature_groups[selection.selected_group]
        selection.table.to_csv(
            validation_dir / f"{model_name}_configuration_search.csv", index=False
        )
        selection.validation_predictions.to_parquet(
            validation_dir / f"{model_name}_predictions.parquet", index=False
        )
        write_nonlinear_selection_plot(selection.table, model_name, validation_dir)
        console.print(
            f"{model_name}: [bold]{selection.selected_group}[/bold] / "
            f"[bold]{selection.selected_candidate}[/bold]"
        )

    console.print("[bold]2/8 Validation-only portfolio implementation selection per model[/bold]")
    for model_name in model_names:
        variant, table = _select_validation_portfolio_variant(
            panel,
            selections[model_name].validation_predictions,
            variants,
            min_assets=min_assets,
            evaluation_end=validation_end,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            annualization_days=annualization_days,
        )
        selected_variants[model_name] = variant
        validation_portfolio_tables[model_name] = table
        table.to_csv(validation_dir / f"{model_name}_portfolio_selection.csv", index=False)
        console.print(
            f"{model_name} portfolio: [bold]{str(variant['name'])}[/bold]"
        )

    console.print("[bold]3/8 Freezing specifications and running the previously defined test period[/bold]")
    pretest_history = pd.concat(
        [monthly["train"], monthly["validation"]], ignore_index=True
    ).sort_values(["date", "ticker"])
    test_predictions_by_model: dict[str, pd.DataFrame] = {}
    test_ic_by_model: dict[str, pd.DataFrame] = {}
    test_deciles_by_model: dict[str, pd.DataFrame] = {}
    test_spread_by_model: dict[str, pd.DataFrame] = {}
    yearly_ic_frames: list[pd.DataFrame] = []
    test_stats: dict[str, dict[str, object]] = {}
    bootstrap_cfg = nonlinear_cfg.get("bootstrap", {})

    for model_name in model_names:
        selection = selections[model_name]
        preds = walk_forward_nonlinear_predictions(
            pretest_history,
            monthly["test"],
            model_name=model_name,
            feature_columns=selected_features[model_name],
            target_column="target_rank",
            params=selection.selected_params,
            candidate_name=selection.selected_candidate,
            min_training_rows=min_training_rows,
            random_seed=random_seed,
        )
        preds.to_parquet(test_dir / f"{model_name}_predictions.parquet", index=False)
        test_predictions_by_model[model_name] = preds

        ic = rank_ic_by_date(
            preds, actual_column=target_column, min_assets=min_assets_for_ic
        )
        ic.to_csv(test_dir / f"{model_name}_rank_ic.csv", index=False)
        test_ic_by_model[model_name] = ic

        deciles = decile_forward_returns(
            preds, actual_column=target_column, n_buckets=int(ml_cfg.get("deciles", 10))
        )
        deciles.to_csv(test_dir / f"{model_name}_decile_forward_returns.csv", index=False)
        test_deciles_by_model[model_name] = deciles

        spread = top_bottom_spread_by_date(
            deciles, top_bucket=int(ml_cfg.get("deciles", 10))
        )
        spread.to_csv(test_dir / f"{model_name}_top_bottom_spread.csv", index=False)
        test_spread_by_model[model_name] = spread

        yearly = yearly_ic_summary(ic).assign(model=model_name)
        yearly.to_csv(test_dir / f"{model_name}_rank_ic_by_year.csv", index=False)
        yearly_ic_frames.append(yearly)

        test_stats[model_name] = statistical_summary(
            ic,
            spread,
            n_bootstrap=int(bootstrap_cfg.get("samples", 5000)),
            confidence=float(bootstrap_cfg.get("confidence", 0.95)),
            seed=int(bootstrap_cfg.get("seed", random_seed)),
        )

    yearly_all = pd.concat(yearly_ic_frames, ignore_index=True)
    yearly_all.to_csv(test_dir / "model_rank_ic_by_year.csv", index=False)

    console.print("[bold]4/8 Fitting pre-test reference models and recording native feature importance[/bold]")
    first_test_date = pd.Timestamp(monthly["test"]["date"].min())
    importance_tables: dict[str, pd.DataFrame] = {}
    for model_name in model_names:
        selection = selections[model_name]
        reference_model, reference_training = fit_reference_nonlinear(
            pretest_history,
            before_date=first_test_date,
            model_name=model_name,
            feature_columns=selected_features[model_name],
            target_column="target_rank",
            params=selection.selected_params,
            random_seed=random_seed,
        )
        importance = native_feature_importance(reference_model, selected_features[model_name])
        importance["pretest_training_rows"] = int(len(reference_training))
        importance_tables[model_name] = importance
        importance.to_csv(test_dir / f"{model_name}_feature_importance.csv", index=False)
        write_native_importance_plot(importance, model_name, test_dir)

    console.print("[bold]5/8 Backtesting frozen nonlinear portfolios[/bold]")
    test_end = min(pd.Timestamp(split_cfg["test_end"]), pd.Timestamp(panel["date"].max()))
    nonlinear_daily: dict[str, pd.DataFrame] = {}
    for model_name in model_names:
        preds = test_predictions_by_model[model_name]
        signal_dates = pd.DatetimeIndex(sorted(preds["date"].unique()))
        targets = _targets_for_ml_variant(
            preds, selected_variants[model_name], min_assets=min_assets
        )
        daily = _evaluate_target_table(
            panel,
            targets,
            signal_dates=signal_dates,
            evaluation_end=test_end,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        nonlinear_daily[model_name] = daily
        daily.to_parquet(portfolio_dir / f"{model_name}_test_daily.parquet")

    console.print("[bold]6/8 Loading frozen Ridge and conventional benchmark portfolios[/bold]")
    ridge_daily = _read_backtest_daily(m2b_dir / "portfolio" / "ridge_selected_test_daily.parquet")
    equal_daily = _read_backtest_daily(m2b_dir / "portfolio" / "equal_weight_test_daily.parquet")
    momentum_daily = _read_backtest_daily(m2b_dir / "portfolio" / "momentum_test_daily.parquet")
    with (m2b_dir / "summary.json").open("r", encoding="utf-8") as handle:
        m2b_summary = json.load(handle)

    daily_by_strategy = {
        "equal_weight": equal_daily,
        "momentum_top_quintile": momentum_daily,
        "ridge_m2b_frozen": ridge_daily,
        **nonlinear_daily,
    }
    portfolio_metrics: dict[str, dict[str, float]] = {}
    for strategy, daily in daily_by_strategy.items():
        benchmark = None if strategy == "equal_weight" else equal_daily["net_return"]
        portfolio_metrics[strategy] = performance_metrics(
            daily["net_return"],
            daily["turnover"],
            benchmark_returns=benchmark,
            annualization_days=annualization_days,
        )
    write_metrics_report(portfolio_metrics, portfolio_dir)
    yearly_portfolios = _yearly_portfolio_metrics(
        daily_by_strategy, annualization_days=annualization_days
    )
    yearly_portfolios.to_csv(portfolio_dir / "yearly_performance_metrics.csv", index=False)

    console.print("[bold]7/8 Building the nonlinear-vs-linear comparison table[/bold]")
    comparison_rows: list[dict[str, object]] = []
    ridge_stats = m2b_summary.get("test_statistical_robustness", {})
    ridge_ic_stats = ridge_stats.get("rank_ic", {}) if isinstance(ridge_stats, dict) else {}
    comparison_rows.extend(
        [
            {"model": "equal_weight", "model_class": "baseline", **portfolio_metrics["equal_weight"]},
            {"model": "momentum_top_quintile", "model_class": "baseline", **portfolio_metrics["momentum_top_quintile"]},
            {
                "model": "ridge_m2b_frozen",
                "model_class": "linear_ml",
                "test_mean_rank_ic": ridge_ic_stats.get("mean_rank_ic"),
                "test_rank_ic_t_stat": ridge_ic_stats.get("t_stat"),
                "test_rank_ic_ci_low": ridge_ic_stats.get("bootstrap_ci_low"),
                "test_rank_ic_ci_high": ridge_ic_stats.get("bootstrap_ci_high"),
                **portfolio_metrics["ridge_m2b_frozen"],
            },
        ]
    )
    for model_name in model_names:
        selection = selections[model_name]
        rank_stats = test_stats[model_name]["rank_ic"]
        spread_stats = test_stats[model_name]["top_bottom_spread"]
        validation_leader = selection.table.iloc[0]
        comparison_rows.append(
            {
                "model": model_name,
                "model_class": "nonlinear_ml",
                "selected_feature_group": selection.selected_group,
                "selected_candidate": selection.selected_candidate,
                "selected_portfolio_variant": str(selected_variants[model_name]["name"]),
                "validation_mean_rank_ic": validation_leader["validation_mean_rank_ic"],
                "test_mean_rank_ic": rank_stats["mean_rank_ic"],
                "test_rank_ic_t_stat": rank_stats["t_stat"],
                "test_rank_ic_ci_low": rank_stats["bootstrap_ci_low"],
                "test_rank_ic_ci_high": rank_stats["bootstrap_ci_high"],
                "test_top_bottom_mean": spread_stats["mean"],
                "test_top_bottom_t_stat": spread_stats["t_stat"],
                **portfolio_metrics[model_name],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(artifacts_dir / "model_comparison.csv", index=False)

    console.print("[bold]8/8 Writing Milestone 2C reports and methodology disclosure[/bold]")
    summary = {
        "milestone": "2C",
        "research_question": (
            "Do validation-selected nonlinear tree models add more persistent cross-sectional "
            "ranking and portfolio value than the frozen Ridge and conventional baselines?"
        ),
        "selection_policy": (
            "For each nonlinear model family, feature group and hyperparameters are selected "
            "only by 2020-2022 validation mean Rank IC. Portfolio implementation is then selected "
            "only by validation Sharpe with lower turnover as tie-breaker. These choices are frozen "
            "before evaluating 2023-2025."
        ),
        "test_reuse_disclosure": (
            "The 2023-2025 evaluation period was already inspected in Milestones 2A and 2B. "
            "Milestone 2C does not tune on it, but it is therefore a frozen historical evaluation "
            "period rather than a pristine never-seen holdout."
        ),
        "selected_models": {
            model_name: {
                "feature_group": selections[model_name].selected_group,
                "rank_features": selected_features[model_name],
                "candidate": selections[model_name].selected_candidate,
                "params": selections[model_name].selected_params,
                "portfolio_variant": selected_variants[model_name],
                "validation_leader": selections[model_name].table.iloc[0].to_dict(),
                "test_statistics": test_stats[model_name],
                "test_portfolio_metrics": portfolio_metrics[model_name],
            }
            for model_name in model_names
        },
        "frozen_ridge_m2b": {
            "selected_feature_group": m2b_summary.get("selected_feature_group"),
            "selected_alpha": m2b_summary.get("selected_alpha"),
            "selected_portfolio_variant": m2b_summary.get("selected_portfolio_variant"),
            "test_statistical_robustness": m2b_summary.get("test_statistical_robustness"),
            "test_portfolio_metrics": portfolio_metrics["ridge_m2b_frozen"],
        },
        "benchmark_portfolio_metrics": {
            "equal_weight": portfolio_metrics["equal_weight"],
            "momentum_top_quintile": portfolio_metrics["momentum_top_quintile"],
        },
        "methodology": {
            "target": "centered cross-sectional rank of 21-trading-day forward return",
            "features": "centered cross-sectional percentile-rank features from the existing 2A monthly dataset",
            "walk_forward": "expanding monthly training with label_end_date strictly before each signal date",
            "models": "Random Forest, XGBoost, and LightGBM with compact predeclared validation grids",
            "importance": "native tree importance from a reference model fitted immediately before the first test signal",
            "portfolio": "model-specific validation-selected long-only portfolio variant using the 1.2 drift-aware cost engine",
        },
    }
    write_json(summary, artifacts_dir / "summary.json")
    write_model_rank_ic_comparison(test_ic_by_model, artifacts_dir)
    write_model_yearly_ic_comparison(yearly_all, artifacts_dir)
    write_model_comparison_equity_curves(
        {
            "Equal weight": equal_daily["equity_curve"],
            "Momentum": momentum_daily["equity_curve"],
            "Ridge 2B frozen": ridge_daily["equity_curve"],
            "Random Forest": nonlinear_daily["random_forest"]["equity_curve"],
            "XGBoost": nonlinear_daily["xgboost"]["equity_curve"],
            "LightGBM": nonlinear_daily["lightgbm"]["equity_curve"],
        },
        artifacts_dir,
    )
    console.print(f"[green]Milestone 2C complete.[/green] Outputs: {artifacts_dir}")


@app.command("run-milestone3a")
def run_milestone3a(
    config_path: Path = typer.Option(ROOT / "configs" / "research.yaml", exists=True),
) -> None:
    """Explain the frozen Milestone 2C nonlinear models with SHAP and permutation tests."""
    cfg = load_config(config_path)
    ml_cfg = cfg.get("ml", {})
    explain_cfg = ml_cfg.get("explainability", {})
    if not explain_cfg:
        raise typer.BadParameter("Missing 'ml.explainability' section in research.yaml.")

    ml_data_dir = ROOT / "data" / "processed" / "ml"
    m2c_dir = ROOT / "artifacts" / "milestone2c"
    artifacts_dir = ROOT / "artifacts" / "milestone3a"
    required = [
        ml_data_dir / "train_monthly.parquet",
        ml_data_dir / "validation_monthly.parquet",
        ml_data_dir / "test_monthly.parquet",
        m2c_dir / "summary.json",
    ]
    for path in required:
        if not path.exists():
            raise typer.BadParameter(
                f"Required prior-milestone file is missing: {path}. "
                "Run 'quant-research run-milestone2c' first."
            )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifacts_dir / "models"
    interaction_dir = artifacts_dir / "interactions"
    for path in (model_dir, interaction_dir):
        path.mkdir(parents=True, exist_ok=True)

    monthly: dict[str, pd.DataFrame] = {}
    for name in ("train", "validation", "test"):
        part = pd.read_parquet(ml_data_dir / f"{name}_monthly.parquet")
        part["date"] = pd.to_datetime(part["date"]).astype("datetime64[ns]")
        part["label_end_date"] = pd.to_datetime(part["label_end_date"]).astype("datetime64[ns]")
        monthly[name] = part
    pretest_history = pd.concat([monthly["train"], monthly["validation"]], ignore_index=True)
    test = monthly["test"].copy()
    first_test_date = pd.Timestamp(test["date"].min())

    with (m2c_dir / "summary.json").open("r", encoding="utf-8") as handle:
        m2c_summary = json.load(handle)
    selected = m2c_summary.get("selected_models", {})
    model_names = [str(x) for x in explain_cfg.get("models", ["random_forest", "xgboost", "lightgbm"])]
    missing = [name for name in model_names if name not in selected]
    if missing:
        raise typer.BadParameter(f"Milestone 2C summary is missing selected models: {missing}")

    random_seed = int(cfg.get("project", {}).get("random_seed", 42))
    repeats = int(explain_cfg.get("permutation_repeats", 10))
    interaction_max_rows = int(explain_cfg.get("shap_interaction_max_rows", 600))
    configured_pairs = [tuple(map(str, pair)) for pair in explain_cfg.get("interaction_pairs", [])]

    console.print("[bold]1/6 Loading frozen 2C specifications and fitting pre-test reference models[/bold]")
    models: dict[str, object] = {}
    training_rows: dict[str, int] = {}
    feature_map: dict[str, list[str]] = {}
    for model_name in model_names:
        spec = selected[model_name]
        features = [str(x) for x in spec["rank_features"]]
        model, training = fit_reference_nonlinear(
            pretest_history,
            before_date=first_test_date,
            model_name=model_name,
            feature_columns=features,
            target_column="target_rank",
            params=dict(spec["params"]),
            random_seed=random_seed,
        )
        models[model_name] = model
        training_rows[model_name] = int(len(training))
        feature_map[model_name] = features

    console.print("[bold]2/6 Computing global SHAP and permutation importance[/bold]")
    shap_long_all: list[pd.DataFrame] = []
    permutation_all: list[pd.DataFrame] = []
    yearly_permutation_all: list[pd.DataFrame] = []
    yearly_shap_all: list[pd.DataFrame] = []
    interaction_all: list[pd.DataFrame] = []
    reference_ic_rows: list[dict[str, object]] = []
    wide_by_model: dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        model = models[model_name]
        features = feature_map[model_name]
        explain_frame = test.dropna(subset=["target_rank", "forward_return_21d", *features]).copy().reset_index(drop=True)
        explain_frame["prediction"] = model.predict(explain_frame[features])
        shap_matrix = tree_shap_values(model, explain_frame[features])
        for j, feature in enumerate(features):
            explain_frame[f"shap__{feature}"] = shap_matrix[:, j]
        wide_by_model[model_name] = explain_frame
        long = shap_long_table(
            explain_frame,
            feature_columns=features,
            values=shap_matrix,
            model_name=model_name,
        )
        shap_long_all.append(long)
        yearly_shap_all.append(summarize_yearly_shap_importance(long))
        perm = permutation_importance_table(
            model,
            explain_frame,
            feature_columns=features,
            target_column="target_rank",
            model_name=model_name,
            repeats=repeats,
            seed=random_seed,
        )
        permutation_all.append(perm)
        yearly_permutation_all.append(
            yearly_permutation_importance(
                model,
                explain_frame,
                feature_columns=features,
                target_column="target_rank",
                model_name=model_name,
                repeats=max(3, repeats // 2),
                seed=random_seed,
            )
        )
        ic = rank_ic_by_date(explain_frame, actual_column="forward_return_21d", min_assets=20)
        summary = ic_summary(ic)
        reference_ic_rows.append({"model": model_name, **summary})
        explain_frame.to_parquet(model_dir / f"{model_name}_reference_explanations.parquet")

    shap_long = pd.concat(shap_long_all, ignore_index=True)
    shap_summary = summarize_shap_importance(shap_long)
    yearly_shap = pd.concat(yearly_shap_all, ignore_index=True)
    permutation = pd.concat(permutation_all, ignore_index=True)
    yearly_permutation = pd.concat(yearly_permutation_all, ignore_index=True)
    shap_summary.to_csv(artifacts_dir / "shap_global_importance.csv", index=False)
    yearly_shap.to_csv(artifacts_dir / "shap_importance_by_year.csv", index=False)
    permutation.to_csv(artifacts_dir / "permutation_importance.csv", index=False)
    yearly_permutation.to_csv(artifacts_dir / "permutation_importance_by_year.csv", index=False)
    pd.DataFrame(reference_ic_rows).to_csv(artifacts_dir / "reference_model_rank_ic.csv", index=False)

    console.print("[bold]3/6 Computing Tree-SHAP interaction strengths[/bold]")
    for model_name in model_names:
        table = shap_interaction_importance(
            models[model_name],
            wide_by_model[model_name],
            feature_columns=feature_map[model_name],
            model_name=model_name,
            max_rows=interaction_max_rows,
            seed=random_seed,
        )
        interaction_all.append(table)
    interactions = pd.concat(interaction_all, ignore_index=True)
    interactions.to_csv(artifacts_dir / "shap_interaction_importance.csv", index=False)

    console.print("[bold]4/6 Building momentum × liquidity / volatility diagnostics[/bold]")
    interaction_manifest: list[dict[str, object]] = []
    for model_name in model_names:
        wide = wide_by_model[model_name]
        for feature_x, feature_y in configured_pairs:
            if feature_x not in feature_map[model_name] or feature_y not in feature_map[model_name]:
                continue
            shap_x = f"shap__{feature_x}"
            grid = interaction_grid(
                wide,
                feature_x=feature_x,
                feature_y=feature_y,
                shap_x=shap_x,
            )
            safe_x = feature_x.replace("rank__", "")
            safe_y = feature_y.replace("rank__", "")
            grid_path = interaction_dir / f"{model_name}_{safe_x}_x_{safe_y}.csv"
            grid.to_csv(grid_path, index=False)
            plot_path = write_shap_dependence_plot(
                wide,
                model_name=model_name,
                feature_x=feature_x,
                feature_y=feature_y,
                shap_x=shap_x,
                out_dir=interaction_dir,
            )
            interaction_manifest.append(
                {
                    "model": model_name,
                    "feature_x": feature_x,
                    "feature_y": feature_y,
                    "grid_csv": str(grid_path.relative_to(artifacts_dir)),
                    "plot": str(plot_path.relative_to(artifacts_dir)),
                }
            )

    console.print("[bold]5/6 Writing feature-importance stability and explainability plots[/bold]")
    write_global_importance_comparison(shap_summary, permutation, artifacts_dir)
    write_yearly_shap_heatmaps(yearly_shap, artifacts_dir)
    write_top_interactions_plot(interactions, artifacts_dir)

    console.print("[bold]6/6 Writing Milestone 3A research summary[/bold]")
    top_features: dict[str, object] = {}
    for model_name in model_names:
        shap_top = shap_summary.loc[shap_summary["model"] == model_name].head(5)
        perm_top = permutation.loc[permutation["model"] == model_name].head(5)
        interaction_top = interactions.loc[
            (interactions["model"] == model_name) & interactions["mean_abs_interaction"].notna()
        ].head(5)
        top_features[model_name] = {
            "pretest_training_rows": training_rows[model_name],
            "top_shap_features": shap_top[["feature", "normalized_mean_abs_shap"]].to_dict("records"),
            "top_permutation_features": perm_top[["feature", "importance_mean"]].to_dict("records"),
            "top_shap_interactions": interaction_top[["feature_a", "feature_b", "mean_abs_interaction"]].to_dict("records"),
        }
    summary = {
        "milestone": "3A",
        "research_question": (
            "What feature structure and interactions drive the validation-selected nonlinear models, "
            "and are those explanations stable across the historical evaluation years?"
        ),
        "policy": (
            "Milestone 3A is post-hoc explanation only. It does not select features, hyperparameters, "
            "portfolio variants, or alter the frozen Milestone 2C models."
        ),
        "reference_model_disclosure": (
            "SHAP and permutation diagnostics use one pre-test reference model per selected family, fitted "
            "only on labels matured before the first 2023 signal. These explanations characterize the frozen "
            "model specification; they are not substitutes for the expanding walk-forward performance results."
        ),
        "shap_dependency": "SHAP TreeExplainer; pyproject pins shap>=0.52,<0.53.",
        "models": top_features,
        "reference_model_rank_ic": pd.DataFrame(reference_ic_rows).to_dict("records"),
        "interaction_diagnostics": interaction_manifest,
    }
    write_json(summary, artifacts_dir / "summary.json")
    console.print(f"[green]Milestone 3A complete.[/green] Outputs: {artifacts_dir}")


@app.command("run-milestone3b")
def run_milestone3b(
    config_path: Path = typer.Option(ROOT / "configs" / "research.yaml", exists=True),
) -> None:
    """Diagnose frozen model and portfolio behaviour across point-in-time market regimes."""
    cfg = load_config(config_path)
    regime_cfg = cfg.get("ml", {}).get("regime_analysis", {})
    if not regime_cfg:
        raise typer.BadParameter("Missing 'ml.regime_analysis' section in research.yaml.")

    panel_path = ROOT / "data" / "processed" / "research_panel.parquet"
    m2b_dir = ROOT / "artifacts" / "milestone2b"
    m2c_dir = ROOT / "artifacts" / "milestone2c"
    m3a_dir = ROOT / "artifacts" / "milestone3a"
    artifacts_dir = ROOT / "artifacts" / "milestone3b"
    required = [
        panel_path,
        m2b_dir / "test_rank_ic.csv",
        m2b_dir / "test_top_bottom_spread.csv",
        m2b_dir / "portfolio" / "equal_weight_test_daily.parquet",
        m2b_dir / "portfolio" / "momentum_test_daily.parquet",
        m2b_dir / "portfolio" / "ridge_selected_test_daily.parquet",
        m2c_dir / "summary.json",
        m3a_dir / "summary.json",
    ]
    for model_name in ("random_forest", "xgboost", "lightgbm"):
        required.extend(
            [
                m2c_dir / "test" / f"{model_name}_rank_ic.csv",
                m2c_dir / "test" / f"{model_name}_top_bottom_spread.csv",
                m2c_dir / "portfolio" / f"{model_name}_test_daily.parquet",
            ]
        )
    for path in required:
        if not path.exists():
            raise typer.BadParameter(
                f"Required prior-milestone file is missing: {path}. "
                "Run Milestones 2B, 2C and 3A first."
            )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]1/6 Building point-in-time market-regime history[/bold]")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    daily_macro = daily_macro_panel(panel)
    regime_daily = build_point_in_time_regimes(
        daily_macro,
        quantile_lookback_days=int(regime_cfg.get("quantile_lookback_days", 504)),
        quantile_min_history_days=int(regime_cfg.get("quantile_min_history_days", 252)),
        lower_quantile=float(regime_cfg.get("lower_quantile", 1 / 3)),
        upper_quantile=float(regime_cfg.get("upper_quantile", 2 / 3)),
        rate_direction_window_days=int(regime_cfg.get("rate_direction_window_days", 63)),
        rate_flat_threshold_pp=float(regime_cfg.get("rate_flat_threshold_pp", 0.10)),
        yield_curve_flat_upper_pp=float(regime_cfg.get("yield_curve_flat_upper_pp", 0.25)),
        market_trend_window_days=int(regime_cfg.get("market_trend_window_days", 63)),
    )
    regime_daily.to_parquet(artifacts_dir / "daily_regime_history.parquet", index=False)

    nonlinear_ic: dict[str, pd.DataFrame] = {}
    nonlinear_spread: dict[str, pd.DataFrame] = {}
    for model_name in ("random_forest", "xgboost", "lightgbm"):
        nonlinear_ic[model_name] = pd.read_csv(
            m2c_dir / "test" / f"{model_name}_rank_ic.csv", parse_dates=["date"]
        )
        nonlinear_spread[model_name] = pd.read_csv(
            m2c_dir / "test" / f"{model_name}_top_bottom_spread.csv", parse_dates=["date"]
        )
    ridge_ic = pd.read_csv(m2b_dir / "test_rank_ic.csv", parse_dates=["date"])
    ridge_spread = pd.read_csv(m2b_dir / "test_top_bottom_spread.csv", parse_dates=["date"])
    signal_dates = sorted(
        set(ridge_ic["date"].tolist()).union(
            *[set(frame["date"].tolist()) for frame in nonlinear_ic.values()]
        )
    )
    signal_regimes = regimes_on_signal_dates(regime_daily, signal_dates)
    configured_regimes = [str(x) for x in regime_cfg.get("regime_columns", DEFAULT_REGIME_COLUMNS)]
    regime_columns = [c for c in configured_regimes if c in signal_regimes.columns]
    if not regime_columns:
        raise ValueError("None of the configured regime columns could be built from the research panel.")
    signal_regimes.to_csv(artifacts_dir / "signal_date_regimes.csv", index=False)
    counts = regime_counts(signal_regimes, regime_columns)
    counts.to_csv(artifacts_dir / "regime_counts.csv", index=False)

    console.print("[bold]2/6 Measuring cross-sectional Rank IC by regime[/bold]")
    ic_frames = {"ridge": ridge_ic, **nonlinear_ic}
    bootstrap_cfg = regime_cfg.get("bootstrap", {})
    regime_ic = metric_summary_by_regime(
        ic_frames,
        signal_regimes,
        value_column="rank_ic",
        regime_columns=regime_columns,
        n_bootstrap=int(bootstrap_cfg.get("samples", 2000)),
        confidence=float(bootstrap_cfg.get("confidence", 0.95)),
        seed=int(bootstrap_cfg.get("seed", cfg.get("project", {}).get("random_seed", 42))),
    )
    regime_ic.to_csv(artifacts_dir / "rank_ic_by_regime.csv", index=False)

    console.print("[bold]3/6 Measuring top-minus-bottom spread by regime[/bold]")
    spread_frames = {"ridge": ridge_spread, **nonlinear_spread}
    regime_spread = metric_summary_by_regime(
        spread_frames,
        signal_regimes,
        value_column="spread",
        regime_columns=regime_columns,
        n_bootstrap=int(bootstrap_cfg.get("samples", 2000)),
        confidence=float(bootstrap_cfg.get("confidence", 0.95)),
        seed=int(bootstrap_cfg.get("seed", cfg.get("project", {}).get("random_seed", 42))) + 1000,
    )
    regime_spread.to_csv(artifacts_dir / "top_bottom_spread_by_regime.csv", index=False)

    console.print("[bold]4/6 Measuring frozen portfolio holding-period behaviour by regime[/bold]")
    daily_by_strategy = {
        "equal_weight": _read_backtest_daily(m2b_dir / "portfolio" / "equal_weight_test_daily.parquet"),
        "momentum_top_quintile": _read_backtest_daily(m2b_dir / "portfolio" / "momentum_test_daily.parquet"),
        "ridge": _read_backtest_daily(m2b_dir / "portfolio" / "ridge_selected_test_daily.parquet"),
        "random_forest": _read_backtest_daily(m2c_dir / "portfolio" / "random_forest_test_daily.parquet"),
        "xgboost": _read_backtest_daily(m2c_dir / "portfolio" / "xgboost_test_daily.parquet"),
        "lightgbm": _read_backtest_daily(m2c_dir / "portfolio" / "lightgbm_test_daily.parquet"),
    }
    holding = holding_period_returns(
        daily_by_strategy,
        signal_regimes,
        regime_columns=regime_columns,
    )
    holding.to_csv(artifacts_dir / "portfolio_holding_period_returns.csv", index=False)
    portfolio_regimes = portfolio_summary_by_regime(
        holding,
        regime_columns=regime_columns,
        periods_per_year=12,
    )
    portfolio_regimes.to_csv(artifacts_dir / "portfolio_metrics_by_regime.csv", index=False)

    console.print("[bold]5/6 Writing regime visualisations[/bold]")
    write_regime_ic_heatmaps(regime_ic, plots_dir)
    write_regime_portfolio_heatmaps(portfolio_regimes, plots_dir)
    write_regime_timeline(signal_regimes, plots_dir)

    console.print("[bold]6/6 Writing diagnostic disclosure and Milestone 3B summary[/bold]")
    min_months = int(regime_cfg.get("min_regime_months_for_interpretation", 6))
    eligible_ic = regime_ic.loc[regime_ic["observations"] >= min_months].copy()
    dispersion_rows: list[dict[str, object]] = []
    if not eligible_ic.empty:
        for (model, dimension), group in eligible_ic.groupby(["model", "regime_dimension"], sort=True):
            if len(group) < 2:
                continue
            high = group.loc[group["mean"].idxmax()]
            low = group.loc[group["mean"].idxmin()]
            dispersion_rows.append(
                {
                    "model": model,
                    "regime_dimension": dimension,
                    "highest_regime": high["regime"],
                    "highest_mean_rank_ic": float(high["mean"]),
                    "lowest_regime": low["regime"],
                    "lowest_mean_rank_ic": float(low["mean"]),
                    "rank_ic_dispersion": float(high["mean"] - low["mean"]),
                }
            )
    dispersion = pd.DataFrame(dispersion_rows)
    dispersion.to_csv(artifacts_dir / "rank_ic_regime_dispersion.csv", index=False)

    summary = {
        "milestone": "3B",
        "research_question": (
            "How do frozen linear/nonlinear ranking signals and portfolios behave across market regimes "
            "defined only with information available at each signal date?"
        ),
        "policy": (
            "Milestone 3B is diagnostic only. Regime labels do not select, retune, switch, ensemble, "
            "or resize any model or portfolio. The 2023-2025 period is already-inspected historical evidence."
        ),
        "regime_methodology": {
            "quantile_levels": (
                "VIX, 10Y Treasury and Fed funds low/medium/high use trailing historical tercile "
                "thresholds estimated from prior observations only."
            ),
            "rate_direction": (
                f"10Y Treasury {int(regime_cfg.get('rate_direction_window_days', 63))}-trading-day change; "
                f"flat when within ±{float(regime_cfg.get('rate_flat_threshold_pp', 0.10)):.2f} percentage points."
            ),
            "yield_curve": (
                f"10Y-2Y spread: inverted < 0, flat 0 to {float(regime_cfg.get('yield_curve_flat_upper_pp', 0.25)):.2f} pp, "
                "normal above the flat band."
            ),
            "market_trend": (
                f"Sign of trailing {int(regime_cfg.get('market_trend_window_days', 63))}-trading-day compounded Fama-French Mkt-RF."
            ),
        },
        "regime_columns": regime_columns,
        "signal_months": int(len(signal_regimes)),
        "regime_counts": counts.to_dict("records"),
        "rank_ic_regime_dispersion_eligible_only": dispersion.to_dict("records"),
        "small_sample_rule": f"Regime cells with fewer than {min_months} signal months are flagged and excluded from dispersion summaries.",
        "multiple_comparisons_warning": (
            "Many model × regime cells are inspected. Confidence intervals and t-statistics are descriptive, "
            "not corrected for multiple testing, and should not be interpreted as discovery p-values."
        ),
        "conditional_portfolio_metric_disclosure": (
            "Conditional Sharpe uses non-contiguous monthly-ish holding-period returns tagged by the regime at the signal date. "
            "Conditional maximum drawdown is intentionally not reported."
        ),
    }
    write_json(summary, artifacts_dir / "summary.json")
    console.print(f"[green]Milestone 3B complete.[/green] Outputs: {artifacts_dir}")


@app.command("run-milestone3c")
def run_milestone3c(
    config_path: Path = typer.Option(ROOT / "configs" / "research.yaml", exists=True),
) -> None:
    """Select and evaluate risk/turnover-aware long-only portfolio optimization."""
    cfg = load_config(config_path)
    research = cfg["research"]
    split_cfg = cfg["split"]
    ml_cfg = cfg.get("ml", {})
    opt_cfg = ml_cfg.get("portfolio_optimization", {})
    if not opt_cfg:
        raise typer.BadParameter("Missing 'ml.portfolio_optimization' section in research.yaml.")

    panel_path = ROOT / "data" / "processed" / "research_panel.parquet"
    m2b_dir = ROOT / "artifacts" / "milestone2b"
    m2c_dir = ROOT / "artifacts" / "milestone2c"
    artifacts_dir = ROOT / "artifacts" / "milestone3c"
    validation_dir = artifacts_dir / "validation"
    evaluation_dir = artifacts_dir / "evaluation"
    portfolio_dir = artifacts_dir / "portfolio"
    plots_dir = artifacts_dir / "plots"
    for path in (validation_dir, evaluation_dir, portfolio_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    prediction_paths = {
        "ridge": {
            "validation": m2b_dir / "selected_validation_predictions.parquet",
            "test": m2b_dir / "selected_test_predictions.parquet",
        },
        "random_forest": {
            "validation": m2c_dir / "validation" / "random_forest_predictions.parquet",
            "test": m2c_dir / "test" / "random_forest_predictions.parquet",
        },
        "xgboost": {
            "validation": m2c_dir / "validation" / "xgboost_predictions.parquet",
            "test": m2c_dir / "test" / "xgboost_predictions.parquet",
        },
        "lightgbm": {
            "validation": m2c_dir / "validation" / "lightgbm_predictions.parquet",
            "test": m2c_dir / "test" / "lightgbm_predictions.parquet",
        },
    }
    required = [panel_path]
    for model in [str(x) for x in opt_cfg.get("alpha_models", [])]:
        if model not in prediction_paths:
            raise typer.BadParameter(f"Unknown optimizer alpha model: {model}")
        required.extend(prediction_paths[model].values())
    required.extend(
        [
            m2b_dir / "portfolio" / "equal_weight_test_daily.parquet",
            m2b_dir / "portfolio" / "momentum_test_daily.parquet",
            m2b_dir / "portfolio" / "ridge_selected_test_daily.parquet",
            m2c_dir / "portfolio" / "random_forest_test_daily.parquet",
            m2c_dir / "portfolio" / "xgboost_test_daily.parquet",
            m2c_dir / "portfolio" / "lightgbm_test_daily.parquet",
        ]
    )
    for path in required:
        if not path.exists():
            raise typer.BadParameter(
                f"Required prior-milestone file is missing: {path}. "
                "Run Milestones 2B and 2C before Milestone 3C."
            )

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    annualization_days = int(research.get("annualization_days", 252))
    validation_end = min(pd.Timestamp(split_cfg["validation_end"]), pd.Timestamp(panel["date"].max()))
    evaluation_end = min(pd.Timestamp(split_cfg["test_end"]), pd.Timestamp(panel["date"].max()))
    bt_cfg = cfg["backtest"]
    transaction_cost_bps = float(bt_cfg["transaction_cost_bps"])
    slippage_bps = float(bt_cfg["slippage_bps"])

    models = [str(x) for x in opt_cfg.get("alpha_models", [])]
    if not models:
        raise ValueError("ml.portfolio_optimization.alpha_models cannot be empty.")
    candidates = list(opt_cfg.get("candidates", []))
    if not candidates:
        raise ValueError("ml.portfolio_optimization.candidates cannot be empty.")

    common_kwargs = {
        "alpha_scale_annual": float(opt_cfg.get("alpha_scale_annual", 0.10)),
        "covariance_lookback_days": int(opt_cfg.get("covariance_lookback_days", 252)),
        "covariance_min_history_days": int(opt_cfg.get("covariance_min_history_days", 126)),
        "annualization_days": annualization_days,
        "max_weight": float(opt_cfg.get("max_weight", 0.10)),
        "min_assets": int(opt_cfg.get("min_assets", 40)),
    }

    console.print("[bold]1/7 Loading frozen alpha predictions and market risk history[/bold]")
    predictions: dict[str, dict[str, pd.DataFrame]] = {}
    for model in models:
        predictions[model] = {}
        for partition in ("validation", "test"):
            frame = pd.read_parquet(prediction_paths[model][partition])
            frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
            predictions[model][partition] = frame

    console.print("[bold]2/7 Validation-only optimizer hyperparameter selection[/bold]")
    selected_candidates: dict[str, dict[str, object]] = {}
    validation_rows: list[dict[str, object]] = []
    selected_validation_diagnostics: list[pd.DataFrame] = []
    for model in models:
        model_rows: list[dict[str, object]] = []
        candidate_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        preds = predictions[model]["validation"]
        signal_dates = pd.DatetimeIndex(sorted(pd.to_datetime(preds["date"]).unique()))
        for order, candidate in enumerate(candidates):
            name = str(candidate["name"])
            targets, diagnostics = build_optimized_targets(
                preds,
                panel,
                risk_aversion=float(candidate["risk_aversion"]),
                turnover_penalty=float(candidate["turnover_penalty"]),
                turnover_cap=(
                    None if candidate.get("turnover_cap") is None else float(candidate["turnover_cap"])
                ),
                **common_kwargs,
            )
            daily = _evaluate_target_table(
                panel,
                targets,
                signal_dates=signal_dates,
                evaluation_end=validation_end,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            )
            metrics = performance_metrics(
                daily["net_return"],
                daily["turnover"],
                annualization_days=annualization_days,
            )
            post_initial_relax = int(diagnostics["turnover_cap_relaxed"].iloc[1:].sum()) if len(diagnostics) > 1 else 0
            row = {
                "model": model,
                "candidate": name,
                "candidate_order": int(order),
                "risk_aversion": float(candidate["risk_aversion"]),
                "turnover_penalty": float(candidate["turnover_penalty"]),
                "turnover_cap": candidate.get("turnover_cap"),
                "avg_ex_ante_volatility": float(diagnostics["ex_ante_volatility"].mean()),
                "avg_effective_n": float(diagnostics["effective_n"].mean()),
                "avg_max_weight": float(diagnostics["max_weight_realized"].mean()),
                "post_initial_cap_relaxations": post_initial_relax,
                **metrics,
            }
            model_rows.append(row)
            validation_rows.append(row)
            candidate_outputs[name] = (targets, diagnostics, daily)

        table = pd.DataFrame(model_rows)
        eligible = table.loc[table["post_initial_cap_relaxations"] == 0].copy()
        if eligible.empty:
            eligible = table.copy()
        eligible = eligible.sort_values(
            ["sharpe", "annualized_turnover", "candidate_order"],
            ascending=[False, True, True],
            na_position="last",
        )
        winner = eligible.iloc[0]
        winner_name = str(winner["candidate"])
        selected_candidates[model] = next(c for c in candidates if str(c["name"]) == winner_name)
        targets, diagnostics, daily = candidate_outputs[winner_name]
        diagnostics = diagnostics.copy()
        diagnostics["model"] = model
        diagnostics["candidate"] = winner_name
        selected_validation_diagnostics.append(diagnostics)
        table.to_csv(validation_dir / f"{model}_optimizer_search.csv", index=False)
        targets.to_csv(validation_dir / f"{model}_selected_targets.csv", index=False)
        diagnostics.to_csv(validation_dir / f"{model}_selected_diagnostics.csv", index=False)
        daily.to_parquet(validation_dir / f"{model}_selected_daily.parquet")
        console.print(f"{model}: [bold]{winner_name}[/bold]")

    validation_table = pd.DataFrame(validation_rows)
    validation_table.to_csv(artifacts_dir / "optimizer_validation_search.csv", index=False)
    write_optimizer_validation_frontier(validation_table, plots_dir)

    # A single primary optimized alpha source is declared from validation only.
    validation_winners = validation_table.loc[
        validation_table.apply(
            lambda row: str(row["candidate"]) == str(selected_candidates[str(row["model"])]["name"]), axis=1
        )
    ].copy()
    primary_row = validation_winners.sort_values(
        ["sharpe", "annualized_turnover", "model"],
        ascending=[False, True, True],
        na_position="last",
    ).iloc[0]
    primary_model = str(primary_row["model"])
    console.print(f"Primary validation-selected optimized alpha source: [bold]{primary_model}[/bold]")

    console.print("[bold]3/7 Freezing optimizer specifications and building historical-evaluation targets[/bold]")
    evaluation_diagnostics: list[pd.DataFrame] = []
    optimized_daily: dict[str, pd.DataFrame] = {}
    optimized_targets: dict[str, pd.DataFrame] = {}
    for model in models:
        candidate = selected_candidates[model]
        preds = predictions[model]["test"]
        signal_dates = pd.DatetimeIndex(sorted(pd.to_datetime(preds["date"]).unique()))
        targets, diagnostics = build_optimized_targets(
            preds,
            panel,
            risk_aversion=float(candidate["risk_aversion"]),
            turnover_penalty=float(candidate["turnover_penalty"]),
            turnover_cap=(None if candidate.get("turnover_cap") is None else float(candidate["turnover_cap"])),
            **common_kwargs,
        )
        daily = _evaluate_target_table(
            panel,
            targets,
            signal_dates=signal_dates,
            evaluation_end=evaluation_end,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        diagnostics = diagnostics.copy()
        diagnostics["model"] = model
        diagnostics["candidate"] = str(candidate["name"])
        evaluation_diagnostics.append(diagnostics)
        optimized_daily[model] = daily
        optimized_targets[model] = targets
        targets.to_csv(evaluation_dir / f"{model}_optimized_targets.csv", index=False)
        diagnostics.to_csv(evaluation_dir / f"{model}_optimizer_diagnostics.csv", index=False)
        daily.to_parquet(portfolio_dir / f"{model}_optimized_daily.parquet")

    diagnostics_all = pd.concat(evaluation_diagnostics, ignore_index=True)
    diagnostics_all.to_csv(artifacts_dir / "optimizer_diagnostics.csv", index=False)

    console.print("[bold]4/7 Loading frozen heuristic portfolios for apples-to-apples comparison[/bold]")
    heuristic_daily = {
        "equal_weight": _read_backtest_daily(m2b_dir / "portfolio" / "equal_weight_test_daily.parquet"),
        "momentum_top_quintile": _read_backtest_daily(m2b_dir / "portfolio" / "momentum_test_daily.parquet"),
        "ridge_heuristic": _read_backtest_daily(m2b_dir / "portfolio" / "ridge_selected_test_daily.parquet"),
        "random_forest_heuristic": _read_backtest_daily(m2c_dir / "portfolio" / "random_forest_test_daily.parquet"),
        "xgboost_heuristic": _read_backtest_daily(m2c_dir / "portfolio" / "xgboost_test_daily.parquet"),
        "lightgbm_heuristic": _read_backtest_daily(m2c_dir / "portfolio" / "lightgbm_test_daily.parquet"),
    }

    console.print("[bold]5/7 Measuring performance, diversification, and turnover efficiency[/bold]")
    all_daily = dict(heuristic_daily)
    for model, daily in optimized_daily.items():
        all_daily[f"{model}_optimized"] = daily
    equal_returns = heuristic_daily["equal_weight"]["net_return"]
    metrics_rows: list[dict[str, object]] = []
    for strategy, daily in all_daily.items():
        benchmark = None if strategy == "equal_weight" else equal_returns
        metrics = performance_metrics(
            daily["net_return"],
            daily["turnover"],
            benchmark_returns=benchmark,
            annualization_days=annualization_days,
        )
        metrics_rows.append({"strategy": strategy, **metrics})
    metrics_table = pd.DataFrame(metrics_rows)
    metrics_table.to_csv(portfolio_dir / "performance_metrics.csv", index=False)
    yearly = _yearly_portfolio_metrics(all_daily, annualization_days=annualization_days)
    yearly.to_csv(portfolio_dir / "yearly_performance_metrics.csv", index=False)

    concentration_rows: list[dict[str, object]] = []
    for model, group in diagnostics_all.groupby("model", sort=True):
        concentration_rows.append(
            {
                "model": model,
                "candidate": str(group["candidate"].iloc[0]),
                "mean_positions": float(group["positions"].mean()),
                "mean_effective_n": float(group["effective_n"].mean()),
                "mean_max_weight": float(group["max_weight_realized"].mean()),
                "mean_ex_ante_volatility": float(group["ex_ante_volatility"].mean()),
                "mean_optimizer_turnover": float(group["optimizer_turnover"].mean()),
                "post_initial_cap_relaxations": int(group["turnover_cap_relaxed"].iloc[1:].sum()) if len(group) > 1 else 0,
            }
        )
    concentration = pd.DataFrame(concentration_rows)
    concentration.to_csv(artifacts_dir / "optimizer_concentration_summary.csv", index=False)

    console.print("[bold]6/7 Writing institutional portfolio-construction diagnostics[/bold]")
    write_optimizer_concentration(diagnostics_all, plots_dir)
    write_optimizer_risk_turnover(diagnostics_all, plots_dir)
    write_optimizer_equity_curves(
        {
            "Equal weight": heuristic_daily["equal_weight"]["equity_curve"],
            "Momentum": heuristic_daily["momentum_top_quintile"]["equity_curve"],
            "Ridge optimized": optimized_daily["ridge"]["equity_curve"] if "ridge" in optimized_daily else pd.Series(dtype=float),
            "RF optimized": optimized_daily["random_forest"]["equity_curve"] if "random_forest" in optimized_daily else pd.Series(dtype=float),
            "XGB optimized": optimized_daily["xgboost"]["equity_curve"] if "xgboost" in optimized_daily else pd.Series(dtype=float),
            "LGB optimized": optimized_daily["lightgbm"]["equity_curve"] if "lightgbm" in optimized_daily else pd.Series(dtype=float),
        },
        plots_dir,
    )

    console.print("[bold]7/7 Writing Milestone 3C methodology disclosure and summary[/bold]")
    metric_lookup = metrics_table.set_index("strategy").to_dict("index")
    summary = {
        "milestone": "3C",
        "research_question": (
            "Can the frozen ML alpha signals be translated into more institutional portfolios by explicitly "
            "balancing predicted alpha, covariance risk, position concentration, and turnover?"
        ),
        "selection_policy": (
            "For each already-frozen alpha model, optimizer risk-aversion/turnover settings are selected only "
            "on 2020-2022 validation net Sharpe. Lower turnover breaks exact Sharpe ties. The primary alpha "
            "source is also declared using validation only. No 2023-2025 optimizer choice is retuned."
        ),
        "historical_evaluation_disclosure": (
            "The 2023-2025 period has been inspected in earlier milestones and remains a frozen historical "
            "evaluation period rather than a pristine holdout."
        ),
        "optimizer": {
            "objective": "maximize alpha'w - risk_aversion*w'Sigma*w - turnover_penalty*|w-w_prev|_1",
            "constraints": {
                "long_only": True,
                "fully_invested": True,
                "max_weight": common_kwargs["max_weight"],
                "candidate_turnover_caps": [c.get("turnover_cap") for c in candidates],
            },
            "covariance": (
                f"Ledoit-Wolf shrinkage on the trailing {common_kwargs['covariance_lookback_days']} trading days, "
                f"requiring at least {common_kwargs['covariance_min_history_days']} observations, using data only through each signal date."
            ),
            "alpha": (
                f"Cross-sectionally standardized frozen model predictions scaled to {common_kwargs['alpha_scale_annual']:.2%} annual alpha per prediction z-score."
            ),
            "pretrade_state": "Previous target weights are drifted through realized returns to the current signal close before turnover is computed.",
        },
        "selected_optimizer_candidates": {
            model: dict(selected_candidates[model]) for model in models
        },
        "primary_validation_selected_alpha_model": primary_model,
        "primary_historical_evaluation_metrics": metric_lookup.get(f"{primary_model}_optimized", {}),
        "optimized_historical_evaluation_metrics": {
            model: metric_lookup.get(f"{model}_optimized", {}) for model in models
        },
        "heuristic_historical_evaluation_metrics": {
            key: metric_lookup.get(key, {}) for key in heuristic_daily
        },
        "concentration_summary": concentration.to_dict("records"),
        "constraint_disclosure": (
            "If a configured turnover cap is mathematically infeasible because of the initial investment or a position-cap repair, "
            "it is relaxed only to the minimum feasible turnover and explicitly flagged in diagnostics."
        ),
    }
    write_json(summary, artifacts_dir / "summary.json")
    console.print(f"[green]Milestone 3C complete.[/green] Outputs: {artifacts_dir}")


@app.command("run-milestone3d")
def run_milestone3d(
    config_path: Path = typer.Option(ROOT / "configs" / "research.yaml", exists=True),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="Exclusive market-data cutoff for the genuine 2026 holdout (YYYY-MM-DD).",
    ),
    reuse_holdout_raw: bool = typer.Option(
        False,
        "--reuse-holdout-raw",
        help="Reuse the previously downloaded 2026 holdout market snapshot.",
    ),
) -> None:
    """Run the frozen methodology on the genuine incremental 2026 holdout."""
    cfg = load_config(config_path)
    research = cfg["research"]
    ml_cfg = cfg.get("ml", {})
    holdout_cfg = ml_cfg.get("holdout2026", {})
    if not holdout_cfg:
        raise typer.BadParameter("Missing 'ml.holdout2026' section in research.yaml.")

    holdout_start = pd.Timestamp(holdout_cfg.get("start_date", "2026-01-01"))
    as_of_date = pd.Timestamp(as_of or holdout_cfg.get("as_of_date", "2026-09-01"))
    if as_of_date < holdout_start:
        raise typer.BadParameter("--as-of must be on or after the holdout start date.")

    historical_market_path = ROOT / "data" / "raw" / "market_ohlcv.parquet"
    m2b_summary_path = ROOT / "artifacts" / "milestone2b" / "summary.json"
    m2c_summary_path = ROOT / "artifacts" / "milestone2c" / "summary.json"
    m3c_summary_path = ROOT / "artifacts" / "milestone3c" / "summary.json"
    required = [historical_market_path, m2b_summary_path, m2c_summary_path, m3c_summary_path]
    for path in required:
        if not path.exists():
            raise typer.BadParameter(
                f"Required frozen-history file is missing: {path}. Run through Milestone 3C first."
            )

    holdout_data_dir = ROOT / "data" / "holdout2026"
    holdout_raw_dir = holdout_data_dir / "raw"
    holdout_processed_dir = holdout_data_dir / "processed"
    artifacts_dir = ROOT / "artifacts" / "milestone3d"
    predictions_dir = artifacts_dir / "predictions"
    portfolio_dir = artifacts_dir / "portfolio"
    optimized_dir = artifacts_dir / "optimized"
    plots_dir = artifacts_dir / "plots"
    for path in (holdout_raw_dir, holdout_processed_dir, predictions_dir, portfolio_dir, optimized_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    incremental_market_path = holdout_raw_dir / "market_ohlcv_2026.parquet"
    universe = [str(x) for x in cfg["universe"]["tickers"]]

    console.print("[bold]1/8 Freezing prior research specifications and holdout cutoff[/bold]")
    spec = load_frozen_holdout_spec(m2b_summary_path, m2c_summary_path, m3c_summary_path)
    write_json(frozen_spec_to_dict(spec), artifacts_dir / "frozen_specifications.json")

    console.print("[bold]2/8 Loading frozen history and incremental 2026 market data[/bold]")
    historical_market = pd.read_parquet(historical_market_path)
    historical_market["date"] = pd.to_datetime(historical_market["date"]).astype("datetime64[ns]")
    if reuse_holdout_raw:
        if not incremental_market_path.exists():
            raise typer.BadParameter(
                f"--reuse-holdout-raw requested but snapshot is missing: {incremental_market_path}"
            )
        incremental_market = pd.read_parquet(incremental_market_path)
        incremental_market["date"] = pd.to_datetime(incremental_market["date"]).astype("datetime64[ns]")
        # Treat --as-of as an exclusive cutoff so a current/in-progress daily bar
        # can never enter the prospective holdout.
        incremental_market = truncate_market_before_as_of(
            incremental_market, as_of_date=as_of_date
        )
        reused_max = pd.Timestamp(incremental_market["date"].max())
        if as_of_date - reused_max > pd.Timedelta(days=7):
            raise typer.BadParameter(
                f"Reused holdout snapshot is stale for --as-of {as_of_date.date()}: latest row is {reused_max.date()}. "
                "Run without --reuse-holdout-raw to refresh it."
            )
        console.print(f"Reused holdout snapshot: {incremental_market_path}")
    else:
        incremental_market = download_market_data_yfinance(
            universe,
            str(holdout_start.date()),
            str(as_of_date.date()),
            auto_adjust=bool(cfg.get("market_data", {}).get("auto_adjust", True)),
        )
        # Yahoo can expose an in-progress daily bar for the as-of session. The
        # prospective holdout uses only completed bars strictly before --as-of.
        incremental_market = truncate_market_before_as_of(
            incremental_market, as_of_date=as_of_date
        )
        validate_market_data(incremental_market)
        incremental_market.to_parquet(incremental_market_path, index=False)

    combined_market = combine_incremental_market(
        historical_market, incremental_market, holdout_start=holdout_start
    )
    validation = validate_market_data(combined_market)
    observed_holdout_tickers = set(
        combined_market.loc[combined_market["date"] >= holdout_start, "ticker"].astype(str)
    )
    missing_holdout_tickers = sorted(set(universe) - observed_holdout_tickers)
    console.print(
        f"Holdout market coverage: {len(observed_holdout_tickers)}/{len(universe)} tickers; "
        f"latest date {validation.end.date()}."
    )

    console.print("[bold]3/8 Rebuilding point-in-time stock features without revising frozen history[/bold]")
    panel = clean_market_data(combined_market)
    panel = add_momentum_features(panel, list(cfg["features"]["momentum_windows"]))
    panel = add_volatility_features(
        panel,
        list(cfg["features"]["volatility_windows"]),
        annualization_days=int(research.get("annualization_days", 252)),
    )
    panel = add_liquidity_features(panel, int(cfg["features"]["liquidity_window"]))
    horizon = int(research["label_horizon_days"])
    target_column = str(ml_cfg.get("target_column", f"forward_return_{horizon}d"))
    panel = add_forward_return_label(panel, horizon)
    panel = add_label_end_dates(panel, horizon)
    panel.to_parquet(holdout_processed_dir / "extended_market_feature_panel.parquet", index=False)

    console.print("[bold]4/8 Building completed-month online cross-sections and frozen predictions[/bold]")
    month_ends = completed_month_end_dates(panel, as_of_date=as_of_date)
    activation_candidates = month_ends[month_ends < holdout_start]
    if activation_candidates.empty:
        raise ValueError("No pre-holdout completed month-end is available for portfolio activation.")
    activation_signal = pd.Timestamp(activation_candidates.max())
    prediction_month_ends = month_ends[month_ends >= activation_signal]
    raw_features = [str(x) for x in ml_cfg["feature_columns"]]
    online_monthly = build_monthly_online_partition(
        panel,
        month_ends=prediction_month_ends.union(month_ends[month_ends < activation_signal]),
        feature_columns=raw_features,
        target_column=target_column,
        min_assets=int(ml_cfg.get("min_assets_per_cross_section", 40)),
    )
    online_monthly.to_parquet(holdout_processed_dir / "monthly_online.parquet", index=False)
    base_history = online_monthly.loc[online_monthly["date"] < activation_signal].copy()
    evaluation = online_monthly.loc[online_monthly["date"] >= activation_signal].copy()
    if evaluation.empty:
        raise ValueError("No completed-month holdout signals are available.")

    min_training_rows = int(ml_cfg.get("min_training_rows", 1000))
    random_seed = int(cfg.get("project", {}).get("random_seed", 42))
    predictions: dict[str, pd.DataFrame] = {}
    predictions["ridge"] = walk_forward_ridge_predictions(
        base_history,
        evaluation,
        feature_columns=spec.ridge_features,
        target_column="target_rank",
        alpha=spec.ridge_alpha,
        min_training_rows=min_training_rows,
    )
    for model_name, model_spec in spec.nonlinear_models.items():
        predictions[model_name] = walk_forward_nonlinear_predictions(
            base_history,
            evaluation,
            model_name=model_name,
            feature_columns=list(model_spec["rank_features"]),
            target_column="target_rank",
            params=dict(model_spec["params"]),
            candidate_name=str(model_spec["candidate"]),
            min_training_rows=min_training_rows,
            random_seed=random_seed,
        )
    for model_name, frame in predictions.items():
        frame.to_parquet(predictions_dir / f"{model_name}_online_predictions.parquet", index=False)

    console.print("[bold]5/8 Scoring only matured 2026 labels and ranking diagnostics[/bold]")
    ic_frames: list[pd.DataFrame] = []
    ic_summary_rows: list[dict[str, object]] = []
    matured_months: dict[str, int] = {}
    for model_name, frame in predictions.items():
        matured = matured_holdout_predictions(
            frame, holdout_start=holdout_start, target_column=target_column
        )
        ic = rank_ic_by_date(
            matured,
            actual_column=target_column,
            min_assets=int(ml_cfg.get("min_assets_for_ic", 20)),
        )
        ic["model"] = model_name
        ic_frames.append(ic)
        summary = ic_summary(ic)
        ic_summary_rows.append({"model": model_name, **summary})
        matured_months[model_name] = int(summary["months"])
    rank_ic_all = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()
    rank_ic_all.to_csv(artifacts_dir / "holdout_rank_ic.csv", index=False)
    ic_summary_table = pd.DataFrame(ic_summary_rows)
    ic_summary_table.to_csv(artifacts_dir / "holdout_rank_ic_summary.csv", index=False)

    console.print("[bold]6/8 Backtesting frozen heuristic and optimized portfolios[/bold]")
    signal_dates = pd.DatetimeIndex(sorted(pd.to_datetime(evaluation["date"]).unique()))
    bt_cfg = cfg["backtest"]
    transaction_cost_bps = float(bt_cfg["transaction_cost_bps"])
    slippage_bps = float(bt_cfg["slippage_bps"])
    annualization_days = int(research.get("annualization_days", 252))
    min_assets = int(ml_cfg.get("portfolio", {}).get("min_assets", 20))

    heuristic_targets: dict[str, pd.DataFrame] = {
        "equal_weight": targets_on_dates(panel, signal_dates, strategy="equal_weight", min_assets=min_assets),
        "momentum_top_quintile": targets_on_dates(
            panel,
            signal_dates,
            strategy="momentum",
            top_fraction=float(cfg["backtest"].get("momentum_top_fraction", 0.20)),
            min_assets=min_assets,
        ),
        "ridge_heuristic": _targets_for_ml_variant(
            predictions["ridge"], spec.ridge_portfolio_variant, min_assets=min_assets
        ),
    }
    for model_name, model_spec in spec.nonlinear_models.items():
        heuristic_targets[f"{model_name}_heuristic"] = _targets_for_ml_variant(
            predictions[model_name], dict(model_spec["portfolio_variant"]), min_assets=min_assets
        )

    evaluation_end = min(as_of_date, pd.Timestamp(panel["date"].max()))
    daily: dict[str, pd.DataFrame] = {}
    for strategy, targets in heuristic_targets.items():
        daily[strategy] = _evaluate_target_table(
            panel,
            targets,
            signal_dates=signal_dates,
            evaluation_end=evaluation_end,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        daily[strategy].to_parquet(portfolio_dir / f"{strategy}_daily.parquet")

    opt_cfg = ml_cfg.get("portfolio_optimization", {})
    common_opt = {
        "alpha_scale_annual": float(opt_cfg.get("alpha_scale_annual", 0.10)),
        "covariance_lookback_days": int(opt_cfg.get("covariance_lookback_days", 252)),
        "covariance_min_history_days": int(opt_cfg.get("covariance_min_history_days", 126)),
        "annualization_days": annualization_days,
        "max_weight": float(opt_cfg.get("max_weight", 0.10)),
        "min_assets": int(opt_cfg.get("min_assets", 40)),
    }
    optimizer_diagnostics: list[pd.DataFrame] = []
    for model_name in ("ridge", "random_forest", "xgboost", "lightgbm"):
        candidate = spec.optimizer_candidates[model_name]
        targets, diagnostics = build_optimized_targets(
            predictions[model_name],
            panel,
            risk_aversion=float(candidate["risk_aversion"]),
            turnover_penalty=float(candidate["turnover_penalty"]),
            turnover_cap=(None if candidate.get("turnover_cap") is None else float(candidate["turnover_cap"])),
            **common_opt,
        )
        strategy = f"{model_name}_optimized"
        diagnostics = diagnostics.copy()
        diagnostics["model"] = model_name
        diagnostics["candidate"] = str(candidate["name"])
        optimizer_diagnostics.append(diagnostics)
        targets.to_csv(optimized_dir / f"{model_name}_targets.csv", index=False)
        diagnostics.to_csv(optimized_dir / f"{model_name}_diagnostics.csv", index=False)
        daily[strategy] = _evaluate_target_table(
            panel,
            targets,
            signal_dates=signal_dates,
            evaluation_end=evaluation_end,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        daily[strategy].to_parquet(portfolio_dir / f"{strategy}_daily.parquet")
    if optimizer_diagnostics:
        pd.concat(optimizer_diagnostics, ignore_index=True).to_csv(
            optimized_dir / "optimizer_diagnostics.csv", index=False
        )

    console.print("[bold]7/8 Measuring genuine-holdout portfolio performance[/bold]")
    equal_returns = daily["equal_weight"]["net_return"]
    metric_rows: list[dict[str, object]] = []
    for strategy, frame in daily.items():
        metrics = performance_metrics(
            frame["net_return"],
            frame["turnover"],
            benchmark_returns=None if strategy == "equal_weight" else equal_returns,
            annualization_days=annualization_days,
        )
        metric_rows.append({"strategy": strategy, **metrics})
    metrics_table = pd.DataFrame(metric_rows).sort_values("sharpe", ascending=False, na_position="last")
    metrics_table.to_csv(portfolio_dir / "performance_metrics.csv", index=False)
    write_holdout_equity_curves(
        {name: frame["equity_curve"] for name, frame in daily.items()}, plots_dir
    )
    write_holdout_rank_ic(rank_ic_all, plots_dir)
    write_holdout_model_comparison(metrics_table, plots_dir)

    console.print("[bold]8/8 Writing immutable 2026 holdout disclosure and manifest[/bold]")
    market_max = pd.Timestamp(panel["date"].max())
    rank_dates = pd.to_datetime(rank_ic_all["date"]) if not rank_ic_all.empty else pd.Series(dtype="datetime64[ns]")
    latest_matured_signal = None if rank_dates.empty else str(pd.Timestamp(rank_dates.max()).date())
    summary = {
        "milestone": "3D",
        "status": "GENUINE_HOLDOUT_PARTIAL_YEAR",
        "holdout_start": str(holdout_start.date()),
        "requested_as_of": str(as_of_date.date()),
        "latest_market_date": str(market_max.date()),
        "activation_signal": str(activation_signal.date()),
        "completed_signal_dates": [str(pd.Timestamp(x).date()) for x in signal_dates],
        "latest_matured_rank_signal": latest_matured_signal,
        "matured_rank_months_by_model": matured_months,
        "missing_universe_tickers_in_holdout": missing_holdout_tickers,
        "frozen_specifications": frozen_spec_to_dict(spec),
        "rank_ic_summary": ic_summary_table.to_dict("records"),
        "portfolio_metrics": metrics_table.to_dict("records"),
        "methodology": {
            "historical_data": "Frozen pre-2026 Yahoo snapshot is never redownloaded or revised by Milestone 3D.",
            "incremental_data": "Only 2026-onward market data are downloaded into a separate holdout snapshot.",
            "features": "Same stock-specific feature definitions as the frozen historical research pipeline.",
            "models": "Feature families and hyperparameters are loaded from the frozen 2B/2C summaries; no 2026 selection occurs.",
            "online_retraining": "Model parameters are refit monthly using the already-declared expanding matured-label rule; model specifications are unchanged.",
            "optimizer": "Risk/turnover settings are loaded from the frozen 3C summary; no 2026 optimizer selection occurs.",
            "rank_evaluation": "Only 2026 signals with fully observed 21-trading-day forward returns are scored.",
            "portfolio_evaluation": "A 31-Dec-2025 activation signal starts a fresh deployment for Jan-2026; transaction costs include initial investment.",
            "current_month": "The as-of calendar month is excluded from signal generation because it is not yet a completed month.",
        },
        "holdout_integrity_warning": (
            "After viewing these 2026 results, changing features, model hyperparameters, portfolio rules, or optimizer settings "
            "because of 2026 performance would contaminate this holdout. Future methodology changes require a new prospective period."
        ),
    }
    write_json(summary, artifacts_dir / "summary.json")
    manifest = {
        "milestone": "3D",
        "as_of": str(as_of_date.date()),
        "historical_market": _snapshot_entry(historical_market_path, len(historical_market), reused=True),
        "incremental_market": _snapshot_entry(incremental_market_path, len(incremental_market), reused=reuse_holdout_raw),
        "frozen_summary_hashes": {
            "milestone2b": sha256_file(m2b_summary_path),
            "milestone2c": sha256_file(m2c_summary_path),
            "milestone3c": sha256_file(m3c_summary_path),
        },
    }
    write_manifest(manifest, holdout_raw_dir / "manifest.json")
    console.print(f"[green]Milestone 3D complete.[/green] Outputs: {artifacts_dir}")


@app.command("export-web")
def export_web(
    artifacts_dir: Path = typer.Option(
        ROOT / "artifacts",
        "--artifacts-dir",
        help="Root directory containing the frozen milestone artifacts.",
    ),
    output_dir: Path = typer.Option(
        ROOT / "web" / "public" / "data",
        "--output-dir",
        help="Destination for stable public dashboard JSON files.",
    ),
) -> None:
    """Validate frozen artifacts and export deterministic public web JSON."""
    try:
        written = ArtifactRepository(artifacts_dir).export_all(output_dir)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        f"[green]Milestone 4A web export complete.[/green] "
        f"Wrote {len(written)} files to {output_dir}."
    )


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("127.0.0.1", help="Host interface to bind."),
    port: int = typer.Option(8000, min=1, max=65535, help="TCP port to bind."),
    reload: bool = typer.Option(False, help="Enable development auto-reload."),
) -> None:
    """Serve the read-only research API and OpenAPI documentation."""
    import uvicorn

    uvicorn.run("quant_platform.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
