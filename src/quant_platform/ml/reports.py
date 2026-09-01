from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_rank_ic_plot(validation_ic: pd.DataFrame, test_ic: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not validation_ic.empty:
        ax.plot(validation_ic["date"], validation_ic["rank_ic"], label="Validation")
    if not test_ic.empty:
        ax.plot(test_ic["date"], test_ic["rank_ic"], label="Test")
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Monthly Cross-Sectional Spearman Rank IC")
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Rank IC")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "rank_ic_by_month.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_decile_plot(deciles: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    means = deciles.groupby("bucket")["mean_forward_return"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(means.index.astype(str), means.values)
    ax.set_title("Test Mean 21-Day Forward Return by Prediction Decile")
    ax.set_xlabel("Prediction decile (10 = highest predicted rank)")
    ax.set_ylabel("Mean realised forward return")
    fig.tight_layout()
    path = out_dir / "test_decile_forward_returns.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_coefficient_plot(coefficients: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = coefficients.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ordered["feature"], ordered["coefficient"])
    ax.axvline(0.0, linewidth=1)
    ax.set_title("Pre-Test Ridge Coefficients")
    ax.set_xlabel("Coefficient on standardized cross-sectional rank feature")
    fig.tight_layout()
    path = out_dir / "ridge_coefficients.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_test_equity_curves(curves: dict[str, pd.Series], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, series in curves.items():
        ax.plot(series.index, series.values, label=name)
    ax.set_title("Milestone 2A Test-Period Portfolio Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "test_equity_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_json(data: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: object):
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value)!r}")

    path.write_text(json.dumps(data, indent=2, default=default, allow_nan=True), encoding="utf-8")
    return path


def write_ablation_plot(ablation: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = ablation.sort_values("validation_mean_rank_ic")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["feature_group"], ordered["validation_mean_rank_ic"])
    ax.axvline(0.0, linewidth=1)
    ax.set_title("Validation Feature Ablation: Mean Rank IC")
    ax.set_xlabel("Validation mean Rank IC")
    fig.tight_layout()
    path = out_dir / "feature_ablation_validation_ic.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_coefficient_stability_plot(coefficients: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if coefficients.empty:
        raise ValueError("Coefficient history is empty.")
    pivot = coefficients.pivot_table(index="feature", columns="date", values="coefficient")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    dates = pd.DatetimeIndex(pivot.columns)
    step = max(1, len(dates) // 8)
    ticks = list(range(0, len(dates), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i].strftime("%Y-%m") for i in ticks], rotation=45, ha="right")
    ax.set_title("Walk-Forward Ridge Coefficient Stability")
    ax.set_xlabel("Prediction month")
    fig.colorbar(image, ax=ax, label="Coefficient")
    fig.tight_layout()
    path = out_dir / "coefficient_stability.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_turnover_frontier_plot(metrics: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(metrics["annualized_turnover"], metrics["sharpe"])
    for _, row in metrics.iterrows():
        ax.annotate(str(row["variant"]), (row["annualized_turnover"], row["sharpe"]), xytext=(4, 4), textcoords="offset points")
    ax.set_title("Validation Ridge Portfolio Efficiency")
    ax.set_xlabel("Annualised turnover")
    ax.set_ylabel("Sharpe ratio")
    fig.tight_layout()
    path = out_dir / "validation_sharpe_vs_turnover.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_yearly_ic_plot(yearly: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(yearly["year"].astype(str), yearly["mean_rank_ic"])
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Selected Ridge Test Mean Rank IC by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Rank IC")
    fig.tight_layout()
    path = out_dir / "test_rank_ic_by_year.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_nonlinear_selection_plot(
    selection: pd.DataFrame, model_name: str, out_dir: Path
) -> Path:
    """Plot validation mean Rank IC for a compact nonlinear configuration search."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if selection.empty:
        raise ValueError("Nonlinear selection table is empty.")
    frame = selection.copy()
    frame["label"] = frame["feature_group"].astype(str) + " / " + frame["candidate"].astype(str)
    frame = frame.sort_values("validation_mean_rank_ic")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(frame))))
    ax.barh(frame["label"], frame["validation_mean_rank_ic"])
    ax.axvline(0.0, linewidth=1)
    ax.set_title(f"{model_name}: Validation Configuration Search")
    ax.set_xlabel("Validation mean Rank IC")
    fig.tight_layout()
    path = out_dir / f"{model_name}_validation_selection.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_native_importance_plot(
    importances: pd.DataFrame, model_name: str, out_dir: Path
) -> Path:
    """Plot pre-test native tree feature importance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if importances.empty:
        raise ValueError("Feature-importance table is empty.")
    frame = importances.sort_values("normalized_importance")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(frame))))
    ax.barh(frame["feature"], frame["normalized_importance"])
    ax.set_title(f"{model_name}: Pre-Test Native Feature Importance")
    ax.set_xlabel("Normalized native importance")
    fig.tight_layout()
    path = out_dir / f"{model_name}_feature_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_model_rank_ic_comparison(ic_by_model: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Plot monthly test Rank IC for the frozen nonlinear models."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for model_name, frame in ic_by_model.items():
        if not frame.empty:
            ax.plot(frame["date"], frame["rank_ic"], label=model_name)
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Milestone 2C Test Monthly Rank IC")
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Spearman Rank IC")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "model_test_rank_ic.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_model_yearly_ic_comparison(yearly: pd.DataFrame, out_dir: Path) -> Path:
    """Plot annual mean Rank IC for each frozen nonlinear model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if yearly.empty:
        raise ValueError("Yearly IC table is empty.")
    pivot = yearly.pivot(index="year", columns="model", values="mean_rank_ic").sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Milestone 2C Test Mean Rank IC by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Rank IC")
    ax.legend(title="Model")
    fig.tight_layout()
    path = out_dir / "model_test_rank_ic_by_year.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_model_comparison_equity_curves(
    curves: dict[str, pd.Series], out_dir: Path
) -> Path:
    """Plot frozen nonlinear models against the already-frozen linear/baseline portfolios."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, series in curves.items():
        ax.plot(series.index, series.values, label=name)
    ax.set_title("Milestone 2C Test-Period Portfolio Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "test_equity_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
