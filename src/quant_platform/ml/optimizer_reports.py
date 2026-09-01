from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_optimizer_validation_frontier(selection: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if selection.empty:
        raise ValueError("Optimizer selection table is empty.")
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, group in selection.groupby("model", sort=True):
        ax.scatter(group["annualized_turnover"], group["sharpe"], label=model)
        for _, row in group.iterrows():
            ax.annotate(
                str(row["candidate"]),
                (row["annualized_turnover"], row["sharpe"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_title("Milestone 3C Validation: Sharpe vs Turnover")
    ax.set_xlabel("Annualised turnover")
    ax.set_ylabel("Validation Sharpe")
    ax.legend(title="Alpha model")
    fig.tight_layout()
    path = out_dir / "validation_sharpe_vs_turnover.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_optimizer_equity_curves(curves: dict[str, pd.Series], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, series in curves.items():
        ax.plot(series.index, series.values, label=name)
    ax.set_title("Milestone 3C Historical Evaluation: Optimized Portfolios")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "optimized_equity_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_optimizer_concentration(diagnostics: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics.empty:
        raise ValueError("Optimizer diagnostics are empty.")
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, group in diagnostics.groupby("model", sort=True):
        ax.plot(group["date"], group["effective_n"], label=model)
    ax.set_title("Milestone 3C Effective Number of Holdings")
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Effective N = 1 / sum(w²)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "effective_n_by_month.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_optimizer_risk_turnover(diagnostics: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics.empty:
        raise ValueError("Optimizer diagnostics are empty.")
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, group in diagnostics.groupby("model", sort=True):
        ax.scatter(group["ex_ante_volatility"], group["optimizer_turnover"], label=model)
    ax.set_title("Milestone 3C Ex-Ante Risk vs Target Turnover")
    ax.set_xlabel("Ex-ante annualized volatility")
    ax.set_ylabel("Signal-date turnover")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "ex_ante_risk_vs_turnover.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
