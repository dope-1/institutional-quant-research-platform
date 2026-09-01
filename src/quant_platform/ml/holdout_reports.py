from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_holdout_equity_curves(curves: dict[str, pd.Series], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, series in curves.items():
        s = series.dropna().copy()
        s.index = pd.to_datetime(s.index)
        ax.plot(s.index, s.values, label=name, linewidth=1.7)
    ax.set_title("Genuine 2026 holdout: portfolio equity curves")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_equity_curves.png", dpi=160)
    plt.close(fig)


def write_holdout_rank_ic(rank_ic: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank_ic.empty:
        return
    table = rank_ic.pivot(index="date", columns="model", values="rank_ic").sort_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for column in table.columns:
        ax.plot(table.index, table[column], marker="o", label=str(column))
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("2026 matured monthly Rank IC")
    ax.set_ylabel("Spearman Rank IC")
    ax.set_xlabel("Signal date")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_rank_ic.png", dpi=160)
    plt.close(fig)


def write_holdout_model_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics.empty or "sharpe" not in metrics.columns:
        return
    table = metrics.sort_values("sharpe", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(table["strategy"].astype(str), table["sharpe"])
    ax.set_title("2026 holdout Sharpe comparison")
    ax.set_ylabel("Sharpe")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_sharpe_comparison.png", dpi=160)
    plt.close(fig)
