from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _pretty(text: str) -> str:
    return text.replace("_regime", "").replace("_", " ").title()


def write_regime_ic_heatmaps(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if table.empty:
        return paths
    for dimension, group in table.groupby("regime_dimension", sort=True):
        pivot = group.pivot(index="model", columns="regime", values="mean").sort_index()
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(pivot.columns)), max(4, 0.7 * len(pivot))))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Mean Rank IC by {_pretty(str(dimension))}")
        fig.colorbar(image, ax=ax, label="Mean Rank IC")
        fig.tight_layout()
        path = out_dir / f"rank_ic_{dimension}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_regime_portfolio_heatmaps(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if table.empty:
        return paths
    for dimension, group in table.groupby("regime_dimension", sort=True):
        pivot = group.pivot(index="strategy", columns="regime", values="conditional_sharpe").sort_index()
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(pivot.columns)), max(4, 0.65 * len(pivot))))
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Conditional Holding-Period Sharpe by {_pretty(str(dimension))}")
        fig.colorbar(image, ax=ax, label="Conditional Sharpe")
        fig.tight_layout()
        path = out_dir / f"portfolio_sharpe_{dimension}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_regime_timeline(signal_regimes: pd.DataFrame, out_dir: Path) -> Path:
    """Write a compact numeric timeline for the three market-state variables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = signal_regimes.copy().sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 5))
    if "vix" in frame.columns:
        ax.plot(frame["date"], frame["vix"], label="VIX")
    if "treasury_10y" in frame.columns:
        ax.plot(frame["date"], frame["treasury_10y"], label="10Y Treasury")
    if "yield_curve_spread" in frame.columns:
        ax.plot(frame["date"], frame["yield_curve_spread"], label="10Y-2Y spread")
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Signal-Date Market Regime Inputs")
    ax.set_xlabel("Signal date")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "regime_inputs_timeline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
