from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_cross_sectional_coverage_plot(coverage: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(coverage["date"]), coverage["assets"], label="Available assets")
    ax.plot(
        pd.to_datetime(coverage["date"]),
        coverage["complete_feature_rows"],
        label="Complete feature rows",
    )
    ax.set_title("Cross-sectional data coverage")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of securities")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "cross_sectional_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_annual_strategy_returns_plot(annual: pd.DataFrame, out_dir: Path) -> Path | None:
    if annual.empty:
        return None
    pivot = annual.pivot(index="year", columns="strategy", values="net_return")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Annual net returns by baseline strategy")
    ax.set_xlabel("Year")
    ax.set_ylabel("Net return")
    fig.tight_layout()
    path = out_dir / "annual_strategy_returns.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
