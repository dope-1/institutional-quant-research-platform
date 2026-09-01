from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_metrics_report(metrics_by_strategy: dict[str, dict[str, float]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(metrics_by_strategy).T
    path = out_dir / "performance_metrics.csv"
    frame.to_csv(path, index_label="strategy")
    return path


def write_equity_curve_plot(curves: dict[str, pd.Series], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, series in curves.items():
        ax.plot(series.index, series.values, label=name)
    ax.set_title("Milestone 1 Baseline Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "equity_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
