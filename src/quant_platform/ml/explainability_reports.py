from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_global_importance_comparison(shap_summary: pd.DataFrame, permutation: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model in sorted(shap_summary["model"].unique()):
        s = shap_summary.loc[shap_summary["model"] == model].copy().sort_values("normalized_mean_abs_shap")
        p = permutation.loc[permutation["model"] == model].copy().sort_values("importance_mean")
        fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(s))))
        ax.barh(s["feature"], s["normalized_mean_abs_shap"])
        ax.set_title(f"{model}: Global Mean |SHAP| Importance")
        ax.set_xlabel("Normalized mean absolute SHAP")
        fig.tight_layout()
        path = out_dir / f"{model}_shap_importance.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(p))))
        ax.barh(p["feature"], p["importance_mean"])
        ax.axvline(0.0, linewidth=1)
        ax.set_title(f"{model}: Permutation Importance by Rank-IC Degradation")
        ax.set_xlabel("Decrease in Spearman Rank IC after permutation")
        fig.tight_layout()
        path = out_dir / f"{model}_permutation_importance.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_yearly_shap_heatmaps(yearly: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model in sorted(yearly["model"].unique()):
        frame = yearly.loc[yearly["model"] == model]
        pivot = frame.pivot(index="feature", columns="year", values="normalized_importance").fillna(0.0)
        fig, ax = plt.subplots(figsize=(8, max(5, 0.42 * len(pivot))))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.astype(str))
        ax.set_title(f"{model}: SHAP Importance Stability by Year")
        ax.set_xlabel("Historical evaluation year")
        fig.colorbar(image, ax=ax, label="Normalized mean |SHAP|")
        fig.tight_layout()
        path = out_dir / f"{model}_yearly_shap_stability.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_shap_dependence_plot(
    wide: pd.DataFrame,
    *,
    model_name: str,
    feature_x: str,
    feature_y: str,
    shap_x: str,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = wide.dropna(subset=[feature_x, feature_y, shap_x])
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(frame[feature_x], frame[shap_x], c=frame[feature_y], alpha=0.55)
    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"{model_name}: SHAP Dependence")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(f"SHAP({feature_x})")
    fig.colorbar(sc, ax=ax, label=feature_y)
    fig.tight_layout()
    safe_x = feature_x.replace("rank__", "")
    safe_y = feature_y.replace("rank__", "")
    path = out_dir / f"{model_name}_dependence_{safe_x}_x_{safe_y}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_top_interactions_plot(interactions: pd.DataFrame, out_dir: Path, top_n: int = 12) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model in sorted(interactions["model"].unique()):
        frame = interactions.loc[interactions["model"] == model].dropna(subset=["mean_abs_interaction"]).head(top_n).copy()
        if frame.empty:
            continue
        frame["pair"] = frame["feature_a"] + " × " + frame["feature_b"]
        frame = frame.sort_values("mean_abs_interaction")
        fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(frame))))
        ax.barh(frame["pair"], frame["mean_abs_interaction"])
        ax.set_title(f"{model}: Strongest Tree-SHAP Interactions")
        ax.set_xlabel("Mean absolute SHAP interaction value")
        fig.tight_layout()
        path = out_dir / f"{model}_top_shap_interactions.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths
