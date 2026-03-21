"""Comparison plots for optimization methods."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthoracle.eval.metrics import ComparisonMetrics


def plot_hv_comparison(
    metrics: ComparisonMetrics,
    output_path: Path,
    title: str = "Hypervolume Convergence",
    n_initial: int = 12,
) -> None:
    """Plot mean ± 1 std HV curves for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, name in enumerate(metrics.method_names):
        mean = metrics.hv_mean[name]
        std = metrics.hv_std[name]
        evals = np.arange(1, len(mean) + 1)
        color = colors[i % len(colors)]

        ax.plot(evals, mean, linewidth=1.5, label=name, color=color)
        ax.fill_between(evals, mean - std, mean + std, alpha=0.2, color=color)

    ax.axvline(x=n_initial, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_hv_boxplot(
    metrics: ComparisonMetrics,
    output_path: Path,
    title: str = "Final Hypervolume Distribution",
) -> None:
    """Plot side-by-side boxplots of final HV per method."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    data = [metrics.final_hv[name] for name in metrics.method_names]
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticklabels(metrics.method_names)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    ax.set_ylabel("Final Hypervolume")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
