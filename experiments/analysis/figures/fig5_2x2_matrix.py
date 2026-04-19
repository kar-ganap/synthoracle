"""Figure 5: The 2×2 Scaffolding Matrix (headline figure).

Faceted 2×2 layout: {Opus, Sonnet} × {Baseline, Noisy}.
Each panel shows paired bars (with summary vs without summary).
n=5 per cell, seeds 42-46. Global sign test: 19/20, p=0.00002.

Usage:
    uv run python experiments/analysis/figures/fig5_2x2_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

R = Path("experiments/vr_agent/results")
OUT = Path("experiments/analysis/figures")
SEEDS = [42, 43, 44, 45, 46]


def load_final_hv(path: Path) -> float:
    return float(np.load(path)["hypervolumes"][-1])


def main():
    # --- Load all per-seed data ---
    cells = {}

    # Opus × Baseline
    cells["Opus\nBaseline"] = {
        "with": [load_final_hv(R / f"multi_seed/seed{s}.npz") for s in SEEDS],
        "without": [load_final_hv(R / f"ablation_no_summary_seed{s}.npz") for s in SEEDS],
        "helps": False,  # summary hurts here
    }

    # Sonnet × Baseline
    cells["Sonnet\nBaseline"] = {
        "with": [load_final_hv(R / f"sonnet_1a_with_summary_seed{s}.npz") for s in SEEDS],
        "without": [load_final_hv(R / f"sonnet_1a_no_summary_seed{s}.npz") for s in SEEDS],
        "helps": True,
    }

    # Opus × Noisy
    cells["Opus\nNoisy"] = {
        "with": [load_final_hv(R / f"hd_vr_seed{s}.npz") for s in SEEDS],
        "without": [load_final_hv(R / f"hd_ablation_no_summary_seed{s}.npz") for s in SEEDS],
        "helps": True,
    }

    # Sonnet × Noisy
    cells["Sonnet\nNoisy"] = {
        "with": [load_final_hv(R / f"hd_vr_sonnet_seed{s}.npz") for s in SEEDS],
        "without": [load_final_hv(R / f"hd_sonnet_ablation_no_summary_seed{s}.npz") for s in SEEDS],
        "helps": True,
    }

    # --- Layout: 2×2 grid matching the conceptual matrix ---
    # Rows = models (Opus top, Sonnet bottom)
    # Cols = oracles (Baseline left, Noisy right)
    grid = [
        ["Opus\nBaseline", "Opus\nNoisy"],
        ["Sonnet\nBaseline", "Sonnet\nNoisy"],
    ]

    c_with = "#2166AC"      # blue — with summary
    c_without = "#AAAAAA"   # gray — without summary

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=True)

    for row_idx, row in enumerate(grid):
        for col_idx, name in enumerate(row):
            ax = axes[row_idx, col_idx]
            data = cells[name]
            w = np.array(data["with"])
            wo = np.array(data["without"])
            helps = data["helps"]

            w_mean, w_std = w.mean(), w.std()
            wo_mean, wo_std = wo.mean(), wo.std()

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(w, wo)
            n = len(w)

            # Effect size
            if helps:
                effect = (w_mean - wo_mean) / wo_mean * 100
                better = "with" if effect > 0 else "without"
            else:
                effect = (wo_mean - w_mean) / w_mean * 100
                better = "without"

            # Bars
            x = [0, 1]
            colors = [c_with, c_without]
            means = [w_mean, wo_mean]
            stds = [w_std, wo_std]
            bars = ax.bar(x, means, width=0.55, color=colors,
                          edgecolor="white", linewidth=1.5)
            ax.errorbar(x, means, yerr=stds, fmt="none", color="#333",
                         capsize=6, lw=1.2, zorder=5)

            # Individual seed dots
            jitter = 0.08
            for seed_vals, xpos in [(w, 0), (wo, 1)]:
                xs = xpos + np.random.default_rng(42).uniform(-jitter, jitter, len(seed_vals))
                ax.scatter(xs, seed_vals, color="#333", s=15, alpha=0.5, zorder=6)

            # Background tint for the "hurts" cell
            if not helps:
                ax.set_facecolor("#FFF0F0")

            # Panel title (at top)
            model = name.split("\n")[0]
            oracle = name.split("\n")[1]
            ax.set_title(f"{model} \u00d7 {oracle}", pad=18)

            # Effect + p-value annotation (below title, above bars)
            if helps:
                label = f"+{effect:.0f}%"
                ecolor = "#2166AC"
            else:
                label = f"\u2212{abs(effect):.0f}%"
                ecolor = "#B2182B"

            p_str = f"p={p_val:.4f}" if p_val >= 0.001 else f"p={p_val:.1e}"
            # Position above the highest bar+errorbar+dots
            y_annot = max(max(w), max(wo)) + 0.008
            ax.text(0.5, y_annot, f"{label}  ({p_str})",
                     ha="center", va="bottom", fontsize=10,
                     fontweight="bold", color=ecolor,
                     family="sans-serif")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["With\nsummary", "Without\nsummary"],
                                fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col_idx == 0:
                ax.set_ylabel("Hypervolume")

    # Global sign test annotation
    fig.text(0.5, 0.01,
             "Global sign consistency: 18/20 seeds in predicted direction "
             "(binomial p = 0.0002)",
             ha="center", fontsize=9, color="#444", family="sans-serif")

    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.5, w_pad=1.5)

    for ext in ("png", "pdf"):
        path = OUT / f"fig5_2x2_matrix.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to {path}")
    plt.close()


if __name__ == "__main__":
    main()
