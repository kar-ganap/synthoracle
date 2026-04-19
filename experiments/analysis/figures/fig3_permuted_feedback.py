"""Figure 3: Permuted-Feedback Test.

Real vs permuted feedback on Baseline (1A), seed 42.
Shows the agent genuinely uses feedback — contra Gupta et al. (2025).

Usage:
    uv run python experiments/analysis/figures/fig3_permuted_feedback.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

R = Path("experiments/vr_agent/results")
OUT = Path("experiments/analysis/figures")


def main():
    real_hv = np.load(R / "ablation_real_seed42.npz")["hypervolumes"]
    perm_hv = np.load(R / "ablation_permuted_seed42.npz")["hypervolumes"]
    n_evals = len(real_hv)
    evals = np.arange(1, n_evals + 1)

    # Colors
    c_real = "#2166AC"
    c_perm = "#888888"

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # LHS region (first 12 evals — shared initial points)
    ax.axvspan(0, 12, color="#F0F0F0", zorder=0)
    ax.text(6, real_hv.max() * 0.02, "LHS\n(shared)",
            ha="center", va="bottom", fontsize=7.5,
            color="#AAA", family="sans-serif")

    # Divergence line at eval 13
    ax.axvline(13, color="#CCCCCC", lw=0.8, ls=":", zorder=1)

    # Plot trajectories
    ax.plot(evals, real_hv, color=c_real, lw=2.2, label="Real feedback",
            zorder=3)
    ax.plot(evals, perm_hv, color=c_perm, lw=2.2, label="Permuted feedback",
            zorder=3)

    # Final values annotation
    real_final = real_hv[-1]
    perm_final = perm_hv[-1]
    effect = (real_final - perm_final) / perm_final * 100

    ax.text(n_evals + 1, real_final, f"{real_final:.3f}",
            va="center", fontsize=8.5, color=c_real, family="sans-serif")
    ax.text(n_evals + 1, perm_final, f"{perm_final:.3f}",
            va="center", fontsize=8.5, color=c_perm, family="sans-serif")

    # Effect size annotation — between the two curves
    mid_y = (real_final + perm_final) / 2
    ax.annotate("", xy=(n_evals - 2, real_final - 0.005),
                xytext=(n_evals - 2, perm_final + 0.005),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.0))
    ax.text(n_evals - 5, mid_y, f"+{effect:.0f}%",
            ha="right", va="center", fontsize=10, fontweight="bold",
            color="#444", family="sans-serif")

    # Divergence label
    ax.text(14.5, real_hv[12] + 0.015, "divergence",
            fontsize=7.5, color="#999", family="sans-serif")

    ax.set_xlabel("Oracle evaluations")
    ax.set_ylabel("Hypervolume")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, n_evals + 8)
    ax.set_ylim(-0.005, real_hv.max() * 1.08)
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"fig3_permuted_feedback.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to {path}")
    plt.close()


if __name__ == "__main__":
    main()
