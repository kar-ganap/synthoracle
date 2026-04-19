"""Figure 4: Learning Trajectories — Surprise Count.

Single panel showing surprise count decreasing over iterations.
Both oracles at 144 evals, matched seeds (42-44), n=3 each.
Truncated to iterations where all seeds have data.

Usage:
    uv run python experiments/analysis/figures/fig4_learning_trajectories.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

R = Path("experiments/vr_agent/results")
OUT = Path("experiments/analysis/figures")

SEEDS = [42, 43, 44, 45, 46]


def load_surprises(log_path: Path) -> list[int]:
    with open(log_path) as f:
        log = json.load(f)
    return [len(s["surprises"]) for s in log["iteration_summaries"]]


def main():
    # Both conditions: 144 evals, seeds 42-44 (n=3 matched)
    base_surp = [load_surprises(R / f"1a_extended_144_seed{s}_log.json")
                  for s in SEEDS]
    noisy_surp = [load_surprises(R / f"hd_vr_ext_seed{s}_log.json")
                   for s in SEEDS]

    # Truncate to where ALL seeds in BOTH conditions have data
    min_iters = min(min(len(s) for s in base_surp),
                    min(len(s) for s in noisy_surp))
    print(f"Truncating to {min_iters} iterations (all seeds × both conditions)")

    def stats(all_series, n_iters):
        result = []
        for i in range(n_iters):
            vals = [float(s[i]) for s in all_series]
            result.append((i + 1, np.mean(vals), np.std(vals)))
        return result

    base_agg = stats(base_surp, min_iters)
    noisy_agg = stats(noisy_surp, min_iters)

    # --- Colors ---
    c_base = "#2166AC"
    c_noisy = "#B2182B"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for agg, color, marker, label in [
        (base_agg, c_base, "o", "Baseline"),
        (noisy_agg, c_noisy, "s", "Noisy"),
    ]:
        iters = [d[0] for d in agg]
        means = [d[1] for d in agg]
        stds = [d[2] for d in agg]
        ax.errorbar(iters, means, yerr=stds, color=color, lw=2,
                     marker=marker, markersize=7, capsize=5, label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Surprise count")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
)
    ax.set_ylim(0, 8)
    ax.set_xlim(0.5, min_iters + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"fig4_learning_trajectories.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to {path}")
    plt.close()


if __name__ == "__main__":
    main()
