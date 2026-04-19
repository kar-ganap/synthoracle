"""Figure 6: Mechanism Isolation — Context Cementing, Not Cognitive Cost.

Three conditions on Baseline (1A), all Opus:
  1. VR full: summary produced AND persisted in context (n=5)
  2. Summary-not-fed: summary produced but NOT in context (n=3)
  3. No summary: summary never produced (n=5)

Key result: conditions 2 ≈ 3 (p=0.80), both >> 1 (p=0.002).
Mechanism: persistence in context, not act of articulation.

Usage:
    uv run python experiments/analysis/figures/fig6_mechanism_isolation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

R = Path("experiments/vr_agent/results")
OUT = Path("experiments/analysis/figures")


def load_final_hv(path: Path) -> float:
    return float(np.load(path)["hypervolumes"][-1])


def main():
    # --- Load data ---
    seeds = [42, 43, 44, 45, 46]
    vr_full = [load_final_hv(R / f"multi_seed/seed{s}.npz") for s in seeds]
    not_fed = [load_final_hv(R / f"ablation_summary_not_fed_seed{s}.npz") for s in seeds]
    no_summ = [load_final_hv(R / f"ablation_no_summary_seed{s}.npz") for s in seeds]

    # BO reference
    bo_hv = load_final_hv(R / "bo_1a_n144_seed42.npz")

    # Stats
    means = [np.mean(vr_full), np.mean(not_fed), np.mean(no_summ)]
    stds = [np.std(vr_full), np.std(not_fed), np.std(no_summ)]
    ns = [len(vr_full), len(not_fed), len(no_summ)]

    # P-values (paired, all n=5 same seeds)
    _, p_nf_vs_ns = stats.ttest_rel(not_fed, no_summ)
    _, p_vr_vs_nf = stats.ttest_rel(vr_full, not_fed)

    # --- Colors ---
    c_vr = "#B2182B"       # red — the bad condition
    c_notfed = "#2166AC"   # blue
    c_nosumm = "#2166AC"   # blue (same — these two are equivalent)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    x = [0, 1, 2]
    colors = [c_vr, c_notfed, c_nosumm]
    labels = [
        "Summary\nproduced &\nin context",
        "Summary\nproduced but\nnot in context",
        "No\nsummary",
    ]
    short_labels = ["Full VR", "Not fed back", "No summary"]

    bars = ax.bar(x, means, width=0.6, color=colors, edgecolor="white",
                   linewidth=1.5, alpha=0.85)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="#333",
                 capsize=6, lw=1.2, zorder=5)

    # Individual seed dots
    rng = np.random.default_rng(42)
    for xpos, vals in [(0, vr_full), (1, not_fed), (2, no_summ)]:
        jitter = rng.uniform(-0.06, 0.06, len(vals))
        ax.scatter(xpos + jitter, vals, color="#333", s=18, alpha=0.5,
                    zorder=6)

    # BO reference line
    ax.axhline(bo_hv, color="#999", lw=1, ls="--", zorder=1)
    ax.text(2.45, bo_hv + 0.002, f"BO ({bo_hv:.3f})",
             fontsize=8, color="#999", family="sans-serif", va="bottom")

    # Bracket: not-fed ≈ no-summary (p=0.35)
    bracket_y = max(means[1] + stds[1], means[2] + stds[2]) + 0.012
    ax.plot([1, 1, 2, 2], [bracket_y - 0.005, bracket_y, bracket_y, bracket_y - 0.005],
             color="#444", lw=1)
    ax.text(1.5, bracket_y + 0.003, f"p = {p_nf_vs_ns:.2f}  (no difference)",
             ha="center", va="bottom", fontsize=8.5, color="#444",
             family="sans-serif")

    # Bracket: VR vs no-summary (spans bar 0 to bar 2)
    bracket_y2 = bracket_y + 0.030
    ax.plot([0, 0, 2, 2], [bracket_y2 - 0.005, bracket_y2, bracket_y2, bracket_y2 - 0.005],
             color="#B2182B", lw=1)
    ax.text(1.0, bracket_y2 + 0.003, f"p = {p_vr_vs_nf:.4f}",
             ha="center", va="bottom", fontsize=8.5, color="#B2182B",
             family="sans-serif")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Hypervolume")
    ax.set_ylim(0, bracket_y2 + 0.055)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"fig6_mechanism_isolation.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to {path}")
    plt.close()


if __name__ == "__main__":
    main()
