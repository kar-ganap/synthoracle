"""Figure 7: Model Capability Gradient on Noisy.

Panel A: HV as % of BO (Opus 97%, Sonnet 96%, Haiku 44%)
Panel B: Info capture vs max noise-edge confidence (decoupling at Haiku)

All conditions: Noisy oracle, 72 evals, n=3 (seeds 42-44).

Usage:
    uv run python experiments/analysis/figures/fig7_model_gradient.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
SEEDS = [42, 43, 44]


def load_final_hv(path: Path) -> float:
    return float(np.load(path)["hypervolumes"][-1])


def main():
    # --- HV data ---
    opus = [load_final_hv(R / f"hd_vr_seed{s}.npz") for s in SEEDS]
    sonnet = [load_final_hv(R / f"hd_vr_sonnet_seed{s}.npz") for s in SEEDS]
    haiku = [load_final_hv(R / f"hd_vr_haiku_seed{s}.npz") for s in SEEDS]
    bo = [load_final_hv(R / f"bo_hd_seed{s}.npz") for s in SEEDS]

    bo_mean = np.mean(bo)

    models = ["Opus", "Sonnet", "Haiku"]
    hv_means = [np.mean(opus), np.mean(sonnet), np.mean(haiku)]
    hv_stds = [np.std(opus), np.std(sonnet), np.std(haiku)]
    hv_pcts = [m / bo_mean * 100 for m in hv_means]

    # --- Info capture (from audit data) ---
    info_capture = [0.996, 0.999, 0.959]  # Opus, Sonnet, Haiku

    # --- Max noise edge confidence per model (across seeds) ---
    import json
    def max_noise_conf(prefix):
        noise_inputs = {"X7", "X8", "X9", "X10", "X11", "X12"}
        confs = []
        for s in SEEDS:
            with open(R / f"{prefix}_seed{s}_log.json") as f:
                log = json.load(f)
            summaries = log.get("iteration_summaries", [])
            if summaries:
                for e in summaries[-1].get("edges", []):
                    if any(n in e["edge"] for n in noise_inputs):
                        confs.append(float(e["confidence"]))
        return max(confs) if confs else 0.0

    noise_confs = [
        max_noise_conf("hd_vr"),
        max_noise_conf("hd_vr_sonnet"),
        max_noise_conf("hd_vr_haiku"),
    ]

    # --- Colors ---
    c_opus = "#2166AC"
    c_sonnet = "#4393C3"
    c_haiku = "#D6604D"
    colors = [c_opus, c_sonnet, c_haiku]

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))

    # Panel A: HV as % of BO
    x = np.arange(3)
    bars = ax1.bar(x, hv_pcts, width=0.55, color=colors,
                    edgecolor="white", linewidth=1.5)
    # Error bars in percentage
    pct_errs = [s / bo_mean * 100 for s in hv_stds]
    ax1.errorbar(x, hv_pcts, yerr=pct_errs, fmt="none", color="#333",
                  capsize=6, lw=1.2, zorder=5)

    # 100% reference line
    ax1.axhline(100, color="#999", lw=1, ls="--", zorder=1)
    ax1.text(2.4, 101, "BO", fontsize=8, color="#999",
              family="sans-serif", va="bottom")

    # Value labels
    for i, (pct, bar) in enumerate(zip(hv_pcts, bars)):
        ax1.text(i, pct + pct_errs[i] + 2, f"{pct:.0f}%",
                  ha="center", va="bottom", fontsize=10,
                  fontweight="bold", color=colors[i], family="sans-serif")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylabel("HV (% of BO)")
    ax1.set_title("Optimization quality")
    ax1.set_ylim(0, 120)

    # Panel B: Info capture (left axis) + noise confidence (right axis)
    bar_width = 0.35

    # Info capture bars (left)
    bars1 = ax2.bar(x - bar_width / 2, info_capture, bar_width,
                     color=colors, alpha=0.7, edgecolor="white",
                     linewidth=1.5, label="Info capture")

    # Value labels for info capture
    for i, v in enumerate(info_capture):
        ax2.text(i - bar_width / 2, v + 0.01, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=8,
                  color=colors[i], family="sans-serif")

    ax2.set_ylabel("Sobol-weighted info capture")
    ax2.set_ylim(0, 1.25)

    # Noise confidence on right axis
    ax2r = ax2.twinx()
    bars2 = ax2r.bar(x + bar_width / 2, noise_confs, bar_width,
                      color=colors, alpha=0.3, edgecolor=colors,
                      linewidth=1.5, hatch="//", label="Max noise confidence")

    # Value labels for noise confidence
    for i, v in enumerate(noise_confs):
        ax2r.text(i + bar_width / 2, v + 0.02, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=8,
                   color=colors[i], family="sans-serif")

    ax2r.set_ylabel("Max noise-edge confidence")
    ax2r.set_ylim(0, 0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_title("Discovery vs. false positives")

    # Combined legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888", alpha=0.7, label="Info capture"),
        Patch(facecolor="#888", alpha=0.3, hatch="//", edgecolor="#888",
              label="Max noise confidence"),
    ]
    ax2.legend(handles=legend_elements, loc="upper center", fontsize=8,
                framealpha=0.9)

    # Shared formatting
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    ax1.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.5)

    for ext in ("png", "pdf"):
        path = OUT / f"fig7_model_gradient.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to {path}")
    plt.close()


if __name__ == "__main__":
    main()
