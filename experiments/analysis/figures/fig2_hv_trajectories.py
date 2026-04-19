"""Figure 2: HV Trajectories — VR vs BO on Baseline and Noisy.

Both panels show 144 evals. VR n=5 (seeds 42-46), BO n=3 (seeds 42-44).
Vertical dashed line at 72 evals marks the budget used in the ablation study.

Usage:
    uv run python experiments/analysis/figures/fig2_hv_trajectories.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

R = Path("experiments/vr_agent/results")
OUT = Path("experiments/analysis/figures")

VR_SEEDS = [42, 43, 44, 45, 46]
BO_SEEDS = [42, 43, 44]


def load_hv(path: Path) -> np.ndarray:
    return np.load(path)["hypervolumes"]


def stack_to_length(trajectories: list[np.ndarray], length: int) -> np.ndarray:
    """Pad shorter trajectories by repeating last value, trim longer ones."""
    result = []
    for h in trajectories:
        if len(h) >= length:
            result.append(h[:length])
        else:
            padded = np.full(length, h[-1])
            padded[: len(h)] = h
            result.append(padded)
    return np.array(result)


def main():
    # --- Load data ---
    vr_base = stack_to_length(
        [load_hv(R / f"1a_extended_144_seed{s}.npz") for s in VR_SEEDS], 144
    )
    bo_base = stack_to_length(
        [load_hv(R / f"bo_1a_n144_seed{s}.npz") for s in BO_SEEDS], 144
    )
    vr_noisy = stack_to_length(
        [load_hv(R / f"hd_vr_ext_seed{s}.npz") for s in VR_SEEDS], 144
    )
    bo_noisy = stack_to_length(
        [load_hv(R / f"bo_hd_n144_seed{s}.npz") for s in BO_SEEDS], 144
    )

    evals = np.arange(1, 145)

    # --- Colors ---
    c_vr = "#2166AC"
    c_bo = "#B2182B"
    c_vr_fill = "#92C5DE"
    c_bo_fill = "#F4A582"

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, vr, bo, title, lhs_end in [
        (ax1, vr_base, bo_base, "Baseline (k/d = 1.17)", 12),
        (ax2, vr_noisy, bo_noisy, "Noisy (k/d = 0.58)", 24),
    ]:
        vr_mean, vr_std = vr.mean(axis=0), vr.std(axis=0)
        bo_mean, bo_std = bo.mean(axis=0), bo.std(axis=0)

        # LHS shading
        ax.axvspan(0, lhs_end, color="#F0F0F0", zorder=0)
        ax.text(lhs_end / 2, 0.003, "LHS", ha="center", fontsize=7.5,
                color="#AAA", family="sans-serif")

        # BO band + line
        ax.fill_between(evals, bo_mean - bo_std, bo_mean + bo_std,
                         color=c_bo_fill, alpha=0.25)
        ax.plot(evals, bo_mean, color=c_bo, lw=2, label=f"BO (n=3)")

        # VR band + line
        ax.fill_between(evals, vr_mean - vr_std, vr_mean + vr_std,
                         color=c_vr_fill, alpha=0.35)
        ax.plot(evals, vr_mean, color=c_vr, lw=2, label=f"VR (n=5)")

        # VR/BO annotation at 144
        vr_final = vr_mean[-1]
        bo_final = bo_mean[-1]
        ratio = vr_final / bo_final * 100
        ax.annotate(
            f"VR/BO = {ratio:.0f}%",
            xy=(144, vr_final),
            xytext=(105, vr_final - 0.035),
            fontsize=9, fontweight="bold", color=c_vr,
            family="sans-serif",
            arrowprops=dict(arrowstyle="-", color=c_vr, lw=0.6),
        )

        ax.set_title(title)
        ax.set_xlabel("Oracle evaluations")
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, 150)
        ax.grid(axis="y", alpha=0.15)

    ax1.set_ylabel("Hypervolume")
    ax1.set_ylim(-0.005, 0.32)

    fig.tight_layout(w_pad=2.0)

    for ext in ("png", "pdf"):
        path = OUT / f"fig2_hv_trajectories.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved to {path}")
    plt.close()


if __name__ == "__main__":
    main()
