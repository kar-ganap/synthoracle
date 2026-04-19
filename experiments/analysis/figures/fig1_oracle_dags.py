"""Figure 1: Oracle Family DAGs.

Four-panel figure showing the causal structure of each oracle.
Focus on input→mechanism wiring (which varies) + output layer.
Mechanism→output edges are uniform and noted in text.

Usage:
    uv run python experiments/analysis/figures/fig1_oracle_dags.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# --- Style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

# Colors
C_INPUT = "#4878CF"
C_MECH = "#6ACC65"
C_OUTPUT = "#D65F5F"
C_NOISE = "#CCCCCC"
C_EDGE = "#555555"
C_EDGE_NEW = "#E69F00"    # orange — new/rewired
C_EDGE_DIRECT = "#56B4E9"  # light blue — direct X→Y
C_BG = "white"

R = 0.028  # node radius


def draw_node(ax, x, y, label, color, fontsize=8.5, radius=None):
    r = radius if radius is not None else R
    circle = plt.Circle((x, y), r, fc=color, ec="black",
                         linewidth=0.8, zorder=10, alpha=0.9)
    ax.add_patch(circle)
    # Dark text on light backgrounds, white on dark
    tc = "white" if color not in (C_NOISE, "#EEEEEE") else "#555"
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", zorder=11, color=tc)


def draw_edge(ax, x1, y1, x2, y2, color=C_EDGE, lw=0.9, style="-",
              alpha=0.55):
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / dist, dy / dist
    sx, sy = x1 + ux * R, y1 + uy * R
    ex, ey = x2 - ux * (R + 0.008), y2 - uy * (R + 0.008)

    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, linestyle=style,
                                shrinkA=0, shrinkB=0,
                                mutation_scale=8),
                zorder=5)


def setup_panel(ax, title, subtitle=""):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(C_BG)
    ax.set_title(title, pad=10)
    if subtitle:
        ax.text(0.5, 0.95, subtitle, ha="center", va="top",
                fontsize=8, color="#555", family="sans-serif",
                transform=ax.transAxes)


# --- Layout positions ---
IX = 0.10   # input column
MX = 0.50   # mechanism column
OX = 0.90   # output column


def spread(n, top=0.88, bot=0.12):
    """Evenly space n items between top and bot."""
    if n == 1:
        return [(top + bot) / 2]
    return [top - i * (top - bot) / (n - 1) for i in range(n)]


MECHS = ["M1", "M2", "M3", "M4"]
MECH_LABELS = {
    "M1": "M1\nthrpt",
    "M2": "M2\nleak",
    "M3": "Z\ncplg",
    "M4": "M4\nthresh",
}
OUTPUTS = ["Y1", "Y2", "Y3", "Y4"]


def draw_baseline(ax):
    setup_panel(ax, "Baseline", "6 inputs, 4 mechanisms, 4 outputs")

    iy = {f"X{i+1}": y for i, y in enumerate(spread(6))}
    my = {m: y for m, y in zip(MECHS, spread(4, top=0.80, bot=0.20))}
    oy = {o: y for o, y in zip(OUTPUTS, spread(4, top=0.80, bot=0.20))}

    for inp, y in iy.items():
        draw_node(ax, IX, y, inp, C_INPUT)
    for m, y in my.items():
        draw_node(ax, MX, y, m, C_MECH, fontsize=7.5)
    for o, y in oy.items():
        draw_node(ax, OX, y, o, C_OUTPUT)

    # X → M (the distinctive wiring)
    xm = [
        ("X1", "M1"), ("X2", "M1"), ("X4", "M1"),
        ("X1", "M2"), ("X3", "M2"),
        ("X4", "M3"), ("X6", "M3"),
        ("X5", "M4"), ("X6", "M4"),
    ]
    for inp, m in xm:
        draw_edge(ax, IX, iy[inp], MX, my[m])

    # M → Y (all mechanisms feed Y1, Y2; subset feed Y3, Y4)
    my_edges = [
        ("M1", "Y1"), ("M1", "Y2"), ("M1", "Y4"),
        ("M2", "Y1"), ("M2", "Y2"),
        ("M3", "Y1"), ("M3", "Y2"), ("M3", "Y4"),
        ("M4", "Y1"), ("M4", "Y2"),
    ]
    for m, o in my_edges:
        draw_edge(ax, MX, my[m], OX, oy[o], color="#AAAAAA", lw=0.5,
                  alpha=0.35)

    # Direct X → Y
    for inp, o in [("X1", "Y3"), ("X3", "Y3")]:
        draw_edge(ax, IX, iy[inp], OX, oy[o],
                  color=C_EDGE_DIRECT, style="--", lw=0.8)


def draw_shifted(ax):
    setup_panel(ax, "Shifted", "Same topology, altered functional forms")

    iy = {f"X{i+1}": y for i, y in enumerate(spread(6))}
    my = {m: y for m, y in zip(MECHS, spread(4, top=0.80, bot=0.20))}
    oy = {o: y for o, y in zip(OUTPUTS, spread(4, top=0.80, bot=0.20))}

    for inp, y in iy.items():
        draw_node(ax, IX, y, inp, C_INPUT)
    for m, y in my.items():
        draw_node(ax, MX, y, m, C_MECH, fontsize=7.5)
    for o, y in oy.items():
        draw_node(ax, OX, y, o, C_OUTPUT)

    # Same X→M as Baseline except X5→M4 removed (threshold disabled)
    xm_shared = [
        ("X1", "M1"), ("X2", "M1"), ("X4", "M1"),
        ("X1", "M2"), ("X3", "M2"),
        ("X4", "M3"), ("X6", "M3"),
        ("X6", "M4"),
    ]
    for inp, m in xm_shared:
        draw_edge(ax, IX, iy[inp], MX, my[m])

    # M→Y (light, same as baseline)
    my_edges = [
        ("M1", "Y1"), ("M1", "Y2"), ("M1", "Y4"),
        ("M2", "Y1"), ("M2", "Y2"),
        ("M3", "Y1"), ("M3", "Y2"), ("M3", "Y4"),
        ("M4", "Y1"), ("M4", "Y2"),
    ]
    for m, o in my_edges:
        draw_edge(ax, MX, my[m], OX, oy[o], color="#AAAAAA", lw=0.5,
                  alpha=0.35)

    # Direct X→Y: original
    for inp, o in [("X1", "Y3"), ("X3", "Y3")]:
        draw_edge(ax, IX, iy[inp], OX, oy[o],
                  color=C_EDGE_DIRECT, style="--", lw=0.8)

    # NEW direct paths (orange, thicker)
    for inp, o in [("X5", "Y1"), ("X5", "Y2"), ("X3", "Y4")]:
        draw_edge(ax, IX, iy[inp], OX, oy[o],
                  color=C_EDGE_NEW, lw=1.4)

    # Mark removed edge: X5→M4 was present in Baseline
    # Draw it faintly with a strike-through
    draw_edge(ax, IX, iy["X5"], MX, my["M4"],
              color="#CC79A7", lw=0.7, style=":", alpha=0.4)
    mid_x = (IX + MX) / 2 - 0.02
    mid_y = (iy["X5"] + my["M4"]) / 2
    ax.text(mid_x, mid_y, "X", fontsize=9, color="#CC79A7",
            ha="center", va="center", fontweight="bold", zorder=15,
            family="sans-serif")


def draw_rewired(ax):
    setup_panel(ax, "Rewired", "Input\u2192mechanism wiring changed")

    iy = {f"X{i+1}": y for i, y in enumerate(spread(6))}
    my = {m: y for m, y in zip(MECHS, spread(4, top=0.80, bot=0.20))}
    oy = {o: y for o, y in zip(OUTPUTS, spread(4, top=0.80, bot=0.20))}

    for inp, y in iy.items():
        draw_node(ax, IX, y, inp, C_INPUT)
    for m, y in my.items():
        draw_node(ax, MX, y, m, C_MECH, fontsize=7.5)
    for o, y in oy.items():
        draw_node(ax, OX, y, o, C_OUTPUT)

    # Shared edges (kept from Baseline)
    xm_shared = [("X1", "M1"), ("X1", "M2")]
    for inp, m in xm_shared:
        draw_edge(ax, IX, iy[inp], MX, my[m])

    # Rewired edges (orange)
    xm_new = [
        ("X3", "M1"), ("X6", "M1"),   # was X2, X4
        ("X4", "M2"),                   # was X3
        ("X2", "M3"), ("X5", "M3"),      # was X4, X6
        ("X6", "M4"), ("X5", "M4"),    # was X5, X6 → X6, X5
    ]
    for inp, m in xm_new:
        draw_edge(ax, IX, iy[inp], MX, my[m],
                  color=C_EDGE_NEW, lw=1.4)

    # M→Y (same structure)
    my_edges = [
        ("M1", "Y1"), ("M1", "Y2"), ("M1", "Y4"),
        ("M2", "Y1"), ("M2", "Y2"),
        ("M3", "Y1"), ("M3", "Y2"), ("M3", "Y4"),
        ("M4", "Y1"), ("M4", "Y2"),
    ]
    for m, o in my_edges:
        draw_edge(ax, MX, my[m], OX, oy[o], color="#AAAAAA", lw=0.5,
                  alpha=0.35)

    # Direct: X1→Y3 (shared), X4→Y3 (new, was X3)
    draw_edge(ax, IX, iy["X1"], OX, oy["Y3"],
              color=C_EDGE_DIRECT, style="--", lw=0.8)
    draw_edge(ax, IX, iy["X4"], OX, oy["Y3"],
              color=C_EDGE_NEW, lw=1.4)


def draw_noisy(ax):
    setup_panel(ax, "Noisy", "Baseline + 6 irrelevant noise dimensions")

    # Real inputs
    real_ys = spread(6, top=0.88, bot=0.42)
    iy = {f"X{i+1}": y for i, y in enumerate(real_ys)}

    # Noise inputs — two columns of 3 to save vertical space
    noise_col1_ys = spread(3, top=0.28, bot=0.04)
    noise_col2_ys = spread(3, top=0.28, bot=0.04)
    noise_x1 = IX - 0.01
    noise_x2 = IX + 0.09
    for i in range(3):
        iy[f"X{i+7}"] = noise_col1_ys[i]
    for i in range(3):
        iy[f"X{i+10}"] = noise_col2_ys[i]

    my = {m: y for m, y in zip(MECHS, spread(4, top=0.80, bot=0.42))}
    oy = {o: y for o, y in zip(OUTPUTS, spread(4, top=0.80, bot=0.42))}

    ax.set_ylim(-0.08, 1.02)

    # Draw nodes
    for i in range(1, 7):
        draw_node(ax, IX, iy[f"X{i}"], f"X{i}", C_INPUT)
    for i in range(7, 10):
        draw_node(ax, noise_x1, iy[f"X{i}"], f"X{i}", C_NOISE, fontsize=8, radius=0.033)
    for i in range(10, 13):
        draw_node(ax, noise_x2, iy[f"X{i}"], f"X{i}", C_NOISE, fontsize=8, radius=0.033)
    for m, y in my.items():
        draw_node(ax, MX, y, m, C_MECH, fontsize=7.5)
    for o, y in oy.items():
        draw_node(ax, OX, y, o, C_OUTPUT)

    # Same X→M as Baseline
    xm = [
        ("X1", "M1"), ("X2", "M1"), ("X4", "M1"),
        ("X1", "M2"), ("X3", "M2"),
        ("X4", "M3"), ("X6", "M3"),
        ("X5", "M4"), ("X6", "M4"),
    ]
    for inp, m in xm:
        draw_edge(ax, IX, iy[inp], MX, my[m])

    # M→Y
    my_edges = [
        ("M1", "Y1"), ("M1", "Y2"), ("M1", "Y4"),
        ("M2", "Y1"), ("M2", "Y2"),
        ("M3", "Y1"), ("M3", "Y2"), ("M3", "Y4"),
        ("M4", "Y1"), ("M4", "Y2"),
    ]
    for m, o in my_edges:
        draw_edge(ax, MX, my[m], OX, oy[o], color="#AAAAAA", lw=0.5,
                  alpha=0.35)

    # Direct X→Y
    for inp, o in [("X1", "Y3"), ("X3", "Y3")]:
        draw_edge(ax, IX, iy[inp], OX, oy[o],
                  color=C_EDGE_DIRECT, style="--", lw=0.8)

    # Noise label — centered below the 2×3 block
    block_cx = (noise_x1 + noise_x2) / 2
    block_bot = min(iy["X9"], iy["X12"]) - 0.05
    ax.text(block_cx, block_bot, "noise (no edges)",
            ha="center", va="top", fontsize=8, color="#555",
            family="sans-serif")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    draw_baseline(axes[0, 0])
    draw_shifted(axes[0, 1])
    draw_rewired(axes[1, 0])
    draw_noisy(axes[1, 1])

    # Legend at bottom
    legend_elements = [
        mpatches.Patch(fc=C_INPUT, ec="black", lw=0.8, label="Inputs (X)"),
        mpatches.Patch(fc=C_MECH, ec="black", lw=0.8, label="Mechanisms (M)"),
        mpatches.Patch(fc=C_OUTPUT, ec="black", lw=0.8, label="Outputs (Y)"),
        mpatches.Patch(fc=C_NOISE, ec="black", lw=0.8, label="Noise (disconnected)"),
        Line2D([0], [0], color=C_EDGE, lw=1.0,
               label="Input\u2192mechanism edges"),
        Line2D([0], [0], color=C_EDGE_NEW, lw=1.5,
               label="New or rewired edges"),
        Line2D([0], [0], color=C_EDGE_DIRECT, lw=1.0, ls="--",
               label="Direct input\u2192output"),
        Line2D([0], [0], color="#AAAAAA", lw=0.8,
               label="Mechanism\u2192output (all oracles)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=8, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 0.005), edgecolor="#CCC")

    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.0, w_pad=1.5)

    out = "experiments/analysis/figures/fig1_oracle_dags.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out}")

    out_pdf = "experiments/analysis/figures/fig1_oracle_dags.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
