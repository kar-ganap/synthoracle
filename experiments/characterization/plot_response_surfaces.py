"""Visualize medium oracle response surfaces.

Key plots:
1. Regime transition: Y1 vs X1 for different X2 values
2. Hidden coupling: Y1, Y2 vs X4 at different X6 values
3. Hidden threshold: Y1 vs X5
4. Reliability threshold: Y3 vs X1
5. 2D heatmaps: Y1(X1, X2), Y1(X4, X6), Y1(X5, X6)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from synthoracle.oracles.medium import MediumOracle

oracle = MediumOracle()
BASE = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
N = 200

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle("Medium Oracle Response Surfaces", fontsize=14, fontweight="bold")

# --- Plot 1: Regime transition — Y1 vs X1 for different X2 ---
ax = axes[0, 0]
x1_range = np.linspace(0.1, 1.0, N)
for x2_val in [0.2, 0.5, 0.8]:
    y1_vals = []
    for x1 in x1_range:
        x = BASE.copy()
        x[0], x[1] = x1, x2_val
        y1_vals.append(oracle.evaluate(x)[0])
    ax.plot(x1_range, y1_vals, label=f"X2={x2_val}")
ax.set_xlabel("X1")
ax.set_ylabel("Y1 (performance)")
ax.set_title("Regime transition: Y1 vs X1")
ax.legend()
ax.axvline(x=0.3, color="gray", linestyle="--", alpha=0.5, label="~regime boundary")

# --- Plot 2: Y2 (cost) vs X1 for different X3 ---
ax = axes[0, 1]
for x3_val in [0.2, 0.5, 0.8]:
    y2_vals = []
    for x1 in x1_range:
        x = BASE.copy()
        x[0], x[2] = x1, x3_val
        y2_vals.append(oracle.evaluate(x)[1])
    ax.plot(x1_range, y2_vals, label=f"X3={x3_val}")
ax.set_xlabel("X1")
ax.set_ylabel("Y2 (cost)")
ax.set_title("Cost vs X1 (leakage regime)")
ax.legend()

# --- Plot 3: Hidden threshold — Y1 vs X5 ---
ax = axes[0, 2]
x5_range = np.linspace(0.1, 1.0, N)
for x6_val in [0.2, 0.5, 0.8]:
    y1_vals = []
    for x5 in x5_range:
        x = BASE.copy()
        x[0] = 0.7  # high X1 so M1 dominates
        x[1] = 0.7  # high X2
        x[4], x[5] = x5, x6_val
        y1_vals.append(oracle.evaluate(x)[0])
    ax.plot(x5_range, y1_vals, label=f"X6={x6_val}")
ax.set_xlabel("X5")
ax.set_ylabel("Y1 (performance)")
ax.set_title("Hidden threshold: M4 activates at X5≈0.38")
ax.axvline(x=0.38, color="gray", linestyle="--", alpha=0.5)
ax.legend()

# --- Plot 4: Hidden coupling — Y1, Y2 vs X4 ---
ax = axes[1, 0]
x4_range = np.linspace(0.1, 1.0, N)
for x6_val in [0.2, 0.5, 0.8]:
    y1_vals, y2_vals = [], []
    for x4 in x4_range:
        x = BASE.copy()
        x[0] = 0.7
        x[3], x[5] = x4, x6_val
        y = oracle.evaluate(x)
        y1_vals.append(y[0])
        y2_vals.append(y[1])
    ax.plot(x4_range, y1_vals, label=f"Y1, X6={x6_val}")
    ax.plot(x4_range, y2_vals, "--", label=f"Y2, X6={x6_val}")
ax.set_xlabel("X4")
ax.set_ylabel("Y1, Y2")
ax.set_title("Hidden coupling: X4 affects Y1 & Y2 through Z")
ax.legend(fontsize=7)

# --- Plot 5: Reliability — Y3 vs X1 ---
ax = axes[1, 1]
for x3_val in [0.2, 0.5, 0.8]:
    y3_vals = []
    for x1 in x1_range:
        x = BASE.copy()
        x[0], x[2] = x1, x3_val
        y3_vals.append(oracle.evaluate(x)[2])
    ax.plot(x1_range, y3_vals, label=f"X3={x3_val}")
ax.axhline(y=0.4, color="red", linestyle="--", alpha=0.5, label="threshold")
ax.set_xlabel("X1")
ax.set_ylabel("Y3 (reliability)")
ax.set_title("Reliability threshold at Y3=0.4")
ax.legend()

# --- Plot 6: Saturation — Y4 vs X2 ---
ax = axes[1, 2]
x2_range = np.linspace(0.1, 1.0, N)
for x1_val in [0.3, 0.6, 0.9]:
    y4_vals = []
    for x2 in x2_range:
        x = BASE.copy()
        x[0], x[1] = x1_val, x2
        y4_vals.append(oracle.evaluate(x)[3])
    ax.plot(x2_range, y4_vals, label=f"X1={x1_val}")
ax.set_xlabel("X2")
ax.set_ylabel("Y4 (speed)")
ax.set_title("Speed saturation: Y4 = M1_eff/(1+2·M1_eff)")
ax.legend()

# --- Plot 7: 2D heatmap Y1(X1, X2) ---
ax = axes[2, 0]
x1_grid = np.linspace(0.1, 1.0, 80)
x2_grid = np.linspace(0.1, 1.0, 80)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
Y1_map = np.zeros_like(X1)
for i in range(80):
    for j in range(80):
        x = BASE.copy()
        x[0], x[1] = X1[i, j], X2[i, j]
        Y1_map[i, j] = oracle.evaluate(x)[0]
c = ax.contourf(X1, X2, Y1_map, levels=20, cmap="viridis")
plt.colorbar(c, ax=ax)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_title("Y1(X1, X2) — regime transition visible")

# --- Plot 8: 2D heatmap Y1(X4, X6) ---
ax = axes[2, 1]
x4_grid = np.linspace(0.1, 1.0, 80)
x6_grid = np.linspace(0.1, 1.0, 80)
X4, X6 = np.meshgrid(x4_grid, x6_grid)
Y1_map2 = np.zeros_like(X4)
for i in range(80):
    for j in range(80):
        x = BASE.copy()
        x[0] = 0.7  # high X1
        x[3], x[5] = X4[i, j], X6[i, j]
        Y1_map2[i, j] = oracle.evaluate(x)[0]
c = ax.contourf(X4, X6, Y1_map2, levels=20, cmap="viridis")
plt.colorbar(c, ax=ax)
ax.set_xlabel("X4")
ax.set_ylabel("X6")
ax.set_title("Y1(X4, X6) — coupling through Z")

# --- Plot 9: 2D heatmap Y1(X5, X6) showing threshold + dual pathway ---
ax = axes[2, 2]
x5_grid = np.linspace(0.1, 1.0, 80)
x6_grid2 = np.linspace(0.1, 1.0, 80)
X5, X6_2 = np.meshgrid(x5_grid, x6_grid2)
Y1_map3 = np.zeros_like(X5)
for i in range(80):
    for j in range(80):
        x = BASE.copy()
        x[0], x[1] = 0.7, 0.7
        x[4], x[5] = X5[i, j], X6_2[i, j]
        Y1_map3[i, j] = oracle.evaluate(x)[0]
c = ax.contourf(X5, X6_2, Y1_map3, levels=20, cmap="viridis")
plt.colorbar(c, ax=ax)
ax.set_xlabel("X5")
ax.set_ylabel("X6")
ax.set_title("Y1(X5, X6) — threshold + dual pathway")

plt.tight_layout()
plt.savefig(
    "experiments/characterization/medium_oracle_response_surfaces.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved to experiments/characterization/medium_oracle_response_surfaces.png")
