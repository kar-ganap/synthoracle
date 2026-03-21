"""Test revised oracle formulas to fix the three structural problems.

Changes:
1. Y2: ratio-based formula where Z enters the denominator
2. M4: partly additive so threshold is detectable independent of M1_eff
3. Y1: M4 has additive + multiplicative components → X6 direction flip works
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(42)
N = 200
N_SAMPLES = 100_000


def evaluate_revised(x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Revised medium oracle. Returns (outputs, mechanism_values)."""
    x1, x2, x3, x4, x5, x6 = x

    # --- Parameters (proposed) ---
    k_m1 = 5.0          # M1 activation steepness (reduced from 7 — see analysis below)
    a_m2 = 0.55         # M2 amplitude
    b_m2 = 1.8          # M2 decay rate
    c_z = 1.0           # Z coupling constant

    # M4 threshold: sharp sigmoid, split into multiplicative + additive
    m4_steepness = 25.0
    m4_threshold = 0.38
    m4_mult_coeff = 0.4   # multiplicative component (smaller)
    m4_add_coeff = 0.35    # additive component (new)

    # --- Mechanisms ---
    m1 = x2**1.37 * x4**0.82 * (1.0 - np.exp(-k_m1 * x1))
    m2 = a_m2 * np.exp(-b_m2 * x3 * np.sqrt(x1))
    z = x4 / (x4 + c_z * x6)
    threshold_gate = 1.0 / (1.0 + np.exp(-m4_steepness * (x5 - m4_threshold)))

    # Effective mechanisms
    m1_eff = m1 * z
    m2_eff = m2 * (1.0 - 0.6 * z)

    # M4 split:
    m4_mult = 1.0 + m4_mult_coeff * x6 * threshold_gate   # multiplies M1_eff
    m4_add = m4_add_coeff * x6 * threshold_gate             # added directly to Y1

    # --- Outputs ---
    # Y1: performance (maximize)
    # M4 enters both multiplicatively (interaction with M1_eff) and additively (direct boost)
    y1 = m1_eff * m4_mult + m4_add - m2_eff

    # Y2: cost (minimize) — REVISED
    # "Cost per unit throughput" — Z in denominator means weak coupling increases cost
    # When Z is low (X4 low or X6 high), throughput-to-cost ratio is poor
    y2 = 0.85 * m2_eff + 0.5 * m1_eff / (m4_mult * z + 0.1)

    # Y3: reliability (threshold > 0.4) — unchanged
    y3 = (1.0 / (1.0 + np.exp(-8.1 * (x1 - 0.27)))) * x3**0.5

    # Y4: speed (maximize, saturates) — increased saturation coefficient
    y4 = m1_eff / (1.0 + 2.0 * m1_eff)

    outputs = np.array([y1, y2, y3, y4])
    mechs = dict(m1=m1, m2=m2, z=z, m1_eff=m1_eff, m2_eff=m2_eff,
                 m4_mult=m4_mult, m4_add=m4_add, threshold_gate=threshold_gate)
    return outputs, mechs


# --- Compute total ranges ---
X = rng.uniform(0.1, 1.0, size=(N_SAMPLES, 6))
Y = np.array([evaluate_revised(x)[0] for x in X])
total_range = Y.max(axis=0) - Y.min(axis=0)

print("=" * 70)
print("REVISED ORACLE — OUTPUT STATISTICS")
print("=" * 70)
for i, name in enumerate(["Y1", "Y2", "Y3", "Y4"]):
    yi = Y[:, i]
    print(f"  {name}: mean={yi.mean():.4f}, std={yi.std():.4f}, "
          f"range=[{yi.min():.4f}, {yi.max():.4f}], total={total_range[i]:.4f}")


# --- Test Problem 1: X4 effect on Y2 ---
print("\n" + "=" * 70)
print("PROBLEM 1: X4 → Y2 COUPLING SURPRISE")
print("=" * 70)
for x1_val in [0.3, 0.5, 0.8]:
    y2_vals = []
    for x4 in np.linspace(0.1, 1.0, N):
        x = np.array([x1_val, 0.5, 0.5, x4, 0.5, 0.5])
        y2_vals.append(evaluate_revised(x)[0][1])
    y2_arr = np.array(y2_vals)
    effect = y2_arr.max() - y2_arr.min()
    pct = effect / total_range[1] * 100
    print(f"  X1={x1_val}: Y2 effect={effect:.4f} ({pct:.1f}%), "
          f"Y2(X4=0.1)={y2_arr[0]:.4f} → Y2(X4=1.0)={y2_arr[-1]:.4f}")

# Also check X6 effect on Y2 (should also be visible through Z)
print("\n  X6 effect on Y2:")
for x4_val in [0.3, 0.5, 0.8]:
    y2_vals = []
    for x6 in np.linspace(0.1, 1.0, N):
        x = np.array([0.7, 0.5, 0.5, x4_val, 0.5, x6])
        y2_vals.append(evaluate_revised(x)[0][1])
    y2_arr = np.array(y2_vals)
    effect = y2_arr.max() - y2_arr.min()
    pct = effect / total_range[1] * 100
    print(f"  X4={x4_val}: Y2 effect of X6={effect:.4f} ({pct:.1f}%)")


# --- Test Problem 2: X5 threshold ---
print("\n" + "=" * 70)
print("PROBLEM 2: X5 THRESHOLD DETECTABILITY")
print("=" * 70)
for x6_val in [0.3, 0.5, 0.7, 0.9]:
    y1_vals = []
    for x5 in np.linspace(0.1, 1.0, N):
        x = np.array([0.8, 0.8, 0.3, 0.5, x5, x6_val])
        y1_vals.append(evaluate_revised(x)[0][0])
    y1_arr = np.array(y1_vals)
    effect = y1_arr.max() - y1_arr.min()
    pct = effect / total_range[0] * 100
    jump = (evaluate_revised(np.array([0.8, 0.8, 0.3, 0.5, 0.50, x6_val]))[0][0] -
            evaluate_revised(np.array([0.8, 0.8, 0.3, 0.5, 0.30, x6_val]))[0][0])
    print(f"  X6={x6_val}: effect={effect:.4f} ({pct:.1f}%), "
          f"jump (0.3→0.5)={jump:.4f}")


# --- Test Problem 3: X6 direction flip ---
print("\n" + "=" * 70)
print("PROBLEM 3: X6 DIRECTION FLIP")
print("=" * 70)

for x5_val, label in [(0.2, "BELOW threshold"), (0.5, "AT threshold"), (0.8, "ABOVE threshold")]:
    y1_vals = []
    for x6 in np.linspace(0.1, 1.0, N):
        x = np.array([0.8, 0.8, 0.3, 0.5, x5_val, x6])
        y1_vals.append(evaluate_revised(x)[0][0])
    y1_arr = np.array(y1_vals)
    effect = y1_arr.max() - y1_arr.min()
    pct = effect / total_range[0] * 100
    direction = "X6↑→Y1↑" if y1_arr[-1] > y1_arr[0] else "X6↑→Y1↓"
    print(f"  {label} (X5={x5_val}): {direction}, effect={effect:.4f} ({pct:.1f}%), "
          f"Y1(X6=0.1)={y1_arr[0]:.4f}, Y1(X6=0.9)={y1_arr[-1]:.4f}")


# --- Full discovery summary ---
print("\n" + "=" * 70)
print("DISCOVERY DETECTABILITY SUMMARY (revised)")
print("=" * 70)
print(f"Y1 range: {total_range[0]:.3f}, Y2 range: {total_range[1]:.3f}")
print()

midpoint = np.full(6, 0.55)

# OAT at midpoint
print("OAT effect sizes at midpoint:")
header = f"  {'Input':>6}"
for name in ["Y1", "Y2", "Y3", "Y4"]:
    header += f"  {name:>10}"
print(header)

for j in range(6):
    x_sweep = np.linspace(0.1, 1.0, 200)
    X_oat = np.tile(midpoint, (200, 1))
    X_oat[:, j] = x_sweep
    Y_oat = np.array([evaluate_revised(x)[0] for x in X_oat])
    row = f"  X{j+1:>5}"
    for i in range(4):
        effect = Y_oat[:, i].max() - Y_oat[:, i].min()
        pct = effect / total_range[i] * 100 if total_range[i] > 0 else 0
        row += f"  {pct:>9.1f}%"
    print(row)

# Conditional effect sizes for hard discoveries
print("\nConditional effects (in favorable regime):")
print(f"  {'Discovery':<45} {'Effect':>8} {'% range':>8} {'OK?':>6}")
print("  " + "-" * 70)

tests = [
    ("1. X2→Y1 (easy, at X1=0.7)", "Y1",
     lambda: np.ptp([evaluate_revised(np.array([0.7, x2, 0.3, 0.5, 0.5, 0.5]))[0][0]
                      for x2 in np.linspace(0.1, 1.0, 100)])),
    ("2. Regime: X1 sweep (medium)", "Y1",
     lambda: np.ptp([evaluate_revised(np.array([x1, 0.6, 0.4, 0.5, 0.5, 0.5]))[0][0]
                      for x1 in np.linspace(0.1, 1.0, 100)])),
    ("3. X3→Y1 leakage (medium, low X1)", "Y1",
     lambda: np.ptp([evaluate_revised(np.array([0.25, 0.5, x3, 0.5, 0.5, 0.5]))[0][0]
                      for x3 in np.linspace(0.1, 1.0, 100)])),
    ("4. X4→Y2 coupling (hard, X1=0.7)", "Y2",
     lambda: np.ptp([evaluate_revised(np.array([0.7, 0.6, 0.4, x4, 0.5, 0.5]))[0][1]
                      for x4 in np.linspace(0.1, 1.0, 100)])),
    ("5. X5 threshold (hard, X1=0.8,X2=0.8,X6=0.7)", "Y1",
     lambda: np.ptp([evaluate_revised(np.array([0.8, 0.8, 0.3, 0.5, x5, 0.7]))[0][0]
                      for x5 in np.linspace(0.1, 1.0, 100)])),
    ("6. X6 flip below thresh (X5=0.2)", "Y1",
     lambda: np.ptp([evaluate_revised(np.array([0.8, 0.8, 0.3, 0.5, 0.2, x6]))[0][0]
                      for x6 in np.linspace(0.1, 1.0, 100)])),
    ("6. X6 flip above thresh (X5=0.8)", "Y1",
     lambda: np.ptp([evaluate_revised(np.array([0.8, 0.8, 0.3, 0.5, 0.8, x6]))[0][0]
                      for x6 in np.linspace(0.1, 1.0, 100)])),
]

for name, out, fn in tests:
    idx = 0 if out == "Y1" else 1
    effect = fn()
    pct = effect / total_range[idx] * 100
    ok = "YES" if pct > 20 else ("~" if pct > 12 else "NO")
    print(f"  {name:<45} {effect:>8.4f} {pct:>7.1f}% {ok:>6}")
