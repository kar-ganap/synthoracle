"""Conditional effect sizes — measure effects in the regimes where they matter.

The OAT analysis at midpoint underestimates regime-dependent effects.
Here we compute effect sizes conditioned on being in the right regime.
"""

from __future__ import annotations

import numpy as np

def evaluate(x: np.ndarray,
             k_m1: float = 7.0,
             a_m2: float = 0.55, b_m2: float = 1.8,
             c_z: float = 1.0,
             m4_steepness: float = 25.0, m4_coeff: float = 1.0,
             y2_m1_coeff: float = 0.55,
             y4_coeff: float = 2.0,
             ) -> np.ndarray:
    x1, x2, x3, x4, x5, x6 = x
    m1 = x2**1.37 * x4**0.82 * (1.0 - np.exp(-k_m1 * x1))
    m2 = a_m2 * np.exp(-b_m2 * x3 * np.sqrt(x1))
    z = x4 / (x4 + c_z * x6)
    sig = 1.0 / (1.0 + np.exp(-m4_steepness * (x5 - 0.38)))
    m4 = 1.0 + m4_coeff * x6 * sig
    m1_eff = m1 * z
    m2_eff = m2 * (1.0 - 0.6 * z)
    y1 = m1_eff * m4 - m2_eff
    y2 = 0.85 * m2_eff + y2_m1_coeff * m1_eff / m4
    y3 = (1.0 / (1.0 + np.exp(-8.1 * (x1 - 0.27)))) * x3**0.5
    y4 = m1_eff / (1.0 + y4_coeff * m1_eff)
    return np.array([y1, y2, y3, y4])


N = 200

# Total output range (for normalizing percentages)
rng = np.random.default_rng(42)
X_rand = rng.uniform(0.1, 1.0, size=(100_000, 6))
Y_rand = np.array([evaluate(x) for x in X_rand])
total_range = Y_rand.max(axis=0) - Y_rand.min(axis=0)
print(f"Total output ranges: Y1={total_range[0]:.3f}, Y2={total_range[1]:.3f}, "
      f"Y3={total_range[2]:.3f}, Y4={total_range[3]:.3f}\n")


print("=" * 70)
print("CONDITIONAL EFFECT SIZES (proposed parameters)")
print("=" * 70)

# --- X5 threshold in throughput-dominated regime ---
print("\n--- X5 threshold effect (M4 activation) ---")
print("  Measured in throughput regime: X1=0.8, X2=0.8")
for x6_val in [0.3, 0.5, 0.7, 0.9]:
    base = np.array([0.8, 0.8, 0.3, 0.5, 0.5, x6_val])
    y1_vals = []
    for x5 in np.linspace(0.1, 1.0, N):
        x = base.copy()
        x[4] = x5
        y1_vals.append(evaluate(x)[0])
    y1_arr = np.array(y1_vals)
    effect = y1_arr.max() - y1_arr.min()
    pct = effect / total_range[0] * 100
    # Also measure the sharpness: what's the jump between X5=0.3 and X5=0.5?
    x_below = base.copy(); x_below[4] = 0.30
    x_above = base.copy(); x_above[4] = 0.50
    jump = evaluate(x_above)[0] - evaluate(x_below)[0]
    print(f"  X6={x6_val}: effect={effect:.4f} ({pct:.1f}% of Y1 range), "
          f"jump at threshold (0.3→0.5)={jump:.4f}")


# --- X6 dual pathway ---
print("\n--- X6 dual pathway effect ---")
print("  Z pathway (below threshold, X5=0.2): X6↑ → Z↓ → M1_eff↓ → Y1↓")
base_below = np.array([0.8, 0.8, 0.3, 0.5, 0.2, 0.5])  # X5 below threshold
y1_below = []
for x6 in np.linspace(0.1, 1.0, N):
    x = base_below.copy()
    x[5] = x6
    y1_below.append(evaluate(x)[0])
y1_below = np.array(y1_below)
effect_z = y1_below.max() - y1_below.min()
direction_z = "decreasing" if y1_below[-1] < y1_below[0] else "increasing"
print(f"  Below threshold: Y1 effect={effect_z:.4f} ({effect_z/total_range[0]*100:.1f}%), "
      f"direction={direction_z}")

print("  M4 pathway (above threshold, X5=0.8): X6↑ → M4↑ → Y1↑")
base_above = np.array([0.8, 0.8, 0.3, 0.5, 0.8, 0.5])  # X5 above threshold
y1_above = []
for x6 in np.linspace(0.1, 1.0, N):
    x = base_above.copy()
    x[5] = x6
    y1_above.append(evaluate(x)[0])
y1_above = np.array(y1_above)
effect_m4 = y1_above.max() - y1_above.min()
direction_m4 = "decreasing" if y1_above[-1] < y1_above[0] else "increasing"
print(f"  Above threshold: Y1 effect={effect_m4:.4f} ({effect_m4/total_range[0]*100:.1f}%), "
      f"direction={direction_m4}")

# Does X6 actually flip direction?
print(f"  Direction flip: below threshold X6 is {direction_z}, above is {direction_m4}")
print(f"  Y1(X6=0.1) below: {y1_below[0]:.4f}, Y1(X6=0.9) below: {y1_below[-1]:.4f}")
print(f"  Y1(X6=0.1) above: {y1_above[0]:.4f}, Y1(X6=0.9) above: {y1_above[-1]:.4f}")


# --- X4 effect on Y2 (coupling surprise) ---
print("\n--- X4 effect on Y2 (coupling through Z) ---")
print("  Agent discovers X4 → Y1 (through M1). Surprise: X4 also → Y2.")
for x1_val in [0.3, 0.5, 0.8]:
    base = np.array([x1_val, 0.5, 0.5, 0.5, 0.5, 0.5])
    y2_vals = []
    for x4 in np.linspace(0.1, 1.0, N):
        x = base.copy()
        x[3] = x4
        y2_vals.append(evaluate(x)[1])
    y2_arr = np.array(y2_vals)
    effect = y2_arr.max() - y2_arr.min()
    pct = effect / total_range[1] * 100
    print(f"  X1={x1_val}: Y2 effect={effect:.4f} ({pct:.1f}%), "
          f"Y2(X4=0.1)={y2_arr[0]:.4f}, Y2(X4=1.0)={y2_arr[-1]:.4f}")


# --- What if we restructure Y2? ---
print("\n--- Y2 FORMULA ALTERNATIVES ---")
print("  Current proposed: Y2 = 0.85*M2_eff + 0.55*M1_eff/M4")
print("  Problem: X4 affects Y2 through Z→M1_eff and Z→M2_eff,")
print("  but these partially cancel (Z↑ → M1_eff↑ but M2_eff↓)")
print()

# Alternative: Y2 = 0.85*M2_eff + coeff*M1_eff*Z/M4
# This makes Z's contribution to Y2 more direct and amplified
# Or: Y2 = a*M2_eff + b*M1_eff/M4 + c*Z  (direct Z term)
# Or: change the M2_eff modulation so Z has a stronger effect on Y2

# Let's try several Y2 formulas and check X4's effect:
for label, y2_func in [
    ("Current: 0.85*M2e+0.55*M1e/M4",
     lambda m1e, m2e, m4, z: 0.85*m2e + 0.55*m1e/m4),
    ("Alt A: 0.85*M2e+0.55*M1e*Z/M4",
     lambda m1e, m2e, m4, z: 0.85*m2e + 0.55*m1e*z/m4),
    ("Alt B: 0.60*M2e+0.40*M1e/M4+0.15*Z",
     lambda m1e, m2e, m4, z: 0.60*m2e + 0.40*m1e/m4 + 0.15*z),
    ("Alt C: 0.85*(M2e+0.2)+0.55*M1e/M4",
     lambda m1e, m2e, m4, z: 0.85*(m2e+0.2) + 0.55*m1e/m4),
]:
    base = np.array([0.7, 0.6, 0.4, 0.5, 0.5, 0.5])
    y2_vals = []
    for x4 in np.linspace(0.1, 1.0, N):
        x = base.copy()
        x[3] = x4
        x1, x2, x3, x4v, x5, x6 = x
        m1 = x2**1.37 * x4v**0.82 * (1.0 - np.exp(-7.0 * x1))
        m2 = 0.55 * np.exp(-1.8 * x3 * np.sqrt(x1))
        z = x4v / (x4v + 1.0 * x6)
        sig = 1.0 / (1.0 + np.exp(-25.0 * (x5 - 0.38)))
        m4 = 1.0 + 1.0 * x6 * sig
        m1e = m1 * z
        m2e = m2 * (1 - 0.6 * z)
        y2_vals.append(y2_func(m1e, m2e, m4, z))
    y2_arr = np.array(y2_vals)
    effect = y2_arr.max() - y2_arr.min()
    y2_range_full = y2_arr.max() - y2_arr.min()
    print(f"  {label}")
    print(f"    X4 effect: {effect:.4f}, Y2 range [{y2_arr.min():.4f}, {y2_arr.max():.4f}]")


# --- Comprehensive: what % of Y1 range is each discovery? ---
print("\n" + "=" * 70)
print("DISCOVERY DETECTABILITY SUMMARY (proposed params)")
print("=" * 70)
print(f"Total Y1 range: {total_range[0]:.3f}")
print(f"Total Y2 range: {total_range[1]:.3f}")
print()
print(f"{'Discovery':<45} {'Effect':>8} {'% Y range':>10} {'Detectable?':>12}")
print("-" * 80)

discoveries = [
    ("1. X2 drives Y1 (M1)", "Y1",
     lambda: max(evaluate(np.array([0.7, xi, 0.3, 0.5, 0.5, 0.5]))[0]
                 for xi in np.linspace(0.1, 1.0, 100)) -
             min(evaluate(np.array([0.7, xi, 0.3, 0.5, 0.5, 0.5]))[0]
                 for xi in np.linspace(0.1, 1.0, 100))),
    ("2. Regime: Y1 flips at X1≈0.3", "Y1",
     lambda: max(evaluate(np.array([xi, 0.6, 0.4, 0.5, 0.5, 0.5]))[0]
                 for xi in np.linspace(0.1, 1.0, 100)) -
             min(evaluate(np.array([xi, 0.6, 0.4, 0.5, 0.5, 0.5]))[0]
                 for xi in np.linspace(0.1, 1.0, 100))),
    ("3. X3 controls leakage (M2)", "Y1",
     lambda: max(evaluate(np.array([0.3, 0.5, xi, 0.5, 0.5, 0.5]))[0]
                 for xi in np.linspace(0.1, 1.0, 100)) -
             min(evaluate(np.array([0.3, xi, 0.5, 0.5, 0.5, 0.5]))[0]
                 for xi in np.linspace(0.1, 1.0, 100))),
    ("4. X4 affects Y2 (coupling surprise)", "Y2",
     lambda: max(evaluate(np.array([0.7, 0.6, 0.4, xi, 0.5, 0.5]))[1]
                 for xi in np.linspace(0.1, 1.0, 100)) -
             min(evaluate(np.array([0.7, 0.6, 0.4, xi, 0.5, 0.5]))[1]
                 for xi in np.linspace(0.1, 1.0, 100))),
    ("5. X5 threshold (high X1,X2)", "Y1",
     lambda: max(evaluate(np.array([0.8, 0.8, 0.3, 0.5, xi, 0.7]))[0]
                 for xi in np.linspace(0.1, 1.0, 100)) -
             min(evaluate(np.array([0.8, 0.8, 0.3, 0.5, xi, 0.7]))[0]
                 for xi in np.linspace(0.1, 1.0, 100))),
    ("6. X6 direction flip (dual pathway)", "Y1",
     lambda: abs(
         (evaluate(np.array([0.8, 0.8, 0.3, 0.5, 0.2, 0.9]))[0] -
          evaluate(np.array([0.8, 0.8, 0.3, 0.5, 0.2, 0.1]))[0]) -
         (evaluate(np.array([0.8, 0.8, 0.3, 0.5, 0.8, 0.9]))[0] -
          evaluate(np.array([0.8, 0.8, 0.3, 0.5, 0.8, 0.1]))[0])
     )),
]

for name, output, effect_fn in discoveries:
    effect = effect_fn()
    idx = 0 if output == "Y1" else 1
    pct = effect / total_range[idx] * 100
    detectable = "YES" if pct > 20 else ("MARGINAL" if pct > 10 else "NO")
    print(f"  {name:<45} {effect:>8.4f} {pct:>9.1f}% {detectable:>12}")
