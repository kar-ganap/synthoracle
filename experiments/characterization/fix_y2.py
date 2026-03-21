"""Focus on making Y2 respond strongly to X4 and X6 through Z.

Current Y2 = 0.85*M2_eff + 0.55*M1_eff/(M4_mult*Z+0.1)
Problem: Z is in both M1_eff (=M1*Z) and the denominator, partially canceling.

Approaches:
A. Use M1 (raw) instead of M1_eff in Y2 second term — breaks the cancellation
B. Add a direct Z penalty term to Y2
C. Increase the denominator sensitivity by reducing the +0.1 offset
D. Use M1_eff/Z in numerator to amplify Z dependence
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(42)
N = 150


def evaluate_y2_variant(x: np.ndarray, y2_variant: str) -> np.ndarray:
    x1, x2, x3, x4, x5, x6 = x

    m1 = x2**1.37 * x4**0.82 * (1.0 - np.exp(-5.0 * x1))
    m2 = 0.55 * np.exp(-1.8 * x3 * np.sqrt(x1))
    z = x4 / (x4 + 1.0 * x6)
    gate = 1.0 / (1.0 + np.exp(-25.0 * (x5 - 0.38)))

    m1_eff = m1 * z
    m2_eff = m2 * (1.0 - 0.6 * z)
    m4_mult = 1.0 + 0.4 * x6 * gate
    m4_add = 0.5 * x6 * gate

    y1 = m1_eff * m4_mult + m4_add - m2_eff

    if y2_variant == "current":
        # 0.85*M2_eff + 0.55*M1_eff/(M4_mult*Z+0.1)
        y2 = 0.85 * m2_eff + 0.55 * m1_eff / (m4_mult * z + 0.1)
    elif y2_variant == "raw_m1":
        # Use M1 (pre-coupling) — Z only in denominator
        y2 = 0.85 * m2_eff + 0.45 * m1 / (m4_mult * z + 0.1)
    elif y2_variant == "direct_z":
        # Add direct Z penalty: cost is higher when coupling is weak
        y2 = 0.85 * m2_eff + 0.35 * m1_eff / m4_mult + 0.25 * (1.0 - z)
    elif y2_variant == "z_denominator_only":
        # M1_eff/M4 + penalty/(Z+offset) — Z purely in denominator
        y2 = 0.85 * m2_eff + 0.30 * m1_eff / m4_mult + 0.12 / (z + 0.15)
    elif y2_variant == "ratio":
        # Cost as ratio: how much leakage per unit throughput, modulated by Z
        y2 = 0.5 * m2_eff / (m1_eff + 0.05) + 0.3 * m1_eff / (m4_mult * z + 0.1)
    else:
        raise ValueError(f"Unknown variant: {y2_variant}")

    y3 = (1.0 / (1.0 + np.exp(-8.1 * (x1 - 0.27)))) * x3**0.5
    y4 = m1_eff / (1.0 + 2.0 * m1_eff)

    return np.array([y1, y2, y3, y4])


# Total ranges
X = rng.uniform(0.1, 1.0, size=(50_000, 6))

variants = ["current", "raw_m1", "direct_z", "z_denominator_only", "ratio"]

print("=" * 70)
print("Y2 FORMULA VARIANTS — SENSITIVITY TO X4 AND X6")
print("=" * 70)

for variant in variants:
    Y = np.array([evaluate_y2_variant(x, variant) for x in X])
    y2_range = Y[:, 1].max() - Y[:, 1].min()

    # X4 effect on Y2 at different X1
    x4_effects = []
    for x1_val in [0.3, 0.5, 0.7, 0.9]:
        base = np.array([x1_val, 0.6, 0.4, 0.5, 0.5, 0.5])
        y2_vals = []
        for x4 in np.linspace(0.1, 1.0, N):
            x = base.copy()
            x[3] = x4
            y2_vals.append(evaluate_y2_variant(x, variant)[1])
        y2_arr = np.array(y2_vals)
        x4_effects.append(np.ptp(y2_arr))

    # X6 effect on Y2
    x6_effects = []
    for x4_val in [0.3, 0.5, 0.7]:
        base = np.array([0.7, 0.6, 0.4, x4_val, 0.5, 0.5])
        y2_vals = []
        for x6 in np.linspace(0.1, 1.0, N):
            x = base.copy()
            x[5] = x6
            y2_vals.append(evaluate_y2_variant(x, variant)[1])
        y2_arr = np.array(y2_vals)
        x6_effects.append(np.ptp(y2_arr))

    # Also check: does X4 still affect Y1 strongly? (shouldn't break Y1)
    base_y1 = np.array([0.7, 0.6, 0.3, 0.5, 0.5, 0.5])
    y1_x4 = [evaluate_y2_variant(np.array([0.7, 0.6, 0.3, x4, 0.5, 0.5]), variant)[0]
              for x4 in np.linspace(0.1, 1.0, N)]
    y1_range = Y[:, 0].max() - Y[:, 0].min()

    print(f"\n  --- {variant} ---")
    print(f"  Y2 range: [{Y[:,1].min():.3f}, {Y[:,1].max():.3f}] = {y2_range:.3f}")
    print(f"  Y1 range: {y1_range:.3f}")
    print(f"  X4→Y2 effect (% of Y2 range):")
    for x1_val, eff in zip([0.3, 0.5, 0.7, 0.9], x4_effects):
        print(f"    X1={x1_val}: {eff:.4f} ({eff/y2_range*100:.1f}%)")
    print(f"  X6→Y2 effect (% of Y2 range):")
    for x4_val, eff in zip([0.3, 0.5, 0.7], x6_effects):
        print(f"    X4={x4_val}: {eff:.4f} ({eff/y2_range*100:.1f}%)")
    print(f"  X4→Y1 effect: {np.ptp(y1_x4):.4f} ({np.ptp(y1_x4)/y1_range*100:.1f}% of Y1 range)")

    # Check all other discoveries still work
    print(f"  Quick check other discoveries:")
    # X2→Y1
    e = np.ptp([evaluate_y2_variant(np.array([0.7, x2, 0.3, 0.5, 0.5, 0.5]), variant)[0]
                for x2 in np.linspace(0.1, 1.0, 100)])
    print(f"    X2→Y1: {e/y1_range*100:.1f}%")
    # X5 threshold
    e = np.ptp([evaluate_y2_variant(np.array([0.8, 0.8, 0.3, 0.5, x5, 0.7]), variant)[0]
                for x5 in np.linspace(0.1, 1.0, 100)])
    print(f"    X5 threshold: {e/y1_range*100:.1f}%")
    # X6 flip
    y1_below = [evaluate_y2_variant(np.array([0.8, 0.8, 0.3, 0.5, 0.2, x6]), variant)[0]
                for x6 in np.linspace(0.1, 1.0, 100)]
    y1_above = [evaluate_y2_variant(np.array([0.8, 0.8, 0.3, 0.5, 0.8, x6]), variant)[0]
                for x6 in np.linspace(0.1, 1.0, 100)]
    dir_below = "↓" if y1_below[-1] < y1_below[0] else "↑"
    dir_above = "↓" if y1_above[-1] < y1_above[0] else "↑"
    print(f"    X6 below thresh: {np.ptp(y1_below)/y1_range*100:.1f}% {dir_below}")
    print(f"    X6 above thresh: {np.ptp(y1_above)/y1_range*100:.1f}% {dir_above}")
    flip = "YES" if dir_below != dir_above else "NO"
    print(f"    Direction flip: {flip}")
