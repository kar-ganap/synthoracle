"""Parameter tuning analysis for medium oracle.

Examines:
1. M1 activation steepness (1-exp(-k*X1)) — regime transition sharpness
2. M2 coefficient and exponential steepness — leakage strength
3. Z coupling constant — dynamic range and centering
4. Combined effect: how do proposed changes affect all outputs?
"""

from __future__ import annotations

import numpy as np

# ============================================================
# 1. M1 ACTIVATION: (1 - exp(-k * X1))
# ============================================================
print("=" * 70)
print("1. M1 ACTIVATION STEEPNESS: (1 - exp(-k * X1))")
print("=" * 70)

x1_range = np.linspace(0.1, 1.0, 200)

print("\n  X1 values where activation reaches 50% and 90%:")
for k in [3.14, 5.0, 7.0, 10.0]:
    activation = 1 - np.exp(-k * x1_range)
    # Find X1 where activation = 0.5 and 0.9
    x1_half = -np.log(0.5) / k
    x1_90 = -np.log(0.1) / k
    # Effective range: activation at X1=0.1 vs X1=1.0
    act_low = 1 - np.exp(-k * 0.1)
    act_high = 1 - np.exp(-k * 1.0)
    # Crossover: where does M1_eff*M4 ≈ M2_eff? (approximate)
    print(f"  k={k:>5.2f}: act(0.1)={act_low:.3f}, act(1.0)={act_high:.3f}, "
          f"50% at X1={x1_half:.3f}, 90% at X1={x1_90:.3f}")

print("\n  Effect on regime transition sharpness:")
print("  (dY1/dX1 at the crossover point, normalized)")
for k in [3.14, 5.0, 7.0, 10.0]:
    # Derivative of (1-exp(-k*x1)) = k*exp(-k*x1)
    # At X1=0.3 (approximate crossover):
    deriv_at_03 = k * np.exp(-k * 0.3)
    # Also: what's the ratio of activation at X1=0.5 vs X1=0.15?
    ratio = (1 - np.exp(-k * 0.5)) / max(1 - np.exp(-k * 0.15), 1e-10)
    print(f"  k={k:>5.2f}: d/dX1 at X1=0.3 = {deriv_at_03:.3f}, "
          f"act(0.5)/act(0.15) = {ratio:.2f}x")


# ============================================================
# 2. M2 PARAMETERS: 0.47 * exp(-2.83 * X3 * sqrt(X1))
# ============================================================
print("\n" + "=" * 70)
print("2. M2 LEAKAGE: a * exp(-b * X3 * sqrt(X1))")
print("=" * 70)

print("\n  M2 values at key points (current: a=0.47, b=2.83):")
for a, b in [(0.47, 2.83), (0.47, 2.0), (0.60, 2.83), (0.60, 2.0), (0.55, 1.8)]:
    m2_low_x1 = a * np.exp(-b * 0.5 * np.sqrt(0.15))   # low X1, mid X3
    m2_mid_x1 = a * np.exp(-b * 0.5 * np.sqrt(0.5))     # mid X1, mid X3
    m2_high_x1 = a * np.exp(-b * 0.5 * np.sqrt(0.9))    # high X1, mid X3
    m2_range = m2_low_x1 - m2_high_x1
    print(f"  a={a}, b={b}: M2(X1=0.15)={m2_low_x1:.4f}, M2(X1=0.5)={m2_mid_x1:.4f}, "
          f"M2(X1=0.9)={m2_high_x1:.4f}, range={m2_range:.4f}")

print("\n  M2 sensitivity to X3 (at X1=0.5):")
for a, b in [(0.47, 2.83), (0.47, 2.0), (0.60, 2.0), (0.55, 1.8)]:
    m2_x3_low = a * np.exp(-b * 0.2 * np.sqrt(0.5))
    m2_x3_high = a * np.exp(-b * 0.8 * np.sqrt(0.5))
    print(f"  a={a}, b={b}: M2(X3=0.2)={m2_x3_low:.4f}, M2(X3=0.8)={m2_x3_high:.4f}, "
          f"ratio={m2_x3_low/m2_x3_high:.2f}x")

print("\n  Key question: is M2 large enough at low X1 to dominate M1_eff?")
print("  For regime transition, need M2_eff > M1_eff*M4 at low X1")
for a, b, k_m1 in [(0.47, 2.83, 3.14), (0.47, 2.83, 5.0),
                     (0.60, 2.0, 5.0), (0.55, 1.8, 7.0)]:
    # At X1=0.15, X2=0.5, X3=0.5, X4=0.5, X5=0.5, X6=0.5
    m1 = 0.5**1.37 * 0.5**0.82 * (1 - np.exp(-k_m1 * 0.15))
    m2 = a * np.exp(-b * 0.5 * np.sqrt(0.15))
    # Z at X4=X6=0.5, with c=0.31 (current) and c=1.0 (proposed)
    z_current = 0.5 / (0.5 + 0.31 * 0.5)
    z_proposed = 0.5 / (0.5 + 1.0 * 0.5)
    m1_eff_cur = m1 * z_current
    m2_eff_cur = m2 * (1 - 0.6 * z_current)
    m1_eff_prop = m1 * z_proposed
    m2_eff_prop = m2 * (1 - 0.6 * z_proposed)
    print(f"  a={a}, b={b}, k_m1={k_m1}: "
          f"M1_eff={m1_eff_cur:.4f}, M2_eff={m2_eff_cur:.4f}, "
          f"ratio M2/M1={m2_eff_cur/max(m1_eff_cur,1e-10):.2f}x "
          f"| Z=1.0: M1_eff={m1_eff_prop:.4f}, M2_eff={m2_eff_prop:.4f}, "
          f"ratio={m2_eff_prop/max(m1_eff_prop,1e-10):.2f}x")


# ============================================================
# 3. Z COUPLING: X4 / (X4 + c * X6)
# ============================================================
print("\n" + "=" * 70)
print("3. Z COUPLING CONSTANT: X4 / (X4 + c * X6)")
print("=" * 70)

print("\n  Z statistics across uniform [0.1, 1.0] for X4, X6:")
rng = np.random.default_rng(42)
x4_samples = rng.uniform(0.1, 1.0, 50000)
x6_samples = rng.uniform(0.1, 1.0, 50000)

for c in [0.31, 0.5, 0.75, 1.0, 1.5, 2.0]:
    z = x4_samples / (x4_samples + c * x6_samples)
    print(f"  c={c:>4.2f}: mean={z.mean():.3f}, std={z.std():.3f}, "
          f"range=[{z.min():.3f}, {z.max():.3f}], "
          f"IQR=[{np.percentile(z,25):.3f}, {np.percentile(z,75):.3f}]")

print("\n  Z at specific points (to understand the interaction):")
for c in [0.31, 1.0, 1.5]:
    print(f"\n  c={c}:")
    for x4, x6 in [(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8), (0.5, 0.5)]:
        z = x4 / (x4 + c * x6)
        print(f"    X4={x4}, X6={x6}: Z={z:.3f}")

print("\n  Effect of Z on M1_eff when M1=0.5 (mid-range):")
for c in [0.31, 1.0, 1.5]:
    z_low = 0.2 / (0.2 + c * 0.8)   # low X4, high X6
    z_high = 0.8 / (0.8 + c * 0.2)  # high X4, low X6
    m1_eff_low = 0.5 * z_low
    m1_eff_high = 0.5 * z_high
    print(f"  c={c:>4.2f}: Z_low={z_low:.3f} → M1_eff={m1_eff_low:.3f}, "
          f"Z_high={z_high:.3f} → M1_eff={m1_eff_high:.3f}, "
          f"ratio={m1_eff_high/m1_eff_low:.2f}x")


# ============================================================
# 4. COMBINED PROPOSED PARAMETERS — FULL SIMULATION
# ============================================================
print("\n" + "=" * 70)
print("4. COMBINED PROPOSED PARAMETERS")
print("=" * 70)

def evaluate_proposed(x: np.ndarray,
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

N = 100_000
X = rng.uniform(0.1, 1.0, size=(N, 6))

configs = {
    "CURRENT": dict(k_m1=3.14, a_m2=0.47, b_m2=2.83, c_z=0.31,
                     m4_steepness=5.2, m4_coeff=0.73,
                     y2_m1_coeff=0.23, y4_coeff=0.45),
    "PROPOSED": dict(k_m1=7.0, a_m2=0.55, b_m2=1.8, c_z=1.0,
                      m4_steepness=25.0, m4_coeff=1.0,
                      y2_m1_coeff=0.55, y4_coeff=2.0),
}

for config_name, params in configs.items():
    print(f"\n  --- {config_name} ---")
    print(f"  params: {params}")
    Y = np.array([evaluate_proposed(x, **params) for x in X])

    print(f"\n  Output statistics:")
    for i, name in enumerate(["Y1", "Y2", "Y3", "Y4"]):
        yi = Y[:, i]
        print(f"    {name}: mean={yi.mean():.4f}, std={yi.std():.4f}, "
              f"range=[{yi.min():.4f}, {yi.max():.4f}]")

    # OAT effect sizes
    midpoint = np.full(6, 0.55)
    print(f"\n  OAT effect sizes (% of output range):")
    header = f"    {'Input':>6}"
    for name in ["Y1", "Y2", "Y3", "Y4"]:
        header += f"  {name:>10}"
    print(header)

    total_ranges = Y.max(axis=0) - Y.min(axis=0)
    for j in range(6):
        x_sweep = np.linspace(0.1, 1.0, 200)
        X_oat = np.tile(midpoint, (200, 1))
        X_oat[:, j] = x_sweep
        Y_oat = np.array([evaluate_proposed(x, **params) for x in X_oat])
        row = f"    X{j+1:>5}"
        for i in range(4):
            effect = Y_oat[:, i].max() - Y_oat[:, i].min()
            pct = effect / total_ranges[i] * 100 if total_ranges[i] > 0 else 0
            row += f"  {pct:>9.1f}%"
        print(row)

    # Mechanism intermediates
    print(f"\n  Mechanism ranges:")
    m1_vals, z_vals, m4_vals, m1e_vals, m2e_vals = [], [], [], [], []
    for x in X[:10000]:
        x1, x2, x3, x4, x5, x6 = x
        m1 = x2**1.37 * x4**0.82 * (1.0 - np.exp(-params['k_m1'] * x1))
        m2 = params['a_m2'] * np.exp(-params['b_m2'] * x3 * np.sqrt(x1))
        z = x4 / (x4 + params['c_z'] * x6)
        sig = 1.0 / (1.0 + np.exp(-params['m4_steepness'] * (x5 - 0.38)))
        m4 = 1.0 + params['m4_coeff'] * x6 * sig
        m1e = m1 * z
        m2e = m2 * (1 - 0.6 * z)
        m1_vals.append(m1)
        z_vals.append(z)
        m4_vals.append(m4)
        m1e_vals.append(m1e)
        m2e_vals.append(m2e)
    for name, vals in [("M1", m1_vals), ("Z", z_vals), ("M4", m4_vals),
                        ("M1_eff", m1e_vals), ("M2_eff", m2e_vals)]:
        arr = np.array(vals)
        print(f"    {name:>8}: [{arr.min():.4f}, {arr.max():.4f}], std={arr.std():.4f}")
