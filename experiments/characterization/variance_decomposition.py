"""Quantitative variance decomposition for the medium oracle.

For each output Yi, compute:
1. Total variance from uniform LHS over the input space
2. Main-effect variance for each input Xj (Sobol first-order)
3. Mechanism-level variance (how much does each mechanism contribute)
4. Effect sizes: max - min when sweeping each input while fixing others at midpoint

This tells us: which mechanisms are detectable from sparse data?
"""

from __future__ import annotations

import numpy as np
from synthoracle.oracles.medium import MediumOracle

oracle = MediumOracle()
rng = np.random.default_rng(42)

N_SAMPLES = 100_000  # large sample for accurate variance estimates
N_SWEEP = 200

# --- 1. Total variance from uniform sampling ---
lo = oracle.bounds[:, 0]
hi = oracle.bounds[:, 1]
X = rng.uniform(lo, hi, size=(N_SAMPLES, 6))
Y = oracle.evaluate_batch(X)

print("=" * 70)
print("MEDIUM ORACLE VARIANCE DECOMPOSITION")
print("=" * 70)

print("\n--- 1. Output Statistics (uniform sampling, N=100k) ---")
for i, name in enumerate(oracle.output_names):
    yi = Y[:, i]
    print(f"  {name}: mean={yi.mean():.4f}, std={yi.std():.4f}, "
          f"min={yi.min():.4f}, max={yi.max():.4f}, range={yi.max()-yi.min():.4f}")

# --- 2. Sobol first-order indices (Monte Carlo estimate) ---
# For each input Xj, estimate Var(E[Y|Xj]) / Var(Y)
# Using pick-freeze method (Saltelli estimator)
print("\n--- 2. Sobol First-Order Sensitivity Indices ---")
print("  (fraction of output variance explained by each input alone)")

X_A = rng.uniform(lo, hi, size=(N_SAMPLES, 6))
X_B = rng.uniform(lo, hi, size=(N_SAMPLES, 6))
Y_A = oracle.evaluate_batch(X_A)
Y_B = oracle.evaluate_batch(X_B)

total_var = Y_A.var(axis=0)

header = f"  {'Input':>6}"
for name in oracle.output_names:
    header += f"  {name:>8}"
print(header)

sobol_indices = np.zeros((6, 4))
for j in range(6):
    # Pick-freeze: replace column j in B with column j from A
    X_AB = X_B.copy()
    X_AB[:, j] = X_A[:, j]
    Y_AB = oracle.evaluate_batch(X_AB)

    # Jansen estimator: S_j = 1 - Var(Y_B - Y_AB) / (2 * Var(Y))
    for i in range(4):
        var_diff = np.var(Y_B[:, i] - Y_AB[:, i])
        sobol_indices[j, i] = 1.0 - var_diff / (2.0 * total_var[i]) if total_var[i] > 0 else 0.0

    row = f"  {oracle.input_names[j]:>6}"
    for i in range(4):
        row += f"  {sobol_indices[j, i]:>8.3f}"
    print(row)

print(f"\n  {'Sum':>6}", end="")
for i in range(4):
    print(f"  {sobol_indices[:, i].sum():>8.3f}", end="")
print("  (>1 indicates interactions)")

# --- 3. One-at-a-time sensitivity (effect size) ---
# Sweep each input from min to max, hold others at midpoint
# Report the range (max-min) of each output
print("\n--- 3. One-at-a-Time Effect Sizes ---")
print("  (output range when sweeping one input, others at midpoint)")

midpoint = (lo + hi) / 2

header = f"  {'Input':>6}"
for name in oracle.output_names:
    header += f"  {name:>8}"
print(header)

for j in range(6):
    x_sweep = np.linspace(lo[j], hi[j], N_SWEEP)
    X_oat = np.tile(midpoint, (N_SWEEP, 1))
    X_oat[:, j] = x_sweep
    Y_oat = oracle.evaluate_batch(X_oat)

    row = f"  {oracle.input_names[j]:>6}"
    for i in range(4):
        effect = Y_oat[:, i].max() - Y_oat[:, i].min()
        row += f"  {effect:>8.4f}"
    print(row)

print(f"\n  {'Total':>6}", end="")
for i in range(4):
    print(f"  {Y[:, i].max() - Y[:, i].min():>8.4f}", end="")
print("  (full output range from uniform sampling)")

# --- 4. Mechanism-level decomposition ---
# Evaluate intermediate mechanism values to understand relative contributions
print("\n--- 4. Mechanism Value Statistics ---")

m1_vals = []
m2_vals = []
z_vals = []
m4_vals = []
m1_eff_vals = []
m2_eff_vals = []

for x in X[:10000]:  # subsample for speed
    x1, x2, x3, x4, x5, x6 = x
    m1 = x2**1.37 * x4**0.82 * (1.0 - np.exp(-3.14 * x1))
    m2 = 0.47 * np.exp(-2.83 * x3 * np.sqrt(x1))
    z = x4 / (x4 + 0.31 * x6)
    sig = 1.0 / (1.0 + np.exp(-5.2 * (x5 - 0.38)))
    m4 = 1.0 + 0.73 * x6 * sig
    m1_eff = m1 * z
    m2_eff = m2 * (1.0 - 0.6 * z)
    m1_vals.append(m1)
    m2_vals.append(m2)
    z_vals.append(z)
    m4_vals.append(m4)
    m1_eff_vals.append(m1_eff)
    m2_eff_vals.append(m2_eff)

for name, vals in [("M1", m1_vals), ("M2", m2_vals), ("Z", z_vals),
                    ("M4", m4_vals), ("M1_eff", m1_eff_vals), ("M2_eff", m2_eff_vals)]:
    arr = np.array(vals)
    print(f"  {name:>8}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
          f"range=[{arr.min():.4f}, {arr.max():.4f}]")

# --- 5. Y1 term decomposition ---
# Y1 = M1_eff * M4 - M2_eff
# What fraction of Y1's variance comes from each term?
print("\n--- 5. Y1 Term Decomposition ---")
m1_eff_arr = np.array(m1_eff_vals)
m2_eff_arr = np.array(m2_eff_vals)
m4_arr = np.array(m4_vals)

term1 = m1_eff_arr * m4_arr  # positive term
term2 = m2_eff_arr            # subtracted term

y1_reconstructed = term1 - term2
print(f"  M1_eff*M4 term: mean={term1.mean():.4f}, std={term1.std():.4f}, "
      f"range=[{term1.min():.4f}, {term1.max():.4f}]")
print(f"  M2_eff term:    mean={term2.mean():.4f}, std={term2.std():.4f}, "
      f"range=[{term2.min():.4f}, {term2.max():.4f}]")
print(f"  Y1 = term1-term2: mean={y1_reconstructed.mean():.4f}, std={y1_reconstructed.std():.4f}")
print(f"  Ratio std(term1)/std(term2) = {term1.std()/term2.std():.2f}")

# --- 6. Y2 term decomposition ---
# Y2 = 0.85 * M2_eff + 0.23 * M1_eff / M4
print("\n--- 6. Y2 Term Decomposition ---")
y2_term1 = 0.85 * m2_eff_arr
y2_term2 = 0.23 * m1_eff_arr / m4_arr

print(f"  0.85*M2_eff term:       mean={y2_term1.mean():.4f}, std={y2_term1.std():.4f}, "
      f"range=[{y2_term1.min():.4f}, {y2_term1.max():.4f}]")
print(f"  0.23*M1_eff/M4 term:    mean={y2_term2.mean():.4f}, std={y2_term2.std():.4f}, "
      f"range=[{y2_term2.min():.4f}, {y2_term2.max():.4f}]")
print(f"  Y2 total:               mean={(y2_term1+y2_term2).mean():.4f}, "
      f"std={(y2_term1+y2_term2).std():.4f}")

# --- 7. M4 threshold sharpness ---
print("\n--- 7. M4 Threshold Characterization ---")
x5_sweep = np.linspace(0.1, 1.0, 200)
for steepness in [5.2, 10, 15, 20, 30]:
    sig_vals = 1.0 / (1.0 + np.exp(-steepness * (x5_sweep - 0.38)))
    # Width of 10-90% transition
    x10 = 0.38 + np.log(9) / steepness  # sigmoid = 0.9
    x90 = 0.38 - np.log(9) / steepness  # sigmoid = 0.1
    width = x10 - x90
    width_pct = width / 0.9 * 100  # as % of input range [0.1, 1.0]
    print(f"  steepness={steepness:>5.1f}: 10-90% transition width={width:.3f} "
          f"({width_pct:.1f}% of input range)")

# --- 8. Y4 saturation analysis ---
print("\n--- 8. Y4 Saturation Analysis ---")
print(f"  M1_eff range: [{m1_eff_arr.min():.4f}, {m1_eff_arr.max():.4f}]")
print(f"  For saturation to be visible, need 0.45*M1_eff ≈ 1, i.e., M1_eff ≈ {1/0.45:.2f}")
print(f"  Current max M1_eff = {m1_eff_arr.max():.4f} — we're at {0.45*m1_eff_arr.max():.1%} of saturation point")

for coeff in [0.45, 1.0, 1.5, 2.0, 3.0]:
    y4_at_max = m1_eff_arr.max() / (1 + coeff * m1_eff_arr.max())
    y4_at_half = (m1_eff_arr.max()/2) / (1 + coeff * m1_eff_arr.max()/2)
    linearity = y4_at_max / (2 * y4_at_half)  # 1.0 = perfectly linear, <1 = saturating
    print(f"  coeff={coeff:.2f}: Y4_max={y4_at_max:.4f}, linearity_ratio={linearity:.3f} "
          f"(1.0=linear, <0.85=visible saturation)")
