"""Fix remaining two issues:
1. X4→Y2 coupling: use M1 (pre-coupling) in Y2 denominator term
2. X6 above-threshold effect: increase m4_add_coeff, optionally reduce c_z

Grid search over (m4_add_coeff, c_z, y2_formula_variant) to find
the parameter set where all 6 discoveries are detectable.
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(42)
N = 150


def evaluate(x: np.ndarray,
             c_z: float = 1.0,
             m4_add_coeff: float = 0.35,
             y2_use_m1_raw: bool = False,
             y2_z_coeff: float = 0.5,
             ) -> np.ndarray:
    x1, x2, x3, x4, x5, x6 = x

    k_m1 = 5.0
    a_m2 = 0.55
    b_m2 = 1.8
    m4_steepness = 25.0
    m4_mult_coeff = 0.4

    m1 = x2**1.37 * x4**0.82 * (1.0 - np.exp(-k_m1 * x1))
    m2 = a_m2 * np.exp(-b_m2 * x3 * np.sqrt(x1))
    z = x4 / (x4 + c_z * x6)
    gate = 1.0 / (1.0 + np.exp(-m4_steepness * (x5 - 0.38)))

    m1_eff = m1 * z
    m2_eff = m2 * (1.0 - 0.6 * z)

    m4_mult = 1.0 + m4_mult_coeff * x6 * gate
    m4_add = m4_add_coeff * x6 * gate

    y1 = m1_eff * m4_mult + m4_add - m2_eff

    # Y2: use M1 (raw, pre-coupling) or M1_eff in denominator term?
    m1_for_y2 = m1 if y2_use_m1_raw else m1_eff
    y2 = 0.85 * m2_eff + y2_z_coeff * m1_for_y2 / (m4_mult * z + 0.1)

    y3 = (1.0 / (1.0 + np.exp(-8.1 * (x1 - 0.27)))) * x3**0.5
    y4 = m1_eff / (1.0 + 2.0 * m1_eff)

    return np.array([y1, y2, y3, y4])


def compute_detectability(c_z: float, m4_add_coeff: float,
                          y2_use_m1_raw: bool, y2_z_coeff: float,
                          ) -> dict[str, tuple[float, float]]:
    """Compute all 6 discovery effect sizes. Returns {name: (effect, pct)}."""
    params = dict(c_z=c_z, m4_add_coeff=m4_add_coeff,
                  y2_use_m1_raw=y2_use_m1_raw, y2_z_coeff=y2_z_coeff)

    # Total ranges
    X_rand = rng.uniform(0.1, 1.0, size=(50_000, 6))
    Y_rand = np.array([evaluate(x, **params) for x in X_rand])
    tr = Y_rand.max(axis=0) - Y_rand.min(axis=0)

    def sweep(base: np.ndarray, dim: int, output: int) -> float:
        vals = []
        for v in np.linspace(0.1, 1.0, N):
            x = base.copy()
            x[dim] = v
            vals.append(evaluate(x, **params)[output])
        return float(np.max(vals) - np.min(vals))

    results = {}

    # 1. X2→Y1 (easy)
    e = sweep(np.array([0.7, 0.5, 0.3, 0.5, 0.5, 0.5]), 1, 0)
    results["1.X2→Y1"] = (e, e / tr[0] * 100)

    # 2. Regime X1 (medium)
    e = sweep(np.array([0.5, 0.6, 0.4, 0.5, 0.5, 0.5]), 0, 0)
    results["2.Regime"] = (e, e / tr[0] * 100)

    # 3. X3→Y1 leakage (medium, low X1)
    e = sweep(np.array([0.25, 0.5, 0.5, 0.5, 0.5, 0.5]), 2, 0)
    results["3.X3leak"] = (e, e / tr[0] * 100)

    # 4. X4→Y2 coupling (hard)
    e = sweep(np.array([0.7, 0.6, 0.4, 0.5, 0.5, 0.5]), 3, 1)
    results["4.X4→Y2"] = (e, e / tr[1] * 100)

    # 5. X5 threshold (hard, favorable regime)
    e = sweep(np.array([0.8, 0.8, 0.3, 0.5, 0.5, 0.7]), 4, 0)
    results["5.X5thr"] = (e, e / tr[0] * 100)

    # 6a. X6 below threshold
    e_below = sweep(np.array([0.8, 0.8, 0.3, 0.5, 0.2, 0.5]), 5, 0)
    results["6a.X6↓"] = (e_below, e_below / tr[0] * 100)

    # 6b. X6 above threshold
    e_above = sweep(np.array([0.8, 0.8, 0.3, 0.5, 0.8, 0.5]), 5, 0)
    # Check direction
    y1_x6_lo = evaluate(np.array([0.8, 0.8, 0.3, 0.5, 0.8, 0.1]), **params)[0]
    y1_x6_hi = evaluate(np.array([0.8, 0.8, 0.3, 0.5, 0.8, 0.9]), **params)[0]
    direction = "↑" if y1_x6_hi > y1_x6_lo else "↓"
    results["6b.X6↑" + direction] = (e_above, e_above / tr[0] * 100)

    # Output ranges
    results["_Y1range"] = (tr[0], 0)
    results["_Y2range"] = (tr[1], 0)

    return results


# --- Grid search ---
print("=" * 70)
print("GRID SEARCH: (c_z, m4_add_coeff, y2_formula, y2_z_coeff)")
print("=" * 70)
print(f"\nTarget: all discoveries >15%, hard ones >12%")
print(f"Constraint: Y1 range > 0.8, Y2 range > 0.3, no output |max| > 5\n")

configs = []
for c_z in [0.7, 0.85, 1.0]:
    for m4_add in [0.40, 0.50, 0.60]:
        for y2_raw in [False, True]:
            for y2_zc in [0.45, 0.55, 0.65]:
                configs.append((c_z, m4_add, y2_raw, y2_zc))

print(f"Testing {len(configs)} configurations...\n")

best_score = 0
best_config = None
best_results = None

header = (f"{'c_z':>4} {'m4a':>5} {'raw':>4} {'y2c':>5} | "
          f"{'X2→Y1':>6} {'Rgme':>6} {'X3lk':>6} {'X4Y2':>6} "
          f"{'X5th':>6} {'X6↓':>6} {'X6↑':>6} | {'min':>5} {'Y1r':>5} {'Y2r':>5}")
print(header)
print("-" * len(header))

for c_z, m4_add, y2_raw, y2_zc in configs:
    try:
        res = compute_detectability(c_z, m4_add, y2_raw, y2_zc)
    except Exception:
        continue

    # Extract percentages
    pcts = {}
    x6_above_key = None
    for k, (e, p) in res.items():
        if k.startswith("_"):
            continue
        pcts[k] = p
        if k.startswith("6b"):
            x6_above_key = k

    y1r = res["_Y1range"][0]
    y2r = res["_Y2range"][0]

    # Skip if output ranges are degenerate
    if y1r < 0.8 or y2r < 0.3:
        continue

    # Score: minimum detectability across all discoveries
    all_pcts = list(pcts.values())
    min_pct = min(all_pcts)

    # Check direction flip
    has_flip = x6_above_key is not None and "↑" in x6_above_key

    label = "raw" if y2_raw else "eff"
    x6_dir = x6_above_key.split("X6")[1][:2] if x6_above_key else "??"

    vals = [pcts.get(k, 0) for k in ["1.X2→Y1", "2.Regime", "3.X3leak", "4.X4→Y2",
                                       "5.X5thr", "6a.X6↓"]]
    vals.append(pcts.get(x6_above_key, 0) if x6_above_key else 0)

    print(f"{c_z:>4.2f} {m4_add:>5.2f} {label:>4} {y2_zc:>5.2f} | "
          f"{vals[0]:>5.1f}% {vals[1]:>5.1f}% {vals[2]:>5.1f}% {vals[3]:>5.1f}% "
          f"{vals[4]:>5.1f}% {vals[5]:>5.1f}% {vals[6]:>5.1f}%{x6_dir} | "
          f"{min_pct:>4.1f}% {y1r:>5.2f} {y2r:>5.2f}"
          f"{'  ★' if has_flip and min_pct > 10 else ''}")

    score = min_pct + (5 if has_flip else 0)
    if score > best_score:
        best_score = score
        best_config = (c_z, m4_add, y2_raw, y2_zc)
        best_results = res


print(f"\n{'='*70}")
print(f"BEST CONFIG: c_z={best_config[0]}, m4_add={best_config[1]}, "
      f"y2_raw={'M1' if best_config[2] else 'M1_eff'}, y2_z_coeff={best_config[3]}")
print(f"{'='*70}")
if best_results:
    for k, (e, p) in sorted(best_results.items()):
        if k.startswith("_"):
            print(f"  {k}: {e:.4f}")
        else:
            ok = "YES" if p > 20 else ("~" if p > 12 else "NO")
            print(f"  {k:<20} effect={e:.4f}  {p:>6.1f}%  {ok}")
