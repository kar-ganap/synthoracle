# Phase 1.1 Retro: Oracle Characterization

**Date:** 2026-03-20
**Branch:** `phase-1.1-oracle-characterization`
**Status:** Complete

---

## 1. What we set out to do

Expand the oracle family (simple oracle, medium variants 1B and 1C) and build a reusable characterization module with correct Sobol estimators, OAT effect analysis, and Pareto front computation. Fix the Phase 1.0 Sobol bug.

## 2. What actually happened

- Built `characterize.py` with correct Saltelli/Jansen Sobol estimators (first-order + total-order), OAT effects, output stats, Pareto front extraction, and timing
- Built `SimpleOracle`: 4 inputs, 2 outputs, smooth power-law mechanisms, 8-edge DAG — all EASY difficulty
- Added `variant="1B"` to `MediumOracle`: M2 formula `sqrt(X1)` → `1/X1`, same DAG structure, shifted regime boundary
- Built `MediumOracle1C`: adds M5 = X3^0.9 * X5^0.6, creating new X3→Y4 and X5→Y4 pathways (29 edges total)
- Built `run_characterization.py` generating 1D sweep plots and Pareto front plots for all 4 oracles
- 87 tests pass, lint clean, mypy strict clean, all characterization plots generated

### Key characterization results

**Simple oracle:** Clean, smooth responses. Sobol sums ~0.91-0.94 (near-additive, minimal interactions). All inputs matter (8-53% OAT effect). 123 Pareto points with convex trade-off. 2 us/eval.

**Medium 1A:** Sobol sums ~0.88-1.00. Meaningful interactions for X2, X4, X5, X6 on Y1 (S_T - S_1 = 0.05-0.07), confirming coupling and threshold mechanisms create non-additive variance. All 6 discoveries detectable at 14-67% OAT. 313 Pareto points. 8.8 us/eval.

**Medium 1B:** Regime shift visible — X1's effect on Y1 drops from 16.1% to 6.3% OAT, while X1's effect on Y2 jumps from 16.9% to 25.9%. The `1/X1` formula makes M2 negligible at low X1, shifting the dominant mechanism. Larger Pareto front (767 points). Y3, Y4 identical to 1A (as designed).

**Medium 1C:** M5 mechanism clearly visible — X3 now has 31.5% OAT effect on Y4 (was 0%), X5 has 22.6% (was 0%). Y4 range nearly doubles (0.31 → 0.59). Y1, Y2, Y3 unchanged from 1A.

## 3. What went well

- **TDD delivered again.** Writing tests first for characterize.py caught the Sobol estimator issues early. The `first_order_sum_leq_one` test directly validates the Phase 1.0 bug fix.
- **Reusable characterization module pays off immediately.** Running `characterize()` on all 4 oracles with one script gave immediate quantitative validation of each variant's design intent.
- **Variant 1C structural surprise works as designed.** The 1D sweep plots show a clear before/after: X3 vs Y4 goes from flat (1A) to a power-law curve (1C). An agent that transfers a "X3 doesn't affect Y4" belief from 1A will be wrong on 1C.
- **Phase 1.0 lessons applied.** Effect sizes verified quantitatively before committing. No placeholder parameters.

## 4. What went poorly

- **Nothing major.** This phase was straightforward because Phase 1.0 did the hard work of parameter tuning. The oracle formulas were already solid.

## 5. Lessons learned

- **Correct Sobol estimators matter.** Phase 1.0's `1 - S_T_j` bug produced plausible-looking but wrong first-order indices. The Saltelli (2010) estimator `S_j = (1/N) * sum(Y_B * (Y_AB_j - Y_A)) / Var(Y)` is the correct one.
- **Interaction strength (S_T - S_1) is a useful diagnostic.** Values of 0.05-0.08 for coupled inputs (X4, X6) vs ~0 for uncoupled inputs (X3 on Y4) confirms the coupling mechanisms work.
- **Simple oracle validates the benchmark design.** BO should easily handle smooth, near-additive landscapes — the simple oracle will serve as a useful control showing VR overhead isn't justified everywhere.

## 6. What to do differently next time

- Consider adding confidence intervals to Sobol estimates (bootstrap). Current point estimates are sufficient for characterization but CIs would strengthen claims.
- The Pareto front computation uses brute-force filtering (200k samples). For harder oracles with more objectives, may need Kung's algorithm or NSGA-II-style sorting.
