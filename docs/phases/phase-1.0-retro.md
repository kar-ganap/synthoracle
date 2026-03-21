# Phase 1.0 Retro: Oracle Core

**Date:** 2026-03-20
**Branch:** `phase-1.0-oracle-core`
**Status:** Complete, merged to main

---

## 1. What we set out to do

Build the oracle base class, DAG representation, and medium oracle ("The Bottleneck Shift") with quantitatively verified mechanisms. The medium oracle needed to have 6 discoverable mechanisms at a graduated difficulty level, all detectable from 50-100 samples.

## 2. What actually happened

- Built the abstract Oracle base, CausalDAG with precision/recall + IO projection, and MediumOracle
- Initial parameters (from conceptual doc) failed visual inspection — the "hard" mechanisms (threshold, coupling, dual pathway) had effect sizes of 2-9% of output range, undetectable from sparse samples
- Iterated through quantitative parameter tuning: variance decomposition, OAT effect sizes, conditional effects, grid search over coefficient space
- Major formula revisions:
  - M4 split from pure multiplicative to multiplicative + additive (enables X6 direction flip)
  - Y2 restructured with direct (1-Z) penalty (breaks coupling cancellation)
  - Z coupling constant 0.31→1.0, M4 sigmoid steepness 5.2→25, Y4 saturation coefficient 0.45→2.0
- Final oracle has all 6 discoveries at 13-29% of output range — marginal for medium difficulty, clear for easy, requires focused exploration for hard
- 36 tests pass, lint clean, mypy strict clean
- Sobol first-order analysis was buggy (wrong formula) — flagged but not fixed, deferred to Phase 1.1 characterization module

## 3. What went well

- **TDD worked.** Tests caught the regime transition shift when we changed k_m1 from 3.14 to 5.0. The test boundary values needed updating, but the test itself was a safety net.
- **Visual inspection before committing.** The user caught 5 issues in the initial response surface plots that the numerical tests missed. Plots should be part of every oracle validation.
- **Quantitative parameter tuning.** Variance decomposition and conditional effect sizes gave objective criteria for "is this mechanism detectable?" — much better than eyeballing plots.
- **Grid search for the final parameter set** found the optimal balance across all 6 discoveries simultaneously.

## 4. What went poorly

- **Initial parameters were placeholders.** The conceptual doc coefficients (3.14, 0.47, 2.83, 0.31, 5.2, 0.73, 0.23, 0.45) were chosen for aesthetics, not detectability. This wasted time on an implementation that needed immediate rework.
- **Sobol estimator bug.** The ad-hoc variance decomposition script used `1 - S_T_j` instead of `S_j` for first-order indices. The values were suspicious (summed to ~5) and were flagged but not investigated deeply enough during Phase 1.0. Should have caught this before committing.
- **Y2 formula required 3 iterations.** The coupling cancellation issue (Z in both numerator and denominator) wasn't anticipated during design. Each iteration required re-running the full analysis.

## 5. Lessons learned

- **Always run response surface plots before committing oracle code.** Numerical tests verify properties but don't catch "this mechanism is too subtle to discover."
- **Design output formulas with detectability in mind.** When an intermediate variable (Z) appears in both terms of an output, it partially cancels. Detect this during formula design, not during parameter tuning.
- **Quantify effect sizes as % of output range** — this is the right metric for "is it detectable from N samples?" The 15-20% threshold worked well as a heuristic.
- **Sobol indices need tests, not just visual inspection.** First-order indices should sum to ≤1. This should be a test in the characterization module.

## 6. What to do differently next time

- For the simple oracle (Phase 1.1): design the formula *and* verify effect sizes *before* implementing. Don't commit placeholder parameters.
- Build the reusable characterization module early (Phase 1.1 Step 0) so all future oracles get rigorous analysis automatically.
- Include Sobol interaction analysis (S_T_j - S_j) as a standard check for any oracle with coupling.
