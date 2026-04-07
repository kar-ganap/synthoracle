# Phase 2.1 Retro: Evaluation Harness + BO Comparison

**Date:** 2026-03-21
**Branch:** `phase-2.1-evaluation-harness`
**Status:** Complete

---

## 1. What we set out to do

Build the evaluation infrastructure: multi-seed runner, comparison metrics (HV convergence, final HV distribution), and plotting. Run a 10-seed BO vs VR (batch) comparison on Medium 1A to establish baselines.

## 2. What actually happened

- Built `eval/runner.py` — multi-seed runner supporting BO and VR methods with shared reference point
- Built `eval/metrics.py` — `compute_comparison()` producing HV matrices, mean/std curves, wall times
- Built `eval/plot.py` — HV convergence envelope and final HV boxplot
- Ran 10-seed comparison: BO reaches 90% reference HV at eval ~19; VR (batch, Sonnet) is competitive early (50% at eval 13) but diverges at 90%
- Comparison results saved in `experiments/comparison/results/`

### 10-seed BO baseline results

| Metric | Mean | Std |
|--------|------|-----|
| Final HV | 0.258 | 0.010 |
| Eval to 50% ref | 13 | 2 |
| Eval to 90% ref | 19 | 3 |

## 3. What went well

- **Shared reference point across seeds** ensures comparable HV values — essential for multi-seed analysis
- **BO baseline is solid** — low variance, consistent behavior, good benchmark to compare against

## 4. What went poorly

- **VR (batch, Sonnet) underperforms BO** at 42 iterations — the directional accuracy issue from Phase 2.0 compounds across seeds
- **Runner only supports batch VR** — tool-use agent wasn't built yet, so the runner abstraction was batch-only

## 5. Lessons learned

- Multi-seed infrastructure is essential before making claims — single-seed VR results were misleading about competitiveness
- The BO baseline variance (~0.010) sets the bar: any VR improvement must exceed this to be detectable

## 6. What to do differently next time

- Build the runner abstraction to support multiple agent types from the start
- Track cost per seed in the runner output
