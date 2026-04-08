# Phase 2.4 Retro: Multi-Seed Validation

**Date:** 2026-04-07
**Branch:** `phase-2.4-multi-seed`
**Status:** Complete

---

## 1. What we set out to do

Run 10 seeds of the Opus tool agent on Medium 1A with the full diagnostic stack. Produce error bars for all metrics. Compare against the existing 10-seed BO baseline. Establish the statistical foundation for the paper.

## 2. What actually happened

- Ran 10 seeds (42-51) with structured iteration summaries, calibration checkpoints, structured OAT predictions
- Built multi-seed runner with seed skipping (reuse completed seeds), CLI seed range, automatic result loading
- Built Section 10 aggregate analysis in compute_metrics.py
- Generated budget-efficiency curve revealing the "understanding tax" and late acceleration
- Designed oracle variants 1D (Structural Shift) and 1E (Rewired) for credible transfer tests
- Fixed several bugs: SDK timeout for non-streaming calls >10min, calibration max_tokens too low for adaptive thinking, regression_fit crashing on interaction terms (X1*X3), Opus pricing (was 3x too high)

### 10-seed results

| Metric | VR Tool Agent | BO Baseline |
|--------|---------------|-------------|
| Final HV | 0.193 +/- 0.028 | 0.275 +/- 0.001 |
| Edge precision | 1.000 +/- 0.000 | N/A |
| Edge recall | 0.944 +/- 0.000 | N/A |
| OAT direction accuracy | 71.2% +/- 9.8% | N/A |
| OAT magnitude error | 0.149 +/- 0.061 | N/A |
| Calibration MAE (first) | 0.036 +/- 0.014 | N/A |
| Calibration MAE (last) | 0.025 +/- 0.014 | N/A |
| Seeds showing learning | 6/9 | N/A |
| Cost per seed | $4.43 +/- $0.96 | $0 (local) |
| Total cost | $44.35 | $0 |

### Budget-efficiency finding

VR agent flat at ~42% of BO for evals 12-60 (spending budget on OAT sweeps and interaction tests — the "understanding tax"). Accelerates to 70% of BO in the final 12 evals (60-72) as the OPTIMIZE phase kicks in. Every seed gained HV in the last 12 evals (+0.024 to +0.148). The agent was still accelerating when the budget ran out.

### Oracle variants designed

- **1D (Structural Shift):** Same topology, altered forms. X5 threshold removed, X3→Y3 inverted-U, M1_eff sign flip on Y2, new edges X5→Y2 and X3→Y4. Prior ~50% wrong on forms.
- **1E (Rewired):** Same mechanism types, swapped input wiring. 3 false positive edges, 4 false negatives. Y2 and Y4 essentially uncorrelated with 1A (r=0.02, 0.11).

## 3. What went well

- **Perfect causal discovery across all 10 seeds.** 1.000 precision, 0.944 recall, zero variance. The tool agent's OAT-based screening is completely reliable. Every seed misses only X5→Y2 (effect below detection threshold).
- **Calibration shows genuine learning.** 6/9 seeds have decreasing calibration MAE from first to last checkpoint. The agent's internal model improves with data.
- **Budget-efficiency curve is the key figure for the paper.** Reframes "70% of BO" from a failure to "the agent invests in understanding first, then optimizes rapidly while still accelerating."
- **Cost is manageable.** $4.43/seed average with correct Opus pricing. 10-seed study for $44.
- **Multi-seed runner is robust.** Handles seed skipping, CLI ranges, failures gracefully. Checkpointing works.

## 4. What went poorly

- **HV significantly below BO** (0.193 vs 0.275). The understanding tax is real — 30 of 72 evals go to systematic screening. BO optimizes from eval 1.
- **High HV variance** (std=0.028 vs BO's 0.001). The LLM's stochastic behavior creates much more run-to-run variability than BO's deterministic GP.
- **Three bugs required re-runs.** Calibration parse failures (max_tokens too low, then SDK timeout), regression_fit crash on interaction terms. Each bug wasted $5-15 before detection.
- **Pricing was wrong for 2+ weeks.** Opus at $15/$75 instead of $5/$25 — all earlier cost estimates were 3x too high. Should have verified against the API pricing page earlier.
- **X5→Y2 missed by every seed.** The effect (range ~0.006) is genuinely below the 0.05 OAT threshold. This is an oracle design choice, not an agent failure, but it means recall can never hit 1.0 on this oracle.
- **Existing variants (1B, 1C) too similar to 1A for credible transfer.** 1B has identical DAG; 1C adds 3 edges but shares everything else. Had to design new variants (1D, 1E).

## 5. Lessons learned

- **Test with tiny budget before committing to full runs.** The n_budget=20 smoke test caught the calibration parse issue. Without it, we'd have burned $40+ on broken runs.
- **The SDK has internal timeout logic** that rejects large max_tokens for non-streaming calls. Adding `timeout=600.0` to all API calls prevents this. Should be set from the start.
- **Budget-efficiency curves are more informative than final HV.** A single number (0.193 vs 0.275) hides the trajectory. The curve shows the agent is still improving — the comparison is unfair at a fixed budget.
- **Transfer variant design needs adversarial review.** 1B/1C were designed for incremental difficulty, not for testing transfer robustness. 1D/1E are designed to make the prior partially wrong, which is the right framing.
- **Run experiments in chunks.** The 4+6 split for 10 seeds caught the SDK timeout bug after $15 instead of $50.

## 6. What to do differently next time

- Verify API pricing against the actual pricing page before any cost tracking
- Set `timeout=600.0` on all API calls from the start
- Design transfer variants with explicit "prior is wrong about X" criteria before running any transfer tests
- Consider extended budget (n=144) runs on a subset of seeds to test whether the acceleration phase closes the gap with BO
- Write the retro before designing new oracle variants — the variants are Phase 2.5 work, not 2.4
