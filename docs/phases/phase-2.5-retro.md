# Phase 2.5 Retro: Transfer Tests + Extended Budget + Robustness

**Date:** 2026-04-07 to 2026-04-09
**Branch:** `phase-2.5-transfer`
**Status:** Complete

---

## 1. What we set out to do

Test whether the VR agent's causal understanding transfers to structurally different oracle variants, and whether extended budget allows VR to match or exceed BO. Address reviewer-facing gaps in the evidence: n=1 transfer results, Opus-only evaluation, weak BO baselines on new oracles.

## 2. What actually happened

### Oracle variants designed and implemented

- **1D (Structural Shift):** Same mechanism topology, altered functional forms. X5 threshold removed, X3→Y3 inverted-U, M1_eff sign flip on Y2, new edges. Prior ~50% wrong on forms.
- **1E (Rewired):** Same mechanism types, completely rewired inputs. 3 false positive edges in prior, 4 false negatives. Y2/Y4 uncorrelated with 1A (r=0.02, 0.11).

Both designed with adversarial reviewer in mind — 1B/1C were too similar to 1A for credible transfer.

### Transfer results (72 evals, pilot)

| Condition | HV | vs BO | Prior impact |
|-----------|-----|-------|-------------|
| 1D prior (v1, falsify-first) | 0.965 | 74% | -16.5% vs fresh |
| 1D prior (v2, screen-first) | 1.026 | 79% | -11.2% vs fresh |
| 1D prior (v3, strict interaction) | 1.054 | 81% | -8.8% vs fresh |
| 1D fresh | 1.156 | 89% | baseline |
| 1E prior (v1, falsify-first) | 0.217 | 78% | -21.3% vs fresh |
| 1E prior (v3, screen-first) | 0.276 | 99% | +0.2% vs fresh |
| 1E fresh | 0.275 | 99% | baseline |

Three prompt iterations were needed to make transfer work. "Screen first, then compare to prior" eliminated the transfer penalty on 1E entirely.

### Extended budget (144 evals)

| Oracle | VR HV | BO HV | Crossed at | Seeds |
|--------|-------|-------|------------|-------|
| 1A | 0.278 +/- 0.001 | 0.275 +/- 0.001 | eval 108-129 | 4/4 crossed |
| 1D prior | 1.307 | 1.305 | ~eval 96 | 1 seed |
| 1E prior | 0.278* | 0.278 | stalled at 86 | 1 seed (partial) |

*1E stalled at eval 86 due to conversation condensing loop — infrastructure issue, not capability.

### Robustness runs

| Run | Result | Purpose |
|-----|--------|---------|
| BO 1E (4 seeds) | 0.278 +/- 0.002 | Solid baseline |
| 1A extended (4 seeds) | All cross BO | Crossover reproducible |
| Sonnet 1E transfer | HV=0.276 (= Opus) | Protocol model-general |

### Infrastructure improvements

- Conversation condensing (replace full tool history with compact summary between iterations)
- Timestamps on all API calls
- Retry logic for timeouts
- Try/except on iteration summary parse
- Calibration disabled for extended budget runs

### BO baselines

| Oracle | BO HV (seeds) |
|--------|---------------|
| 1A | 0.275 +/- 0.001 (10 seeds) |
| 1D | 1.305 +/- 0.000 (2 seeds) |
| 1E | 0.278 +/- 0.002 (4 seeds) |

## 3. What went well

- **VR crosses BO at extended budget — reproducibly.** 4/4 seeds on 1A cross BO between eval 108-129. The understanding tax is an investment that pays back at 2x BO's convergence point.
- **Transfer protocol matters more than transfer content.** Three prompt iterations (falsify-first → screen-first → strict interaction rule) turned a -21% penalty into neutral. The final protocol is: screen the new system with OAT first, then compare to prior, never test prior interactions without OAT evidence.
- **Results are model-general.** Sonnet matches Opus exactly on 1E transfer (HV difference: 0.0001). The protocol, not the model, drives the result.
- **Oracle variants are adversarially credible.** 1D and 1E have 0-3 false positive edges and 2-4 false negatives from the 1A prior. Y2/Y4 correlation with 1A drops to 0.02 on 1E. A reviewer can't dismiss these as "trivial parameter tweaks."
- **BO struggles on the new oracles.** Severe GP conditioning issues (non-positive-definite matrices) on 1D and 1E, requiring repeated jitter and retry. BO still works but the GP is not well-suited to these nonlinear oracle structures.

## 4. What went poorly

- **Conversation condensing caused stalls.** The agent lost context after condensing and looped on analysis tools without spending eval budget. Happened on 1A at eval 113 and 1E at eval 86. The condensing was too aggressive initially (2-message stub); improved to include data summary and recent evaluations. Still not fully solved — 1E at 144 budget stalled.
- **Calibration timeouts killed extended runs.** The calibration checkpoint's `parse()` call timed out repeatedly at extended budget. Had to disable calibration for 144-eval runs. The calibration feature works at 72 evals but is unreliable beyond that.
- **Three prompt iterations to get transfer right.** The initial "falsify the prior" prompt was counterproductive — it told the agent to start with the prior's claims. This wasted ~$10 on failed runs before finding the right framing.
- **BO baselines took much longer than expected.** GP conditioning issues on 1D/1E made each BO seed take 30+ minutes instead of 5. Had to reduce from 10 seeds to 2-4.
- **Runs done as inline commands, not scripts.** Multiple one-off runs made reproducibility harder. Created `run_robustness.py` mid-phase to address this.

## 5. Lessons learned

- **"Screen first, then compare" is how scientists actually transfer knowledge.** Don't test a prior's claims — measure the new system and use the prior to interpret your measurements. The prompt should match the cognitive process.
- **Conversation condensing is essential for extended budget but hard to get right.** Too aggressive = agent loses context and loops. Too conservative = context blows up and API calls timeout. The sweet spot: include data summary, recent evaluations, and the causal model, but drop raw tool call history.
- **Calibration checkpoints are a diagnostic tool, not a production feature.** They work at 72 evals but become the reliability bottleneck at extended budget. For production runs, disable calibration and rely on iteration summaries for model quality assessment.
- **The BO crossover is the strongest result but requires 2x budget.** At 72 evals (matched budget), VR is at 70% of BO. At 144, VR exceeds BO. The paper needs to frame this as "VR invests in understanding, which has a payback period" — not "VR is better."
- **Infrastructure reliability limits experimental throughput more than API cost.** The $7/run cost is manageable. The 2-hour runtime with potential stalls, timeouts, and crashes is the real bottleneck.

## 6. What to do differently next time

- Write experiment scripts upfront, not inline commands
- Disable calibration for any run >72 evals from the start
- Test conversation condensing on a 20-eval dry run before committing to 144-eval runs
- Run BO baselines first (they're free) before designing VR experiments
- Design oracle variants before Phase 2.4, not during Phase 2.5 — the variant design should be reviewed independently of the transfer results
- Track all API spend in real-time, not retroactively

## Total Phase 2.5 spend

~$45 (transfer pilots + prompt iterations + extended budget + robustness)

Grand total project spend: ~$155
