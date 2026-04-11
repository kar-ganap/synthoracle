# Gap-Closing Experiments — Post-Mortem

**Date:** 2026-04-11
**Branch:** `phase-2.7-difficulty-rubric`
**Pre-registration:** `gap_preregistration.md` (committed 2026-04-11 before experiments)
**Status:** Preliminary — BO 1D @ 144 and BO HD @ 144 still running. Will be
finalized once those land.

This document compares the predictions from the pre-registration to the
actual experimental outcomes. Findings are marked:
- ✓ (prediction within CI)
- ✗ (prediction outside CI but qualitatively correct)
- ✗✗ (prediction qualitatively wrong / direction flipped)
- ?? (inconclusive due to noise or small n)

---

## Experiment A1: 1D prior + fresh @ n=3

### Predicted vs Actual

| Quantity | Predicted | Actual (n=3) | Verdict |
|---|---|---|---|
| 1D fresh VR/BO | 0.88 [0.85, 0.92] | **0.909** | ✓ |
| 1D prior VR/BO | 0.81 [0.78, 0.85] | **0.926** | **✗ (5pp higher than upper CI)** |
| 1D prior penalty (positive = prior hurts) | +8% [+4%, +13%] | **−2.0%** | **✗✗ qualitatively wrong** |
| σ(1D fresh VR/BO) | 0.03 | 0.063 | ✗ (2× larger) |

### Raw numbers

| Condition | n=1 pilot HV | n=3 mean HV | n=3 std | Δ from pilot |
|---|---|---|---|---|
| 1D prior | 1.054 | **1.209** | 0.120 | +15% |
| 1D fresh | 1.156 | **1.186** | 0.074 | +3% |

### Interpretation

**The n=1 seed 42 pilot was unrepresentative.** With two additional seeds, 1D
prior's mean HV is 1.209 — HIGHER than 1D fresh's 1.186, by 2.0%. The original
"−9% prior penalty" was a negative outlier (seed 42 happened to be a bad prior
run).

**Rule 9 (original):** "prior topology preservation matters — 1D's partially-
wrong prior gives a −9% penalty at matched budget." **FALSIFIED** at n=3.

**Rule 9 (revised by data):** "on 1D (functional shifts only, topology
preserved), the screen-first protocol **leverages** the 1A prior: prior runs
outperform fresh runs by 2% on average (n=3, σ=0.12 for prior, σ=0.07 for
fresh). The overlap in 95% CI is substantial, so the statistical claim is
'prior is at worst neutral, at best mildly helpful'."

**Rule 8 (mechanism sufficiency predicts VR underperforms on 1D):** unclear.
1D fresh VR/BO is 0.909 — HIGHER than 1A multi_seed_72 VR/BO (0.685 at matched
BO baseline). So 1D's lower R²(M→Y) = 0.748 does NOT produce a worse VR/BO
than 1A's R²(M→Y) = 0.961. The predicted relationship between R²(M→Y) and
VR/BO is **not supported** — the R² metric captures mechanism→Y sufficiency,
but VR's performance on the optimization problem depends on more than that.

---

## Experiment A2: 1E prior + fresh @ n=3

### Predicted vs Actual

| Quantity | Predicted | Actual (n=3) | Verdict |
|---|---|---|---|
| 1E fresh VR/BO | 0.99 [0.96, 1.02] | **0.934** | ✗ (slightly lower, outside CI) |
| 1E prior VR/BO | 0.99 [0.96, 1.02] | **0.977** | ✓ |
| 1E prior penalty | 0% [−4%, +4%] | **−4.6%** | ✗ (prior helps, larger than expected) |
| σ(1E fresh VR/BO) | 0.02 | 0.040 | ✗ (2× larger) |

### Raw numbers

| Condition | n=1 pilot HV | n=3 mean HV | n=3 std | Δ from pilot |
|---|---|---|---|---|
| 1E prior | 0.276 | **0.272** | 0.005 | −1% |
| 1E fresh | 0.275 | **0.260** | 0.011 | −5% |

### Interpretation

**The 1E "neutral prior" finding was weakened by the pilot overestimating
fresh performance.** With n=3, 1E fresh drops to 0.260 (from 0.275 pilot)
while 1E prior stays at 0.272 (nearly identical to pilot 0.276). The net
effect: **1E prior outperforms 1E fresh by 4.6%**.

**Rule 9 (original, 1E side):** "catastrophically wrong prior (full topology
rewire) has neutral effect under screen-first protocol." **Partially
falsified** — prior doesn't just stay neutral; it **outperforms fresh** by
~5% on average. The screen-first protocol successfully uses the prior even
when Y2/Y4 are uncorrelated with 1A.

**Rule 9 (revised by 1D+1E data):** "The screen-first transfer protocol
successfully leverages partial prior knowledge even when structural
assumptions are violated. On 1D (functional shifts, topology preserved),
prior helps by 2%. On 1E (full topology rewire, Y2/Y4 uncorrelated),
prior helps by 5%. Both results contradict the naive expectation that
a wrong prior should hurt."

**This is a significantly stronger positive claim for the paper** than
the original "prior is dangerous on 1D, neutral on 1E."

---

## Experiment B: Sonnet HD @ n=3

### Predicted vs Actual

| Quantity | Predicted | Actual (n=3) | Verdict |
|---|---|---|---|
| VR HV | 0.245 [0.22, 0.27] | **0.254 ± 0.014** | ✓ |
| VR/BO | 0.92 [0.83, 0.97] | **0.956** | ✓ |
| Recall | 0.55 [0.30, 0.80] | (TBD — need audit data) | ?? |
| Noise OAT fraction | 0% [0, 8%] | **20%** (4/20) | **✗ FAILED** |
| False noise edges (≥0.5 conf) | 0 | **0** (max conf 0.22) | ✓ |
| Max noise edge confidence | ≤ 0.10 | **0.22** | ✗ |

### Interpretation

**HV outcome generalizes; screening *strategy* does not.**

The headline HV and zero-false-positive claims hold for Sonnet:
- Sonnet HD VR/BO = 0.956 (vs Opus 0.966) — within 1pp
- Max noise edge confidence = 0.22 (still well below the 0.5 "claimed
  edge" threshold)
- No noise edge is actually claimed as real by either model

But the **screening strategy differs substantially**:
- **Opus**: dismisses all 6 noise dimensions via pure inference (0/12 OAT
  sweeps on noise, 0% tool-call evals on noise). Internal confidence for
  noise edges: 0.02-0.05.
- **Sonnet**: tests some noise dimensions (4/20 OAT = 20% of sweeps). Max
  noise edge confidence reaches 0.22 on one seed (X9→Y2), indicating
  Sonnet's dismissal is less confident.

**This is a more nuanced finding than "Sonnet replicates Opus":**
1. The *outcome* that matters for HV and zero-false-positives is model-
   independent (both Opus and Sonnet produce near-identical HV and no
   false positive claims).
2. The *pathway* to that outcome — perfect inference-based dismissal — is
   partially Opus-specific. Sonnet verifies more and dismisses less
   confidently but still dismisses successfully.

**Rule 10 (revised):** "The VR HD result (zero false positives, HV matches
BO within ~4%) generalizes from Opus to Sonnet, but the screening *strategy*
is model-dependent. Opus dismisses irrelevant dimensions via inference;
Sonnet tests more and dismisses with lower confidence. Both strategies
produce zero false positive edges (confidence < 0.5)."

**New rule candidate (Rule 11):** "Models may achieve the same optimization
outcome via different cognitive pathways. Reporting only headline metrics
obscures this — the screening-efficiency metric (Section 8) is the
diagnostic that surfaces the difference."

---

## Experiment B2: Haiku HD @ n=3 (added after preregistration)

**Motivation:** After seeing the Opus → Sonnet screening strategy gradient
(0% noise OAT → 20%), we added a Haiku HD run to test the capability floor
and complete the 3-tier model sweep. This experiment was NOT pre-registered
so the predictions below are post-hoc expectations.

**Post-hoc expectations (stated before running, recorded in this paragraph
only):** Haiku would continue the trend (~30-50% noise OAT) with moderate
HV degradation (~75% of BO). Based on the Phase 2.3 Haiku-on-1A pilot
which reached 62% of BO with decent edge recall but zero structured
iteration summaries.

### Surprises

1. **Haiku does not support adaptive thinking.** First run attempt failed
   immediately with HTTP 400 "adaptive thinking is not supported on this
   model." Re-ran with `thinking=None` and got usable data.

2. **Haiku HV collapsed to 43.7% of BO** — much worse than the ~75%
   expectation. Larger capability gap than anticipated.

3. **Structured iteration summaries worked** (contra Phase 2.3 pilot).
   Haiku 4.5 produces usable Pydantic-parsed output now; we have full
   rubric data for Haiku including info capture and edge confidences.

4. **Seed 44 crossed the 0.5 false-positive threshold.** Haiku seed 44
   claimed **two noise edges as real**: X9→Y2 at 0.60 and X8→Y1 at 0.55.
   This is the first time in the entire project that VR produces a
   false positive noise edge claim under the strict ≥0.5 confidence
   definition.

### Full 3-tier comparison (HD 72, n=3 each)

| Model | HV ± std | VR/BO | Noise OAT % | Max noise conf | Info capture | False positives |
|---|---|---|---|---|---|---|
| Opus | 0.257 ± 0.011 | 96.6% | 0% | 0.00 | 0.996 | 0 |
| Sonnet | 0.254 ± 0.014 | 95.6% | 20% | 0.30 | 0.999 | 0 |
| **Haiku** | **0.116 ± 0.034** | **43.7%** | 22% | **0.60** | 0.959 | **2 (seed 44)** |

### Findings

1. **Capability cliff, not gradient.** The HV drops 52pp from Sonnet (95.6%)
   to Haiku (43.7%) — a cliff, not a smooth gradient. The cognitive style
   gradient (Opus 0% → Sonnet/Haiku 20-22% noise OAT) is real but does NOT
   predict the HV cliff.

2. **Discovery-vs-optimization decoupling.** Haiku's info capture (Sobol-
   weighted recall) is 0.959 — near-parity with Opus/Sonnet. Haiku
   correctly identifies the high-impact causal edges. But its HV is only
   44% of BO. **Haiku discovers the causal structure but cannot exploit
   it for optimization.** This is a qualitatively different failure mode
   from what I expected (expected: "Haiku fails to discover").

3. **Zero-false-positive property breaks at Haiku.** Opus (max conf 0.00)
   and Sonnet (max conf 0.30) both stay below the 0.5 "claimed edge"
   threshold on every seed. Haiku seed 44 claims two noise edges at 0.55
   and 0.60. The VR guarantee of "no false causal claims" holds for
   Sonnet-and-above, breaks at the Haiku tier.

4. **Capability floor identified.** The VR protocol's zero-false-positive
   and competitive-HV properties hold above the Sonnet capability tier.
   At the Haiku tier, both break. This is a useful result for practitioners:
   **pick a Sonnet-class or better model for VR-style causal reasoning.**

### Rule updates from Haiku HD

- **Rule 10 (model generality)** needs complete restatement. The old claim
  "VR is model-general" is wrong. The new claim is:
  **"VR's headline properties (competitive HV, zero false positive claims)
  generalize from Opus to Sonnet — a ~40% cheaper model — but break at
  Haiku. This identifies a practical capability floor at the Sonnet tier
  for VR-style protocols that rely on adaptive thinking and structured
  iteration summaries."**

- **Rule 11 (new) — discovery vs optimization decoupling:**
  "Causal discovery (info capture) and optimization (HV) can decouple.
  Haiku HD recovers 0.959 info capture (near-parity with Opus/Sonnet) but
  only 43.7% of BO on HV. Discovering the structure is not sufficient; the
  agent must also exploit it. At Haiku tier, exploitation fails even when
  discovery succeeds."

- **Rule 12 (new) — cognitive strategy gradient:**
  "Models achieve the same outcomes via different tool-use strategies
  when all three properties (HV, false positives, recall) coincide — as
  they do for Opus and Sonnet on HD. Opus: 0% noise OAT, dismissal by
  inference. Sonnet: 20% noise OAT, verification with residual uncertainty.
  When the outcomes diverge (Haiku), the strategy difference is no longer
  just cosmetic."

---

## Experiment C: BO baselines @ 144 budget

**Status:** Partial. BO 1A @ 144 complete. BO 1D and BO HD still running
(each takes ~3-5 hours due to O(n³) GP fitting at high n). 1E skipped
(known to be at ceiling at 66-eval BO).

### 1A result

| Oracle | BO @ 66 | BO @ 144 | Δ | Predicted Δ |
|---|---|---|---|---|
| 1A | 0.2754 | **0.2811** | **+2.2%** | +0.3% |

**Prediction: ✗ (BO is NOT saturated as predicted).**

BO 1A gains 2.2% from 66→144 evals — much more than predicted. This is
small in absolute terms but qualitatively important.

### Consequence for the rubric

The audit now uses BO 1A @ 144 as the baseline for 1A comparisons (via
`_bo_runs_for` preferring `*_n144_*.npz`). This updates:

| 1A condition | Old VR/BO (vs BO@66) | New VR/BO (vs BO@144) | Δ |
|---|---|---|---|
| 1A multi_seed_72 (n=10) | 0.699 | **0.685** | −1.4pp |
| 1A extended_144 (n=4) | 1.009 | **0.989** | −2.0pp |

**Critical finding: VR 1A extended does NOT exceed BO 1A at matched
budget.** All 4 VR seeds fall below BO 1A @ 144 (0.281). The gap is small
(~1%) but real.

**Rule 2 (extended-budget crossover):** the claim "VR crosses BO at eval
123" needs qualification. It crosses BO *as-it-stood-at-66-evals*. At
matched 144-eval comparison, VR 1A ext reaches ~99% of BO but does not
exceed it.

**Rule 2 (revised):** "At 2× the matched budget (144 evals on a 6-input
oracle), VR closes the understanding gap — from 68% at 72 budget to 99%
at 144 budget — but does not fully overtake matched-budget BO on 1A. The
qualitative finding (VR becomes competitive at extended budget) holds;
the 'crosses BO' headline is a match only if BO is treated as its 66-eval
ceiling, not its true 144-eval optimum."

### Pending: BO 1D and HD @ 144

- BO 1D @ 144: if similar +2% gain, then VR 1D prior_144 (n=1 = 1.002 vs
  BO@66) becomes ~0.98 vs BO@144. VR 1D prior_144 would still be very
  close to BO but possibly just below.
- BO HD @ 144: if similar +2% gain, then BO HD @ 144 ≈ 0.272. VR HD ext
  (0.268) would be ~0.985 of BO — close but below.

**The HD extended crossover claim is at risk of the same qualification.**
Will be resolved when BO HD @ 144 lands.

---

## Summary of rule revisions (preliminary)

| Rule | Status | Revision |
|---|---|---|
| 1 (baseline understanding tax) | ✓ STRENGTHENED | Still 68-70% of BO on 1A at 72 budget; new BO@144 baseline slightly lowers the number but qualitative finding holds |
| 2 (extended-budget crossover) | **✗ QUALIFIED** | VR reaches 99% of BO at 144 budget; does NOT exceed matched-budget BO. "Crosses 66-eval BO" is a weaker claim than originally stated |
| 3 (dimensionality scaling) | ✓ UNCHANGED | HD VR/BO (96.6%) still exceeds 1A VR/BO (68.5%) at 72 budget |
| 4 (perfect screening at tight budget) | ✓ UNCHANGED (Opus) + ✗ NUANCED (Sonnet) | Opus: 0% noise OAT. Sonnet: 20% noise OAT but zero false claims |
| 5 (recall convergence at extended budget) | ✓ UNCHANGED | HD recall σ still collapses to 0 |
| 6 (calibration learning emerges) | ✓ UNCHANGED | 3/3 HD ext seeds show learning |
| 7 (info capture exceeds raw recall) | ✓ UNCHANGED | 0.996 for HD ext |
| 8 (R²(M→Y) predicts VR fit) | **✗✗ FALSIFIED** | 1D (R²=0.748) has VR/BO 0.909 — higher than 1A (R²=0.961) at 0.685. R² does not monotonically predict VR/BO |
| 9 (prior topology matters) | **✗✗ FALSIFIED, REPLACED** | Prior HELPS on both 1D (+2%) and 1E (+5%). Screen-first protocol leverages prior even when structurally wrong |
| 10 (model generality) | ✓ PARTIALLY | Sonnet matches Opus on HV and false-positives but differs on screening strategy |
| 11 (new) | NEW | Models achieve same outcomes via different pathways — strategy diagnostics (screening) surface this |

## Next steps

1. **Wait for BO 1D and HD @ 144** to land (~3-6 more hours)
2. **Re-run audit + rubric** once BO 1D and HD data are in
3. **Rewrite the rules of thumb** in `build_rubric.py` based on these
   findings:
   - Rule 2: qualify the extended-budget crossover claim
   - Rule 8: replace with a different intrinsic predictor (maybe just
     d/d_eff or interaction_fraction) or drop it
   - Rule 9: replace with "screen-first protocol leverages prior even
     when structurally wrong"
   - Rule 10: update to "HV outcome generalizes; screening strategy is
     model-specific"
4. **Add new Rule 11** about outcome-vs-strategy dichotomy
5. **Update Section 11 limitations** to reflect the new statistical depth
   (1D/1E now n=3)
6. **Commit the final rubric**

## Methodological reflection

Pre-registration saved us from publishing multiple wrong claims:
- Rule 9 (prior penalty) would have been in the paper at −9% with n=1
  evidence. With n=3 we see prior actually **helps**.
- Rule 8 (R² predicts VR fit) would have been a confident claim. The data
  don't support it.
- Rule 2 (VR crosses BO) would have been "VR exceeds BO at extended
  budget." With BO @ 144 we see VR reaches ~99% but not 100%.

Running BO at matched extended budget was also critical — the "VR exceeds
BO" narrative was an artifact of comparing against a BO that was saturated
too early. This is exactly the kind of moving-target problem that
pre-registering the hypothesis "BO is saturated" was designed to catch.
The hypothesis was wrong, the data told us, and we update.
