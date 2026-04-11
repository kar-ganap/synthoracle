# Gap-Closing Experiments — Preregistration

**Date written:** 2026-04-11
**Branch:** `phase-2.7-difficulty-rubric`
**Audit data version:** post-commit `9806ffc` (rubric drops 1B/1C)

This document is committed **before** the gap-closing experiments are run.
It records what we expect to see, so we can honestly verify after the fact
whether the new data actually closes the gaps it was designed to close, or
whether the rubric's rules of thumb were wrong.

The rubric's current scope (after dropping 1B/1C as stepping stones):
**1A, 1D, 1E, HD**.

## Background

Five rules in the current rubric rest partially or fully on n=1 evidence:
- **Rule 8** (mechanism sufficiency predicts where VR works) — relies on n=1 1D data
- **Rule 9** (prior topology preservation matters) — relies on n=1 1D and n=1 1E
- **Rule 10** (model generality) — relies on n=1 Sonnet 1E (which is at HV ceiling so it's an uninformative test)

The crossover_eval metric for HD ext (134) and 1A ext (123) is computed against
BO baselines that only ran for 54-66 evals — shorter than the VR runs themselves.

This preregistration covers four experiments designed to address these gaps.
For each, we predict the expected outcome and identify what would falsify the
corresponding rule.

---

## Experiment A1: 1D prior + fresh, n=3 (seeds 42 already on disk; add 43, 44)

**Cost:** ~$12-20 (4 new Opus 72-eval runs)

**Currently on disk (n=1):**
- 1D prior @ 72: VR HV = 1.054, VR/BO = **0.808**
- 1D fresh @ 72: VR HV = 1.156, VR/BO = **0.886**
- Implied prior penalty: **−9% (1 − 0.808/0.886)**

**Predictions for n=3:**

| Quantity | Predicted mean | Predicted σ | 95% CI |
|---|---|---|---|
| 1D fresh VR/BO | 0.88 | 0.03 | [0.85, 0.92] |
| 1D prior VR/BO | 0.81 | 0.04 | [0.78, 0.85] |
| Prior penalty | 8% | 0.04 | [4%, 13%] |

**What we're testing:**

1. **Rule 9 (1D side):** "1D's partially-wrong prior produces a measurable
   penalty at matched budget." Pre-registered as TRUE if prior penalty
   95% CI excludes 0.
2. **Rule 8 (1D side):** "Low R²(M→Y) = 0.748 predicts VR underperforms
   on 1D even with correct prior." Pre-registered as TRUE if 1D fresh VR/BO
   < 1A multi-seed (0.70) — wait, this is wrong direction. Let me restate:
   1A multi-seed VR/BO = 0.70. If 1D's lower R²(M→Y) hurts VR, we'd expect
   1D fresh VR/BO ≤ 1A. But our n=1 says 1D fresh VR/BO = 0.886 (HIGHER than
   1A). This is suspicious — let me think.

   Actually 1D's reference HV is 1.10 (much higher than 1A's 0.26), so the
   normalization changes the comparison. Better metric: **1D fresh HV / 1D
   ref HV** vs **1A fresh HV / 1A ref HV**. From the audit:
   - 1A multi-seed: VR HV (%ref) = 0.735
   - 1D fresh: VR HV (%ref) = 1.049 (n=1)

   So 1D fresh actually exceeds its ref HV (because BO also exceeds it; the
   reference HV from characterize() with 200k samples isn't the true Pareto
   front). The ratio that matters for Rule 8 is **VR/BO**, not VR/ref.

   On VR/BO, 1A = 0.70, 1D fresh = 0.886. So 1D fresh > 1A on VR/BO. This
   actually CONTRADICTS the naive "low R² = VR worse" prediction.

   **Revised Rule 8 prediction:** I should weaken this. The relationship
   between R²(M→Y) and VR/BO is not simply monotonic — 1D's higher reference
   HV may give VR more headroom. Pre-register as: **the rubric will need to
   refine Rule 8** based on n=3 data, possibly to "low R²(M→Y) shifts which
   outputs are hardest to predict (Y3 in 1D's case)" rather than "low R²
   reduces VR/BO."

3. **Variance check:** if σ on VR/BO > 0.10, the 1D HV trajectory is
   unusually noisy and the n=1 anchor was misleading.

**Falsification criteria:**
- 1D prior penalty < 0.03 (no detectable penalty): Rule 9's 1D claim fails
- 1D prior penalty > 0.20 (much larger than estimated): the n=1 was an
  underestimate; the rule still holds but the magnitude shifts substantially
- σ(1D fresh VR/BO) > 0.10: HV is too noisy to support specific rule magnitudes

---

## Experiment A2: 1E prior + fresh, n=3 (seeds 42 on disk; add 43, 44)

**Cost:** ~$12-20 (4 new Opus 72-eval runs)

**Currently on disk (n=1):**
- 1E prior @ 72: VR HV = 0.276, VR/BO = **0.992**
- 1E fresh @ 72: VR HV = 0.275, VR/BO = **0.990**
- Implied prior penalty: **+0.2% (essentially 0)**

**Predictions for n=3:**

| Quantity | Predicted mean | Predicted σ | 95% CI |
|---|---|---|---|
| 1E fresh VR/BO | 0.99 | 0.02 | [0.96, 1.02] |
| 1E prior VR/BO | 0.99 | 0.02 | [0.96, 1.02] |
| Prior penalty | 0% | 0.03 | [−4%, +4%] |

**What we're testing:**

1. **Rule 9 (1E side):** "Catastrophically wrong prior (full topology rewire)
   has neutral effect under screen-first protocol." Pre-registered as TRUE
   if |prior penalty| < 5% with overlapping CI for prior and fresh.

2. **Ceiling effect concern:** 1E's BO baseline is 0.278 vs ref HV 0.259, so
   BO already exceeds the reference. Both VR conditions are at this ceiling
   (~0.275-0.276). The n=3 comparison may continue to be uninformative because
   there's no headroom. If both fresh and prior cluster at the ceiling, the
   "neutral" claim is consistent but weakly supported.

3. **Variance check:** 1E n=1 std is 0 by construction. With n=3 we'll
   finally see real σ. If σ > 0.05, 1E HV is unexpectedly noisy and the
   ceiling-effect interpretation needs revision.

**Falsification criteria:**
- 1E prior CI bound < 0.95 (clear penalty): screen-first protocol doesn't
  fully neutralize a topology-rewired prior → Rule 9 weakens significantly
- 1E fresh > 1E prior by > 5%: small but real penalty exists → Rule 9 needs
  qualification

---

## Experiment B: Sonnet HD n=3 (3 new Sonnet 72-eval runs)

**Cost:** ~$6-9

**Currently on disk:**
- Opus HD @ 72 (n=3): VR HV = 0.2569 ± 0.011, VR/BO = **0.966**
- Opus HD screening: **0/12 OAT sweeps on noise inputs (0.0%)**
- Opus HD false noise edges: **0 across all seeds**
- Sonnet HD: **zero data**

**Predictions for Sonnet HD n=3:**

| Quantity | Predicted mean | Predicted σ | 95% CI |
|---|---|---|---|
| VR HV | 0.245 | 0.015 | [0.22, 0.27] |
| VR/BO | 0.92 | 0.05 | [0.83, 0.97] |
| Recall | 0.55 | 0.15 | [0.30, 0.80] |
| Noise OAT fraction | 0.0% | small | [0%, 8%] |
| False noise edges | 0 | — | — |

**Calibration:** Opus 1A → Sonnet 1A degraded by ~5pp (model sweep result).
Sonnet HD should be similar: ~5pp below Opus HD. Opus HD/BO = 0.966, so
Sonnet HD/BO ≈ 0.92.

**What we're testing:**

1. **Rule 10 (model generality):** "VR's behavior on HD is not Opus-specific."
   Pre-registered as TRUE if:
   - Sonnet HD VR/BO ≥ 0.85 (within 10pp of Opus)
   - Sonnet HD noise OAT fraction ≤ 0.10
   - Sonnet HD false noise edges = 0
2. **Stronger claim — screening behavior generalizes:** "0% noise OAT" is the
   most surprising HD result. Pre-registered as TRUE if Sonnet HD noise OAT
   fraction ≤ 0.10.
3. **Recall expectation:** Sonnet should have lower recall than Opus
   (~0.55 vs 0.685) because Sonnet's hypothesis generation is less systematic.

**Falsification criteria:**
- Sonnet HD VR/BO < 0.80: model gap is much larger on HD than 1A → screening
  has Opus-specific dependencies (would weaken Rule 10 substantially)
- Sonnet HD noise OAT fraction > 0.20: Sonnet doesn't replicate the screening
  behavior → screening is partially Opus-specific
- Sonnet HD has any false noise edges: zero-false-positive claim becomes
  Opus-specific
- Sonnet HD recall < 0.30 with high σ: Sonnet is much worse at discovery
  on HD; model gap is qualitative not just quantitative

---

## Experiment C: BO baselines at 144 budget (free, local BoTorch)

**Cost:** $0 (local compute, ~30-60 min total)

**Currently on disk:**
- BO 1A @ 66 (10 seeds): 0.275 ± 0.001
- BO 1D @ 66 (2 seeds): 1.305 ± 0.000
- BO 1E @ 66 (4 seeds): 0.278 ± 0.002
- BO HD @ 66 (3 seeds): 0.266 ± 0.002

**Predictions for BO at 144:**

| Oracle | Current (66) | Predicted (144) | Predicted Δ |
|---|---|---|---|
| 1A | 0.275 | 0.276 | +0.001 |
| 1D | 1.305 | 1.305 | 0.000 |
| 1E | 0.278 | 0.279 | +0.001 |
| HD | 0.266 | 0.267 | +0.001 |

**What we're testing:**

1. **BO saturation:** "BO's qNEHVI converges within ~50 evals on these
   oracles." Pre-registered as TRUE if all 144-eval BO HVs are within
   0.005 of their 66-eval counterparts.
2. **Crossover_eval interpretation:** if BO is genuinely saturated, then
   "VR catches BO at eval 123/134" means VR catches a static target —
   stronger claim. If BO improved meaningfully from 66→144, the crossover
   metric is comparing VR to a moving target.

**Falsification criteria:**
- Any BO oracle gains > 0.01 from 66→144: BO is not saturated; the
  crossover_eval claim weakens (we'd need to recompute it against the
  longer BO curve)
- Any BO oracle gains > 0.02: BO is genuinely improving at extended budget,
  and VR's "exceeds BO by 1%" claim may be meaningless

---

## Pre-registered post-mortem checklist

After all experiments land, write `gap_postmortem.md` with:

1. **For each prediction**, mark: ✓ (within CI), ✗ (outside CI but qualitatively
   correct), ✗✗ (qualitatively wrong), or ?? (inconclusive due to noise).
2. **For each rule of thumb** the experiment was designed to test, mark:
   STRENGTHENED, UNCHANGED (still directional), WEAKENED, or REPLACED.
3. **For each new gap revealed by the data**, propose what experiment would
   close it (without running it).
4. Update the rubric with the new statistical claims and remove "directional"
   caveats where appropriate.

This document is my honest commitment. If outcomes differ from predictions,
the rubric will be updated to reflect what we actually saw, not what we hoped.
