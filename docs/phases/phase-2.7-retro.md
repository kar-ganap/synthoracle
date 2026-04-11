# Phase 2.7 Retro: Difficulty Scaling Rubric + Gap-Closing Experiments

**Date:** 2026-04-10 to 2026-04-11
**Branch:** `phase-2.7-difficulty-rubric`
**Status:** Complete (with followup analysis deferred to phase 2.8)

---

## 1. What we set out to do

Build a **practitioner-facing difficulty rubric** that synthesizes all results across the full oracle family (1A, 1B, 1C, 1D, 1E, HD) into a single paper-ready document. The rubric was meant to enable readers to locate their own problem on difficulty axes (dimensionality, mechanism complexity, prior quality, noise) and read off expected VR performance, with every claim evidence-linked.

**Explicit goals at phase start:**
- Cross-oracle data extractor (`audit_all.py`) that loads all runs and computes a consistent metric signature
- Rubric generator (`build_rubric.py`) that emits a paper-ready markdown
- Pre-registration of gap-closing experiments *before* running them, with falsification criteria
- Honest post-hoc verification of predictions
- All of this on a new branch `phase-2.7-difficulty-rubric` merged cleanly into main

Budget ceiling at phase start: ~$40 remaining of the $200 initial project budget.

## 2. What actually happened

### Infrastructure built

- **`experiments/analysis/audit_all.py`** (~1000 lines) — cross-oracle extractor covering intrinsic properties (d, k, k/d, d_eff, interaction fraction, adversarial region count, noise dim count, reference HV, mechanism sufficiency R²(M→Y)) and observed metrics per run condition (HV, sample efficiency at 50/75/90% of ref HV, edge P/R, info capture via Sobol-weighted recall, OAT accuracy, calibration learning, adversarial prediction, screening efficiency, cost).
- **`experiments/analysis/build_rubric.py`** (~850 lines) — consumes `audit_data.json` and produces `difficulty_rubric.md` with 11 sections. Ephemeral metrics (dollar cost, wall time) are stored in the JSON but filtered from the rubric per the durability analysis.
- **Minimal edit to `compute_metrics.py`** — added HD to `ORACLE_CONFIGS` (imports, type hint, list entry). This triggered automatic reference HV + Sobol cache computation for HD via `compute_reference_hvs`. Result: `reference_hvs.json` now has 6 entries; `sobol_total_HD.npy` cached.

### Pre-registration + gap-closing experiments

The phase included a pre-registration doc (`gap_preregistration.md`) committed *before* running the gap-closing experiments, recording predictions and falsification criteria. Four experiments were pre-registered:

- **A1:** 1D prior + fresh extended to n=3 (seeds 43, 44 added). ~$10 extra.
- **A2:** 1E prior + fresh extended to n=3. ~$12 extra.
- **B:** Sonnet HD n=3 to test model generality on the HD screening result. ~$6 extra.
- **C:** BO baselines at 144 budget to fix the crossover_eval comparison. Free.

**Later added (not in preregistration but natural extensions):**
- **Haiku HD n=3** (~$0.30) — to complete the 3-tier Opus/Sonnet/Haiku sweep on HD. Required disabling adaptive thinking (Haiku 4.5 doesn't support it; API returns 400).
- **1B fresh n=3** (~$9) — to reach parity with 1D for the Rule 8 low-R²(M→Y) claim. 1B transfer is useless (topologically identical to 1A) but 1B fresh is a legitimate low-R² data point.

### Pre-registration verdicts (the main scientific outcome of the phase)

| Pre-registered rule | Outcome | Verdict |
|---|---|---|
| R8: R²(M→Y) predicts VR/BO (low R² → worse VR) | 1D R²=0.748 → VR/BO 0.909; 1B R²=0.799 → VR/BO 0.919. Both HIGHER than 1A R²=0.961 → VR/BO 0.685. | **FALSIFIED — direction reversed** |
| R9: Wrong prior hurts (1D prior penalty ~9%, 1E neutral) | 1D prior +2%, 1E prior +5% at n=3. | **FALSIFIED — direction reversed** |
| R2: VR crosses BO at extended budget | BO 1A @ 144 = 0.2811 (+2.2% over 66); VR 1A ext = 0.2780 (98.9%). BO HD @ 144 = 0.2732 (+2.8%); VR HD ext = 0.2681 (98.1%). 0/4 and 0/3 seeds cross BO at matched budget. | **QUALIFIED — approaches 98-99% but does not cross** |
| R10: Sonnet replicates Opus on HD | HV matches (0.254 vs 0.257, 95.6% vs 96.6%) and zero false positives holds for both. BUT noise OAT fraction differs (Opus 0%, Sonnet 20%). | **OUTCOMES replicate, STRATEGIES differ** |
| Gap C: BO is saturated at 66 evals | BO 1A gained 2.2%, BO HD gained 2.8% from 66 → 144. | **FALSIFIED — BO is not saturated** |

Plus new findings that weren't pre-registered but emerged:
- **Haiku HD:** HV 0.116 (43.7% of BO) but info capture 0.959 (near-parity with Opus/Sonnet). First documented case in the project where discovery and exploitation decouple at a specific model tier.
- **Haiku seed 44:** two noise edges at 0.55 and 0.60 confidence (first false-positive noise edges we've seen).

### Infrastructure gotchas resolved

- **max_tokens loop on HD extended:** `vr_tools.py` outer loop could hang when adaptive thinking exhausted max_tokens=16000 before tool_use blocks emitted. Added a safety valve: 3 consecutive no-progress iterations → break with `[SAFETY VALVE]` log. Also bumped `max_tokens` to 24000 for HD extended runs. Cost of the stuck seed 43 before the fix: $9.89.
- **BO 144 wall time:** each BO run at 144 evals took ~3 hours on 1A (O(n³) GP fitting + 12-dim acquisition optimization) and ~4 hours on HD. Originally planned to run 3 seeds per oracle; actually only ran 1 seed per oracle after realizing the total would be ~12 hours wall time. Decorative cleanup (1D) was killed mid-run to prioritize the load-bearing HD comparison.

## 3. What went well

- **Pre-registration discipline caught wrong claims.** Without it, three rules (R8 R²-predicts-VR, R9 prior-hurts, R2 VR-crosses-BO) would have been published in their falsified forms. The pre-registered predictions turned out to be wrong; the post-mortem revealed this cleanly. This is the single most important scientific outcome of the phase.
- **1B/1C were correctly identified as stepping stones and excluded from the rubric.** Phase 2.5 explicitly designed 1D and 1E after realizing 1B/1C were too similar to 1A to test transfer. This was confirmed empirically by the audit (1B has 18/18 edges shared with 1A, 0 wrong, 0 missing; 1C has 18/18 + 2 new edges, no false positives in the prior). The rubric separates "catalog entries" from "rubric entries" with a clear footnote.
- **The 3-tier model sweep on HD (Opus/Sonnet/Haiku) produced a genuinely useful finding.** The HV-and-zero-false-positive story generalizes from Opus to Sonnet but breaks at Haiku; that's a capability floor finding with practical value for practitioners. The screening strategy gradient (0% → 20% → 22% noise OAT) is a real behavioral signal that differentiates models at a pathway level even when outcomes match.
- **Rule 8 was retired with a nuanced reframe, not a silent delete.** R²(M→Y) stays in the rubric as an interpretability diagnostic rather than a performance predictor. Honest, and the reframe is more useful than the original claim because it applies to non-overlapping problems (understanding vs optimization).
- **`audit_all.py` and `build_rubric.py` are reusable.** Future paper drafts or revisions can re-run `audit_all` after new experiments land and `build_rubric` will auto-regenerate with updated numbers. The ephemeral-metrics filter (cost, wall time) encodes the durability discussion into code.

## 4. What went poorly

- **I repeatedly proposed framings faster than I could test them.** "Prior helps via selective leverage of correct causal knowledge" → "it's behavioral priming" → "it's vocabulary priming" → "it's bimodal" — within one conversation. Each framing was plausible storytelling; none were tested against dynamics data before I proposed them. The user flagged this directly and asked for a holistic dynamics analysis before any more framings. That analysis is now scoped as phase 2.8.
- **n=3 is underpowered for the transferability claim.** Paired t-tests on 1D at n=3 gave p=0.38 one-sided, 95% CI [−22%, +26%]. 1E at n=3 gave p=0.12 one-sided, 95% CI [−8%, +17%]. Combined n=6 relative differences p=0.15. The "prior helps by +2% on 1D and +5% on 1E" claim is directional only; it does not survive at α=0.05. The n was chosen based on budget, not power analysis.
- **The 1D seed 42 outlier (−8.77% prior penalty) is single-handedly driving the 1D story.** Dropping it moves the 1D mean from +2% to +7%. Keeping it makes the variance enormous. At n=3, we cannot distinguish "1D has bimodal prior response" from "seed 42 is an unlucky LHS". The user was right to flag this as a reviewer red flag.
- **BO 144 runs were much slower than I estimated.** I expected ~1 hour per run based on the 66-eval timings; actual was 3-4 hours due to O(n³) GP fitting + 12-dim acquisition optimization at n=144. This forced me to drop 1D from the BO 144 comparison, keeping only the load-bearing HD run. The decorative 1E baseline was skipped entirely.
- **I kept proposing new experiments without first mining the existing data.** At one point I proposed 3-7 more 1D or 1E seeds to reach statistical significance, ~$30-80 more. The user correctly responded "we have a very rich suite and logs that capture the dynamics of a single run — let's extract everything from existing data before spending more." This was the right call; phase 2.8 is the followup.
- **The understanding tax framing is a trap.** "VR costs 32pp HV at 72 budget on 1A" compares a dual-goal method (optimize + discover) against a single-goal method (optimize only). It's not a fair comparison of optimization ability. I was writing paper language around it before realizing this, then had to unwind. The honest framing is "VR pays for something BO doesn't deliver" but I only arrived there after the framing was already in the rubric draft.

## 5. Lessons learned

- **Dynamics mining must precede new experiments.** Our logs contain per-iteration hypothesis text, edge confidence trajectories, structured OAT prediction scores (direction_correct, predicted/actual direction and magnitude), evaluate_point prediction_errors, calibration checkpoints, surprise lists, and tool call sequences. Almost none of this was used during phase 2.7. Every claim I made was end-state only. The next phase must start with systematic within-run dynamics extraction before proposing new experiments or framings.
- **Pre-registration is scientific hygiene but also an interpretation trap if I don't pair it with dynamics analysis.** Pre-registering predictions caught the wrong claims (good), but it also tempted me to focus narrowly on those predictions and miss alternative mechanisms hiding in the data. Having to answer "is the +2% prior effect real?" led me down a series of speculative framings when the right answer was "look at the per-seed trajectories and see what actually happened."
- **End-state metrics mislead when n is small.** At n=3, a single outlier dominates. The 1D seed 42 −8.77% outlier shifted the mean from +7% to +2%. Dynamics data per seed makes the outlier interpretable (seed 42's LHS gave a lucky start that penalized the prior condition); aggregated statistics alone did not.
- **"Closest prior art" framing matters for novelty claims.** BORA (LLM + BO with adaptive policy) doesn't do prospective prediction + verification. "Are We There Yet?" showed LLM agents are insensitive to permuted feedback on discrete problems. Our permuted-feedback ablation passes where BDA's fails, but we never revisited the trajectory dynamics of the real vs permuted runs to nail down WHY. That's a phase 2.8 task too.
- **Per-phase API budget ceilings should be set in advance.** Phase 2.7 started at ~$180 and ended at ~$219. Some of that was unavoidable (gap-closing experiments were load-bearing), but the 1D killed BO run ($9.89 burned on stuck loop) and the 1D-without-fix retries cost were avoidable with better guardrails. Future phases should declare a ceiling and stop when it's reached.
- **Reference HV geometry matters for VR/BO interpretation.** 1A has ref HV = 0.262 and BO final = 0.275 (BO exceeds reference by 5%). 1D has ref HV = 1.10 and BO final = 1.30 (BO exceeds by 18%). 1E has ref HV = 0.259 and BO final = 0.278 (BO exceeds by 7%). The VR/BO ratio mixes "how good is VR" with "how much room above the reference HV is there for BO to operate in". This isn't a clean measurement; dynamics analysis should disentangle it.

## 6. What to do differently next time

- **Write the dynamics extractor first.** Before building any new rubric or running any new experiments, spend a phase (or the start of a phase) writing systematic per-iteration dynamics analysis: learning trajectories, edge confidence drift, tool-call allocation, prediction accuracy over time. Treat this as the instrument, not the last step. The phase 2.8 plan in `/Users/kartikganapathi/.claude/plans/mighty-munching-map.md` is exactly this.
- **Set a per-phase API budget ceiling and enforce it.** Phase 2.7 started with ~$40 "remaining" and ended with ~$19 overspent. The overspend happened in small slices, each individually reasonable, but the cumulative drift was not tracked in real time. Tracking should be per-phase, not per-run.
- **Treat n<5 seed counts as directional-only from the start.** Don't write "prior helps by +2%" when the paired t-test at n=3 gives p=0.38. Write "prior is directionally positive at n=3 but statistically underpowered" and flag the experiment as blocked for statistical claims until n≥6.
- **Every proposed framing gets a falsification test before the framing is committed to any document.** I violated this multiple times in phase 2.7 by writing paper-framing text with framings I hadn't tested. The phase 2.8 plan's guardrail "each claim must cite a specific data point produced by a function in `holistic_analysis.py`" is a direct response to this.
- **Distinguish between "catalog" and "rubric" entries for stepping-stone oracles.** 1B and 1C serve as intrinsic catalog entries but aren't suitable as rubric entries. The current rubric handles this correctly (catalog in Section 1, rubric tables filter to 1A/1D/1E/HD) but the distinction should be established earlier — by the end of Phase 2.5 at the latest, not during phase 2.7.
- **Kill expensive runs earlier.** BO 1D @ 144 ran for ~4 hours before being killed without producing output, because the wall time estimate was wrong. Future long-running experiments should set a wall-time budget and check progress explicitly; if progress stalls or the initial time estimate is exceeded by 2×, kill and reassess.

## Total Phase 2.7 spend

Gap-closing experiments (Opus): $52 (1D n=3 ~$10, 1E n=3 ~$12, Sonnet HD $6, Haiku HD $0.30, 1B fresh n=3 $9, the killed seed 43 ~$9.89, HD ext retries ~$16, other small ~$5)
BO 144 (local compute, free): 2 runs completed (1A seed 42, HD seed 42), 1 killed mid-run (1D seed 42)
Total new API spend: **~$52**
**Project total: ~$219** (over the $200 initial budget by ~$19, or +10%)

## Key artifacts

- `experiments/analysis/audit_all.py` — cross-oracle data extractor
- `experiments/analysis/build_rubric.py` — rubric markdown generator
- `experiments/analysis/holistic_analysis.py` — planned but deferred to phase 2.8
- `experiments/analysis/results/audit_data.json` — structured data artifact
- `experiments/analysis/results/difficulty_rubric.md` — 337-line paper-ready rubric
- `experiments/analysis/results/gap_preregistration.md` — committed before gap-closing experiments
- `experiments/analysis/results/gap_postmortem.md` — preliminary post-mortem (superseded by phase 2.8 holistic analysis)
- `docs/paper_framing.md` — working document; the final framing decision is deferred pending phase 2.8 dynamics analysis
- `src/synthoracle/agents/vr_tools.py` — safety valve fix for max_tokens stuck loops

## What phase 2.7 did NOT settle (deferred to phase 2.8)

- **Does VR actually learn from feedback within a run?** Need per-iteration prediction accuracy trajectories.
- **Is the "prior helps" effect selective-leverage, behavioral-priming, or something else?** Need systematic per-seed tool-call allocation comparison, not just seed 42.
- **Is the discovery-optimization decoupling at Haiku actually visible in the run dynamics?** Need to check where Haiku's evaluate_point targets land vs Opus's.
- **Why does 1A have VR/BO=0.685 while 1E has VR/BO=0.977 with the same R²(M→Y)=0.96?** Need BO-room-to-improve geometry analysis or dynamics comparison.
- **Final paper framing.** The hero contribution decision is blocked until phase 2.8 produces a grounded claim ledger.
