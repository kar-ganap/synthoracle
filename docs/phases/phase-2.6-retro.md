# Phase 2.6 Retro: High-Dimensional Oracle

**Date:** 2026-04-10
**Branch:** `phase-2.6-high-dim`
**Status:** Complete

---

## 1. What we set out to do

Address the practitioner-facing gap: *"My problem has 20-30 parameters, not 6. Does your result transfer?"*

Specifically, test whether the VR agent can **screen and dismiss irrelevant variables** in a higher-dimensional space, and whether the "understanding tax" observed on 1A (VR at 70% of BO at 72 evals) shrinks when the extra dimensions are noise.

The mechanism-bottleneck argument from the CBM literature predicts this directly: mechanism bottlenecks are most effective when `k << d` (few mechanisms, many parameters). 1A has `k/d ≈ 1.3` (unfavorable). HD with `d=12` has `k/d ≈ 0.67` (more favorable). This is a falsifiable prediction that VR should perform *relatively better* (vs BO) as the irrelevant-dimension count grows.

## 2. What actually happened

### Oracle design

**`MediumOracleHD`** — 12 inputs, 4 outputs.
- X1-X6: real inputs, identical to 1A (delegates to `MediumOracle(variant="1A")`)
- X7-X12: **pure zero-effect** noise dimensions (not near-zero — clean test)
- Ground truth DAG: 1A's 26 edges + 6 disconnected noise input nodes
- IO projection: same 18 edges as 1A
- Adversarial regions: 1A's + new `noise_extreme` region
- n_initial = 24 (2 × d) — realistic budget pressure; agent gets 48 evals for tools vs 60 on 1A

Implementation is ~90 lines wrapping `MediumOracle`. Tests cover dimensions, noise-zero-effect, match-1A-on-real-inputs, DAG structure, IO projection equivalence to 1A.

### Base results: HD at 72 budget (n=3 Opus seeds)

| Metric | Value |
|---|---|
| HV | **0.2569 ± 0.011** |
| BO HD (n=3) | 0.2658 ± 0.002 |
| **VR/BO** | **96.6%** |
| Precision | 1.000 (zero false positives) |
| Recall | 0.685 ± 0.146 |
| OAT sweeps on noise | **0 / 12** across all seeds |
| Tool-call evals on noise | **0 / 96 (0.0%)** |
| Cost | $9.74 for 3 seeds, $3.25/seed |

**Headline:** perfect screening (0% of tool budget spent on noise dimensions), zero false-positive edges, 96.6% of BO — vs 70% on 1A. The k/d prediction is validated: the advantage of the mechanism bottleneck grows when irrelevant dimensions are added.

### Extended results: HD at 144 budget (n=3 Opus seeds)

| Metric | 72 budget | **144 budget** | Δ |
|---|---|---|---|
| HV | 0.2569 ± 0.011 | **0.2681 ± 0.0015** | +4.4% |
| VR/BO | 96.6% | **100.9%** | **crosses BO** |
| Precision | 1.000 | 1.000 | — |
| **Recall** | 0.685 ± 0.146 | **0.944 ± 0.000** | +0.26, uniform |
| Missed edges | 13 unique | **X5→Y2 only** | collapsed to 1 edge |
| Calibration learning | 0/2 seeds | **3/3 seeds** | emerged |
| First → last MAE | 0.055 → 0.055 | 0.110 → 0.053 | −52% |
| Adversarial MAE (X2×X4) | 0.114 | **0.046** | −60% |
| Adversarial MAE (Z coupling) | 0.111 | **0.037** | −67% |
| Cost/seed | $3.25 | $7.58 | 2.3× |

**Headline:** the extended-budget crossover pattern holds on HD (matches 1A extended). Three qualitatively new signals emerged that weren't present at 72 budget:
1. **Recall becomes uniform at 0.944 across seeds** (σ collapses from 0.146 to 0.000). Every seed misses only X5→Y2 — the same hard edge that 9/10 seeds miss in 1A multi-seed.
2. **Calibration learning in 3/3 seeds** (not observable at 72 because only 1-2 checkpoints fire). MAE drops >20% in every seed.
3. **Adversarial prediction accuracy improves 60-67%** in the hard regions. The agent's mental model gets measurably better in the places it matters most.

### Screening strategy shifts with budget

- At 72 evals: agent does **0 noise OAT sweeps** (pure inference-based dismissal)
- At 144 evals: agent does **7 noise OAT sweeps** (~15% of tool budget) for **explicit verification**

Both strategies produce **zero false-positive noise edges**. The agent is rational about budget allocation: tight budget → infer noise is irrelevant and skip it; loose budget → verify systematically.

### Noise edge confidence

When listed in iteration summaries, noise edges get explicit low confidence:
- Seed 42 (72 budget): omitted noise edges entirely
- Seed 43 (72 budget): enumerated all 24 at **0.00** confidence
- Seed 44 (72 budget): omitted
- Seed 42 (144 budget): all 24 at **0.02**
- Seed 43 (144 budget): all 24 at **0.05**
- Seed 44 (144 budget): omitted

**Max noise-edge confidence across all 6 runs: 0.05.** The agent unambiguously dismisses X7-X12 regardless of strategy.

### Infrastructure incident: the max_tokens stuck loop

HD ext seed 43 (first attempt) hit an infinite loop at eval 127/144:
- Adaptive thinking + 12-dim tool contexts exhausted `max_tokens=16000`
- API responses were truncated before any `tool_use` blocks emitted
- Iteration summary parse failed repeatedly
- Outer loop `while eval_count < n_budget` kept starting new iterations that made no progress
- Burned ~$3.40 on the stuck tail before manual termination (total $9.89 for a run that reached HV=0.2691 at eval 127 but had no tool_calls/iteration_summaries in the final log)

**Root-cause fixes** (both in place):
1. `vr_tools.py`: **safety valve** — if 3 consecutive iterations add zero evals, break out with a `[SAFETY VALVE]` log. This is the general library improvement.
2. `run_hd_test.py`: **`max_tokens=24000`** for `vr-ext` mode, giving adaptive thinking + 12-dim contexts more headroom. This is the targeted config fix.

Rerun completed cleanly: seed 43 in 52 min at $5.70 (vs $9.89 killed attempt), seed 44 in 73 min at $10.27.

## 3. What went well

- **The k/d prediction was validated cleanly.** Perfect screening (0% noise tool-call fraction) on all 3 base-budget seeds is a stronger result than the plan anticipated. The paper can report this as a crisp, falsifiable prediction from CBM theory that held on the first attempt.
- **Extended budget reveals three new signals at once.** Recall becoming uniform, calibration learning emerging, and adversarial prediction error dropping 60-67% all happened at 144 evals. None were visible at 72. This suggests the extended-budget regime is where the VR methodology's distinctive properties become observable, which matters for how we frame the paper.
- **Oracle implementation was trivial** (~90 lines wrapping 1A). Tests passed on the first run. The screen-by-embedding approach is a good template for future dimensionality tests.
- **The analysis script (`analyze_hd.py`) is reusable and reads cleanly.** It reuses `_extract_edges_from_tool_calls` from `compute_metrics.py` for edge P/R, and implements HD-specific screening and noise-edge-confidence analysis.
- **Only one oracle variant was needed.** HD alone answers the dimensionality-scaling question; no HD1A/HD1B/HDv2 needed. One variant, clean result.

## 4. What went poorly

- **max_tokens=16000 was silently too low for extended HD runs.** The 16k limit was inherited from 1A extended, which worked because 1A has 6 inputs and therefore shorter tool contexts. On 12-dim HD the agent's adaptive thinking could fill 16k output tokens before emitting any tool calls. This was not caught in the 72-budget runs because contexts were shorter, and was not caught by calibration parse failures because those were treated as non-fatal.
- **No safety valve existed in `vr_tools.py`** before this phase. The outer `while eval_count < n_budget` loop had no way to detect stuck states. A run that burns tokens with zero progress would loop forever, only stopping on manual intervention or budget exhaustion. This is a general library gap that only surfaced on HD.
- **Calibration parse failures were known but tolerated.** At 72 budget on HD, seed 43 had one calibration parse failure (max_tokens hit on the structured parse). We noted it as "non-fatal" and moved on. In hindsight, this was an early warning that max_tokens was close to the limit — the same underlying issue that broke the ext run. Should have bumped max_tokens then.
- **$9.89 wasted on the killed seed.** Most of this (~$6.50) was productive up through eval 127, but ~$3.40 was pure waste on the stuck loop before termination. Budget impact: non-trivial given we're near the project ceiling.
- **Structured iteration summary parse is still fragile at extended budget.** Even in the successful rerun, the parse hit max_tokens on calibration at eval 120 on seed 44. The run continued normally (non-fatal), but the parse reliability isn't at 100%. This is a known issue carried from Phase 2.5.

## 5. Lessons learned

- **Context grows with dimensionality.** 16k output tokens is sufficient for 6-input extended runs but not for 12-input extended runs. For future higher-dimensional work, set max_tokens based on `d` (e.g., `16000 + 1000*d`), not a fixed default.
- **Safety valves are cheap insurance.** A 10-line loop-detector in vr_tools would have saved $3.40 and 30 minutes of wall time. Add safety valves proactively to any outer loop that depends on a counter advancing — if the counter can stall, add a stall detector.
- **"Non-fatal" parse failures are tail-end warning signals.** When calibration parse fails with `stop=max_tokens` on one seed, the fix isn't "keep going" — it's "the max_tokens budget is too tight, bump it now before it becomes fatal somewhere else." Treat max_tokens exhaustion as a capacity-planning signal, not a transient error.
- **Screening via inference vs verification are both rational strategies.** The agent's strategy shift (0 noise sweeps at 72 → 7 at 144) is not a bug or inconsistency — it's reasonable budget allocation. Tight budget: infer and skip. Loose budget: verify. Both produce zero false positives. This is the kind of behavior that emerges from the prompt + tool API and is worth highlighting in the paper.
- **The "understanding tax pays off" story is cleaner on HD than on 1A.** On 1A extended, the crossover is a quantitative shift (0.275 → 0.278, VR matches BO). On HD extended, three qualitatively new properties emerge (uniform recall, calibration learning, adversarial MAE drop). If the paper wants a single "what does extended budget buy you?" story, HD is the clearer example.
- **Variance-collapse-to-zero is a publication-worthy signal.** When recall goes from σ=0.146 to σ=0.000 at extended budget and every seed misses the same edge, that's not a noisy result getting better — it's the agent converging to the same rational behavior regardless of initial randomness. This is a distinctive VR property (a GP would not show this).

## 6. What to do differently next time

- **Set `max_tokens` based on `d` for future higher-d oracles.** Formula: `16000 + 1000*d` (so 28k for d=12, 36k for d=20).
- **Treat any max_tokens parse failure as a capacity signal**, not a transient. If one seed hits it, bump the limit before the next seed starts.
- **Add safety valves to outer loops proactively.** Any `while counter < target` loop where the counter advancement depends on external API behavior should have a no-progress detector.
- **Check checkpoint coverage.** The vr_tools checkpoint dir writes X, Y, hypervolumes but **not** tool_calls, iteration_summaries, or calibration_checks. For the killed seed 43, the optimization data was on disk but the diagnostic data was lost. For future runs, either write intermediate log JSONs or expand the checkpoint.
- **Run the single-seed sanity check first on any new budget configuration**, not just new oracles. The 144-budget sanity check on seed 42 worked perfectly, but we didn't catch that max_tokens=16000 was too tight because seed 42's specific trajectory didn't exhaust it. Running 2 sanity checks (different seeds) would have caught the issue before the full multi-seed run.

## Total Phase 2.6 spend

- HD 72 multi-seed: $9.74 (3 seeds)
- HD ext seed 42 (sanity check): $6.77
- HD ext seed 43 (killed, max_tokens loop): $9.89
- HD ext seeds 43, 44 rerun (with fix): $15.97 ($5.70 + $10.27)
- **Phase total: $42.37**

**Grand project total: ~$180** (of ~$200 initial budget).

## Key artifacts

- `src/synthoracle/oracles/medium_hd.py` — HD oracle (~90 lines)
- `tests/test_medium_hd.py` — 15 tests
- `experiments/vr_agent/run_hd_test.py` — `bo`, `vr`, `vr-ext` modes
- `experiments/analysis/analyze_hd.py` — standalone HD diagnostic suite
- `experiments/analysis/results/hd_hv_envelope.png`, `hd_ext_hv_envelope.png` — HV plots
- `src/synthoracle/agents/vr_tools.py` — safety valve (general library improvement)
- Run data: `hd_vr_seed{42-44}.npz/.json`, `hd_vr_ext_seed{42-44}.npz/.json` (6 runs)
