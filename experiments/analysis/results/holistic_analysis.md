# Holistic Within-Run Dynamics Analysis

Systematic mining of per-iteration data from every run in the dataset. 
Every claim in this document cites a specific data point produced by 
`experiments/analysis/holistic_analysis.py`. No new experiments.

**Data sources:**
- `iteration_summaries` (hypothesis, edges with confidence, surprises, findings, next_plan)
- `tool_calls` (grouped by iteration via cumulative cost matching against iteration eval_counts)
- `calibration_checks` (per-output errors at specific eval counts)
- HV trajectories (from `npz['hypervolumes']`)

**Not analyzed:** per-LLM-call thinking budgets (only aggregate tokens), 
ablation runs `ablation_real/permuted_seed42.npz` (no iteration_summaries — 
only HV + X/Y trajectories).

---

## Section A — Per-run learning trajectories

**Question:** Does per-iteration prediction accuracy actually improve during a run? If the agent is using feedback, we expect OAT direction accuracy to trend upward and surprise counts to trend downward across iterations. If these are flat or noisy, the "VR learns from feedback" claim is weakened.

**Method:** For each run, group tool calls by iteration using cumulative cost matching against `iteration_summaries[i].eval_count`, then compute (a) mean `direction_correct` across all OAT outputs in the iteration, (b) mean `magnitude_error`, (c) mean |`prediction_errors[Yj]`| across `evaluate_point` calls, (d) surprise count from `iteration_summaries[i].surprises`.

### A.1 Learning signal summary per condition

Per-condition summary. Note: agents typically **front-load OAT sweeps in iteration 0** and switch to exploitation afterward, so many iterations have `None` for OAT accuracy. All "first/last" comparisons use first-non-None and last-non-None values. Conditions with `n iters = 0` are old Phase 2.3 runs without structured iteration summaries (1B/1C transfer_72, 1C fresh_72).

| Condition | n seeds | n iters (median) | OAT dir acc (fnn→lnn) | eval_point MAE (fnn→lnn) | Surprise (first→last) | seeds ↑ eval_MAE↓ | seeds ↓ surprise |
|---|---|---|---|---|---|---|---|
| 1A/multi_seed_72 | 10 | 3 | 0.71 → 0.73 | 0.043 → 0.042 | 4.7 → 3.3 | 2/3 | 8/10 |
| 1A/extended_144 | 4 | 7 | 0.71 → 0.94 | 0.034 → 0.004 | 4.8 → 1.8 | 4/4 | 4/4 |
| 1B/transfer_72 | 1 | 0 | n/a | n/a | n/a | — | 0/1 |
| 1B/fresh_72 | 3 | 3 | 0.60 → 0.60 | 0.021 → 0.013 | 5.0 → 3.5 | 1/1 | 2/3 |
| 1C/transfer_72 | 1 | 0 | n/a | n/a | n/a | — | 0/1 |
| 1C/fresh_72 | 1 | 0 | n/a | n/a | n/a | — | 0/1 |
| 1D/prior_72 | 3 | 3 | 0.78 → 0.78 | 0.035 → 0.019 | 4.7 → 2.3 | 1/2 | 3/3 |
| 1D/fresh_72 | 3 | 3 | 0.69 → 1.00 | 0.040 → 0.032 | 5.0 → 3.0 | 1/1 | 3/3 |
| 1D/prior_144 | 1 | 14 | 0.88 → 0.88 | 0.048 → 0.000 | 4.0 → 1.0 | 1/1 | 1/1 |
| 1E/prior_72 | 3 | 5 | 0.68 → 0.68 | 0.034 → 0.009 | 5.3 → 3.0 | 2/3 | 3/3 |
| 1E/fresh_72 | 3 | 3 | 0.67 → 0.81 | 0.029 → 0.014 | 4.7 → 3.7 | 2/2 | 3/3 |
| 1E/sonnet_prior_72 | 1 | 9 | 0.54 → 0.54 | 0.110 → 0.025 | 5.0 → 4.0 | 1/1 | 1/1 |
| HD/base_72 | 3 | 3 | 0.93 → 0.89 | 0.045 → 0.013 | 3.3 → 2.0 | 2/2 | 3/3 |
| HD/extended_144 | 3 | 5 | 0.72 → 0.81 | 0.063 → 0.003 | 5.3 → 1.3 | 3/3 | 3/3 |
| HD/sonnet_72 | 3 | 2 | 0.85 → 0.85 | 0.096 → 0.082 | 4.0 → 4.7 | 1/1 | 1/3 |
| HD/haiku_72 | 3 | 1 | 0.66 → 0.66 | n/a | 4.7 → 4.3 | — | 1/3 |

### A.2 Per-seed learning trajectories (plots)

4-panel trajectories for every seed of every condition. Panels: (a) OAT direction accuracy, (b) OAT magnitude MAE, (c) evaluate_point MAE, (d) surprise count. Saved to `holistic_plots/`.

_Generated 40 per-seed trajectory plots._

### A.4 Does the learning loop connect to exploitation?

The key confound: predictions improve (eval_point MAE drops) but Section B showed within-condition prediction accuracy doesn't correlate with final HV. Is the learning loop disconnected from exploitation, or does it help in a way Section B couldn't detect?

#### Test 1: HV gain rate per iteration

If improved predictions feed into better exploitation, HV gain per eval should be **higher in later iterations** (where the model is better). Computed as (HV_end − HV_start) / n_evals for each iteration.

Note: diminishing returns near the Pareto front can also cause declining HV gain rate, so a decreasing rate is ambiguous. An INCREASING rate is strong evidence of exploitation improving.

| Condition | Seed | Iter 0 gain/eval | Last iter gain/eval | Trend | Notes |
|---|---|---|---|---|---|
| 1A/multi_seed_72 | 42 | 0.00250 | 0.00640 | ↑ INCREASING | max at iter 2 (0.00640), 3 iters |
| 1A/multi_seed_72 | 43 | 0.00250 | 0.01851 | ↑ INCREASING | max at iter 2 (0.01851), 3 iters |
| 1A/multi_seed_72 | 44 | 0.00322 | 0.01026 | ↑ INCREASING | max at iter 2 (0.01026), 3 iters |
| 1A/multi_seed_72 | 45 | 0.00239 | 0.01356 | ↑ INCREASING | max at iter 2 (0.01356), 3 iters |
| 1A/multi_seed_72 | 46 | 0.00370 | 0.00546 | → flat | max at iter 2 (0.00546), 3 iters |
| 1A/multi_seed_72 | 48 | 0.00222 | 0.03259 | ↑ INCREASING | max at iter 2 (0.03259), 3 iters |
| 1A/multi_seed_72 | 49 | 0.00239 | 0.01467 | ↑ INCREASING | max at iter 2 (0.01467), 3 iters |
| 1A/multi_seed_72 | 50 | 0.00269 | 0.01570 | ↑ INCREASING | max at iter 2 (0.01570), 3 iters |
| 1A/multi_seed_72 | 51 | 0.00250 | 0.00964 | ↑ INCREASING | max at iter 2 (0.00964), 3 iters |
| 1A/extended_144 | 42 | 0.00250 | 0.00012 | ↓ decreasing | max at iter 1 (0.00438), 6 iters |
| 1A/extended_144 | 43 | 0.00250 | 0.00020 | ↓ decreasing | max at iter 1 (0.00499), 6 iters |
| 1A/extended_144 | 44 | 0.00322 | 0.00010 | ↓ decreasing | max at iter 1 (0.00363), 8 iters |
| 1A/extended_144 | 45 | 0.00239 | 0.00002 | ↓ decreasing | max at iter 1 (0.01039), 6 iters |
| 1B/fresh_72 | 43 | 0.00249 | 0.00089 | ↓ decreasing | max at iter 1 (0.00467), 3 iters |
| 1B/fresh_72 | 44 | 0.00308 | 0.00569 | ↑ INCREASING | max at iter 2 (0.00569), 3 iters |
| 1D/prior_72 | 42 | 0.00938 | 0.02627 | ↑ INCREASING | max at iter 2 (0.02627), 3 iters |
| 1D/prior_72 | 43 | 0.00639 | 0.00887 | → flat | max at iter 1 (0.07495), 3 iters |
| 1D/prior_72 | 44 | 0.01954 | 0.01274 | → flat | max at iter 0 (0.01954), 3 iters |
| 1D/fresh_72 | 43 | 0.00867 | 0.00694 | → flat | max at iter 1 (0.02952), 3 iters |
| 1D/fresh_72 | 44 | 0.00915 | 0.00517 | → flat | max at iter 1 (0.05809), 3 iters |
| 1D/prior_144 | 42 | 0.00898 | 0.00019 | ↓ decreasing | max at iter 1 (0.04030), 8 iters |
| 1E/prior_72 | 42 | 0.00290 | 0.00064 | ↓ decreasing | max at iter 1 (0.00828), 3 iters |
| 1E/prior_72 | 43 | 0.00249 | 0.00041 | ↓ decreasing | max at iter 1 (0.01750), 4 iters |
| 1E/prior_72 | 44 | 0.00221 | 0.00213 | → flat | max at iter 1 (0.01137), 4 iters |
| 1E/fresh_72 | 42 | 0.00290 | 0.00051 | ↓ decreasing | max at iter 1 (0.00830), 3 iters |
| 1E/fresh_72 | 44 | 0.00221 | 0.00057 | ↓ decreasing | max at iter 1 (0.00719), 3 iters |
| 1E/sonnet_prior_72 | 42 | 0.00345 | 0.00011 | ↓ decreasing | max at iter 4 (0.00809), 5 iters |
| HD/base_72 | 43 | 0.00317 | 0.00255 | → flat | max at iter 1 (0.00385), 3 iters |
| HD/base_72 | 44 | 0.00198 | 0.00074 | ↓ decreasing | max at iter 1 (0.01248), 3 iters |
| HD/extended_144 | 42 | 0.00108 | 0.00029 | ↓ decreasing | max at iter 1 (0.00585), 5 iters |
| HD/extended_144 | 43 | 0.00187 | 0.00006 | ↓ decreasing | max at iter 1 (0.00413), 4 iters |
| HD/extended_144 | 44 | 0.00165 | 0.00023 | ↓ decreasing | max at iter 1 (0.00460), 7 iters |
| HD/sonnet_72 | 42 | 0.00147 | 0.01601 | ↑ INCREASING | max at iter 3 (0.01601), 3 iters |

#### Test 2: Does prediction improvement precede HV breakthroughs?

For extended-budget runs, identify (a) the iteration where eval_point MAE first drops below 50% of its initial value ('prediction clicks'), and (b) the iteration with the largest single-iteration HV gain ('exploitation breakthrough'). If (a) precedes or coincides with (b), the learning loop connects to exploitation.

| Condition | Seed | Pred clicks (iter) | HV breakthrough (iter) | Clicks before breakthrough? |
|---|---|---|---|---|
| 1A/multi_seed_72 | 42 | n/a | 0 | n/a — MAE never dropped 50% |
| 1A/multi_seed_72 | 43 | n/a | 0 | n/a — MAE never dropped 50% |
| 1A/multi_seed_72 | 44 | n/a | 0 | n/a — MAE never dropped 50% |
| 1A/multi_seed_72 | 45 | n/a | 0 | n/a — MAE never dropped 50% |
| 1A/multi_seed_72 | 46 | n/a | 0 | n/a — MAE never dropped 50% |
| 1A/multi_seed_72 | 48 | n/a | 0 | n/a — MAE never dropped 50% |
| 1A/multi_seed_72 | 49 | n/a | 0 | n/a — MAE never dropped 50% |
| 1A/multi_seed_72 | 50 | n/a | 0 | n/a — MAE never dropped 50% |
| 1A/multi_seed_72 | 51 | n/a | 0 | n/a — MAE never dropped 50% |
| 1A/extended_144 | 42 | 5 | 1 | NO — breakthrough first |
| 1A/extended_144 | 43 | 3 | 1 | NO — breakthrough first |
| 1A/extended_144 | 44 | 2 | 0 | NO — breakthrough first |
| 1A/extended_144 | 45 | n/a | 1 | n/a — MAE never dropped 50% |
| 1B/fresh_72 | 43 | 2 | 1 | NO — breakthrough first |
| 1B/fresh_72 | 44 | n/a | 0 | n/a — MAE never dropped 50% |
| 1D/prior_72 | 42 | n/a | 1 | n/a — MAE never dropped 50% |
| 1D/prior_72 | 43 | 2 | 1 | NO — breakthrough first |
| 1D/prior_72 | 44 | n/a | 0 | n/a — MAE never dropped 50% |
| 1D/fresh_72 | 43 | 2 | 1 | NO — breakthrough first |
| 1D/fresh_72 | 44 | n/a | 1 | n/a — MAE never dropped 50% |
| 1D/prior_144 | 42 | 3 | 1 | NO — breakthrough first |
| 1E/prior_72 | 42 | 2 | 0 | NO — breakthrough first |
| 1E/prior_72 | 43 | 4 | 1 | NO — breakthrough first |
| 1E/prior_72 | 44 | 2 | 1 | NO — breakthrough first |
| 1E/fresh_72 | 42 | 2 | 0 | NO — breakthrough first |
| 1E/fresh_72 | 44 | 2 | 1 | NO — breakthrough first |
| 1E/sonnet_prior_72 | 42 | 4 | 0 | NO — breakthrough first |
| HD/base_72 | 43 | 2 | 0 | NO — breakthrough first |
| HD/base_72 | 44 | 2 | 1 | NO — breakthrough first |
| HD/extended_144 | 42 | 2 | 1 | NO — breakthrough first |
| HD/extended_144 | 43 | 2 | 0 | NO — breakthrough first |
| HD/extended_144 | 44 | 2 | 1 | NO — breakthrough first |
| HD/sonnet_72 | 42 | n/a | 3 | n/a — MAE never dropped 50% |

#### Test 3: Does the articulated model guide exploitation targets?

If the rough model from iter 0 OAT sweeps guides iter 1+ exploitation, then inputs identified as high-effect by OAT should be varied MORE in evaluate_point targets. Compute per-input: (a) OAT importance = max `actual_magnitude` across outputs from iter 0 sweeps, (b) exploitation emphasis = range of that input across iter 1+ evaluate_point X targets. Spearman correlation between (a) and (b) should be positive if model guides exploitation.

| Condition | Seed | Spearman ρ (OAT importance vs exploit emphasis) | p-value | n inputs | Interpretation |
|---|---|---|---|---|---|
| 1D/prior_72 | 42 | +0.754 | 0.084 | 6 | model guides exploitation |
| 1D/prior_72 | 43 | -0.123 | 0.816 | 6 | inverse — exploits LOW-effect inputs? |
| 1D/prior_72 | 44 | -0.123 | 0.816 | 6 | inverse — exploits LOW-effect inputs? |
| 1D/fresh_72 | 42 | -0.030 | 0.954 | 6 | no clear connection |
| 1D/fresh_72 | 43 | +0.541 | 0.268 | 6 | no clear connection |
| 1D/fresh_72 | 44 | -0.123 | 0.816 | 6 | inverse — exploits LOW-effect inputs? |
| 1D/prior_144 | 42 | -0.123 | 0.816 | 6 | inverse — exploits LOW-effect inputs? |
| 1E/prior_72 | 42 | -0.370 | 0.470 | 6 | inverse — exploits LOW-effect inputs? |
| 1E/prior_72 | 43 | -0.293 | 0.573 | 6 | inverse — exploits LOW-effect inputs? |
| 1E/prior_72 | 44 | -0.370 | 0.470 | 6 | inverse — exploits LOW-effect inputs? |
| 1E/fresh_72 | 42 | -0.370 | 0.470 | 6 | inverse — exploits LOW-effect inputs? |
| 1E/fresh_72 | 43 | -0.293 | 0.573 | 6 | inverse — exploits LOW-effect inputs? |
| 1E/fresh_72 | 44 | -0.213 | 0.686 | 6 | inverse — exploits LOW-effect inputs? |
| 1E/sonnet_prior_72 | 42 | -0.293 | 0.573 | 6 | inverse — exploits LOW-effect inputs? |

**Summary:** 1/14 seeds show significant positive correlation (ρ > 0.3, p < 0.1) between OAT-discovered importance and exploitation emphasis. 11/14 show inverse pattern.

#### Test 4: Are exploitation targets better than random (LHS)?

Compare the mean output values of iter 1+ evaluate_point targets to the mean output values of initial LHS points, on the maximize objectives (Y1 and Y4). If evaluate_point targets are systematically better, the agent is directing exploitation to promising regions (regardless of whether it's model-guided or just following gradients).

| Condition | Seed | LHS mean(Y1) | Exploit mean(Y1) | LHS mean(Y4) | Exploit mean(Y4) | Y1 better? | Y4 better? |
|---|---|---|---|---|---|---|---|
| 1D/prior_72 | 42 | 0.781 | 1.379 | 0.237 | 0.496 | YES | YES |
| 1D/prior_72 | 43 | 0.653 | 1.852 | 0.178 | 0.521 | YES | YES |
| 1D/prior_72 | 44 | 0.797 | 1.744 | 0.238 | 0.533 | YES | YES |
| 1D/fresh_72 | 42 | 0.781 | 1.753 | 0.237 | 0.509 | YES | YES |
| 1D/fresh_72 | 43 | 0.653 | 1.599 | 0.178 | 0.541 | YES | YES |
| 1D/fresh_72 | 44 | 0.797 | 1.741 | 0.238 | 0.526 | YES | YES |
| 1D/prior_144 | 42 | 0.781 | 1.879 | 0.237 | 0.540 | YES | YES |
| 1E/prior_72 | 42 | 0.146 | 0.595 | 0.093 | 0.230 | YES | YES |
| 1E/prior_72 | 43 | 0.060 | 0.608 | 0.067 | 0.213 | YES | YES |
| 1E/prior_72 | 44 | 0.179 | 0.673 | 0.102 | 0.229 | YES | YES |
| 1E/fresh_72 | 42 | 0.146 | 0.487 | 0.093 | 0.205 | YES | YES |
| 1E/fresh_72 | 43 | 0.060 | 0.465 | 0.067 | 0.196 | YES | YES |
| 1E/fresh_72 | 44 | 0.179 | 0.504 | 0.102 | 0.227 | YES | YES |
| 1E/sonnet_prior_72 | 42 | 0.146 | 0.660 | 0.093 | 0.224 | YES | YES |

**Summary:** 14/14 seeds have exploitation targets with higher mean Y1 than LHS. 14/14 for Y4. (Both are maximize objectives — higher = better.)

**Summary:** 0/18 seeds show prediction improvement preceding or coinciding with the HV breakthrough. 18/18 show the breakthrough happening before prediction clicks.

### A.5 Falsification verdicts (updated with A.4)

#### Test 5: Is the agent surprised when exploring and accurate when exploiting?

If VR's articulation-prediction-reconciliation loop works, we expect: (a) exploration tool calls (OAT sweeps) should have HIGH surprise rate (probing new territory), (b) exploitation tool calls (evaluate_point) should have LOW surprise rate that DECREASES over iterations (model improving). 'Surprise' for evaluate_point = max|prediction_error| > 0.05. 'Surprise' for OAT = any output with direction_correct == False.

| Condition | Seed | OAT surprise rate (iter 0) | eval_point surprise rate (iter 1) | eval_point surprise rate (last iter) | Exploit surprise drops? |
|---|---|---|---|---|---|
| 1D/prior_72 | 42 | 67% | 75% | 75% | n/a (≤1 iter with data) |
| 1D/prior_72 | 43 | 83% | 82% | 0% | YES |
| 1D/prior_72 | 44 | 71% | 0% | 20% | no |
| 1D/fresh_72 | 42 | 67% | 67% | 67% | n/a (≤1 iter with data) |
| 1D/fresh_72 | 43 | 50% | 75% | 0% | YES |
| 1D/fresh_72 | 44 | 83% | 100% | 100% | n/a (≤1 iter with data) |
| 1D/prior_144 | 42 | 50% | 50% | 0% | YES |
| 1E/prior_72 | 42 | 67% | 60% | 0% | YES |
| 1E/prior_72 | 43 | 83% | 62% | 0% | YES |
| 1E/prior_72 | 44 | 83% | 0% | 36% | no |
| 1E/fresh_72 | 42 | 100% | 53% | 0% | YES |
| 1E/fresh_72 | 43 | 67% | 73% | 73% | n/a (≤1 iter with data) |
| 1E/fresh_72 | 44 | 100% | 14% | 0% | YES |
| 1E/sonnet_prior_72 | 42 | 83% | 100% | 70% | YES |

**OAT surprise rate in iter 0 (exploration):** mean = 75%, range [50%, 100%] across 14 seeds. HIGH as expected — agent is discovering new information.

**Exploitation surprise rate drops over iterations:** 8/10 seeds (80%) show evaluate_point surprise rate lower in last exploitation iter than first. PASS — agent becomes less surprised as it exploits.

**Exploration vs exploitation surprise:** OAT surprise rate (iter 0) = 75%. evaluate_point surprise rate (first exploitation iter) = 58%. Exploration IS more surprising than exploitation, as expected.

- **VR learns from feedback (evaluate_point MAE ↓, primary signal)**: **PASS** — 21/24 seeds (88%) show evaluate_point MAE lower in last iteration with data than first. Additionally, 17/24 seeds show a >50% MAE reduction (first iter to last iter).
- **OAT direction accuracy improves over iterations (secondary; OATs are front-loaded)**: **PASS** — 10/13 seeds (77%) show OAT direction accuracy higher in last non-None iter than first. Note: many seeds only have 1 iteration with OAT sweeps, so this metric is noisier than eval_point MAE.
- **Surprise rate decreases over iterations**: **PASS** — 36/40 seeds (90%) show surprise count lower in last iteration than first
- **Prediction improvement precedes HV breakthroughs (learning → exploitation coupling)**: **FALSIFIED** — Only 0/18 seeds (0%) show coupling — breakthroughs happen independent of prediction improvement.
- **OAT-discovered input importance guides exploitation targets (model → exploitation causal link)**: **FALSIFIED** — Only 1/14 seeds (7%) show the expected correlation.
- **Exploitation targets are better than random (LHS) on maximize objectives**: **PASS** — 14/14 seeds (100%) have exploitation targets with higher mean Y1 or Y4 than LHS. 14/14 are better on BOTH.
- **Exploration (OAT) is more surprising than exploitation (evaluate_point)**: **PASS** — OAT surprise rate (iter 0) = 75% vs evaluate_point surprise rate (first exploit iter) = 58%. Agent encounters more surprises when probing new territory than when using its model, consistent with the model providing useful guidance.
- **Exploitation surprise rate decreases over iterations (model improves during exploitation)**: **PASS** — 8/10 seeds (80%) show exploitation surprise decreasing.

## Section B — Cross-seed dynamics correlations

**Question:** Do per-seed dynamics metrics correlate with per-seed final HV within the same condition? If calibration learning predicts better HV, that's evidence the confidence-tracking mechanism works. If final recall correlates with final HV, that's evidence discovery helps exploitation (or at least that they track together).

**Caveat:** With n=3 per condition, correlations are extremely noisy. n=10 (1A multi_seed) is the only condition with real statistical power. For n=3 conditions, individual r values should be treated as directional only. We pool across conditions where possible for more robust signals.

### B.1 Per-condition summary

| Condition | n | r(HV, late_eval_MAE) | r(HV, final_recall) | r(HV, surprise_decrease) | r(HV, n_iters) |
|---|---|---|---|---|---|
| 1A/multi_seed_72 | 10 | -0.27 (p=0.48) | +0.09 (p=0.80) | -0.39 (p=0.26) | +0.26 (p=0.46) |
| 1A/extended_144 | 4 | -0.51 (p=0.49) | -0.58 (p=0.42) | +0.62 (p=0.38) | -0.27 (p=0.73) |
| 1B/fresh_72 | 2 | n/a | n/a | n/a | n/a |
| 1D/prior_72 | 3 | -0.96 (p=0.17) | const | -0.72 (p=0.49) | const |
| 1D/fresh_72 | 3 | +1.00 (p=0.06) | +0.69 (p=0.52) | -0.23 (p=0.85) | +0.29 (p=0.81) |
| 1E/prior_72 | 3 | +0.34 (p=0.78) | const | -0.92 (p=0.26) | -0.80 (p=0.40) |
| 1E/fresh_72 | 3 | -0.29 (p=0.81) | const | const | +0.32 (p=0.79) |
| HD/base_72 | 3 | +0.41 (p=0.73) | -0.35 (p=0.77) | +0.64 (p=0.56) | -0.35 (p=0.77) |
| HD/extended_144 | 3 | +0.17 (p=0.89) | -0.10 (p=0.94) | +0.41 (p=0.73) | -0.15 (p=0.91) |
| HD/sonnet_72 | 3 | -0.99 (p=0.08) | const | -0.94 (p=0.21) | -1.00 (p=0.00) |
| HD/haiku_72 | 3 | n/a | +0.32 (p=0.80) | -0.66 (p=0.54) | -0.66 (p=0.54) |

### B.2 Pooled cross-condition analysis

Pool all seeds from n>=3 conditions. **Warning:** pooling across conditions mixes different oracles and budgets, so correlation structure may be driven by between-condition differences rather than within-condition variation. Report for completeness but interpret cautiously.

```
  final_hv vs late_eval_MAE (n=36): Pearson r=-0.038 (p=0.826), Spearman ρ=-0.413 (p=0.012)
  final_hv vs final_recall (n=40): Pearson r=+0.375 (p=0.017), Spearman ρ=+0.445 (p=0.004)
  final_hv vs surprise_decrease (n=40): Pearson r=+0.152 (p=0.350), Spearman ρ=+0.387 (p=0.014)
  final_hv vs n_iterations (n=40): Pearson r=-0.085 (p=0.603), Spearman ρ=+0.415 (p=0.008)
```

**Expected signs:** HV vs late_eval_MAE should be **negative** (better predictions → higher HV). HV vs final_recall should be **positive** (more discovered edges → better HV). HV vs surprise_decrease should be **positive** (more learning → higher HV). HV vs n_iters is ambiguous (more iters could mean more learning OR more failed attempts).

### B.3 Falsification verdicts

- **Better prediction accuracy predicts higher HV**: **INCONCLUSIVE** — Pooled Pearson r=-0.038, p=0.826, n=36. Weak or near-zero correlation — may be driven by within-condition noise or between-condition mixing.
- **Discovery (final recall) predicts exploitation (final HV)**: **PASS** — Pooled Pearson r=0.375, p=0.017, n=40.

## Section C — Prior vs fresh: systematic, not just seed 42

**Question:** Is the 'prior helps' effect (1D +2%, 1E +5% at n=3) due to (a) selective leverage of correct causal claims in the prior, (b) behavioral priming that changes tool-call allocation, or (c) something else? The seed 42 spot-check (Section A of the phase 2.7 analysis) showed prior → 12 local_gradients vs fresh → 3. Is this consistent across seeds 43 and 44?

### C.1D — 1D prior vs fresh (n=3 each)

#### C.1D.1 Tool call type distribution

Per-seed tool call counts for prior vs fresh. The seed 42 finding was: 1D prior had 12 `local_gradients` vs fresh's 3. Is this consistent?

| Tool type | P s42 | P s43 | P s44 | F s42 | F s43 | F s44 | P mean | F mean |
|---|---|---|---|---|---|---|---|---|
| **correlation_matrix** | 2 | 1 | 1 | 2 | 2 | 2 | **1.3** | **2.0** |
| current_pareto_front | 4 | 4 | 5 | 3 | 6 | 5 | 4.3 | 4.7 |
| **evaluate_point** | 4 | 13 | 21 | 3 | 8 | 7 | **12.7** | **6.0** |
| **interaction_test** | 2 | 1 | 0 | 2 | 2 | 1 | **1.0** | **1.7** |
| **local_gradients** | 12 | 3 | 9 | 3 | 4 | 2 | **8.0** | **3.0** |
| oat_sweep | 6 | 6 | 7 | 8 | 7 | 7 | 6.3 | 7.3 |
| **regression_fit** | 4 | 7 | 0 | 3 | 8 | 8 | **3.7** | **6.3** |
| **sensitivity_report** | 2 | 1 | 3 | 2 | 4 | 3 | **2.0** | **3.0** |

#### C.1D.2 First OAT sweep targets (input ordering)

The order in which the agent probes inputs via OAT sweeps reveals its exploration strategy. Does prior change which inputs are probed first?

- **prior seed 42**: X1 → X2 → X3 → X4 → X5 → X6
- **prior seed 43**: X1 → X2 → X3 → X4 → X5 → X6
- **prior seed 44**: X1 → X2 → X3 → X4 → X5 → X6 → X3
- **fresh seed 42**: X5 → X3 → X4 → X6 → X1 → X2 → X5 → X6
- **fresh seed 43**: X6 → X3 → X2 → X4 → X5 → X1 → X5
- **fresh seed 44**: X1 → X2 → X3 → X4 → X5 → X6 → X5

#### C.1D.3 Hypothesis text at iteration 0 (prior references)

Does the prior-equipped agent's initial hypothesis explicitly reference the prior? Searching for keywords: 'prior', 'related system', 'previous', 'from the'.

- **prior seed 42**: mentions: ['prior']. Snippet: "The system has 6 inputs all positively driving Y1. Y2 is reduced by X1-X4 (beneficial) but increased by X5 and X6 (harmful trade-off with Y1). Y3 depe..."
- **prior seed 43**: no prior keywords. Snippet: "The system has a mostly additive structure for Y1, Y2, and Y4, with the critical exception of Y3 which depends ONLY on X1 and X3 with a strong nonline..."
- **prior seed 44**: mentions: ['prior', 'from the']. Snippet: "The system has modular structure with key differences from the prior. Y1 is driven by ALL 6 inputs positively (with X6 sign-dependent on regime), domi..."
- **fresh seed 42**: no prior keywords. Snippet: "The system has a mostly additive structure with one key interaction. ALL six inputs increase Y1 (with varying strengths). Y2 is decreased by X1,X2,X3,..."
- **fresh seed 43**: no prior keywords. Snippet: "The system has a largely additive structure with the following causal relationships:  **Y1** (maximize): Driven by ALL 6 inputs with positive, approxi..."
- **fresh seed 44**: no prior keywords. Snippet: "Y1 is driven by all 6 inputs additively (all positive), with X5 and X1 having the strongest effects. Y2 is driven by all 6 inputs additively: X4, X3, ..."

#### C.1D.4 Edge confidence initialization: GT-aligned vs variant-specific

Does the prior-equipped agent start with higher confidence on edges that are correct (shared with 1A GT) and lower confidence on edges that are wrong (1A-specific, not in this variant's GT)? This is the direct test of 'screen-first selectively leverages correct priors.'

Edge classification for 1D:
- Shared with 1A (prior should help): 18
- Wrong from 1A (prior should dismiss): 0
- Missing from 1A (prior doesn't know): 1

| Condition | Seed | Mean conf (shared) | Mean conf (wrong from 1A) | Mean conf (missing from 1A) |
|---|---|---|---|---|
| prior | 42 | 0.91 (n=18) | — | 0.90 (n=1) |
| prior | 43 | 0.91 (n=18) | — | 0.90 (n=1) |
| prior | 44 | 0.90 (n=18) | — | 0.90 (n=1) |
| fresh | 42 | 0.89 (n=18) | — | 0.85 (n=1) |
| fresh | 43 | 0.84 (n=18) | — | 0.85 (n=1) |
| fresh | 44 | 0.85 (n=18) | — | 0.82 (n=1) |

### C.1E — 1E prior vs fresh (n=3 each)

#### C.1E.1 Tool call type distribution

Per-seed tool call counts for prior vs fresh. The seed 42 finding was: 1D prior had 12 `local_gradients` vs fresh's 3. Is this consistent?

| Tool type | P s42 | P s43 | P s44 | F s42 | F s43 | F s44 | P mean | F mean |
|---|---|---|---|---|---|---|---|---|
| correlation_matrix | 2 | 3 | 3 | 2 | 3 | 2 | 2.7 | 2.3 |
| current_pareto_front | 4 | 7 | 7 | 5 | 2 | 8 | 6.0 | 5.0 |
| **evaluate_point** | 19 | 28 | 19 | 19 | 11 | 14 | **22.0** | **14.7** |
| **interaction_test** | 1 | 0 | 1 | 1 | 2 | 1 | **0.7** | **1.3** |
| **local_gradients** | 6 | 19 | 10 | 5 | 2 | 4 | **11.7** | **3.7** |
| oat_sweep | 6 | 6 | 6 | 6 | 6 | 7 | 6.0 | 6.3 |
| regression_fit | 0 | 2 | 8 | 0 | 6 | 6 | 3.3 | 4.0 |
| **sensitivity_report** | 2 | 5 | 6 | 2 | 3 | 4 | **4.3** | **3.0** |

#### C.1E.2 First OAT sweep targets (input ordering)

The order in which the agent probes inputs via OAT sweeps reveals its exploration strategy. Does prior change which inputs are probed first?

- **prior seed 42**: X1 → X2 → X3 → X4 → X5 → X6
- **prior seed 43**: X1 → X2 → X3 → X4 → X5 → X6
- **prior seed 44**: X1 → X2 → X3 → X4 → X5 → X6
- **fresh seed 42**: X3 → X6 → X4 → X2 → X1 → X5
- **fresh seed 43**: X3 → X4 → X6 → X1 → X2 → X5
- **fresh seed 44**: X3 → X4 → X6 → X1 → X2 → X5 → X3

#### C.1E.3 Hypothesis text at iteration 0 (prior references)

Does the prior-equipped agent's initial hypothesis explicitly reference the prior? Searching for keywords: 'prior', 'related system', 'previous', 'from the'.

- **prior seed 42**: mentions: ['prior', 'from the']. Snippet: "This system has modular structure distinct from the prior. Y3 is driven ONLY by X1 and X4 with strong synergistic interaction (std=0.100); Y3=0.997 wh..."
- **prior seed 43**: mentions: ['prior', 'from the']. Snippet: "This system shows significant structural differences from the prior. Y3 is driven by X1 and X4 (NOT X3) with synergistic interaction (X4 effect 5× str..."
- **prior seed 44**: mentions: ['prior', 'from the']. Snippet: "The system has modular structure different from the prior. Y3 is driven ONLY by X1(+) and X4(+), with X1 saturating above ~0.775 and X4 being the stro..."
- **fresh seed 42**: mentions: ['causal model']. Snippet: "The system has a structured causal model with key features: (1) Y3 depends ONLY on X1 and X4, with a nonlinear saturating relationship and moderate X1..."
- **fresh seed 43**: no prior keywords. Snippet: "The system has a clean causal structure: Y1 is driven positively by ALL 6 inputs (X6 and X3 strongest). Y2 is driven negatively by X4, X2, X1 and posi..."
- **fresh seed 44**: no prior keywords. Snippet: "The system has a clear causal structure: Y1 is driven positively by ALL six inputs (X6 strongest). Y2 is driven by all inputs with mixed signs (X4 dec..."

#### C.1E.4 Edge confidence initialization: GT-aligned vs variant-specific

Does the prior-equipped agent start with higher confidence on edges that are correct (shared with 1A GT) and lower confidence on edges that are wrong (1A-specific, not in this variant's GT)? This is the direct test of 'screen-first selectively leverages correct priors.'

Edge classification for 1E:
- Shared with 1A (prior should help): 16
- Wrong from 1A (prior should dismiss): 2
- Missing from 1A (prior doesn't know): 3

| Condition | Seed | Mean conf (shared) | Mean conf (wrong from 1A) | Mean conf (missing from 1A) |
|---|---|---|---|---|
| prior | 42 | 0.87 (n=16) | 0.02 (n=2) | 0.92 (n=3) |
| prior | 43 | 0.85 (n=16) | 0.02 (n=2) | 0.88 (n=3) |
| prior | 44 | 0.88 (n=16) | 0.02 (n=2) | 0.93 (n=3) |
| fresh | 42 | 0.90 (n=16) | — | 0.93 (n=3) |
| fresh | 43 | 0.80 (n=16) | — | 0.86 (n=3) |
| fresh | 44 | 0.77 (n=16) | — | 0.79 (n=3) |

### C.3 Falsification verdicts

- **Prior primes toward local_gradients on 1D**: **QUALIFIED** — 1D prior uses 8.0 local_gradients vs fresh 3.0 — directionally higher but not consistent across seeds ([12, 3, 9] vs [3, 4, 2])
- **Prior primes toward local_gradients on 1E**: **PASS** — 1E prior uses 11.7 local_gradients vs fresh 3.7 — consistently higher across all seeds ([6, 19, 10] vs [5, 2, 4])
- **Screen-first selectively leverages correct priors (edge confidence initialization)**: **SEE TABLE C.{1D,1E}.4** — Inspect the edge confidence initialization tables above. If prior-run shared-edge confidence > fresh-run shared-edge confidence at iteration 0, the selective-leverage claim has support. If they're similar, the prior isn't being used for edge-specific initialization.
- **1D prior-fresh bimodality is real (not just seed 42 outlier)**: **QUALIFIED** — Paired diffs (prior − fresh): ['-0.1014', '+0.1103', '+0.0608']. Signs: ['−', '+', '+']. Mixed signs confirm bimodality.

## Section E — Permuted-feedback revisit

**Question:** The 'Are We There Yet?' paper showed LLM agents are insensitive to permuted feedback on discrete problems. Does our agent use feedback in a continuous structured setting?

**Data:** Phase 2.0 ablation runs on 1A (n=1 each, seed 42, 54 evals). Only HV + X + Y arrays available — no iteration_summaries, tool_calls, or hypothesis text. Earlier protocol version (pre-structured-summaries).

### E.1 HV trajectory comparison


| Eval | Real HV | Permuted HV | Δ |
|---|---|---|---|
| 1 | 0.0321 | 0.0321 | +0.0000 |
| 6 | 0.0776 | 0.0776 | +0.0000 |
| 12 | 0.1004 | 0.1004 | +0.0000 |
| 18 | 0.1855 | 0.1213 | +0.0642 |
| 24 | 0.2187 | 0.1227 | +0.0960 |
| 36 | 0.2187 | 0.1305 | +0.0882 |
| 48 | 0.2187 | 0.1317 | +0.0870 |
| 54 | 0.2187 | 0.1317 | +0.0870 |

**Divergence point:** eval 13 (first eval where |Δ| > 0.01). LHS phase (evals 1-12) is identical; divergence begins immediately when the agent starts using tools.

**Final HV:** real = 0.2187, permuted = 0.1317. Real is 166% of permuted (+0.0870 absolute).

**Real run plateaus at eval 22** (HV=0.2183, no further improvement in remaining 32 evals). This is 79% of BO@66 — notably higher than the current VR multi_seed average (0.193, 70% of BO). This was an earlier protocol version (Phase 2.0) that may not have had forced structured iteration summaries.

### E.2 X-space targeting comparison

Does the real agent target different regions than the permuted agent? Compare mean X values in the post-LHS phase (evals 13+).

| Input | Real mean | Permuted mean | Real range | Permuted range |
|---|---|---|---|---|
| X1 | 0.971 | 0.772 | 0.250 | 0.600 |
| X2 | 0.999 | 0.595 | 0.050 | 0.770 |
| X3 | 0.993 | 0.666 | 0.150 | 0.700 |
| X4 | 0.954 | 0.611 | 0.350 | 0.750 |
| X5 | 0.993 | 0.509 | 0.150 | 0.730 |
| X6 | 0.101 | 0.454 | 0.050 | 0.810 |

### E.3 Output quality comparison (post-LHS)

| Output | Direction | Real mean | Permuted mean | Real better? |
|---|---|---|---|---|
| Y1 | maximize | 0.9026 | 0.2681 | YES |
| Y2 | minimize | 0.3511 | 0.2724 | no |
| Y3 | threshold | 0.9923 | 0.7858 | — |
| Y4 | maximize | 0.3153 | 0.1299 | YES |

_HV trajectory plot saved to `E_permuted_feedback_hv.png`._

### E.4 Falsification verdicts

- **LLM agent uses feedback in continuous structured settings (counter to 'Are We There Yet?')**: **PASS** — Real HV (0.2187) is 166% of permuted (0.1317). Divergence begins at eval 13 (immediately after LHS). When feedback is real, the agent exploits it; when permuted, it stalls. CAVEAT: n=1, one oracle, earlier protocol version.

## Section I — Claim ledger — what survives, what doesn't

One-line verdict for every claim made in the rubric, postmortem, paper_framing, or conversation. Based on Sections A-E dynamics analysis plus the critical ablation finding (VR without forced summary outperforms VR with summary by 35%, p=0.0003, n=5).

### I.1 THE HEADLINE: Forced articulation hurts

The ablation (same LLM, same tools, same oracle, same budget, skip only the structured iteration summary) produces 35% higher HV than VR with summaries (0.259 vs 0.193, paired t-test p=0.0003, n=5, 5/5 seeds positive, 95% CI [+0.048, +0.086]). The 'understanding tax' decomposes as ~8pp from exploration overhead (OAT sweeps vs BO acquisition) and ~23pp from the forced articulation penalty.

**Mechanism (from Sections A and C):** The forced iteration summary cements the agent's iter 0 model (including wrong dismissals like X5→Y2 at confidence 0.1) into the condensed conversation context. This anchors subsequent iterations to potentially wrong beliefs and imposes systematic rather than data-driven exploration. Seed 43 controls for iteration count and tool-call count, confirming the effect is from HOW tools are used, not how many.

### I.2 Claim-by-claim verdicts


| # | Claim | Verdict | Key evidence | Paper action |
|---|---|---|---|---|
| 1 | VR's forced articulation (structured iteration summary) improves optimization | **FALSIFIED** | Ablation n=5: removing summary improves HV by 35% (p=0.0003). The summary HURTS. | Central finding of the paper. VR protocol's design choice is counterproductive. |
| 2 | The agent's predictions improve over iterations (feedback loop is real) | **PASS** | eval_point MAE drops in 88% of seeds (21/24). Extended budget: 8-20× improvement. | The feedback loop works — but it works BETTER without forced summary. |
| 3 | Exploration (OAT) is more surprising than exploitation (evaluate_point) | **PASS** | OAT surprise 75% vs evaluate_point surprise 58%. Exploitation surprise drops in 80% of seeds. | The explore→exploit transition is real and measurable. |
| 4 | Exploitation targets are better than random (LHS) | **PASS** | 14/14 seeds, 2-6× improvement on maximize objectives vs LHS baseline. | The agent IS directed, regardless of articulation. |
| 5 | OAT-discovered input importance guides exploitation at per-input level | **FALSIFIED** | 1/14 seeds show positive Spearman ρ. 11/14 show inverse (agent FIXES important inputs at optimal, varies unimportant one... | Agent exploits at region level, not per-input emphasis level. Inverse pattern consistent with fixing... |
| 6 | Within-condition prediction accuracy predicts HV | **INCONCLUSIVE** | 1A multi_seed n=10: r=-0.27, p=0.48. No significant within-condition correlation. Likely because articulation penalty do... | Not a useful predictor. The binding constraint is the summary overhead, not prediction quality. |
| 7 | Discovery (recall) predicts exploitation (HV) pooled across conditions | **QUALIFIED** | Pooled r=+0.375, p=0.017 — but driven by between-condition variation (different oracles), not within-condition. The corr... | Not a causal claim. Different oracles produce different HV and recall. |
| 8 | Prior shifts tool allocation toward exploitation | **PASS (within VR regime)** | local_gradients: prior 8-12× vs fresh 3-4×, consistent across 1D and 1E. BUT: this is within the VR protocol. The summar... | Evidence for belief anchoring mechanism — the summary creates overcommitment to exploitation. |
| 9 | Screen-first protocol selectively leverages correct priors | **QUALIFIED** | Section C.4: prior correctly dismisses wrong edges on 1E (confidence 0.02). But raises ALL edges by ~5pp (general inflat... | The 'selective leverage' story is partially real (1E wrong-edge dismissal) but the dominant effect i... |
| 10 | LLM agent uses feedback in continuous structured settings | **PASS (n=1 caveat)** | Real HV 0.219 vs permuted 0.132 (66% better). Divergence at eval 13 (immediately after LHS). Agent exploits real feedbac... | Counter to 'Are We There Yet?' on discrete problems. LLMs DO use feedback here. But n=1, one oracle,... |
| 11 | VR reaches 68% of BO at 72 budget on 1A (understanding tax) | **PASS but REINTERPRETED** | 0.193/0.281 = 68.5% (n=10 vs BO@144). But ablation shows 0.259/0.281 = 92%. The 'tax' is mostly articulation overhead (~... | The tax exists but its decomposition changes the story completely. |
| 12 | VR approaches BO at extended budget (144 evals) | **PASS** | VR reaches 98.9% of BO@144 on 1A (n=4). HD: 98.1% (n=3). Never exceeds BO at matched budget. | The convergence is real but the ablation suggests it's the tools converging, not the articulation. |
| 13 | HD dimensionality scaling: VR advantage grows when k/d shrinks | **PASS** | 1A VR/BO=68.5%, HD VR/BO=94.0% at matched budgets. 25.6pp gap. Screening is effective (0% noise OAT on Opus). | The dimensionality result is real. Question: does it hold for the ablation agent too? (Not tested.) |
| 14 | R²(M→Y) predicts VR/BO ratio | **FALSIFIED** | Low-R² oracles (1B R²=0.80, 1D R²=0.75) have VR/BO≈0.91 — HIGHER than 1A (R²=0.96) at 0.69. Direction reversed. | Retired. R²(M→Y) characterizes oracle structure but doesn't predict VR/BO. |
| 15 | Prior knowledge helps on structurally different oracles (+2% 1D, +5% 1E) | **STATISTICALLY UNDERPOWERED** | Paired t-test: 1D p=0.38, 1E p=0.12. Combined p=0.15. Direction is consistent (5/6 positive) but not significant at α=0.... | Directional only. Would need n=6+ on 1E to reach significance, but the finding is secondary to the a... |
| 16 | Capability floor at Sonnet tier (Haiku breaks zero-false-positive property) | **PASS** | Haiku HD: 43.7% of BO, 2 false positive noise edges (conf 0.55, 0.60). Opus/Sonnet: 95-97% of BO, zero false positives. | Real finding, independent of VR articulation. The TOOLS require Sonnet-tier capability. |
| 17 | Discovery and optimization decouple at Haiku (info capture 0.96 but HV 44%) | **PASS (not tested against ablation)** | Haiku HD info capture 0.959 vs Opus 0.996, but HV 43.7% vs 96.6%. Discovery succeeds, exploitation fails. | Real finding. Would be interesting to test ablation on Haiku HD. |

### I.3 What the paper should be about (grounded in the claim ledger)

The benchmark (SynthOracle) + diagnostic suite is the primary contribution. It enabled the discovery that forced articulation hurts — a finding that could not have been made without ground-truth evaluation.

VR is the example protocol: well-motivated a priori (CBMs, CoT, Reflexion, scientific method analogy), tested rigorously, found to be counterproductive in its central design choice. The tools (OAT sweeps, evaluate_point, local_gradients) provide value; the forced structured summary does not.

The ablation result (35% improvement from removing summaries, p=0.0003) is the headline finding. The mechanism (belief anchoring via condensed context) is supported by per-seed dynamics analysis (Section C).

Supporting findings that survive: agent uses feedback (Section E), dimensionality scaling works (Rule 3), capability floor at Sonnet (Rule 10), discovery-optimization decoupling at Haiku (Rule 11).

