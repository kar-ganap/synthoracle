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

### A.3 Falsification verdicts

- **VR learns from feedback (evaluate_point MAE ↓, primary signal)**: **PASS** — 21/24 seeds (88%) show evaluate_point MAE lower in last iteration with data than first. Additionally, 17/24 seeds show a >50% MAE reduction (first iter to last iter).
- **OAT direction accuracy improves over iterations (secondary; OATs are front-loaded)**: **PASS** — 10/13 seeds (77%) show OAT direction accuracy higher in last non-None iter than first. Note: many seeds only have 1 iteration with OAT sweeps, so this metric is noisier than eval_point MAE.
- **Surprise rate decreases over iterations**: **PASS** — 36/40 seeds (90%) show surprise count lower in last iteration than first

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

