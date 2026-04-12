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

