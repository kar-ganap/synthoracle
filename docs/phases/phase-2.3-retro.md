# Phase 2.3 Retro: Tool-Use Agent + Transfer Tests

**Date:** 2026-03-21 to 2026-03-22
**Branch:** `phase-2.3-tool-use-agent`
**Status:** Complete

---

## 1. What we set out to do

Build a tool-use VR agent where the LLM autonomously designs experiments by calling oracle tools (OAT sweeps, interaction tests, point evaluations) and analysis tools (correlations, gradients, regressions). Test whether structured tool calls improve exploration over batch proposals. Run transfer tests on oracle variants 1B and 1C.

## 2. What actually happened

- Built `agents/vr_tools.py` with agentic tool loop: LLM calls tools, executes them, returns results, repeats until budget exhausted or agent stops
- 7 tools: 3 oracle (evaluate_point, oat_sweep, interaction_test — cost budget) + 4 analysis (correlation_matrix, sensitivity_report, local_gradients, regression_fit, current_pareto_front — free)
- All oracle tools require predictions before observing results (accountability)
- Transfer tests: 1B and 1C with and without prior knowledge from 1A
- Built comprehensive analysis pipeline (`experiments/analysis/compute_metrics.py`) with 6 metric sections
- Later extended with model sweep (Phase B), structured summaries, calibration, T6

### Tool agent results (Opus, seed 42)

| Metric | Batch (Phase 2.2) | Tool (Phase 2.3) |
|--------|-------------------|-------------------|
| Final HV | 0.273 | 0.268 |
| Edge precision | 0.750 | 1.000 |
| Edge recall | 1.000 | 0.944 |
| False positives | 6 | 0 |
| Mechanism discoveries | 24 (from X/Y bins) | 19 (from OAT replay) |
| Cost (Opus) | ~$2.00 | $2.82 |

### Transfer results

| Test | HV | Key finding |
|------|-----|-------------|
| 1B with prior | 0.247 | Prior helped early exploration |
| 1B fresh | 0.230 | Slightly worse without prior |
| 1C with prior | 0.480 | Near reference HV |
| 1C fresh | 0.465 | Comparable — 1C is easier |

### Model sweep results (Phase B)

| Agent+Model | Final HV | Cost |
|-------------|----------|------|
| Opus tool | 0.268 | $2.82 |
| Sonnet tool | 0.267 | $1.32 |
| Haiku tool | 0.170 | $0.13 |
| Sonnet batch | 0.243 | $3.50* |
| Haiku batch | 0.191 | $0.30* |

## 3. What went well

- **Tool architecture produces zero false positives.** OAT sweeps give clean, controlled sensitivity data. Batch agent infers edges from noisy X/Y bins and gets 6 FPs.
- **Tool agent is robust to model capability.** Sonnet tool matches Opus tool on HV and edge P/R (99.6% HV). The structured tools enforce good exploration regardless of model quality.
- **Transfer works.** Prior knowledge from 1A helps on 1B, confirming the causal model transfers.
- **Analysis pipeline grew into a 9-section diagnostic suite** (later extended to 11). This became the main contribution.

## 4. What went poorly

- **Tool agent HV is slightly below batch** on single seed (0.268 vs 0.273). Not statistically significant, but the tool agent's advantage is in understanding quality, not optimization speed.
- **Mechanism log was free text** — unparseable for quantitative analysis. Fixed in structured-iteration-summaries branch with Pydantic `_IterationSummary`.
- **`biggest_surprise` field was still broken** in save code. Carried over from Phase 2.2.
- **Sonnet batch cost exploded** ($3.50+) due to conversation context growth — more expensive than Opus despite cheaper per-token pricing.
- **Haiku not viable** (<70% HV on both agents).

## 5. Lessons learned

- **Architecture matters more than model scale** for the tool agent. The structured tools (OAT, interaction test) force systematic exploration that compensates for weaker reasoning.
- **The diagnostic metrics are the contribution, not the HV number.** Edge P/R, OAT direction accuracy, adversarial MAE, calibration error, and confidence calibration — these measure what HV alone can't distinguish.
- **Cost is dominated by conversation context**, not model pricing. Sonnet uses more iterations and more tokens per iteration, erasing its per-token cost advantage.
- **n=6 initial points expose a VR limitation.** Even Opus trusts spurious correlations from 6 points rather than recognizing insufficient sample size.

## 6. What to do differently next time

- Every LLM output that feeds a metric must go through Pydantic structured output
- Test with minimal budget (n_budget=20) before committing to full runs
- Track cost per run automatically in the runner, not manually after the fact
- Write retros incrementally, not in batch — the work on this branch spans 2 weeks and 20 commits
