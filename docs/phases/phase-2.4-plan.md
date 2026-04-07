# Phase 2.4 Plan: Multi-Seed Validation

## Objective

Run 10 seeds of the Opus tool agent on Medium 1A with the full diagnostic stack. Produce error bars for all metrics. Compare against the existing 10-seed BO baseline. This is the statistical foundation for the paper — every claim must be backed by this data.

## What we already have

- **BO baseline:** 10 seeds (42-51) in `experiments/comparison/results/`, shared reference point (seed=0)
- **VR seed 42:** Already ran as smoke test via `run_multi_seed.py`. $4.82, 3 iterations, HV=0.178, 2 calibration checkpoints (MAE=0.027, 0.010). Uses same shared reference point — reusable.
- **Runner script:** `experiments/vr_agent/run_multi_seed.py` — tested end-to-end
- **Analysis pipeline:** `experiments/analysis/compute_metrics.py` — 11 sections, needs multi-seed runs registered

## Experiment

| Parameter | Value |
|-----------|-------|
| Model | claude-opus-4-6 |
| Agent | Tool-use (`vr_tools.py`) |
| Oracle | Medium 1A |
| Seeds | 42-51 (reuse seed 42 from smoke test) |
| Budget | 72 evals per seed (12 initial + 60 tool) |
| Thinking | `{"type": "adaptive"}` |
| Calibration | Every 20 evals |
| Reference point | Shared, seed=0 |
| Estimated cost | ~$48 ($4.82 × 10, variance from iteration count) |

### Execution

Run seeds 43-51 (seed 42 already complete). The script skips seeds that already have saved results, or we run a modified invocation for seeds 43-51 only.

### Data saved per seed

- `multi_seed/seed{N}.npz` — X, Y, hypervolumes, Pareto front
- `multi_seed/seed{N}_log.json` — tool_calls (with results), mechanism_log, calibration_checks, iteration_summaries, token counts

## Metrics to report (aggregate over 10 seeds)

### Primary (paper headline)

1. **Final HV:** mean ± std, compared to BO (mean ± std). 95% CI overlap test.
2. **Edge precision/recall:** mean ± std. Do all seeds find the same edges?
3. **Budget-normalized HV:** evals to reach 50%/75%/90% of reference, mean ± std.

### Diagnostic (paper depth)

4. **OAT direction accuracy:** mean ± std across seeds. How consistent is prediction quality?
5. **OAT magnitude error:** mean ± std. Does functional form understanding vary?
6. **Calibration MAE:** per-checkpoint mean ± std. Does the learning trajectory replicate?
7. **Surprise rate:** early/mid/late phase rates, mean ± std. Is the LEARNING trend consistent?
8. **Adversarial MAE:** mean in adversarial vs non-adversarial regions.
9. **Confidence calibration error:** mean ± std per iteration.
10. **Information capture (T6):** Sobol proxy per iteration, mean ± std.

### Cost

11. **Cost per seed:** mean ± std. Is iteration count stable or variable?
12. **Tokens per seed:** input/output mean ± std.

## Analysis updates needed

Register multi-seed runs in `compute_metrics.py` TOOL_USE_RUNS. Two options:

**Option A:** Add all 10 seeds individually as separate runs. Pro: full per-seed detail. Con: tables get long.

**Option B:** Add aggregate analysis functions that load all seeds from the `multi_seed/` directory. Pro: clean aggregate reporting. Con: new code.

Recommendation: **Option B** — add a `compute_multi_seed_aggregate()` function that loads all `multi_seed/seed*.npz` + `seed*_log.json` files, computes per-metric distributions, and prints summary tables + plots. Register a few representative seeds in TOOL_USE_RUNS for the per-run detail sections.

## Plots to produce

1. **HV envelope:** VR mean ± std overlaid with BO mean ± std (already in `run_multi_seed.py`)
2. **Final HV boxplot:** VR vs BO side by side
3. **Edge P/R distribution:** per-seed P/R scatter or violin
4. **Calibration trajectory:** all 10 seeds overlaid, mean highlighted
5. **Causal convergence:** per-iteration P/R across seeds (mean ± std envelope)

## Validation gates

1. All 10 seeds complete without crashes
2. Final HV mean within 15% of BO mean (not a strict requirement — the paper's claim is about understanding, not optimization superiority)
3. Edge recall ≥ 0.90 on ≥ 8/10 seeds (causal discovery is consistent)
4. Calibration MAE decreases from first to last checkpoint on ≥ 6/10 seeds (learning trend replicates)
5. Cost per seed ≤ $8 on average (budget feasibility)

## Verification

```bash
# Run remaining seeds (43-51)
ANTHROPIC_API_KEY=... uv run --extra vr python experiments/vr_agent/run_multi_seed.py

# Or run selectively if seed 42 should be skipped:
# Modify SEEDS in the script to range(43, 52), then run

# After completion:
uv run python experiments/analysis/compute_metrics.py

# Review: aggregate tables, HV envelope, calibration trajectories
```
