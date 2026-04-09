# Phase 2.5 Plan: Transfer Tests (1D + 1E)

## Context

Phase 2.4 showed the VR agent consistently discovers causal structure (100% precision, 94.4% recall) but reaches only 70% of BO's HV. The paper's thesis is that this structural understanding *transfers* — justifying the understanding tax. We need transfer tests on variants where the 1A prior is partially wrong, not just slightly different.

Variants 1D (Structural Shift) and 1E (Rewired) are designed for this. 1D tests falsification of wrong functional forms; 1E tests unlearning of confidently-held false edges.

## Experiment matrix

Pilot: 1 seed (42) per condition, 4 runs total.

| Condition | Oracle | Prior knowledge | Tests |
|-----------|--------|----------------|-------|
| 1D with prior | MediumOracle1D | 1A causal model from seed 42 | Does prior help despite wrong forms? |
| 1D fresh | MediumOracle1D | None | Baseline without prior |
| 1E with prior | MediumOracle1E | 1A causal model from seed 42 | Does prior hurt with 3 false edges? |
| 1E fresh | MediumOracle1E | None | Baseline without prior |

Budget: 72 evals each. Estimated cost: ~$18 ($4.50 × 4).

## Prior knowledge format

Extract from the 1A multi-seed seed 42 iteration summaries: the final `hypothesis` + `edges` (with confidence scores). This is the "causal model" the agent transfers. Passed via the `prior_knowledge` parameter in `run_vr_tools()`, which already exists and includes the anti-confirmation-bias prompt.

## What to measure

Per-run (same as Phase 2.4 diagnostic stack):
1. HV trajectory + final HV
2. Edge P/R against the *target* oracle's ground truth (not 1A's)
3. OAT prediction accuracy (structured)
4. Calibration checkpoints
5. Surprise rate
6. Iteration summaries with edge confidence
7. Cost

Cross-condition comparisons:
- **1D with prior vs fresh:** Does knowing 1A edges help? HV higher early? Edge recall faster?
- **1E with prior vs fresh:** Does the 1A prior hurt? More surprises? Slower to correct false edges?
- **Prior falsification rate:** How quickly does the agent correct wrong priors? Track which 1A claims it abandons and when.

## Implementation

Single new script: `experiments/vr_agent/run_transfer_1d_1e.py`

Pattern follows `run_head_to_head.py` — run 4 conditions sequentially, save per-condition results, print comparison table.

The prior knowledge string is extracted from `multi_seed/seed42_log.json` → last iteration summary → hypothesis + edges formatted as text.

Results saved to:
- `experiments/vr_agent/results/transfer_1d_prior_seed42.{npz,log.json}`
- `experiments/vr_agent/results/transfer_1d_fresh_seed42.{npz,log.json}`
- `experiments/vr_agent/results/transfer_1e_prior_seed42.{npz,log.json}`
- `experiments/vr_agent/results/transfer_1e_fresh_seed42.{npz,log.json}`

## Files to create/modify

1. `experiments/vr_agent/run_transfer_1d_1e.py` — new experiment script
2. `experiments/analysis/compute_metrics.py` — register new runs in TOOL_USE_RUNS (4 entries)
3. `docs/phases/phase-2.5-plan.md` — this plan

## Verification

```bash
# Smoke test: verify oracles + prior extraction work
uv run python -c "
from synthoracle.oracles.medium_1d import MediumOracle1D
from synthoracle.oracles.medium_1e import MediumOracle1E
import json
with open('experiments/vr_agent/results/multi_seed/seed42_log.json') as f:
    log = json.load(f)
last = log['iteration_summaries'][-1]
print(f'Prior: {len(last[\"edges\"])} edges, hypothesis length: {len(last[\"hypothesis\"])}')
print(f'1D IO edges: {len(MediumOracle1D().ground_truth().project_to_io().edges)}')
print(f'1E IO edges: {len(MediumOracle1E().ground_truth().project_to_io().edges)}')
"

# Run pilot (4 conditions)
source .env && uv run --extra vr python experiments/vr_agent/run_transfer_1d_1e.py

# Review results
uv run python experiments/analysis/compute_metrics.py 2>&1 | grep -A 10 "transfer_1d\|transfer_1e"
```

## Go/no-go after pilot

- **Prior helps on 1D:** HV with prior > HV fresh by eval 36 (prior gives head start on correct edges)
- **Prior hurts on 1E early but recovers:** HV with prior < HV fresh at eval 36, but agent falsifies wrong edges by eval 60 and recovers
- **If both priors hurt:** The VR agent can't transfer — report as negative result
- **If both priors help equally:** Variants aren't different enough — need harder E variant
- **Scale up:** If pilot shows interesting differential, run 5 seeds per condition (~$89)
