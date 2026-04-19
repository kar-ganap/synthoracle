# SynthOracle

A benchmark for evaluating structured reasoning in LLM optimization agents.

**Paper:** [Structured Reasoning in LLM Optimization Agents: Scaffolding, Not Regularization](paper/main.tex)

## What is SynthOracle?

SynthOracle is a family of synthetic multi-objective optimization oracles with known causal structure. Each oracle defines a directed acyclic graph (DAG) from inputs through mechanisms to outputs, enabling separate measurement of:

- **Optimization quality** (hypervolume relative to Bayesian optimization)
- **Reasoning quality** (precision and recall of discovered causal edges against the ground-truth DAG)

The benchmark includes four oracles:

| Oracle | Inputs | Description |
|--------|--------|-------------|
| **Baseline** | 6 | Reference oracle with regime transitions, hidden coupling, threshold activation |
| **Shifted** | 6 | Same topology, altered functional forms (tests belief unlearning) |
| **Rewired** | 6 | Input-mechanism wiring swapped (tests prior dismissal) |
| **Noisy** | 12 | Baseline + 6 irrelevant noise dimensions (tests screening) |

Oracle evaluations are deterministic, take <1ms, and require no API calls.

## Key Finding

Forcing LLM agents to produce structured iteration summaries is **scaffolding, not regularization**:

- It helps less capable models (Sonnet +26%) and harder tasks (Noisy +14% to +35%)
- It hurts the most capable model on the simplest oracle (Opus on Baseline, -35%, p=0.0006)
- The mechanism is **context cementing**: the summary's persistence in conversation context anchors early beliefs. An agent that produces the summary but doesn't receive it back performs identically to one that never produces it (p=0.35).

## Installation

```bash
# Clone and install
git clone https://github.com/kar-ganap/synthoracle.git
cd synthoracle
uv sync

# Run tests
make test

# Run oracle evaluation (no API key needed)
uv run python -c "
from synthoracle.oracles.medium import MediumOracle
oracle = MediumOracle(variant='1A')
import numpy as np
x = np.array([0.5, 0.7, 0.3, 0.6, 0.5, 0.4])
y = oracle.evaluate(x)
print(f'Inputs: {x}')
print(f'Outputs: {y}')
print(f'Ground truth edges: {len(oracle.ground_truth().edges)}')
"
```

## Running Experiments

### BO baseline (free, local compute)
```bash
uv run --extra bo python experiments/vr_agent/run_hd_test.py bo
```

### VR agent (requires Anthropic API key)
```bash
source .env  # ANTHROPIC_API_KEY=...
uv run --extra vr python experiments/vr_agent/run_hd_test.py vr
```

## Repository Structure

```
src/synthoracle/
  oracles/          # Oracle implementations (medium, medium_1d, medium_1e, medium_hd)
  agents/           # VR agent (vr_tools.py)
  baselines/        # BO baseline (bo.py)
  dag.py            # Causal DAG representation
  oracle.py         # Base oracle interface
  optim_utils.py    # Hypervolume, Pareto front, reference point utilities

experiments/
  analysis/         # Audit, rubric, holistic analysis scripts
    figures/        # Paper figure generation scripts
  vr_agent/         # Experiment runner scripts
    results/        # Run data (.npz, logs)

paper/
  main.tex          # Paper source

tests/              # Unit tests
```

## Reproducing Paper Results

The paper's figures can be regenerated from the run data:

```bash
# Generate all figures
for fig in experiments/analysis/figures/fig*.py; do
    uv run python "$fig"
done
```

All experiment data (.npz files and JSON logs) is in `experiments/vr_agent/results/`.

## License

MIT
