"""Transfer test: run tool-use VR agent on Medium 1B and 1C.

The agent starts with the causal model learned from Medium 1A
and must adapt to the variant oracle.

Usage:
    ANTHROPIC_API_KEY=... uv run --extra vr python experiments/vr_agent/run_transfer_test.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthoracle.agents.vr_tools import VRToolsResult, run_vr_tools
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_1c import MediumOracle1C

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BUDGET = 72

# The causal model learned from Medium 1A (summary of tool-use run)
PRIOR_MODEL_1A = """## Prior Knowledge (from a related system)
You previously studied a SIMILAR system and discovered:
- Y1 is driven primarily by X2*X4 (strong multiplicative interaction) + X5 sigmoid threshold
- X5 has a sigmoid/step activation around X5~0.4: below this Y1 drops sharply
- X6 has a context-dependent effect: at high X2, increasing X6 boosts Y1 and REDUCES Y2
- Y3 depends ONLY on X1 and X3 (no other inputs affect it)
- Y4 is driven by X2 and X4, inversely by X6
- Y2 is minimized by high X3, but X2 and X4 increase Y2 (tradeoff with Y1)
- X1 has a moderate positive effect on Y1 but is not the dominant driver
- Key Pareto template: [X1=1, X2=varies, X3=1, X4=1, X5~0.55, X6=varies by X2]

THIS IS A VARIANT of that system. Some relationships may have CHANGED.
Your first priority is to VERIFY which relationships still hold and
DISCOVER what has changed. Do NOT assume the prior model is correct — TEST it.
Focus your screening budget on confirming or falsifying the prior model."""


def run_one(
    name: str,
    oracle: MediumOracle | MediumOracle1C,
    prior: str,
) -> VRToolsResult:
    """Run transfer test on one oracle variant."""
    checkpoint = str(RESULTS_DIR / f"transfer_{name}_checkpoint")

    print(f"\n{'=' * 60}")
    print(f"  Transfer test: {name}")
    print(f"{'=' * 60}")

    result = run_vr_tools(
        oracle,
        n_budget=N_BUDGET,
        seed=SEED,
        thresholds={"Y3": 0.4},
        model="claude-opus-4-6",
        thinking={"type": "adaptive"},
        max_tokens=16000,
        max_tool_calls_per_iteration=15,
        checkpoint_dir=checkpoint,
        prior_knowledge=prior,
    )

    # Summary
    init_hv = result.hypervolumes[result.n_initial - 1] if result.n_initial > 0 else 0.0
    print(f"\n  Results for {name}:")
    print(f"    Evals: {result.eval_count}/{result.n_budget}")
    print(f"    Final HV: {result.hypervolumes[-1]:.6f}")
    print(f"    Initial HV: {init_hv:.6f}")
    print(f"    HV gain: {result.hypervolumes[-1] - init_hv:.6f}")
    print(f"    Pareto points: {result.pareto_Y.shape[0]}")
    print(f"    LLM calls: {result.total_llm_calls}")
    print(f"    Tokens: {result.total_input_tokens:,} in, {result.total_output_tokens:,} out")

    # Tool usage
    tool_counts: dict[str, int] = {}
    tool_costs: dict[str, int] = {}
    for tc in result.tool_calls:
        n = str(tc["name"])
        cost = int(tc.get("cost", 0))
        tool_counts[n] = tool_counts.get(n, 0) + 1
        tool_costs[n] = tool_costs.get(n, 0) + cost
    print(f"    Tool usage:")
    for tn in sorted(tool_counts.keys()):
        print(f"      {tn}: {tool_counts[tn]}x (cost: {tool_costs[tn]})")

    # Mechanism log
    print(f"    Mechanism log ({len(result.mechanism_log)} entries):")
    for i, m in enumerate(result.mechanism_log):
        print(f"      [{i + 1}] {m[:120]}")

    # Save
    np.savez(
        RESULTS_DIR / f"transfer_{name}_seed42.npz",
        X=result.X, Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
    )
    with open(RESULTS_DIR / f"transfer_{name}_seed42_log.json", "w") as f:
        json.dump({
            "tool_calls": result.tool_calls,
            "mechanism_log": result.mechanism_log,
            "calibration_checks": result.calibration_checks,
            "iteration_summaries": result.iteration_summaries,
            "eval_count": result.eval_count,
            "total_llm_calls": result.total_llm_calls,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
        }, f, indent=2)

    return result


def main() -> None:
    import sys

    variant = sys.argv[1] if len(sys.argv) > 1 else "1B"

    if variant.upper() == "1B":
        run_one("medium_1b", MediumOracle(variant="1B"), PRIOR_MODEL_1A)
    elif variant.upper() == "1C":
        run_one("medium_1c", MediumOracle1C(), PRIOR_MODEL_1A)
    else:
        print(f"Usage: {sys.argv[0]} [1B|1C]")
        sys.exit(1)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
