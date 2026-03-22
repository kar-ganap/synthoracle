"""Single-seed Opus proof-of-concept with tool-use VR agent.

Usage:
    ANTHROPIC_API_KEY=... uv run --extra vr python experiments/vr_agent/run_tools_test.py

Tests whether tool-use architecture improves exploration efficiency
over batch proposals (Phase 2.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthoracle.agents.vr_tools import VRToolsResult, run_vr_tools
from synthoracle.oracles.medium import MediumOracle

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BUDGET = 72  # 12 initial + 60 oracle evals


def print_summary(result: VRToolsResult) -> None:
    """Print detailed summary."""
    print(f"\n{'=' * 70}")
    print(f"  Tool-Use VR Agent: Medium 1A")
    print(f"{'=' * 70}")
    print(f"  Evals: {result.eval_count}/{result.n_budget} "
          f"({result.n_initial} initial + {result.eval_count - result.n_initial} tools)")
    print(f"  Final HV: {result.hypervolumes[-1]:.6f}")
    init_hv = result.hypervolumes[result.n_initial - 1] if result.n_initial > 0 else 0.0
    print(f"  Initial HV: {init_hv:.6f}")
    print(f"  HV gain: {result.hypervolumes[-1] - init_hv:.6f}")
    print(f"  Pareto front: {result.pareto_Y.shape[0]} points")
    print(f"  Wall time: {result.total_seconds:.1f}s")

    # Tool usage
    print(f"\n  Tool usage ({len(result.tool_calls)} total calls):")
    tool_counts: dict[str, int] = {}
    tool_costs: dict[str, int] = {}
    for tc in result.tool_calls:
        name = str(tc["name"])
        cost = int(tc.get("cost", 0))
        tool_counts[name] = tool_counts.get(name, 0) + 1
        tool_costs[name] = tool_costs.get(name, 0) + cost
    for name in sorted(tool_counts.keys()):
        print(f"    {name}: {tool_counts[name]}x (cost: {tool_costs[name]} evals)")

    # Token usage
    print(f"\n  Token usage:")
    print(f"    LLM calls: {result.total_llm_calls}")
    print(f"    Input tokens: {result.total_input_tokens:,}")
    print(f"    Output tokens: {result.total_output_tokens:,}")

    # Mechanism log
    print(f"\n  Mechanism log ({len(result.mechanism_log)} entries):")
    for i, m in enumerate(result.mechanism_log):
        print(f"    [{i + 1}] {m[:120]}")


def save_result(result: VRToolsResult, output_dir: Path) -> None:
    """Save results."""
    np.savez(
        output_dir / "tools_test_seed42.npz",
        X=result.X, Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
    )
    with open(output_dir / "tools_test_seed42_log.json", "w") as f:
        json.dump({
            "tool_calls": result.tool_calls,
            "mechanism_log": result.mechanism_log,
            "calibration_checks": result.calibration_checks,
            "eval_count": result.eval_count,
            "total_llm_calls": result.total_llm_calls,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
        }, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    evals = list(range(1, len(result.hypervolumes) + 1))
    ax.plot(evals, result.hypervolumes, linewidth=1.5)
    ax.axvline(x=result.n_initial, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Tool-Use VR Agent: Medium 1A")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "tools_test_hv.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    oracle = MediumOracle()
    checkpoint = str(RESULTS_DIR / "tools_checkpoint")

    print(f"Running tool-use VR agent (Opus 4.6 + adaptive thinking)...")
    print(f"Budget: {N_BUDGET} oracle evals")
    print()

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
    )

    print_summary(result)
    save_result(result, RESULTS_DIR)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
