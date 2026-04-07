"""Single-seed Opus 4.6 proof-of-concept with disciplined exploration.

Usage:
    uv run --extra vr python experiments/vr_agent/run_opus_test.py

Tests whether the improved prompt + Opus + extended thinking produces
genuinely scientific exploration behavior (OAT sweeps, mechanism discovery,
falsification testing).

Primary metric: mechanism log quality, NOT HV.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthoracle.agents.vr import VRResult, run_vr
from synthoracle.oracles.medium import MediumOracle

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_ITERATIONS = 60  # generous budget (20 LLM calls × 3 points)


def print_summary(result: VRResult) -> None:
    """Print detailed summary focused on exploration quality."""
    n_total = result.n_initial + result.n_vr_iterations
    print(f"\n{'=' * 70}")
    print(f"  Opus 4.6 Proof-of-Concept: Medium 1A")
    print(f"{'=' * 70}")
    print(f"  Total evals: {n_total} ({result.n_initial} initial + {result.n_vr_iterations} VR)")
    print(f"  Final HV: {result.hypervolumes[-1]:.6f}")
    print(f"  Initial HV: {result.hypervolumes[result.n_initial - 1]:.6f}")
    hv_gain = result.hypervolumes[-1] - result.hypervolumes[result.n_initial - 1]
    print(f"  HV gain: {hv_gain:.6f}")
    print(f"  Pareto front: {result.pareto_Y.shape[0]} points")
    print(f"  Wall time: {result.total_seconds:.1f}s")

    # Explore vs exploit counts
    explore = sum(1 for s in result.step_logs if s.explore_or_exploit == "explore")
    exploit = sum(1 for s in result.step_logs if s.explore_or_exploit == "exploit")
    print(f"\n  Exploration behavior:")
    print(f"    Explore points: {explore}")
    print(f"    Exploit points: {exploit}")
    print(f"    Explore ratio: {explore / len(result.step_logs):.1%}")

    # Explore ratio by phase (first/middle/last third)
    n = len(result.step_logs)
    third = n // 3
    for phase_name, start, end in [("SCREEN", 0, third), ("PROBE", third, 2*third),
                                    ("OPTIMIZE", 2*third, n)]:
        phase_logs = result.step_logs[start:end]
        if phase_logs:
            phase_explore = sum(1 for s in phase_logs if s.explore_or_exploit == "explore")
            print(f"    {phase_name}: {phase_explore}/{len(phase_logs)} explore")

    # Prediction accuracy
    if result.step_logs:
        flat = [a for s in result.step_logs for a in s.directional_accuracy]
        overall = sum(flat) / len(flat) if flat else 0.0
        print(f"\n  Prediction accuracy: {overall:.1%} directional")
        mae = np.mean(np.abs(result.prediction_errors), axis=0)
        print(f"  Mean |error| per output: {', '.join(f'{e:.4f}' for e in mae)}")

    # Token usage and cost
    print(f"\n  Token usage:")
    print(f"    LLM calls: {result.total_llm_calls}")
    print(f"    Input tokens: {result.total_input_tokens:,}")
    print(f"    Output tokens: {result.total_output_tokens:,}")
    # Opus 4.6 pricing: $5/M input, $25/M output
    cost = result.total_input_tokens * 5 / 1e6 + result.total_output_tokens * 25 / 1e6
    print(f"    Estimated cost: ${cost:.2f}")


def print_mechanism_log(result: VRResult) -> None:
    """Print the full mechanism log for qualitative review."""
    print(f"\n{'=' * 70}")
    print(f"  Mechanism Discovery Log")
    print(f"{'=' * 70}")
    for log in result.step_logs:
        label = "🔍" if log.explore_or_exploit == "explore" else "⚡"
        print(f"\n  [{label} Step {log.step + 1}] ({log.explore_or_exploit})")
        print(f"  Hypothesis: {log.hypothesis[:200]}")
        print(f"  Reasoning: {log.reasoning[:150]}")
        if log.falsification:
            print(f"  Falsification: {log.falsification[:150]}")
        if log.reconciliation and log.step > 0:
            print(f"  Reconciliation: {log.reconciliation[:150]}")


def save_result(result: VRResult, output_dir: Path) -> None:
    """Save results."""
    np.savez(
        output_dir / "opus_test_seed42.npz",
        X=result.X, Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
        predictions=result.predictions,
        prediction_errors=result.prediction_errors,
    )
    log_data = [
        {
            "step": s.step,
            "hypothesis": s.hypothesis,
            "reasoning": s.reasoning,
            "reconciliation": s.reconciliation,
            "falsification": s.falsification,
            "explore_or_exploit": s.explore_or_exploit,
            "biggest_surprise": s.biggest_surprise,
            "prediction_error": s.prediction_error.tolist(),
            "directional_accuracy": s.directional_accuracy,
            "x": s.x.tolist(),
        }
        for s in result.step_logs
    ]
    with open(output_dir / "opus_test_seed42_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # HV curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    evals = list(range(1, len(result.hypervolumes) + 1))
    ax.plot(evals, result.hypervolumes, linewidth=1.5)
    ax.axvline(x=result.n_initial, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Opus 4.6 + Disciplined Exploration: Medium 1A")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "opus_test_hv.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    oracle = MediumOracle()

    print("Running VR agent with Opus 4.6 + adaptive thinking...")
    print(f"Budget: {N_ITERATIONS} oracle evals (batch_size=3, ~{N_ITERATIONS // 3} LLM calls)")
    print()

    checkpoint = str(RESULTS_DIR / "opus_checkpoint")
    result = run_vr(
        oracle,
        n_iterations=N_ITERATIONS,
        batch_size=3,
        seed=SEED,
        thresholds={"Y3": 0.4},
        model="claude-opus-4-6",
        thinking={"type": "adaptive"},
        max_tokens=16000,
        checkpoint_dir=checkpoint,
    )

    print_summary(result)
    print_mechanism_log(result)
    save_result(result, RESULTS_DIR)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
