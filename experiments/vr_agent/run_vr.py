"""Run VR agent on MediumOracle and generate results.

Usage:
    uv run --extra vr python experiments/vr_agent/run_vr.py

Requires ANTHROPIC_API_KEY in environment.

Produces:
- Printed summary (HV, prediction accuracy, token usage, cost)
- Saved to experiments/vr_agent/results/
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
N_ITERATIONS = 42


def save_result(name: str, result: VRResult, output_dir: Path) -> None:
    """Save VRResult to .npz and mechanism log to .json."""
    np.savez(
        output_dir / f"{name}_seed{result.seed}.npz",
        X=result.X,
        Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X,
        pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
        predictions=result.predictions,
        prediction_errors=result.prediction_errors,
        seed=result.seed,
        n_initial=result.n_initial,
        n_vr_iterations=result.n_vr_iterations,
        total_seconds=result.total_seconds,
        total_llm_calls=result.total_llm_calls,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
    )

    # Save mechanism log as JSON for readability
    log_data = [
        {
            "step": s.step,
            "hypothesis": s.hypothesis,
            "reasoning": s.reasoning,
            "reconciliation": s.reconciliation,
            "prediction_error": s.prediction_error.tolist(),
            "directional_accuracy": s.directional_accuracy,
        }
        for s in result.step_logs
    ]
    with open(output_dir / f"{name}_seed{result.seed}_log.json", "w") as f:
        json.dump(log_data, f, indent=2)


def print_summary(name: str, result: VRResult) -> None:
    """Print a formatted summary."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    n_total = result.n_initial + result.n_vr_iterations
    print(f"  Total evals: {n_total} ({result.n_initial} initial + {result.n_vr_iterations} VR)")
    print(f"  Final HV: {result.hypervolumes[-1]:.6f}")
    print(f"  Initial HV: {result.hypervolumes[result.n_initial - 1]:.6f}")
    hv_gain = result.hypervolumes[-1] - result.hypervolumes[result.n_initial - 1]
    print(f"  HV gain from VR: {hv_gain:.6f}")
    print(f"  Pareto front: {result.pareto_Y.shape[0]} points")
    print(f"  Wall time: {result.total_seconds:.1f}s")

    # Prediction accuracy
    n_steps = len(result.step_logs)
    if n_steps > 0:
        all_acc = [s.directional_accuracy for s in result.step_logs]
        # Overall directional accuracy
        flat = [a for step_acc in all_acc for a in step_acc]
        overall = sum(flat) / len(flat) if flat else 0.0
        print(f"\n  Prediction accuracy:")
        print(f"    Overall directional: {overall:.1%}")

        # First half vs second half
        mid = n_steps // 2
        first_half = [a for s in result.step_logs[:mid] for a in s.directional_accuracy]
        second_half = [a for s in result.step_logs[mid:] for a in s.directional_accuracy]
        if first_half and second_half:
            acc1 = sum(first_half) / len(first_half)
            acc2 = sum(second_half) / len(second_half)
            print(f"    First half: {acc1:.1%}")
            print(f"    Second half: {acc2:.1%}")

        # Mean absolute prediction error
        mae = np.mean(np.abs(result.prediction_errors), axis=0)
        print(f"    Mean |error| per output: {', '.join(f'{e:.4f}' for e in mae)}")

    # Token usage
    print(f"\n  Token usage:")
    print(f"    LLM calls: {result.total_llm_calls}")
    print(f"    Input tokens: {result.total_input_tokens:,}")
    print(f"    Output tokens: {result.total_output_tokens:,}")


def plot_hypervolume(name: str, result: VRResult, output_dir: Path) -> None:
    """Plot HV convergence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    evals = list(range(1, len(result.hypervolumes) + 1))
    ax.plot(evals, result.hypervolumes, linewidth=1.5, label="VR Agent")
    ax.axvline(x=result.n_initial, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"VR Agent: {name} Hypervolume Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_hv_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_prediction_error(name: str, result: VRResult, output_dir: Path) -> None:
    """Plot prediction error over iterations."""
    if len(result.step_logs) == 0:
        return
    errors = np.abs(result.prediction_errors)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for i in range(errors.shape[1]):
        ax.plot(range(1, len(errors) + 1), errors[:, i], label=f"Y{i + 1}", alpha=0.7)
    ax.set_xlabel("VR Iteration")
    ax.set_ylabel("|Prediction Error|")
    ax.set_title(f"VR Agent: {name} Prediction Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_pred_error.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    oracle = MediumOracle()
    name = "medium_1a"

    print(f"Running VR agent on {name} (seed={SEED})...")
    result = run_vr(
        oracle,
        n_iterations=N_ITERATIONS,
        seed=SEED,
        thresholds={"Y3": 0.4},
    )
    print_summary(name, result)
    save_result(name, result, RESULTS_DIR)
    plot_hypervolume(name, result, RESULTS_DIR)
    plot_prediction_error(name, result, RESULTS_DIR)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
