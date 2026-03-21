"""Run BO baseline on all oracles and generate results.

Usage:
    uv run --extra bo python experiments/bo_baseline/run_bo.py

Produces:
- Printed summary (HV, Pareto size, timing)
- Hypervolume convergence curves
- Saved to experiments/bo_baseline/results/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthoracle.baselines.bo import BOResult, run_bo
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_1c import MediumOracle1C
from synthoracle.oracles.simple import SimpleOracle

RESULTS_DIR = Path("experiments/bo_baseline/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_ITERATIONS = 42  # + initial = ~50 total evals

ORACLES = [
    ("simple", SimpleOracle(), None),
    ("medium_1a", MediumOracle(), {"Y3": 0.4}),
    ("medium_1b", MediumOracle(variant="1B"), {"Y3": 0.4}),
    ("medium_1c", MediumOracle1C(), {"Y3": 0.4}),
]

SEED = 42


def save_result(name: str, result: BOResult, output_dir: Path) -> None:
    """Save BOResult to .npz file."""
    np.savez(
        output_dir / f"{name}_seed{result.seed}.npz",
        X=result.X,
        Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X,
        pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
        seed=result.seed,
        n_initial=result.n_initial,
        n_bo_iterations=result.n_bo_iterations,
        total_seconds=result.total_seconds,
    )


def print_summary(name: str, result: BOResult) -> None:
    """Print a formatted summary."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Total evals: {len(result.X)} ({result.n_initial} initial + {result.n_bo_iterations} BO)")
    print(f"  Final HV: {result.hypervolumes[-1]:.6f}")
    print(f"  Initial HV: {result.hypervolumes[result.n_initial - 1]:.6f}")
    hv_gain = result.hypervolumes[-1] - result.hypervolumes[result.n_initial - 1]
    print(f"  HV gain from BO: {hv_gain:.6f}")
    print(f"  Pareto front: {result.pareto_Y.shape[0]} points")
    print(f"  Wall time: {result.total_seconds:.1f}s")


def plot_hypervolume_curves(
    results: dict[str, BOResult], output_dir: Path
) -> None:
    """Plot HV vs evaluation number for all oracles."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, result in results.items():
        evals = list(range(1, len(result.hypervolumes) + 1))
        ax.plot(evals, result.hypervolumes, label=name, linewidth=1.5)
        ax.axvline(
            x=result.n_initial, color="gray", linestyle="--",
            alpha=0.3, linewidth=0.8,
        )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title("BO Baseline: Hypervolume Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "hypervolume_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nHypervolume curves saved to {output_dir / 'hypervolume_curves.png'}")


def main() -> None:
    results: dict[str, BOResult] = {}

    for name, oracle, thresholds in ORACLES:
        print(f"\nRunning BO on {name}...")
        result = run_bo(
            oracle,
            n_iterations=N_ITERATIONS,
            seed=SEED,
            thresholds=thresholds,
        )
        results[name] = result
        print_summary(name, result)
        save_result(name, result, RESULTS_DIR)

    plot_hypervolume_curves(results, RESULTS_DIR)
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
