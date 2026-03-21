"""Run BO vs VR comparison on Medium 1A with multiple seeds.

Usage:
    uv run --extra bo --extra vr python experiments/comparison/run_comparison.py

Requires ANTHROPIC_API_KEY in environment for VR runs.

Produces:
- HV convergence plot (mean ± std) for BO and VR
- Final HV boxplot
- Summary table with statistics
- Saved to experiments/comparison/results/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthoracle.eval.metrics import compute_comparison, print_summary
from synthoracle.eval.plot import plot_final_hv_boxplot, plot_hv_comparison
from synthoracle.eval.runner import run_experiment
from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracles.medium import MediumOracle

RESULTS_DIR = Path("experiments/comparison/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(42, 52))  # 10 seeds: 42, 43, ..., 51
N_ITERATIONS = 42
THRESHOLDS = {"Y3": 0.4}


def main() -> None:
    oracle = MediumOracle()

    # Shared reference point for comparable HV
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    print(f"Reference point: {ref_point}")
    print(f"Seeds: {SEEDS}")
    print(f"Iterations per run: {N_ITERATIONS}")
    print()

    # Run BO (10 seeds)
    print("=" * 50)
    print("Running BO baseline (10 seeds)...")
    print("=" * 50)
    bo_results = run_experiment(
        "bo", oracle, seeds=SEEDS, n_iterations=N_ITERATIONS,
        thresholds=THRESHOLDS, reference_point=ref_point,
    )

    # Run VR (10 seeds)
    print()
    print("=" * 50)
    print("Running VR agent (10 seeds)...")
    print("=" * 50)
    vr_results = run_experiment(
        "vr", oracle, seeds=SEEDS, n_iterations=N_ITERATIONS,
        thresholds=THRESHOLDS, reference_point=ref_point,
    )

    # Compute metrics
    all_results = {"BO": bo_results, "VR": vr_results}
    metrics = compute_comparison(all_results)
    print_summary(metrics)

    # Save results
    for name, result_list in all_results.items():
        for r in result_list:
            np.savez(
                RESULTS_DIR / f"{name.lower()}_seed{r.seed}.npz",
                X=r.X, Y=r.Y,
                hypervolumes=np.array(r.hypervolumes),
                reference_point=r.reference_point,
                total_seconds=r.total_seconds,
            )

    # Plots
    plot_hv_comparison(
        metrics, RESULTS_DIR / "hv_comparison.png",
        title="BO vs VR: Medium 1A Hypervolume Convergence",
    )
    plot_final_hv_boxplot(
        metrics, RESULTS_DIR / "final_hv_boxplot.png",
        title="BO vs VR: Final Hypervolume (Medium 1A)",
    )

    print(f"\nResults and plots saved to {RESULTS_DIR}/")

    # Estimate VR cost
    total_vr_input = sum(
        getattr(r, "total_input_tokens", 0) for r in vr_results
    )
    total_vr_output = sum(
        getattr(r, "total_output_tokens", 0) for r in vr_results
    )
    print(f"\nVR API usage: {total_vr_input:,} input + {total_vr_output:,} output tokens")


if __name__ == "__main__":
    main()
