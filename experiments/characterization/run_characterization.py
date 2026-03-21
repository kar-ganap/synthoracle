"""Run characterization on all oracles and generate results.

Produces:
- Printed summary (Sobol indices, OAT effects, Pareto front sizes)
- Response surface plots for each oracle
- Saved to experiments/characterization/results/
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from synthoracle.characterize import CharacterizationResult, characterize
from synthoracle.oracle import Oracle
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_1c import MediumOracle1C
from synthoracle.oracles.simple import SimpleOracle

RESULTS_DIR = Path("experiments/characterization/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def print_summary(name: str, oracle: Oracle, result: CharacterizationResult) -> None:
    """Print a formatted summary of characterization results."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    print(f"\n  Output statistics:")
    for oname, stats in result.output_stats.items():
        print(f"    {oname}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    print(f"\n  Sobol first-order indices:")
    header = f"    {'':>6}"
    for oname in oracle.output_names:
        header += f"  {oname:>8}"
    print(header)
    for j in range(oracle.n_inputs):
        row = f"    {oracle.input_names[j]:>6}"
        for i in range(oracle.n_outputs):
            row += f"  {result.sobol_first_order[j, i]:>8.3f}"
        print(row)
    sums = result.sobol_first_order.sum(axis=0)
    print(f"    {'Sum':>6}" + "".join(f"  {s:>8.3f}" for s in sums))

    print(f"\n  Interaction strength (S_T - S_1):")
    interactions = result.sobol_total_order - result.sobol_first_order
    print(header)
    for j in range(oracle.n_inputs):
        row = f"    {oracle.input_names[j]:>6}"
        for i in range(oracle.n_outputs):
            row += f"  {interactions[j, i]:>8.3f}"
        print(row)

    print(f"\n  OAT effect sizes (% of output range):")
    print(header)
    total_ranges = np.array([
        result.output_stats[name]["range"] for name in oracle.output_names
    ])
    for j in range(oracle.n_inputs):
        row = f"    {oracle.input_names[j]:>6}"
        for i in range(oracle.n_outputs):
            pct = result.oat_effects[j, i] / total_ranges[i] * 100 if total_ranges[i] > 0 else 0
            row += f"  {pct:>7.1f}%"
        print(row)

    print(f"\n  Pareto front: {result.pareto_front.shape[0]} points")
    print(f"  Timing: {result.timing_us:.1f} us/eval")


def plot_1d_sweeps(
    name: str, oracle: Oracle, output_dir: Path, n_sweep: int = 200
) -> None:
    """Generate 1D sweep plots for each input vs each output."""
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    midpoint = (lo + hi) / 2

    n_inputs = oracle.n_inputs
    n_outputs = oracle.n_outputs
    fig, axes = plt.subplots(n_outputs, n_inputs, figsize=(4 * n_inputs, 3 * n_outputs))
    if n_outputs == 1:
        axes = axes[np.newaxis, :]
    if n_inputs == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f"{name}: 1D Sweeps", fontsize=14, fontweight="bold")

    for j in range(n_inputs):
        sweep = np.linspace(lo[j], hi[j], n_sweep)
        X_oat = np.tile(midpoint, (n_sweep, 1))
        X_oat[:, j] = sweep
        Y_oat = oracle.evaluate_batch(X_oat)

        for i in range(n_outputs):
            ax = axes[i, j]
            ax.plot(sweep, Y_oat[:, i])
            ax.set_xlabel(oracle.input_names[j])
            ax.set_ylabel(oracle.output_names[i])
            if i == 0:
                ax.set_title(oracle.input_names[j])

    plt.tight_layout()
    plt.savefig(output_dir / f"{name.lower().replace(' ', '_')}_1d_sweeps.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_pareto_front(
    name: str, oracle: Oracle, result: CharacterizationResult, output_dir: Path
) -> None:
    """Plot Pareto front projections (2D scatter for each pair of objectives)."""
    front = result.pareto_front
    directions = oracle.output_directions
    obj_names = oracle.output_names
    obj_indices = [i for i, d in enumerate(directions) if d != "threshold"]

    n_obj = len(obj_indices)
    if n_obj < 2:
        return

    n_pairs = n_obj * (n_obj - 1) // 2
    fig, axes = plt.subplots(1, max(n_pairs, 1), figsize=(5 * max(n_pairs, 1), 4))
    if n_pairs == 1:
        axes = [axes]

    fig.suptitle(f"{name}: Pareto Front", fontsize=14, fontweight="bold")

    pair_idx = 0
    for a in range(n_obj):
        for b in range(a + 1, n_obj):
            ax = axes[pair_idx]
            ia, ib = obj_indices[a], obj_indices[b]
            ax.scatter(front[:, ia], front[:, ib], s=2, alpha=0.5)
            ax.set_xlabel(f"{obj_names[ia]} ({directions[ia]})")
            ax.set_ylabel(f"{obj_names[ib]} ({directions[ib]})")
            pair_idx += 1

    plt.tight_layout()
    plt.savefig(output_dir / f"{name.lower().replace(' ', '_')}_pareto.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    oracles: list[tuple[str, Oracle, dict[str, float] | None]] = [
        ("Simple", SimpleOracle(), None),
        ("Medium 1A", MediumOracle(), {"Y3": 0.4}),
        ("Medium 1B", MediumOracle(variant="1B"), {"Y3": 0.4}),
        ("Medium 1C", MediumOracle1C(), {"Y3": 0.4}),
    ]

    for name, oracle, thresholds in oracles:
        print(f"\nCharacterizing {name}...")
        result = characterize(
            oracle,
            n_samples=100_000,
            n_sobol=50_000,
            n_pareto=200_000,
            thresholds=thresholds,
            seed=42,
        )
        print_summary(name, oracle, result)
        plot_1d_sweeps(name, oracle, RESULTS_DIR)
        plot_pareto_front(name, oracle, result, RESULTS_DIR)

    print(f"\n\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
