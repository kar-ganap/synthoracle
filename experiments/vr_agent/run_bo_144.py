"""Run BO baselines at 144-eval budget for 1A, 1D, 1E, HD.

Free local BoTorch runs. Produces longer BO curves so crossover_eval
comparisons against extended-budget VR runs are apples-to-apples.

Output: experiments/vr_agent/results/bo_{oracle}_n144_seed{N}.npz

Usage:
    uv run --extra bo python experiments/vr_agent/run_bo_144.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthoracle.baselines.bo import run_bo
from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_1d import MediumOracle1D
from synthoracle.oracles.medium_1e import MediumOracle1E
from synthoracle.oracles.medium_hd import MediumOracleHD

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {"Y3": 0.4}
# n=1 per oracle is sufficient to verify the pre-registered hypothesis
# that BO is saturated by ~50-60 evals. Each BO run at 144 evals takes
# ~30-60 min due to O(n³) GP fitting; 12 runs would be ~9 hours.
SEEDS = [42]
TOTAL_EVALS = 144


def run_bo_for_oracle(label: str, oracle, seeds=SEEDS) -> list[float]:
    """Run BO at 144 evals for given oracle, save to disk, return final HVs."""
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    n_initial = 2 * oracle.n_inputs  # default
    n_iterations = TOTAL_EVALS - n_initial
    print(f"\n{label}: n_initial={n_initial}, n_iterations={n_iterations}, "
          f"target total={TOTAL_EVALS}")

    finals = []
    for seed in seeds:
        out_path = RESULTS_DIR / f"bo_{label}_n144_seed{seed}.npz"
        if out_path.exists():
            d = np.load(out_path)
            hv = float(d["hypervolumes"][-1])
            print(f"  seed {seed} (existing): HV={hv:.6f} "
                  f"({len(d['hypervolumes'])} evals)")
            finals.append(hv)
            continue

        print(f"  Running BO {label} seed {seed} (target {TOTAL_EVALS} evals)...",
              flush=True)
        result = run_bo(
            oracle, n_iterations=n_iterations, seed=seed,
            reference_point=ref_point, thresholds=THRESHOLDS,
        )
        np.savez(
            out_path, X=result.X, Y=result.Y,
            hypervolumes=np.array(result.hypervolumes),
            pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
            reference_point=ref_point,
        )
        print(f"  seed {seed}: HV={result.hypervolumes[-1]:.6f} "
              f"({len(result.hypervolumes)} evals, {result.total_seconds:.0f}s)")
        finals.append(float(result.hypervolumes[-1]))

    return finals


def main() -> None:
    print("=" * 60)
    print("  BO baselines at 144-eval budget")
    print("=" * 60)

    # 1A done in first run. Skip 1E: known to be at BO ceiling (~0.278)
    # where BO @ 66 ≈ BO @ 144 to within noise. Focus on 1D and HD, which
    # are the critical comparisons for the transfer + HD headline claims.
    oracles = [
        ("1a", MediumOracle(variant="1A")),  # seed 42 on disk, auto-skipped
        ("1d", MediumOracle1D()),
        ("hd", MediumOracleHD()),
    ]

    summary = {}
    for label, oracle in oracles:
        finals = run_bo_for_oracle(label, oracle, SEEDS)
        summary[label] = finals

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(f"  {'oracle':<8} {'n':>3} {'mean HV':>12} {'std':>10}")
    for label, finals in summary.items():
        n = len(finals)
        if n == 0:
            continue
        mean = float(np.mean(finals))
        std = float(np.std(finals))
        print(f"  {label:<8} {n:>3} {mean:>12.6f} {std:>10.6f}")


if __name__ == "__main__":
    main()
