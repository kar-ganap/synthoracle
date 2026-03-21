"""Multi-seed experiment runner for optimization methods."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracle import Oracle


def run_experiment(
    method: Literal["bo", "vr"],
    oracle: Oracle,
    *,
    seeds: list[int],
    n_iterations: int = 42,
    thresholds: dict[str, float] | None = None,
    reference_point: npt.NDArray[np.float64] | None = None,
    vr_model: str = "claude-sonnet-4-6",
) -> list[Any]:
    """Run an optimization method across multiple seeds.

    Computes a shared reference point so HV values are comparable
    across seeds and methods.

    Parameters
    ----------
    method : "bo" or "vr"
        Which optimizer to run.
    oracle : Oracle
        The oracle to optimize.
    seeds : list of int
        Random seeds for each run.
    n_iterations : int
        Number of optimization iterations per run.
    thresholds : dict, optional
        Threshold constraints.
    reference_point : ndarray, optional
        Shared reference point. Auto-computed if None.
    vr_model : str
        Anthropic model for VR agent.

    Returns
    -------
    list
        One BOResult or VRResult per seed.
    """
    if thresholds is None:
        thresholds = {}

    # Compute shared reference point
    if reference_point is None:
        obj_indices, _, signs = parse_directions(oracle)
        reference_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    results: list[Any] = []

    for seed in seeds:
        print(f"  Running {method.upper()} seed={seed}...")
        if method == "bo":
            from synthoracle.baselines.bo import run_bo

            r: Any = run_bo(
                oracle,
                n_iterations=n_iterations,
                seed=seed,
                reference_point=reference_point,
                thresholds=thresholds if thresholds else None,
            )
        elif method == "vr":
            from synthoracle.agents.vr import run_vr

            r = run_vr(
                oracle,
                n_iterations=n_iterations,
                seed=seed,
                reference_point=reference_point,
                thresholds=thresholds if thresholds else None,
                model=vr_model,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        results.append(r)

    return results
