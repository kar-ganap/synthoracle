"""Reusable characterization module for synthetic oracles.

Computes output statistics, Sobol sensitivity indices, one-at-a-time
effect sizes, reference Pareto fronts, and timing benchmarks for any
Oracle subclass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from synthoracle.oracle import Oracle


@dataclass
class CharacterizationResult:
    """Results from characterizing an oracle."""

    output_stats: dict[str, dict[str, float]]
    sobol_first_order: npt.NDArray[np.float64]  # (n_inputs, n_outputs)
    sobol_total_order: npt.NDArray[np.float64]  # (n_inputs, n_outputs)
    oat_effects: npt.NDArray[np.float64]  # (n_inputs, n_outputs)
    pareto_front: npt.NDArray[np.float64]  # (n_pareto, n_outputs)
    pareto_inputs: npt.NDArray[np.float64]  # (n_pareto, n_inputs)
    timing_us: float  # microseconds per evaluation


def characterize(
    oracle: Oracle,
    *,
    n_samples: int = 100_000,
    n_sobol: int = 50_000,
    n_sweep: int = 200,
    n_pareto: int = 200_000,
    thresholds: dict[str, float] | None = None,
    seed: int = 42,
) -> CharacterizationResult:
    """Characterize an oracle with statistics, sensitivity, and Pareto analysis."""
    rng = np.random.default_rng(seed)
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]

    # Output statistics
    X_stats = rng.uniform(lo, hi, size=(n_samples, oracle.n_inputs))
    Y_stats = oracle.evaluate_batch(X_stats)
    output_stats = compute_output_stats(oracle, Y_stats)

    # Sobol indices
    s1, st = compute_sobol_indices(oracle, n_sobol, rng)

    # OAT effects
    oat = compute_oat_effects(oracle, n_sweep)

    # Pareto front
    X_pareto = rng.uniform(lo, hi, size=(n_pareto, oracle.n_inputs))
    Y_pareto = oracle.evaluate_batch(X_pareto)
    p_front, p_inputs = compute_pareto_front(
        oracle, X_pareto, Y_pareto, thresholds=thresholds
    )

    # Timing
    timing = _benchmark_timing(oracle, rng)

    return CharacterizationResult(
        output_stats=output_stats,
        sobol_first_order=s1,
        sobol_total_order=st,
        oat_effects=oat,
        pareto_front=p_front,
        pareto_inputs=p_inputs,
        timing_us=timing,
    )


def compute_output_stats(
    oracle: Oracle, Y: npt.NDArray[np.float64]
) -> dict[str, dict[str, float]]:
    """Compute per-output statistics from evaluated samples."""
    stats: dict[str, dict[str, float]] = {}
    for i, name in enumerate(oracle.output_names):
        yi = Y[:, i]
        stats[name] = {
            "mean": float(yi.mean()),
            "std": float(yi.std()),
            "min": float(yi.min()),
            "max": float(yi.max()),
            "range": float(yi.max() - yi.min()),
        }
    return stats


def compute_sobol_indices(
    oracle: Oracle, n: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute first-order and total-order Sobol sensitivity indices.

    Uses Saltelli (2010) estimator for first-order and Jansen (1999)
    estimator for total-order indices via the pick-freeze method.

    Returns (first_order, total_order), each shape (n_inputs, n_outputs).
    """
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    d = oracle.n_inputs
    m = oracle.n_outputs

    X_A = rng.uniform(lo, hi, size=(n, d))
    X_B = rng.uniform(lo, hi, size=(n, d))
    Y_A = oracle.evaluate_batch(X_A)
    Y_B = oracle.evaluate_batch(X_B)

    total_var = Y_A.var(axis=0)

    first_order = np.zeros((d, m), dtype=np.float64)
    total_order = np.zeros((d, m), dtype=np.float64)

    for j in range(d):
        # C_j = X_B with column j replaced by column j from X_A
        # C_j shares all columns with X_B except j (which comes from X_A)
        X_Cj = X_B.copy()
        X_Cj[:, j] = X_A[:, j]
        Y_Cj = oracle.evaluate_batch(X_Cj)

        for i in range(m):
            if total_var[i] < 1e-12:
                continue

            # Saltelli (2010) first-order estimator:
            # S_j = mean(Y_A * (Y_Cj - Y_B)) / Var(Y)
            # Y_Cj and Y_B share ~j, differ in j. Y_A shares j with Y_Cj.
            first_order[j, i] = float(
                np.mean(Y_A[:, i] * (Y_Cj[:, i] - Y_B[:, i])) / total_var[i]
            )

            # Jansen (1999) total-order estimator:
            # S_T_j = mean((Y_B - Y_Cj)^2) / (2 * Var(Y))
            # Y_B and Y_Cj differ ONLY in input j.
            total_order[j, i] = float(
                np.mean((Y_B[:, i] - Y_Cj[:, i]) ** 2) / (2.0 * total_var[i])
            )

    return first_order, total_order


def compute_oat_effects(
    oracle: Oracle, n_sweep: int = 200
) -> npt.NDArray[np.float64]:
    """One-at-a-time effect sizes: sweep each input, others at midpoint.

    Returns shape (n_inputs, n_outputs) with the range (max - min) of
    each output when sweeping each input.
    """
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    midpoint = (lo + hi) / 2.0
    d = oracle.n_inputs
    m = oracle.n_outputs

    effects = np.zeros((d, m), dtype=np.float64)
    for j in range(d):
        sweep = np.linspace(lo[j], hi[j], n_sweep)
        X_oat = np.tile(midpoint, (n_sweep, 1))
        X_oat[:, j] = sweep
        Y_oat = oracle.evaluate_batch(X_oat)
        for i in range(m):
            effects[j, i] = float(Y_oat[:, i].max() - Y_oat[:, i].min())

    return effects


def compute_pareto_front(
    oracle: Oracle,
    X: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
    thresholds: dict[str, float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the reference Pareto front from dense sampling.

    Handles maximize/minimize/threshold objectives. Threshold objectives
    are used to filter feasible points but don't participate in dominance.

    Returns (pareto_Y, pareto_X) — the non-dominated outputs and their inputs.
    """
    directions = oracle.output_directions
    output_names = oracle.output_names
    n = Y.shape[0]
    m = Y.shape[1]

    # Filter by thresholds
    mask = np.ones(n, dtype=bool)
    if thresholds:
        for name, threshold in thresholds.items():
            idx = output_names.index(name)
            mask &= Y[:, idx] >= threshold

    X_feas = X[mask]
    Y_feas = Y[mask]

    if len(Y_feas) == 0:
        return np.empty((0, m), dtype=np.float64), np.empty((0, oracle.n_inputs), dtype=np.float64)

    # Identify which objectives participate in dominance (not threshold)
    obj_indices = [i for i, d in enumerate(directions) if d != "threshold"]

    # Transform to minimization: negate "maximize" objectives
    Y_min = np.empty((len(Y_feas), len(obj_indices)), dtype=np.float64)
    for col, idx in enumerate(obj_indices):
        if directions[idx] == "maximize":
            Y_min[:, col] = -Y_feas[:, idx]
        else:  # minimize
            Y_min[:, col] = Y_feas[:, idx]

    # Non-dominated sorting
    is_pareto = _is_pareto_efficient(Y_min)

    return Y_feas[is_pareto], X_feas[is_pareto]


def _is_pareto_efficient(costs: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Find Pareto-efficient points (minimization).

    Uses iterative filtering for efficiency. For each candidate point,
    remove all points dominated by it, then move to the next.
    """
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # Check if any other efficient point dominates i
        # Point j dominates i if j <= i on all objectives and j < i on at least one
        dominated = np.all(costs[is_efficient] <= costs[i], axis=1)
        strictly = np.any(costs[is_efficient] < costs[i], axis=1)
        dominators = dominated & strictly

        if np.any(dominators):
            is_efficient[i] = False
        else:
            # Remove points dominated by i
            other_mask = is_efficient.copy()
            other_mask[i] = False
            dominated_by_i = np.all(costs[i] <= costs, axis=1) & np.any(
                costs[i] < costs, axis=1
            )
            is_efficient[dominated_by_i] = False
            is_efficient[i] = True  # Restore i

    return is_efficient


def _benchmark_timing(oracle: Oracle, rng: np.random.Generator) -> float:
    """Measure microseconds per evaluation (average of 1000 evals)."""
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(1000, oracle.n_inputs))

    start = time.perf_counter()
    oracle.evaluate_batch(X)
    elapsed = time.perf_counter() - start

    return elapsed / 1000.0 * 1e6  # convert to microseconds per eval
