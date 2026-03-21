"""Shared numpy-only optimization utilities.

Provides direction parsing, reference-point computation, non-dominated
sorting, hypervolume calculation, and Pareto-front extraction — all
without torch or botorch dependencies.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from synthoracle.oracle import Oracle


def parse_directions(
    oracle: Oracle,
) -> tuple[list[int], list[int], npt.NDArray[np.float64]]:
    """Extract objective and constraint indices from oracle directions.

    Returns
    -------
    obj_indices : list of int
        Indices of maximize/minimize outputs (the true objectives).
    constraint_indices : list of int
        Indices of threshold outputs.
    signs : ndarray of shape (n_obj,)
        +1 for maximize, -1 for minimize (to convert to all-maximize).
    """
    directions = oracle.output_directions
    obj_indices: list[int] = []
    constraint_indices: list[int] = []
    signs_list: list[float] = []
    for i, d in enumerate(directions):
        if d == "threshold":
            constraint_indices.append(i)
        else:
            obj_indices.append(i)
            signs_list.append(1.0 if d == "maximize" else -1.0)
    signs = np.array(signs_list, dtype=np.float64)
    return obj_indices, constraint_indices, signs


def compute_reference_point(
    oracle: Oracle,
    obj_indices: list[int],
    signs: npt.NDArray[np.float64],
    seed: int,
    n_samples: int = 10_000,
) -> npt.NDArray[np.float64]:
    """Compute reference point from random oracle samples.

    Samples the oracle, transforms to all-maximize space, and returns
    worst_per_objective - 0.1 * range.
    """
    rng = np.random.default_rng(seed + 1000)
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(n_samples, oracle.n_inputs))
    Y = oracle.evaluate_batch(X)
    Y_obj = Y[:, obj_indices] * signs
    worst = Y_obj.min(axis=0)
    ranges = Y_obj.max(axis=0) - Y_obj.min(axis=0)
    ref: npt.NDArray[np.float64] = worst - 0.1 * ranges
    return ref


def is_non_dominated(Y: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Find non-dominated points in all-maximize convention.

    A point j dominates point i if j >= i on all objectives and j > i
    on at least one.

    Parameters
    ----------
    Y : ndarray of shape (n_points, n_objectives)
        Points in all-maximize space.

    Returns
    -------
    mask : boolean ndarray of shape (n_points,)
        True for non-dominated points.
    """
    n = Y.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # Check if any other efficient point dominates i
        # In maximization: j dominates i if j >= i on all and j > i on at least one
        dominated = np.all(Y[is_efficient] >= Y[i], axis=1)
        strictly = np.any(Y[is_efficient] > Y[i], axis=1)
        dominators = dominated & strictly

        if np.any(dominators):
            is_efficient[i] = False
        else:
            # Remove points dominated by i
            dominated_by_i = np.all(Y[i] >= Y, axis=1) & np.any(Y[i] > Y, axis=1)
            is_efficient[dominated_by_i] = False
            is_efficient[i] = True  # Restore i
    return is_efficient


def _hypervolume_2d(
    points: npt.NDArray[np.float64],
    ref: npt.NDArray[np.float64],
) -> float:
    """Compute 2D hypervolume by sorting and sweeping.

    Assumes all points dominate the reference point.
    """
    # Sort by first objective descending
    order = np.argsort(-points[:, 0])
    sorted_pts = points[order]

    hv = 0.0
    y_max = ref[1]  # current "floor" for second objective
    for pt in sorted_pts:
        if pt[1] > y_max:
            hv += (pt[0] - ref[0]) * (pt[1] - y_max)
            y_max = pt[1]
    return hv


def _hypervolume_3d(
    points: npt.NDArray[np.float64],
    ref: npt.NDArray[np.float64],
) -> float:
    """Compute 3D hypervolume using slicing algorithm.

    Sorts by third objective descending, then computes incremental 2D
    hypervolumes for each slice.
    """
    # Sort by third objective descending
    order = np.argsort(-points[:, 2])
    sorted_pts = points[order]

    hv = 0.0
    prev_z = sorted_pts[0, 2]
    # Collect points seen so far (projected onto first 2 objectives)
    seen_2d: list[npt.NDArray[np.float64]] = []
    ref_2d = ref[:2]

    for idx in range(len(sorted_pts)):
        pt = sorted_pts[idx]
        cur_z = pt[2]

        if idx > 0 and cur_z < prev_z:
            # Compute 2D HV of accumulated points and multiply by z-slab height
            pts_2d = np.array(seen_2d, dtype=np.float64)
            mask_2d = is_non_dominated(pts_2d)
            pareto_2d = pts_2d[mask_2d]
            hv_2d = _hypervolume_2d(pareto_2d, ref_2d)
            slab_height = prev_z - cur_z
            hv += hv_2d * slab_height
            prev_z = cur_z

        seen_2d.append(pt[:2].copy())

    # Final slab from last z down to ref[2]
    if len(seen_2d) > 0 and prev_z > ref[2]:
        pts_2d = np.array(seen_2d, dtype=np.float64)
        mask_2d = is_non_dominated(pts_2d)
        pareto_2d = pts_2d[mask_2d]
        hv_2d = _hypervolume_2d(pareto_2d, ref_2d)
        slab_height = prev_z - ref[2]
        hv += hv_2d * slab_height

    return hv


def compute_hypervolume(
    Y_all: npt.NDArray[np.float64],
    obj_indices: list[int],
    signs: npt.NDArray[np.float64],
    constraint_indices: list[int],
    thresholds: dict[str, float],
    output_names: tuple[str, ...],
    ref_point: npt.NDArray[np.float64],
) -> float:
    """Compute hypervolume of non-dominated feasible points.

    Full pipeline: filter feasible, transform to all-maximize, find
    non-dominated, compute HV.

    Parameters
    ----------
    Y_all : ndarray of shape (n_points, n_outputs)
        All raw oracle outputs.
    obj_indices : list of int
        Indices of objective outputs.
    signs : ndarray of shape (n_obj,)
        +1 for maximize, -1 for minimize.
    constraint_indices : list of int
        Indices of threshold outputs.
    thresholds : dict mapping output name to threshold value.
    output_names : tuple of str
        Names of all outputs.
    ref_point : ndarray of shape (n_obj,)
        Reference point in all-maximize space.

    Returns
    -------
    float
        Hypervolume indicator. Returns 0.0 if no feasible points.
    """
    # Filter by threshold constraints
    feasible = np.ones(len(Y_all), dtype=bool)
    for idx in constraint_indices:
        name = output_names[idx]
        if name in thresholds:
            feasible &= Y_all[:, idx] >= thresholds[name]

    if not feasible.any():
        return 0.0

    # Transform objectives to all-maximize
    Y_obj = Y_all[feasible][:, obj_indices] * signs

    # Find non-dominated points
    pareto_mask = is_non_dominated(Y_obj)
    pareto_Y = Y_obj[pareto_mask]

    if len(pareto_Y) == 0:
        return 0.0

    # Filter out points that don't dominate the reference point
    dominates_ref = np.all(pareto_Y > ref_point, axis=1)
    pareto_Y = pareto_Y[dominates_ref]

    if len(pareto_Y) == 0:
        return 0.0

    n_obj = len(obj_indices)
    if n_obj == 2:
        return _hypervolume_2d(pareto_Y, ref_point)
    elif n_obj == 3:
        return _hypervolume_3d(pareto_Y, ref_point)
    else:
        raise NotImplementedError(
            f"Numpy-only hypervolume supports 2D and 3D only, got {n_obj}D. "
            "Use botorch for higher dimensions."
        )


def extract_pareto_front(
    X_all: npt.NDArray[np.float64],
    Y_all: npt.NDArray[np.float64],
    oracle: Oracle,
    thresholds: dict[str, float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract Pareto front from evaluated data.

    Filters by threshold constraints, finds non-dominated points in
    all-maximize objective space, and returns results in raw space.

    Parameters
    ----------
    X_all : ndarray of shape (n_points, n_inputs)
        All evaluated inputs.
    Y_all : ndarray of shape (n_points, n_outputs)
        All raw oracle outputs.
    oracle : Oracle
        The oracle (used for direction/name metadata).
    thresholds : dict, optional
        Threshold constraints, e.g. {"Y3": 0.4}.

    Returns
    -------
    pareto_X : ndarray of shape (n_pareto, n_inputs)
        Pareto-optimal inputs in raw space.
    pareto_Y : ndarray of shape (n_pareto, n_outputs)
        Pareto-optimal outputs in raw space.
    """
    if thresholds is None:
        thresholds = {}

    obj_indices, constraint_indices, signs = parse_directions(oracle)

    # Filter by threshold constraints
    feasible = np.ones(len(Y_all), dtype=bool)
    for idx in constraint_indices:
        name = oracle.output_names[idx]
        if name in thresholds:
            feasible &= Y_all[:, idx] >= thresholds[name]

    if not feasible.any():
        return (
            np.empty((0, oracle.n_inputs), dtype=np.float64),
            np.empty((0, oracle.n_outputs), dtype=np.float64),
        )

    # Transform objectives to all-maximize
    Y_obj_feas = Y_all[feasible][:, obj_indices] * signs

    # Find non-dominated
    pareto_mask = is_non_dominated(Y_obj_feas)
    feas_indices = np.where(feasible)[0]
    pareto_global = feas_indices[pareto_mask]

    pareto_X: npt.NDArray[np.float64] = X_all[pareto_global]
    pareto_Y: npt.NDArray[np.float64] = Y_all[pareto_global]
    return pareto_X, pareto_Y
