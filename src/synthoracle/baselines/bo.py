"""Multi-objective Bayesian optimization baseline using BoTorch qNEHVI.

Provides run_bo() which takes any Oracle and returns a BOResult with
hypervolume convergence tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from synthoracle.oracle import Oracle


@dataclass
class BOResult:
    """Results from a multi-objective Bayesian optimization run."""

    X: npt.NDArray[np.float64]
    """All evaluated inputs, shape (n_total, n_inputs)."""

    Y: npt.NDArray[np.float64]
    """All raw oracle outputs, shape (n_total, n_outputs)."""

    hypervolumes: list[float]
    """Hypervolume after each evaluation, length n_total."""

    pareto_X: npt.NDArray[np.float64]
    """Pareto-optimal inputs at termination, shape (n_pareto, n_inputs)."""

    pareto_Y: npt.NDArray[np.float64]
    """Pareto-optimal raw outputs at termination, shape (n_pareto, n_outputs)."""

    reference_point: npt.NDArray[np.float64]
    """Reference point in all-maximize objective space, shape (n_obj,)."""

    seed: int
    n_initial: int
    n_bo_iterations: int
    total_seconds: float


def _parse_directions(
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


def _compute_reference_point(
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


def _generate_initial_design(
    oracle: Oracle,
    n_initial: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Generate Sobol quasi-random initial design within oracle bounds.

    Returns
    -------
    X : ndarray of shape (n_initial, n_inputs)
    """
    sobol = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
        dimension=oracle.n_inputs, scramble=True, seed=seed
    )
    X_unit = sobol.draw(n_initial).numpy().astype(np.float64)
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    X: npt.NDArray[np.float64] = lo + X_unit * (hi - lo)
    return X


def _compute_hypervolume(
    Y_all: npt.NDArray[np.float64],
    obj_indices: list[int],
    signs: npt.NDArray[np.float64],
    constraint_indices: list[int],
    thresholds: dict[str, float],
    output_names: tuple[str, ...],
    ref_point: npt.NDArray[np.float64],
) -> float:
    """Compute hypervolume of non-dominated feasible points.

    Returns 0.0 if no feasible points exist.
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
    Y_tensor = torch.tensor(Y_obj, dtype=torch.double)
    ref_tensor = torch.tensor(ref_point, dtype=torch.double)

    # Find non-dominated points
    pareto_mask = is_non_dominated(Y_tensor)
    pareto_Y = Y_tensor[pareto_mask]

    if len(pareto_Y) == 0:
        return 0.0

    hv = Hypervolume(ref_point=ref_tensor)
    return float(hv.compute(pareto_Y))


def _build_constraint_callables(
    constraint_indices: list[int],
    thresholds: dict[str, float],
    output_names: tuple[str, ...],
) -> list[Callable[[torch.Tensor], torch.Tensor]]:
    """Build constraint callables for qNEHVI.

    Each callable takes posterior samples of shape (..., q, m) where m is
    the total number of modeled outputs, and returns (..., q) where
    negative values indicate feasibility.
    """
    callables: list[Callable[[torch.Tensor], torch.Tensor]] = []
    for idx in constraint_indices:
        name = output_names[idx]
        if name in thresholds:
            thresh = thresholds[name]
            # Capture idx and thresh in closure
            def _constraint(
                samples: torch.Tensor, _idx: int = idx, _thresh: float = thresh
            ) -> torch.Tensor:
                # samples[..., _idx] is the predicted value
                # negative means feasible (constraint satisfied)
                return -1.0 * (samples[..., _idx] - _thresh)

            callables.append(_constraint)
    return callables


def run_bo(
    oracle: Oracle,
    *,
    n_initial: int | None = None,
    n_iterations: int = 30,
    seed: int = 42,
    reference_point: npt.NDArray[np.float64] | None = None,
    thresholds: dict[str, float] | None = None,
) -> BOResult:
    """Run multi-objective Bayesian optimization on an oracle.

    Parameters
    ----------
    oracle : Oracle
        The oracle to optimize.
    n_initial : int, optional
        Number of initial Sobol points. Defaults to 2 * n_inputs.
    n_iterations : int
        Number of BO iterations after initial design.
    seed : int
        Random seed for reproducibility.
    reference_point : ndarray, optional
        Reference point in all-maximize space. Auto-computed if None.
    thresholds : dict, optional
        Threshold constraints, e.g. {"Y3": 0.4}.
    """
    t0 = time.monotonic()
    torch.manual_seed(seed)

    if n_initial is None:
        n_initial = 2 * oracle.n_inputs
    if thresholds is None:
        thresholds = {}

    obj_indices, constraint_indices, signs = _parse_directions(oracle)

    if reference_point is None:
        reference_point = _compute_reference_point(oracle, obj_indices, signs, seed)

    # Initial design
    X_all = _generate_initial_design(oracle, n_initial, seed)
    Y_all = oracle.evaluate_batch(X_all)

    # Track hypervolumes
    hypervolumes: list[float] = []
    for k in range(1, n_initial + 1):
        hv = _compute_hypervolume(
            Y_all[:k], obj_indices, signs, constraint_indices,
            thresholds, oracle.output_names, reference_point,
        )
        hypervolumes.append(hv)

    # Bounds in normalized [0, 1] space for acquisition optimization
    bounds_norm = torch.stack([
        torch.zeros(oracle.n_inputs, dtype=torch.double),
        torch.ones(oracle.n_inputs, dtype=torch.double),
    ])
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]

    # Build constraint callables
    constraint_callables = _build_constraint_callables(
        constraint_indices, thresholds, oracle.output_names,
    )

    # BO loop
    for _step in range(n_iterations):
        # Prepare training data as torch tensors
        # Normalize X to [0, 1]
        train_X = torch.tensor(
            (X_all - lo) / (hi - lo), dtype=torch.double
        )

        # Transform Y: negate minimize objectives, keep thresholds as-is
        Y_train = Y_all.copy()
        for j, oi in enumerate(obj_indices):
            if signs[j] < 0:
                Y_train[:, oi] *= -1.0
        train_Y = torch.tensor(Y_train, dtype=torch.double)

        # Fit ModelListGP (one GP per output)
        models = []
        for i in range(oracle.n_outputs):
            gp = SingleTaskGP(
                train_X, train_Y[:, i : i + 1],
                outcome_transform=Standardize(m=1),
            )
            models.append(gp)
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Map from model output indices to objective indices
        # Model outputs: all oracle outputs (in transformed space)
        # Objectives: obj_indices within the model outputs
        obj_idx_tensor = torch.tensor(obj_indices, dtype=torch.long)

        # Build qNEHVI
        ref_tensor = torch.tensor(reference_point, dtype=torch.double)
        acqf_kwargs: dict[str, object] = {
            "model": model,
            "ref_point": ref_tensor,
            "X_baseline": train_X,
            "prune_baseline": True,
        }
        if constraint_callables:
            acqf_kwargs["constraints"] = constraint_callables
        # Use objective to select which model outputs are objectives
        if len(obj_indices) < oracle.n_outputs:
            from botorch.acquisition.multi_objective.objective import (
                MCMultiOutputObjective,
            )

            class _SelectObjectives(MCMultiOutputObjective):  # type: ignore[misc]
                def forward(
                    self, samples: torch.Tensor, X: torch.Tensor | None = None
                ) -> torch.Tensor:
                    return samples[..., obj_idx_tensor]

            acqf_kwargs["objective"] = _SelectObjectives()

        acqf = qLogNoisyExpectedHypervolumeImprovement(**acqf_kwargs)

        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds_norm,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        # Unnormalize and evaluate
        x_new_norm = candidates.detach().numpy().astype(np.float64).squeeze(0)
        x_new: npt.NDArray[np.float64] = lo + x_new_norm * (hi - lo)
        y_new = oracle.evaluate(x_new)

        X_all = np.vstack([X_all, x_new[np.newaxis, :]])
        Y_all = np.vstack([Y_all, y_new[np.newaxis, :]])

        hv = _compute_hypervolume(
            Y_all, obj_indices, signs, constraint_indices,
            thresholds, oracle.output_names, reference_point,
        )
        hypervolumes.append(hv)

    # Extract final Pareto front
    feasible = np.ones(len(Y_all), dtype=bool)
    for idx in constraint_indices:
        name = oracle.output_names[idx]
        if name in thresholds:
            feasible &= Y_all[:, idx] >= thresholds[name]

    if feasible.any():
        Y_obj_feas = Y_all[feasible][:, obj_indices] * signs
        Y_tensor = torch.tensor(Y_obj_feas, dtype=torch.double)
        pareto_mask = is_non_dominated(Y_tensor).numpy()
        feas_indices = np.where(feasible)[0]
        pareto_global = feas_indices[pareto_mask]
        pareto_X = X_all[pareto_global]
        pareto_Y = Y_all[pareto_global]
    else:
        pareto_X = np.empty((0, oracle.n_inputs), dtype=np.float64)
        pareto_Y = np.empty((0, oracle.n_outputs), dtype=np.float64)

    total_seconds = time.monotonic() - t0

    return BOResult(
        X=X_all,
        Y=Y_all,
        hypervolumes=hypervolumes,
        pareto_X=pareto_X,
        pareto_Y=pareto_Y,
        reference_point=reference_point,
        seed=seed,
        n_initial=n_initial,
        n_bo_iterations=n_iterations,
        total_seconds=total_seconds,
    )
