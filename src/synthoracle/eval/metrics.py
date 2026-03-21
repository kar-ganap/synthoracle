"""Comparison metrics for optimization methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


class OptResult(Protocol):
    """Common interface for optimization results (BOResult, VRResult)."""

    X: npt.NDArray[np.float64]
    Y: npt.NDArray[np.float64]
    hypervolumes: list[float]
    pareto_X: npt.NDArray[np.float64]
    pareto_Y: npt.NDArray[np.float64]
    reference_point: npt.NDArray[np.float64]
    seed: int
    n_initial: int
    total_seconds: float


@dataclass
class ComparisonMetrics:
    """Metrics comparing multiple optimization methods across seeds."""

    method_names: list[str]
    hv_curves: dict[str, npt.NDArray[np.float64]]
    """method → (n_seeds, n_evals) HV matrix."""

    final_hv: dict[str, npt.NDArray[np.float64]]
    """method → (n_seeds,) final HV per seed."""

    hv_mean: dict[str, npt.NDArray[np.float64]]
    """method → (n_evals,) mean HV curve."""

    hv_std: dict[str, npt.NDArray[np.float64]]
    """method → (n_evals,) std HV curve."""

    wall_times: dict[str, npt.NDArray[np.float64]]
    """method → (n_seeds,) seconds per run."""

    pareto_sizes: dict[str, npt.NDArray[np.float64]]
    """method → (n_seeds,) number of Pareto points."""


def compute_comparison(
    results: dict[str, list[OptResult]],
) -> ComparisonMetrics:
    """Compute comparison metrics from multi-seed results.

    Parameters
    ----------
    results : dict
        Maps method name → list of OptResult (one per seed).

    Returns
    -------
    ComparisonMetrics
        Aggregated metrics with mean/std curves.
    """
    method_names = list(results.keys())
    hv_curves: dict[str, npt.NDArray[np.float64]] = {}
    final_hv: dict[str, npt.NDArray[np.float64]] = {}
    hv_mean: dict[str, npt.NDArray[np.float64]] = {}
    hv_std: dict[str, npt.NDArray[np.float64]] = {}
    wall_times: dict[str, npt.NDArray[np.float64]] = {}
    pareto_sizes: dict[str, npt.NDArray[np.float64]] = {}

    for name, result_list in results.items():
        n_seeds = len(result_list)
        n_evals = len(result_list[0].hypervolumes)

        # HV matrix: (n_seeds, n_evals)
        hv_matrix = np.zeros((n_seeds, n_evals), dtype=np.float64)
        for i, r in enumerate(result_list):
            hvs = r.hypervolumes
            # Pad or truncate to n_evals (in case seeds produce different lengths)
            n = min(len(hvs), n_evals)
            hv_matrix[i, :n] = hvs[:n]
            if n < n_evals:
                hv_matrix[i, n:] = hvs[-1]  # pad with final value

        hv_curves[name] = hv_matrix
        final_hv[name] = hv_matrix[:, -1]
        hv_mean[name] = hv_matrix.mean(axis=0)
        hv_std[name] = hv_matrix.std(axis=0)

        wall_times[name] = np.array(
            [r.total_seconds for r in result_list], dtype=np.float64,
        )
        pareto_sizes[name] = np.array(
            [r.pareto_Y.shape[0] for r in result_list], dtype=np.float64,
        )

    return ComparisonMetrics(
        method_names=method_names,
        hv_curves=hv_curves,
        final_hv=final_hv,
        hv_mean=hv_mean,
        hv_std=hv_std,
        wall_times=wall_times,
        pareto_sizes=pareto_sizes,
    )


def print_summary(metrics: ComparisonMetrics) -> None:
    """Print a formatted summary table."""
    print(f"\n{'=' * 70}")
    print(f"  {'Method':<12} {'Final HV':>14} {'Wall Time (s)':>14} {'Pareto Pts':>12}")
    print(f"  {'-' * 66}")
    for name in metrics.method_names:
        fhv = metrics.final_hv[name]
        wt = metrics.wall_times[name]
        ps = metrics.pareto_sizes[name]
        print(
            f"  {name:<12} "
            f"{fhv.mean():>6.4f} ± {fhv.std():>5.4f} "
            f"{wt.mean():>7.1f} ± {wt.std():>5.1f} "
            f"{ps.mean():>6.1f} ± {ps.std():>4.1f}"
        )
    print(f"{'=' * 70}")
