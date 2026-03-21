"""Tests for the BO baseline module."""

from __future__ import annotations

import numpy as np
import pytest

botorch = pytest.importorskip("botorch")
torch = pytest.importorskip("torch")

from synthoracle.baselines.bo import (
    BOResult,
    _build_constraint_callables,
    _compute_hypervolume,
    _compute_reference_point,
    _generate_initial_design,
    _parse_directions,
    run_bo,
)
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.simple import SimpleOracle


# ---------------------------------------------------------------------------
# TestDirectionHandling
# ---------------------------------------------------------------------------

class TestDirectionHandling:
    def test_simple_has_no_constraints(self) -> None:
        oracle = SimpleOracle()
        obj_indices, constraint_indices, signs = _parse_directions(oracle)
        assert len(constraint_indices) == 0
        assert len(obj_indices) == 2

    def test_medium_has_one_constraint(self) -> None:
        oracle = MediumOracle()
        obj_indices, constraint_indices, signs = _parse_directions(oracle)
        assert len(constraint_indices) == 1
        assert constraint_indices[0] == 2  # Y3 is threshold
        assert len(obj_indices) == 3  # Y1, Y2, Y4

    def test_signs_correct(self) -> None:
        """maximize → +1, minimize → -1."""
        oracle = SimpleOracle()
        _, _, signs = _parse_directions(oracle)
        # SimpleOracle: (maximize, minimize)
        np.testing.assert_array_equal(signs, [1.0, -1.0])

    def test_medium_signs_correct(self) -> None:
        oracle = MediumOracle()
        _, _, signs = _parse_directions(oracle)
        # Medium: (maximize, minimize, threshold, maximize) → obj signs: [+1, -1, +1]
        np.testing.assert_array_equal(signs, [1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# TestInitialDesign
# ---------------------------------------------------------------------------

class TestInitialDesign:
    def test_sobol_within_bounds(self) -> None:
        oracle = SimpleOracle()
        X = _generate_initial_design(oracle, n_initial=20, seed=42)
        lo = oracle.bounds[:, 0]
        hi = oracle.bounds[:, 1]
        assert np.all(X >= lo)
        assert np.all(X <= hi)

    def test_sobol_correct_shape(self) -> None:
        oracle = MediumOracle()
        X = _generate_initial_design(oracle, n_initial=12, seed=42)
        assert X.shape == (12, 6)

    def test_sobol_deterministic(self) -> None:
        oracle = SimpleOracle()
        X1 = _generate_initial_design(oracle, n_initial=10, seed=42)
        X2 = _generate_initial_design(oracle, n_initial=10, seed=42)
        np.testing.assert_array_equal(X1, X2)

    def test_initial_count_default(self) -> None:
        """Default n_initial should be 2 * n_inputs."""
        oracle = SimpleOracle()
        # Not testing run_bo default directly, just verifying the convention
        assert 2 * oracle.n_inputs == 8
        oracle_m = MediumOracle()
        assert 2 * oracle_m.n_inputs == 12


# ---------------------------------------------------------------------------
# TestReferencePoint
# ---------------------------------------------------------------------------

class TestReferencePoint:
    def test_reference_point_below_all_data(self) -> None:
        oracle = SimpleOracle()
        obj_indices, _, signs = _parse_directions(oracle)
        ref = _compute_reference_point(oracle, obj_indices, signs, seed=42)
        # Sample some points and verify ref is below all of them
        rng = np.random.default_rng(99)
        lo, hi = oracle.bounds[:, 0], oracle.bounds[:, 1]
        X = rng.uniform(lo, hi, size=(1000, oracle.n_inputs))
        Y = oracle.evaluate_batch(X)
        Y_obj = Y[:, obj_indices] * signs
        assert np.all(ref < Y_obj.min(axis=0))

    def test_reference_point_deterministic(self) -> None:
        oracle = SimpleOracle()
        obj_indices, _, signs = _parse_directions(oracle)
        r1 = _compute_reference_point(oracle, obj_indices, signs, seed=42)
        r2 = _compute_reference_point(oracle, obj_indices, signs, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_reference_point_shape(self) -> None:
        oracle = MediumOracle()
        obj_indices, _, signs = _parse_directions(oracle)
        ref = _compute_reference_point(oracle, obj_indices, signs, seed=42)
        assert ref.shape == (3,)  # 3 objectives (Y1, Y2, Y4)


# ---------------------------------------------------------------------------
# TestHypervolume
# ---------------------------------------------------------------------------

class TestHypervolume:
    def test_hypervolume_positive(self) -> None:
        """HV should be positive with some data."""
        oracle = SimpleOracle()
        obj_indices, constraint_indices, signs = _parse_directions(oracle)
        ref = _compute_reference_point(oracle, obj_indices, signs, seed=42)
        X = _generate_initial_design(oracle, n_initial=20, seed=42)
        Y = oracle.evaluate_batch(X)
        hv = _compute_hypervolume(
            Y, obj_indices, signs, constraint_indices,
            {}, oracle.output_names, ref,
        )
        assert hv > 0

    def test_hypervolume_nondecreasing(self) -> None:
        """HV of cumulative data can only increase."""
        oracle = SimpleOracle()
        obj_indices, constraint_indices, signs = _parse_directions(oracle)
        ref = _compute_reference_point(oracle, obj_indices, signs, seed=42)
        X = _generate_initial_design(oracle, n_initial=30, seed=42)
        Y = oracle.evaluate_batch(X)

        hvs = []
        for k in range(1, len(Y) + 1):
            hv = _compute_hypervolume(
                Y[:k], obj_indices, signs, constraint_indices,
                {}, oracle.output_names, ref,
            )
            hvs.append(hv)

        for i in range(1, len(hvs)):
            assert hvs[i] >= hvs[i - 1] - 1e-10

    def test_hypervolume_constraint_filtering(self) -> None:
        """Infeasible points should not contribute to HV."""
        oracle = MediumOracle()
        obj_indices, constraint_indices, signs = _parse_directions(oracle)
        ref = _compute_reference_point(oracle, obj_indices, signs, seed=42)
        X = _generate_initial_design(oracle, n_initial=100, seed=42)
        Y = oracle.evaluate_batch(X)

        # HV with constraint should be <= HV without constraint
        hv_no_constraint = _compute_hypervolume(
            Y, obj_indices, signs, [], {}, oracle.output_names, ref,
        )
        hv_with_constraint = _compute_hypervolume(
            Y, obj_indices, signs, constraint_indices,
            {"Y3": 0.4}, oracle.output_names, ref,
        )
        assert hv_with_constraint <= hv_no_constraint + 1e-10


# ---------------------------------------------------------------------------
# TestConstraintCallable
# ---------------------------------------------------------------------------

class TestConstraintCallable:
    def test_feasible_returns_negative(self) -> None:
        oracle = MediumOracle()
        _, constraint_indices, _ = _parse_directions(oracle)
        callables = _build_constraint_callables(
            constraint_indices, {"Y3": 0.4}, oracle.output_names,
        )
        assert len(callables) == 1
        # Simulate posterior samples: shape (1, 1, 4) where Y3=0.5
        samples = torch.zeros(1, 1, 4, dtype=torch.double)
        samples[..., 2] = 0.5  # Y3 = 0.5 > 0.4 → feasible
        result = callables[0](samples)
        assert result.item() < 0  # negative = feasible

    def test_infeasible_returns_positive(self) -> None:
        oracle = MediumOracle()
        _, constraint_indices, _ = _parse_directions(oracle)
        callables = _build_constraint_callables(
            constraint_indices, {"Y3": 0.4}, oracle.output_names,
        )
        samples = torch.zeros(1, 1, 4, dtype=torch.double)
        samples[..., 2] = 0.3  # Y3 = 0.3 < 0.4 → infeasible
        result = callables[0](samples)
        assert result.item() > 0  # positive = infeasible


# ---------------------------------------------------------------------------
# TestRunBO (slow — require GP fitting)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRunBO:
    def test_bo_completes_simple(self) -> None:
        result = run_bo(SimpleOracle(), n_iterations=3, seed=42)
        assert isinstance(result, BOResult)

    def test_bo_completes_medium(self) -> None:
        result = run_bo(
            MediumOracle(), n_iterations=3, seed=42,
            thresholds={"Y3": 0.4},
        )
        assert isinstance(result, BOResult)

    def test_bo_result_shapes(self) -> None:
        oracle = SimpleOracle()
        result = run_bo(oracle, n_initial=8, n_iterations=3, seed=42)
        n_total = 8 + 3
        assert result.X.shape == (n_total, oracle.n_inputs)
        assert result.Y.shape == (n_total, oracle.n_outputs)
        assert len(result.hypervolumes) == n_total
        assert result.reference_point.shape == (2,)  # 2 objectives
        assert result.pareto_X.shape[1] == oracle.n_inputs
        assert result.pareto_Y.shape[1] == oracle.n_outputs
        assert result.pareto_X.shape[0] == result.pareto_Y.shape[0]
        assert result.n_initial == 8
        assert result.n_bo_iterations == 3
        assert result.total_seconds > 0

    def test_bo_deterministic(self) -> None:
        oracle = SimpleOracle()
        r1 = run_bo(oracle, n_iterations=3, seed=42)
        r2 = run_bo(oracle, n_iterations=3, seed=42)
        np.testing.assert_array_equal(r1.X, r2.X)
        np.testing.assert_array_equal(r1.Y, r2.Y)
        assert r1.hypervolumes == r2.hypervolumes

    def test_bo_hypervolume_improves(self) -> None:
        """HV at end should be >= HV at start (BO should help)."""
        result = run_bo(SimpleOracle(), n_iterations=10, seed=42)
        assert result.hypervolumes[-1] >= result.hypervolumes[0]

    def test_bo_simple_sanity(self) -> None:
        """Within 50 evals, BO should find a reasonable solution."""
        oracle = SimpleOracle()
        result = run_bo(oracle, n_initial=8, n_iterations=42, seed=42)
        # Best Y1 found should be > 0.7 (max possible ~1.18)
        assert result.Y[:, 0].max() > 0.7
        # HV should be meaningfully above zero
        assert result.hypervolumes[-1] > 0

    def test_bo_medium_has_feasible_pareto(self) -> None:
        """Some Pareto points should satisfy the Y3 > 0.4 constraint."""
        result = run_bo(
            MediumOracle(), n_iterations=5, seed=42,
            thresholds={"Y3": 0.4},
        )
        if result.pareto_Y.shape[0] > 0:
            assert np.any(result.pareto_Y[:, 2] >= 0.4)
