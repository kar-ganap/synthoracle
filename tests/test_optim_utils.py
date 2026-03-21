"""Tests for shared optimization utilities (numpy-only)."""

from __future__ import annotations

import numpy as np
import pytest

from synthoracle.optim_utils import (
    compute_hypervolume,
    compute_reference_point,
    extract_pareto_front,
    is_non_dominated,
    parse_directions,
)
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.simple import SimpleOracle


@pytest.fixture
def simple_oracle() -> SimpleOracle:
    return SimpleOracle()


@pytest.fixture
def medium_oracle() -> MediumOracle:
    return MediumOracle()


# ---------------------------------------------------------------------------
# TestParseDirections
# ---------------------------------------------------------------------------


class TestParseDirections:
    """Tests for parse_directions."""

    def test_simple_no_constraints(self, simple_oracle: SimpleOracle) -> None:
        """SimpleOracle: 2 objectives, 0 constraints."""
        obj_idx, con_idx, _signs = parse_directions(simple_oracle)
        assert len(obj_idx) == 2
        assert len(con_idx) == 0
        assert obj_idx == [0, 1]

    def test_medium_one_constraint(self, medium_oracle: MediumOracle) -> None:
        """MediumOracle: 3 objectives, 1 constraint (Y3 at index 2)."""
        obj_idx, con_idx, _signs = parse_directions(medium_oracle)
        assert len(obj_idx) == 3
        assert len(con_idx) == 1
        assert con_idx == [2]
        assert obj_idx == [0, 1, 3]

    def test_signs_simple(self, simple_oracle: SimpleOracle) -> None:
        """SimpleOracle signs: [+1, -1] for (maximize, minimize)."""
        _obj, _con, signs = parse_directions(simple_oracle)
        np.testing.assert_array_equal(signs, [1.0, -1.0])

    def test_signs_medium(self, medium_oracle: MediumOracle) -> None:
        """MediumOracle signs: [+1, -1, +1] for (max, min, max)."""
        _obj, _con, signs = parse_directions(medium_oracle)
        np.testing.assert_array_equal(signs, [1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# TestComputeReferencePoint
# ---------------------------------------------------------------------------


class TestComputeReferencePoint:
    """Tests for compute_reference_point."""

    def test_below_all_data(self, simple_oracle: SimpleOracle) -> None:
        """Reference point must be below all observed data in all-maximize space."""
        obj_idx, _con_idx, signs = parse_directions(simple_oracle)
        ref = compute_reference_point(simple_oracle, obj_idx, signs, seed=42, n_samples=1000)

        # Sample some data and verify ref is below all in all-maximize space
        rng = np.random.default_rng(99)
        lo = simple_oracle.bounds[:, 0]
        hi = simple_oracle.bounds[:, 1]
        X = rng.uniform(lo, hi, size=(1000, simple_oracle.n_inputs))
        Y = simple_oracle.evaluate_batch(X)
        Y_obj = Y[:, obj_idx] * signs

        # ref should be below the minimum on each objective
        for j in range(len(obj_idx)):
            assert ref[j] < Y_obj[:, j].min()

    def test_deterministic(self, simple_oracle: SimpleOracle) -> None:
        """Same seed produces same reference point."""
        obj_idx, _con_idx, signs = parse_directions(simple_oracle)
        ref1 = compute_reference_point(simple_oracle, obj_idx, signs, seed=123)
        ref2 = compute_reference_point(simple_oracle, obj_idx, signs, seed=123)
        np.testing.assert_array_equal(ref1, ref2)

    def test_shape(self, medium_oracle: MediumOracle) -> None:
        """MediumOracle: reference point has shape (3,) for 3 objectives."""
        obj_idx, _con_idx, signs = parse_directions(medium_oracle)
        ref = compute_reference_point(medium_oracle, obj_idx, signs, seed=42)
        assert ref.shape == (3,)


# ---------------------------------------------------------------------------
# TestIsNonDominated
# ---------------------------------------------------------------------------


class TestIsNonDominated:
    """Tests for is_non_dominated (all-maximize convention)."""

    def test_known_pareto_2d(self) -> None:
        """Hand-crafted 2D points with known Pareto set."""
        # In all-maximize space:
        # [3, 1] and [1, 3] are non-dominated
        # [2, 2] is dominated by neither (it's also non-dominated)
        # [1, 1] is dominated by [2, 2] or [3, 1]
        Y = np.array([
            [3.0, 1.0],
            [1.0, 3.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ])
        mask = is_non_dominated(Y)
        # [3,1], [1,3], [2,2] are non-dominated; [1,1] is dominated
        assert mask[0]  # [3, 1]
        assert mask[1]  # [1, 3]
        assert mask[2]  # [2, 2]
        assert not mask[3]  # [1, 1] dominated by [2, 2]

    def test_single_point(self) -> None:
        """A single point is always non-dominated."""
        Y = np.array([[5.0, 3.0]])
        mask = is_non_dominated(Y)
        assert mask[0]

    def test_all_same(self) -> None:
        """Identical points: all non-dominated (no strict domination)."""
        Y = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ])
        mask = is_non_dominated(Y)
        assert mask.all()


# ---------------------------------------------------------------------------
# TestComputeHypervolume
# ---------------------------------------------------------------------------


class TestComputeHypervolume:
    """Tests for compute_hypervolume."""

    def test_2d_known_value_single_point(self) -> None:
        """Single point at origin with ref=[-1,-1]: HV = 1.0."""
        # 2 objectives, both maximize, no constraints
        Y_all = np.array([[0.0, 0.0]])
        obj_indices = [0, 1]
        signs = np.array([1.0, 1.0])
        ref = np.array([-1.0, -1.0])
        hv = compute_hypervolume(
            Y_all, obj_indices, signs,
            constraint_indices=[], thresholds={},
            output_names=("O1", "O2"), ref_point=ref,
        )
        assert abs(hv - 1.0) < 1e-10

    def test_2d_known_value_two_points(self) -> None:
        """Two points with known analytical HV.

        ref=[0, 0], points=[[1, 2], [2, 1]]
        Non-dominated in all-max: both points.
        HV = area of L-shaped region:
          Rectangle from (0,0) to (1,2): 1*2 = 2
          Rectangle from (1,0) to (2,1): 1*1 = 1
          Total = 3
        """
        Y_all = np.array([[1.0, 2.0], [2.0, 1.0]])
        obj_indices = [0, 1]
        signs = np.array([1.0, 1.0])
        ref = np.array([0.0, 0.0])
        hv = compute_hypervolume(
            Y_all, obj_indices, signs,
            constraint_indices=[], thresholds={},
            output_names=("O1", "O2"), ref_point=ref,
        )
        assert abs(hv - 3.0) < 1e-10

    def test_3d_known_value(self) -> None:
        """Single point in 3D with known HV.

        ref=[0,0,0], point=[1,1,1]: HV = 1*1*1 = 1.0
        """
        Y_all = np.array([[1.0, 1.0, 1.0]])
        obj_indices = [0, 1, 2]
        signs = np.array([1.0, 1.0, 1.0])
        ref = np.array([0.0, 0.0, 0.0])
        hv = compute_hypervolume(
            Y_all, obj_indices, signs,
            constraint_indices=[], thresholds={},
            output_names=("O1", "O2", "O3"), ref_point=ref,
        )
        assert abs(hv - 1.0) < 1e-10

    def test_returns_zero_no_feasible(self) -> None:
        """All points infeasible: HV = 0."""
        # Constraint on column 2 with threshold 0.5, all values below
        Y_all = np.array([
            [1.0, 2.0, 0.1],
            [2.0, 1.0, 0.2],
        ])
        obj_indices = [0, 1]
        signs = np.array([1.0, 1.0])
        ref = np.array([0.0, 0.0])
        hv = compute_hypervolume(
            Y_all, obj_indices, signs,
            constraint_indices=[2], thresholds={"C1": 0.5},
            output_names=("O1", "O2", "C1"), ref_point=ref,
        )
        assert hv == 0.0

    def test_constraint_filtering(self, medium_oracle: MediumOracle) -> None:
        """Threshold Y3 > 0.4 filters out some points."""
        rng = np.random.default_rng(42)
        lo = medium_oracle.bounds[:, 0]
        hi = medium_oracle.bounds[:, 1]
        X = rng.uniform(lo, hi, size=(200, medium_oracle.n_inputs))
        Y = medium_oracle.evaluate_batch(X)

        obj_idx, con_idx, signs = parse_directions(medium_oracle)
        ref = compute_reference_point(medium_oracle, obj_idx, signs, seed=42)

        # With threshold
        hv_constrained = compute_hypervolume(
            Y, obj_idx, signs, con_idx,
            thresholds={"Y3": 0.4},
            output_names=medium_oracle.output_names,
            ref_point=ref,
        )
        # Without threshold
        hv_unconstrained = compute_hypervolume(
            Y, obj_idx, signs, constraint_indices=[],
            thresholds={},
            output_names=medium_oracle.output_names,
            ref_point=ref,
        )
        # Constrained HV <= unconstrained HV (fewer feasible points)
        assert hv_constrained <= hv_unconstrained + 1e-12

    def test_nondecreasing(self) -> None:
        """Adding points can only increase (or maintain) HV."""
        rng = np.random.default_rng(7)
        # Generate random 2D points in all-maximize space
        Y_all = rng.uniform(0.5, 2.0, size=(20, 2))
        obj_indices = [0, 1]
        signs = np.array([1.0, 1.0])
        ref = np.array([0.0, 0.0])

        prev_hv = 0.0
        for k in range(1, len(Y_all) + 1):
            hv = compute_hypervolume(
                Y_all[:k], obj_indices, signs,
                constraint_indices=[], thresholds={},
                output_names=("O1", "O2"), ref_point=ref,
            )
            assert hv >= prev_hv - 1e-12
            prev_hv = hv


# ---------------------------------------------------------------------------
# TestExtractParetoFront
# ---------------------------------------------------------------------------


class TestExtractParetoFront:
    """Tests for extract_pareto_front."""

    def test_simple_oracle_pareto(self, simple_oracle: SimpleOracle) -> None:
        """Pareto front is non-empty with correct shapes."""
        rng = np.random.default_rng(42)
        lo = simple_oracle.bounds[:, 0]
        hi = simple_oracle.bounds[:, 1]
        X = rng.uniform(lo, hi, size=(500, simple_oracle.n_inputs))
        Y = simple_oracle.evaluate_batch(X)

        p_X, p_Y = extract_pareto_front(X, Y, simple_oracle)
        assert p_X.shape[0] > 0
        assert p_X.shape[1] == simple_oracle.n_inputs
        assert p_Y.shape[1] == simple_oracle.n_outputs
        assert p_X.shape[0] == p_Y.shape[0]

    def test_pareto_truly_nondominated(self, simple_oracle: SimpleOracle) -> None:
        """Verify no Pareto point dominates another (in all-maximize space)."""
        rng = np.random.default_rng(42)
        lo = simple_oracle.bounds[:, 0]
        hi = simple_oracle.bounds[:, 1]
        X = rng.uniform(lo, hi, size=(500, simple_oracle.n_inputs))
        Y = simple_oracle.evaluate_batch(X)

        p_X, p_Y = extract_pareto_front(X, Y, simple_oracle)

        obj_idx, _con_idx, signs = parse_directions(simple_oracle)
        p_obj = p_Y[:, obj_idx] * signs  # all-maximize space

        n = p_obj.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # j should NOT dominate i
                dominates = bool(
                    np.all(p_obj[j] >= p_obj[i]) and np.any(p_obj[j] > p_obj[i])
                )
                assert not dominates, (
                    f"Point {j} dominates point {i} on Pareto front"
                )
