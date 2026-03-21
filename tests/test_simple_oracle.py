"""Tests for the Simple Oracle (smooth, no regimes/coupling/thresholds)."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from synthoracle.dag import NodeType
from synthoracle.oracles.simple import SimpleOracle

from .conftest import random_inputs


@pytest.fixture
def simple_oracle() -> SimpleOracle:
    return SimpleOracle()


class TestDeterminism:
    def test_deterministic(self, simple_oracle: SimpleOracle) -> None:
        x = np.array([0.5, 0.5, 0.5, 0.5])
        reference = simple_oracle.evaluate(x)
        for _ in range(100):
            np.testing.assert_array_equal(simple_oracle.evaluate(x), reference)

    def test_deterministic_across_instances(self) -> None:
        o1 = SimpleOracle()
        o2 = SimpleOracle()
        x = np.array([0.3, 0.7, 0.2, 0.8])
        np.testing.assert_array_equal(o1.evaluate(x), o2.evaluate(x))


class TestNumericalSafety:
    def test_no_nan_inf(
        self, simple_oracle: SimpleOracle, rng: np.random.Generator
    ) -> None:
        X = random_inputs(simple_oracle, rng, n=1000)
        Y = simple_oracle.evaluate_batch(X)
        assert not np.any(np.isnan(Y))
        assert not np.any(np.isinf(Y))

    def test_outputs_bounded(
        self, simple_oracle: SimpleOracle, rng: np.random.Generator
    ) -> None:
        X = random_inputs(simple_oracle, rng, n=1000)
        Y = simple_oracle.evaluate_batch(X)
        assert np.all(np.abs(Y) < 10)

    def test_boundary_inputs(self, simple_oracle: SimpleOracle) -> None:
        lo = simple_oracle.bounds[:, 0]
        hi = simple_oracle.bounds[:, 1]
        for corner in itertools.product(*zip(lo, hi)):
            x = np.array(corner)
            y = simple_oracle.evaluate(x)
            assert not np.any(np.isnan(y))
            assert not np.any(np.isinf(y))


class TestInputRelevance:
    def test_all_inputs_matter(self, simple_oracle: SimpleOracle) -> None:
        """Each input has OAT effect > 0 on at least one output."""
        midpoint = (simple_oracle.bounds[:, 0] + simple_oracle.bounds[:, 1]) / 2
        for j in range(simple_oracle.n_inputs):
            sweep = np.linspace(
                simple_oracle.bounds[j, 0], simple_oracle.bounds[j, 1], 50
            )
            X_oat = np.tile(midpoint, (50, 1))
            X_oat[:, j] = sweep
            Y_oat = simple_oracle.evaluate_batch(X_oat)
            max_effect = max(
                Y_oat[:, i].max() - Y_oat[:, i].min()
                for i in range(simple_oracle.n_outputs)
            )
            assert max_effect > 0.01, f"Input X{j+1} has no detectable effect"

    def test_smooth_landscape(self, simple_oracle: SimpleOracle) -> None:
        """Finite differences between adjacent sweep points are bounded (no jumps)."""
        midpoint = (simple_oracle.bounds[:, 0] + simple_oracle.bounds[:, 1]) / 2
        for j in range(simple_oracle.n_inputs):
            sweep = np.linspace(
                simple_oracle.bounds[j, 0], simple_oracle.bounds[j, 1], 200
            )
            X_oat = np.tile(midpoint, (200, 1))
            X_oat[:, j] = sweep
            Y_oat = simple_oracle.evaluate_batch(X_oat)
            # Max step-to-step change should be small (smooth)
            diffs = np.abs(np.diff(Y_oat, axis=0))
            assert np.all(diffs < 0.1), f"Input X{j+1} causes discontinuous jump"


class TestTradeoff:
    def test_pareto_tradeoff_exists(
        self, simple_oracle: SimpleOracle, rng: np.random.Generator
    ) -> None:
        """Cannot simultaneously maximize Y1 and minimize Y2."""
        X = random_inputs(simple_oracle, rng, n=5000)
        Y = simple_oracle.evaluate_batch(X)
        # Find point with best Y1
        best_y1_idx = np.argmax(Y[:, 0])
        # Find point with best (lowest) Y2
        best_y2_idx = np.argmin(Y[:, 1])
        # These should be different points
        assert best_y1_idx != best_y2_idx, "Y1 max and Y2 min at same point — no trade-off"


class TestGroundTruthDAG:
    def test_dag_edge_count(self, simple_oracle: SimpleOracle) -> None:
        dag = simple_oracle.ground_truth()
        assert len(dag.edges) == 8

    def test_dag_node_types(self, simple_oracle: SimpleOracle) -> None:
        dag = simple_oracle.ground_truth()
        node_map = {n.name: n.node_type for n in dag.nodes}
        for name in ("X1", "X2", "X3", "X4"):
            assert node_map[name] == NodeType.INPUT
        for name in ("M_A", "M_B"):
            assert node_map[name] == NodeType.MECHANISM
        for name in ("Y1", "Y2"):
            assert node_map[name] == NodeType.OUTPUT

    def test_dag_contains_all_edges(self, simple_oracle: SimpleOracle) -> None:
        dag = simple_oracle.ground_truth()
        edge_pairs = {(e.source, e.target) for e in dag.edges}
        expected = {
            ("X1", "M_A"), ("X2", "M_A"),
            ("X3", "M_B"), ("X4", "M_B"),
            ("M_A", "Y1"), ("M_B", "Y1"),
            ("M_A", "Y2"), ("M_B", "Y2"),
        }
        assert edge_pairs == expected

    def test_io_projection(self, simple_oracle: SimpleOracle) -> None:
        """All 4 inputs reach both outputs."""
        dag = simple_oracle.ground_truth()
        projected = dag.project_to_io()
        edge_pairs = {(e.source, e.target) for e in projected.edges}
        for xi in ("X1", "X2", "X3", "X4"):
            for yj in ("Y1", "Y2"):
                assert (xi, yj) in edge_pairs


class TestOracleContract:
    def test_evaluate_shape(self, simple_oracle: SimpleOracle) -> None:
        x = np.full(simple_oracle.n_inputs, 0.5)
        y = simple_oracle.evaluate(x)
        assert y.shape == (2,)

    def test_bounds_shape(self, simple_oracle: SimpleOracle) -> None:
        assert simple_oracle.bounds.shape == (4, 2)

    def test_output_directions(self, simple_oracle: SimpleOracle) -> None:
        assert simple_oracle.output_directions == ("maximize", "minimize")


class TestPerformance:
    @pytest.mark.slow
    def test_evaluation_speed(self, simple_oracle: SimpleOracle) -> None:
        rng = np.random.default_rng(42)
        X = random_inputs(simple_oracle, rng, n=1000)
        start = time.perf_counter()
        simple_oracle.evaluate_batch(X)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0
