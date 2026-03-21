"""Tests for medium oracle transfer variants (1B and 1C)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from synthoracle.dag import NodeType
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_1c import MediumOracle1C

from .conftest import random_inputs


class TestVariant1B:
    @pytest.fixture
    def oracle_1a(self) -> MediumOracle:
        return MediumOracle()

    @pytest.fixture
    def oracle_1b(self) -> MediumOracle:
        return MediumOracle(variant="1B")

    def test_1b_different_from_1a(
        self, oracle_1a: MediumOracle, oracle_1b: MediumOracle
    ) -> None:
        """Same input produces different outputs for 1A vs 1B."""
        x = np.array([0.3, 0.5, 0.5, 0.5, 0.5, 0.5])
        y_1a = oracle_1a.evaluate(x)
        y_1b = oracle_1b.evaluate(x)
        assert not np.allclose(y_1a, y_1b)

    def test_1b_dag_identical_to_1a(
        self, oracle_1a: MediumOracle, oracle_1b: MediumOracle
    ) -> None:
        """Same DAG structure — same causal graph, different parameters."""
        dag_1a = oracle_1a.ground_truth()
        dag_1b = oracle_1b.ground_truth()
        pairs_1a = {(e.source, e.target) for e in dag_1a.edges}
        pairs_1b = {(e.source, e.target) for e in dag_1b.edges}
        assert pairs_1a == pairs_1b
        assert len(dag_1a.nodes) == len(dag_1b.nodes)

    def test_1b_deterministic(self, oracle_1b: MediumOracle) -> None:
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ref = oracle_1b.evaluate(x)
        for _ in range(50):
            np.testing.assert_array_equal(oracle_1b.evaluate(x), ref)

    def test_1b_no_nan_inf(
        self, oracle_1b: MediumOracle, rng: np.random.Generator
    ) -> None:
        X = random_inputs(oracle_1b, rng, n=1000)
        Y = oracle_1b.evaluate_batch(X)
        assert not np.any(np.isnan(Y))
        assert not np.any(np.isinf(Y))

    def test_1b_outputs_bounded(
        self, oracle_1b: MediumOracle, rng: np.random.Generator
    ) -> None:
        X = random_inputs(oracle_1b, rng, n=1000)
        Y = oracle_1b.evaluate_batch(X)
        assert np.all(np.abs(Y) < 10)

    def test_1b_boundary_inputs(self, oracle_1b: MediumOracle) -> None:
        lo = oracle_1b.bounds[:, 0]
        hi = oracle_1b.bounds[:, 1]
        for corner in itertools.product(*zip(lo, hi)):
            x = np.array(corner)
            y = oracle_1b.evaluate(x)
            assert not np.any(np.isnan(y))
            assert not np.any(np.isinf(y))

    def test_1b_regime_transition_shifted(
        self, oracle_1a: MediumOracle, oracle_1b: MediumOracle
    ) -> None:
        """M2 at low X1 is dramatically different between 1A and 1B.

        1A: M2 = 0.55 * exp(-1.8 * 0.5 * sqrt(0.1)) ≈ 0.55 * exp(-0.28) ≈ 0.41
        1B: M2 = 0.55 * exp(-1.8 * 0.5 / 0.1)       ≈ 0.55 * exp(-9.0) ≈ 0.00007
        """
        x = np.array([0.1, 0.5, 0.5, 0.5, 0.5, 0.5])
        y_1a = oracle_1a.evaluate(x)
        y_1b = oracle_1b.evaluate(x)
        # Y2 should be very different (M2 contributes to Y2)
        assert abs(y_1a[1] - y_1b[1]) > 0.05

    def test_invalid_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown variant"):
            MediumOracle(variant="1Z")


class TestVariant1C:
    @pytest.fixture
    def oracle_1a(self) -> MediumOracle:
        return MediumOracle()

    @pytest.fixture
    def oracle_1c(self) -> MediumOracle1C:
        return MediumOracle1C()

    def test_1c_y4_differs_from_1a(
        self, oracle_1a: MediumOracle, oracle_1c: MediumOracle1C
    ) -> None:
        """Y4 changes due to M5; Y1, Y2, Y3 should be identical."""
        x = np.array([0.5, 0.5, 0.7, 0.5, 0.7, 0.5])  # X3, X5 non-trivial
        y_1a = oracle_1a.evaluate(x)
        y_1c = oracle_1c.evaluate(x)
        # Y1, Y2, Y3 unchanged
        np.testing.assert_array_almost_equal(y_1c[:3], y_1a[:3])
        # Y4 is different (M5 adds to it)
        assert y_1c[3] > y_1a[3]

    def test_1c_dag_has_more_edges(
        self, oracle_1a: MediumOracle, oracle_1c: MediumOracle1C
    ) -> None:
        dag_1a = oracle_1a.ground_truth()
        dag_1c = oracle_1c.ground_truth()
        assert len(dag_1c.edges) == len(dag_1a.edges) + 3  # 26 + 3 = 29

    def test_1c_m5_edges_present(self, oracle_1c: MediumOracle1C) -> None:
        dag = oracle_1c.ground_truth()
        edge_pairs = {(e.source, e.target) for e in dag.edges}
        assert ("X3", "M5") in edge_pairs
        assert ("X5", "M5") in edge_pairs
        assert ("M5", "Y4") in edge_pairs

    def test_1c_m5_node_exists(self, oracle_1c: MediumOracle1C) -> None:
        dag = oracle_1c.ground_truth()
        node_map = {n.name: n.node_type for n in dag.nodes}
        assert "M5" in node_map
        assert node_map["M5"] == NodeType.MECHANISM

    def test_1c_x3_reaches_y4_in_projection(
        self, oracle_1c: MediumOracle1C
    ) -> None:
        """X3 should now reach Y4 (new IO connection via M5)."""
        dag = oracle_1c.ground_truth()
        projected = dag.project_to_io()
        edge_pairs = {(e.source, e.target) for e in projected.edges}
        assert ("X3", "Y4") in edge_pairs

    def test_1c_x5_reaches_y4_in_projection(
        self, oracle_1c: MediumOracle1C
    ) -> None:
        """X5 should now reach Y4 (new IO connection via M5)."""
        dag = oracle_1c.ground_truth()
        projected = dag.project_to_io()
        edge_pairs = {(e.source, e.target) for e in projected.edges}
        assert ("X5", "Y4") in edge_pairs

    def test_1c_deterministic(self, oracle_1c: MediumOracle1C) -> None:
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ref = oracle_1c.evaluate(x)
        for _ in range(50):
            np.testing.assert_array_equal(oracle_1c.evaluate(x), ref)

    def test_1c_no_nan_inf(
        self, oracle_1c: MediumOracle1C, rng: np.random.Generator
    ) -> None:
        X = random_inputs(oracle_1c, rng, n=1000)
        Y = oracle_1c.evaluate_batch(X)
        assert not np.any(np.isnan(Y))
        assert not np.any(np.isinf(Y))

    def test_1c_outputs_bounded(
        self, oracle_1c: MediumOracle1C, rng: np.random.Generator
    ) -> None:
        X = random_inputs(oracle_1c, rng, n=1000)
        Y = oracle_1c.evaluate_batch(X)
        assert np.all(np.abs(Y) < 10)

    def test_1c_y4_range_reasonable(
        self, oracle_1c: MediumOracle1C, rng: np.random.Generator
    ) -> None:
        """Y4 should stay in O(0.1-1) range even with M5 addition."""
        X = random_inputs(oracle_1c, rng, n=1000)
        Y = oracle_1c.evaluate_batch(X)
        assert Y[:, 3].max() < 2.0
        assert Y[:, 3].min() > -0.5
