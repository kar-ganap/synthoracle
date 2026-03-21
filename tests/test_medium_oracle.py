"""Tests for the Medium Oracle ('The Bottleneck Shift')."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from synthoracle.dag import NodeType
from synthoracle.oracles.medium import MediumOracle

from .conftest import random_inputs


class TestDeterminism:
    def test_deterministic(self, medium_oracle: MediumOracle) -> None:
        """Same input produces identical output across 100 calls."""
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        reference = medium_oracle.evaluate(x)
        for _ in range(100):
            np.testing.assert_array_equal(medium_oracle.evaluate(x), reference)

    def test_deterministic_across_instances(self) -> None:
        """Two independent MediumOracle instances agree."""
        o1 = MediumOracle()
        o2 = MediumOracle()
        x = np.array([0.3, 0.7, 0.2, 0.8, 0.4, 0.6])
        np.testing.assert_array_equal(o1.evaluate(x), o2.evaluate(x))


class TestNumericalSafety:
    def test_no_nan_inf(
        self, medium_oracle: MediumOracle, rng: np.random.Generator
    ) -> None:
        """1000 random valid inputs produce no NaN or Inf."""
        X = random_inputs(medium_oracle, rng, n=1000)
        Y = medium_oracle.evaluate_batch(X)
        assert not np.any(np.isnan(Y))
        assert not np.any(np.isinf(Y))

    def test_outputs_bounded(
        self, medium_oracle: MediumOracle, rng: np.random.Generator
    ) -> None:
        """All outputs have magnitude < 10 across 1000 random inputs."""
        X = random_inputs(medium_oracle, rng, n=1000)
        Y = medium_oracle.evaluate_batch(X)
        assert np.all(np.abs(Y) < 10)

    def test_boundary_inputs(self, medium_oracle: MediumOracle) -> None:
        """All 64 corners of the input hypercube produce valid outputs."""
        lo = medium_oracle.bounds[:, 0]
        hi = medium_oracle.bounds[:, 1]
        for corner in itertools.product(*zip(lo, hi)):
            x = np.array(corner)
            y = medium_oracle.evaluate(x)
            assert not np.any(np.isnan(y))
            assert not np.any(np.isinf(y))


class TestRegimeTransitions:
    def test_regime_transition_exists(self, medium_oracle: MediumOracle) -> None:
        """Sweeping X1 changes the sensitivity of Y1 to X2.

        At low X1, M2 (leakage) dominates, so X2's effect (through M1) is weak.
        At high X1, M1 (throughput) dominates, so X2's effect is strong.
        """
        base = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        def dy1_dx2(x1_val: float) -> float:
            x_lo = base.copy()
            x_lo[0] = x1_val
            x_lo[1] = 0.2
            x_hi = base.copy()
            x_hi[0] = x1_val
            x_hi[1] = 0.9
            return float(medium_oracle.evaluate(x_hi)[0] - medium_oracle.evaluate(x_lo)[0])

        sensitivity_low_x1 = dy1_dx2(0.1)
        sensitivity_high_x1 = dy1_dx2(0.9)
        # At high X1, Y1 should be notably more sensitive to X2
        assert abs(sensitivity_high_x1) > abs(sensitivity_low_x1) * 1.5

    def test_m2_dominance_at_low_x1(self, medium_oracle: MediumOracle) -> None:
        """At X1=0.1, leakage (M2) should be significant relative to throughput (M1)."""
        x = np.array([0.1, 0.5, 0.5, 0.5, 0.5, 0.5])
        y = medium_oracle.evaluate(x)
        # At low X1: M1 activation (1-exp(-5*0.1))≈0.39, M2 still significant
        # Y1 should be small or negative (M2_eff subtracts from Y1)
        assert y[0] < 0.5  # Y1 is suppressed at low X1

    def test_m1_dominance_at_high_x1(self, medium_oracle: MediumOracle) -> None:
        """At X1=0.9, throughput (M1) should dominate."""
        x = np.array([0.9, 0.8, 0.5, 0.5, 0.5, 0.5])
        y = medium_oracle.evaluate(x)
        # At high X1 with high X2: M1 is large, M2 decays exponentially
        assert y[0] > 0.3  # Y1 is substantial at high X1


class TestHiddenCoupling:
    def test_z_coupling_x4_affects_both_objectives(
        self, medium_oracle: MediumOracle
    ) -> None:
        """Changing X4 affects both Y1 and Y2 through the hidden coupling Z."""
        base = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        x_lo = base.copy()
        x_lo[3] = 0.2  # X4 = 0.2
        x_hi = base.copy()
        x_hi[3] = 0.9  # X4 = 0.9

        y_lo = medium_oracle.evaluate(x_lo)
        y_hi = medium_oracle.evaluate(x_hi)

        # Both Y1 and Y2 should change
        assert abs(y_hi[0] - y_lo[0]) > 0.01  # Y1 changes
        assert abs(y_hi[1] - y_lo[1]) > 0.01  # Y2 changes


class TestHiddenThreshold:
    def test_m4_threshold(self, medium_oracle: MediumOracle) -> None:
        """M4 activates around X5=0.38 — Y1 should increase above the threshold."""
        base = np.array([0.7, 0.7, 0.3, 0.5, 0.0, 0.7])  # X5 placeholder

        x_below = base.copy()
        x_below[4] = 0.2  # X5 well below threshold
        x_above = base.copy()
        x_above[4] = 0.6  # X5 well above threshold

        y_below = medium_oracle.evaluate(x_below)
        y_above = medium_oracle.evaluate(x_above)

        # Y1 should be at least 10% higher above the threshold
        assert y_above[0] / max(y_below[0], 1e-10) > 1.1


class TestReliabilityThreshold:
    def test_y3_threshold_achievable(self, medium_oracle: MediumOracle) -> None:
        """Y3 > 0.4 is achievable for some inputs and not for others."""
        # High X1 (above regime boundary) + moderate X3 → Y3 should be high
        x_high = np.array([0.8, 0.5, 0.6, 0.5, 0.5, 0.5])
        y_high = medium_oracle.evaluate(x_high)
        assert y_high[2] > 0.4  # Y3 achievable

        # Low X1 (below regime boundary) → sigmoid suppresses Y3
        x_low = np.array([0.15, 0.5, 0.6, 0.5, 0.5, 0.5])
        y_low = medium_oracle.evaluate(x_low)
        assert y_low[2] < 0.4  # Y3 not achievable


class TestGroundTruthDAG:
    def test_dag_edge_count(self, medium_oracle: MediumOracle) -> None:
        """Medium oracle DAG has exactly 26 edges."""
        dag = medium_oracle.ground_truth()
        assert len(dag.edges) == 26

    def test_dag_node_types(self, medium_oracle: MediumOracle) -> None:
        """X nodes are INPUT, mechanism nodes are MECHANISM, Y nodes are OUTPUT."""
        dag = medium_oracle.ground_truth()
        node_map = {n.name: n.node_type for n in dag.nodes}
        for name in ("X1", "X2", "X3", "X4", "X5", "X6"):
            assert node_map[name] == NodeType.INPUT
        for name in ("M1", "M2", "Z", "M4_mult", "M4_add", "M1_eff", "M2_eff"):
            assert node_map[name] == NodeType.MECHANISM
        for name in ("Y1", "Y2", "Y3", "Y4"):
            assert node_map[name] == NodeType.OUTPUT

    def test_dag_contains_key_edges(self, medium_oracle: MediumOracle) -> None:
        """Verify critical edges are present."""
        dag = medium_oracle.ground_truth()
        edge_pairs = {(e.source, e.target) for e in dag.edges}

        # Throughput pathway
        assert ("X2", "M1") in edge_pairs
        assert ("M1", "M1_eff") in edge_pairs
        assert ("M1_eff", "Y1") in edge_pairs

        # Leakage pathway
        assert ("X3", "M2") in edge_pairs
        assert ("M2_eff", "Y1") in edge_pairs

        # Hidden coupling through Z
        assert ("X4", "Z") in edge_pairs
        assert ("X6", "Z") in edge_pairs
        assert ("Z", "M1_eff") in edge_pairs
        assert ("Z", "M2_eff") in edge_pairs
        # Z also enters Y2 denominator
        assert ("Z", "Y2") in edge_pairs

        # Hidden threshold — split into mult and add
        assert ("X5", "M4_mult") in edge_pairs
        assert ("X5", "M4_add") in edge_pairs
        assert ("M4_mult", "Y1") in edge_pairs
        assert ("M4_add", "Y1") in edge_pairs

        # Dual pathway for X6
        assert ("X6", "M4_mult") in edge_pairs
        assert ("X6", "M4_add") in edge_pairs

    def test_io_projection_edges(self, medium_oracle: MediumOracle) -> None:
        """Projected IO DAG should show which inputs affect which outputs."""
        dag = medium_oracle.ground_truth()
        projected = dag.project_to_io()
        edge_pairs = {(e.source, e.target) for e in projected.edges}

        # X2 should reach Y1 (through M1 -> M1_eff -> Y1)
        assert ("X2", "Y1") in edge_pairs
        # X5 should reach Y1 (through M4 -> Y1)
        assert ("X5", "Y1") in edge_pairs
        # X3 should reach Y3 (direct)
        assert ("X3", "Y3") in edge_pairs
        # X1 should reach Y3 (direct)
        assert ("X1", "Y3") in edge_pairs


class TestPerformance:
    @pytest.mark.slow
    def test_evaluation_speed(self, medium_oracle: MediumOracle) -> None:
        """1000 evaluations complete in < 1 second."""
        rng = np.random.default_rng(42)
        X = random_inputs(medium_oracle, rng, n=1000)
        start = time.perf_counter()
        medium_oracle.evaluate_batch(X)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0
