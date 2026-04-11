"""Tests for the high-dimensional embedding oracle."""

from __future__ import annotations

import numpy as np
import pytest

from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_hd import MediumOracleHD


@pytest.fixture
def oracle() -> MediumOracleHD:
    return MediumOracleHD()


@pytest.fixture
def oracle_1a() -> MediumOracle:
    return MediumOracle()


class TestBasicProperties:
    def test_dimensions(self, oracle: MediumOracleHD) -> None:
        assert oracle.n_inputs == 12
        assert oracle.n_outputs == 4

    def test_input_names(self, oracle: MediumOracleHD) -> None:
        assert len(oracle.input_names) == 12
        assert oracle.input_names[:6] == ("X1", "X2", "X3", "X4", "X5", "X6")
        assert oracle.input_names[6:] == ("X7", "X8", "X9", "X10", "X11", "X12")

    def test_bounds(self, oracle: MediumOracleHD) -> None:
        assert oracle.bounds.shape == (12, 2)
        assert np.all(oracle.bounds[:, 0] == 0.1)
        assert np.all(oracle.bounds[:, 1] == 1.0)

    def test_output_directions(self, oracle: MediumOracleHD) -> None:
        assert oracle.output_directions == ("maximize", "minimize", "threshold", "maximize")


class TestNoiseDimensions:
    def test_noise_dims_have_zero_effect(
        self, oracle: MediumOracleHD,
    ) -> None:
        base = np.array([0.5] * 12)
        y_base = oracle.evaluate(base)

        for noise_idx in range(6, 12):
            for val in [0.1, 0.5, 0.9]:
                x = base.copy()
                x[noise_idx] = val
                y = oracle.evaluate(x)
                np.testing.assert_array_equal(
                    y, y_base,
                    err_msg=f"X{noise_idx + 1}={val} changed output",
                )

    def test_matches_1a_on_real_inputs(
        self, oracle: MediumOracleHD, oracle_1a: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            x6 = rng.uniform(0.1, 1.0, size=6)
            x12 = np.concatenate([x6, rng.uniform(0.1, 1.0, size=6)])
            y_1a = oracle_1a.evaluate(x6)
            y_hd = oracle.evaluate(x12)
            np.testing.assert_array_almost_equal(y_hd, y_1a)

    def test_evaluate_batch(self, oracle: MediumOracleHD) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(10, 12))
        Y = oracle.evaluate_batch(X)
        assert Y.shape == (10, 4)
        # Each row should match single evaluate
        for i in range(10):
            np.testing.assert_array_almost_equal(Y[i], oracle.evaluate(X[i]))


class TestMechanisms:
    def test_evaluate_with_mechanisms(self, oracle: MediumOracleHD) -> None:
        x = np.array([0.5] * 12)
        y, m = oracle.evaluate_with_mechanisms(x)
        assert "M1" in m
        assert "Z" in m
        assert "gate" in m
        assert len(y) == 4

    def test_mechanisms_match_1a(
        self, oracle: MediumOracleHD, oracle_1a: MediumOracle,
    ) -> None:
        x6 = np.array([0.3, 0.7, 0.5, 0.8, 0.4, 0.6])
        x12 = np.concatenate([x6, np.array([0.5] * 6)])
        _, m_1a = oracle_1a.evaluate_with_mechanisms(x6)
        _, m_hd = oracle.evaluate_with_mechanisms(x12)
        for key in m_1a:
            assert abs(m_1a[key] - m_hd[key]) < 1e-10, f"Mechanism {key} differs"


class TestGroundTruth:
    def test_dag_has_noise_nodes(self, oracle: MediumOracleHD) -> None:
        dag = oracle.ground_truth()
        node_names = {n.name for n in dag.nodes}
        for i in range(7, 13):
            assert f"X{i}" in node_names

    def test_noise_nodes_have_no_edges(self, oracle: MediumOracleHD) -> None:
        dag = oracle.ground_truth()
        noise_names = {f"X{i}" for i in range(7, 13)}
        for edge in dag.edges:
            assert edge.source not in noise_names, (
                f"Noise node {edge.source} has outgoing edge to {edge.target}"
            )

    def test_io_projection_same_as_1a(
        self, oracle: MediumOracleHD, oracle_1a: MediumOracle,
    ) -> None:
        io_hd = oracle.ground_truth().project_to_io()
        io_1a = oracle_1a.ground_truth().project_to_io()
        hd_pairs = {(e.source, e.target) for e in io_hd.edges}
        a_pairs = {(e.source, e.target) for e in io_1a.edges}
        assert hd_pairs == a_pairs

    def test_edge_count(self, oracle: MediumOracleHD) -> None:
        dag = oracle.ground_truth()
        # Same 26 edges as 1A, just more nodes
        assert len(dag.edges) == 26
        # 17 original nodes + 6 noise = 23
        assert len(dag.nodes) == 23


class TestAdversarialRegions:
    def test_has_noise_extreme_region(self, oracle: MediumOracleHD) -> None:
        regions = oracle.adversarial_regions()
        names = [r["name"] for r in regions]
        assert "noise_extreme" in names

    def test_noise_extreme_triggers(self, oracle: MediumOracleHD) -> None:
        regions = oracle.adversarial_regions()
        noise_region = next(r for r in regions if r["name"] == "noise_extreme")
        # 3+ noise dims at extremes
        x_extreme = np.array([0.5] * 6 + [0.1, 0.1, 0.9, 0.5, 0.5, 0.5])
        assert noise_region["test"](x_extreme)  # 3 extreme noise dims
        x_mild = np.array([0.5] * 12)
        assert not noise_region["test"](x_mild)  # no extreme noise dims
