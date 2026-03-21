"""Tests for Oracle base class contract, using MediumOracle as concrete implementation."""

from __future__ import annotations

import numpy as np

from synthoracle.dag import CausalDAG
from synthoracle.oracles.medium import MediumOracle


class TestOracleContract:
    def test_evaluate_returns_correct_shape(self, medium_oracle: MediumOracle) -> None:
        x = np.full(medium_oracle.n_inputs, 0.5)
        y = medium_oracle.evaluate(x)
        assert y.shape == (medium_oracle.n_outputs,)

    def test_evaluate_batch_returns_correct_shape(
        self, medium_oracle: MediumOracle
    ) -> None:
        X = np.full((10, medium_oracle.n_inputs), 0.5)
        Y = medium_oracle.evaluate_batch(X)
        assert Y.shape == (10, medium_oracle.n_outputs)

    def test_bounds_shape(self, medium_oracle: MediumOracle) -> None:
        assert medium_oracle.bounds.shape == (medium_oracle.n_inputs, 2)

    def test_bounds_ordering(self, medium_oracle: MediumOracle) -> None:
        for i in range(medium_oracle.n_inputs):
            assert medium_oracle.bounds[i, 0] < medium_oracle.bounds[i, 1]

    def test_input_names_length(self, medium_oracle: MediumOracle) -> None:
        assert len(medium_oracle.input_names) == medium_oracle.n_inputs

    def test_output_names_length(self, medium_oracle: MediumOracle) -> None:
        assert len(medium_oracle.output_names) == medium_oracle.n_outputs

    def test_output_directions_valid(self, medium_oracle: MediumOracle) -> None:
        valid = {"maximize", "minimize", "threshold"}
        for d in medium_oracle.output_directions:
            assert d in valid

    def test_ground_truth_returns_dag(self, medium_oracle: MediumOracle) -> None:
        dag = medium_oracle.ground_truth()
        assert isinstance(dag, CausalDAG)
