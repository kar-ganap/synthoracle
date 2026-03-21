"""Tests for the reusable characterization module."""

from __future__ import annotations

import numpy as np
import pytest

from synthoracle.characterize import (
    CharacterizationResult,
    characterize,
)
from synthoracle.oracles.medium import MediumOracle


@pytest.fixture
def medium_result() -> CharacterizationResult:
    """Characterize medium oracle with small sample sizes for fast tests."""
    oracle = MediumOracle()
    return characterize(oracle, n_samples=5_000, n_sobol=5_000, n_pareto=10_000, seed=42)


class TestOutputStats:
    def test_stats_has_all_outputs(self, medium_result: CharacterizationResult) -> None:
        for name in ("Y1", "Y2", "Y3", "Y4"):
            assert name in medium_result.output_stats

    def test_stats_has_required_keys(self, medium_result: CharacterizationResult) -> None:
        for stats in medium_result.output_stats.values():
            for key in ("mean", "std", "min", "max", "range"):
                assert key in stats

    def test_stats_values_finite(self, medium_result: CharacterizationResult) -> None:
        for stats in medium_result.output_stats.values():
            for v in stats.values():
                assert np.isfinite(v)

    def test_stats_range_consistent(self, medium_result: CharacterizationResult) -> None:
        for stats in medium_result.output_stats.values():
            assert stats["range"] == pytest.approx(stats["max"] - stats["min"])


class TestSobolIndices:
    def test_sobol_shape(self, medium_result: CharacterizationResult) -> None:
        oracle = MediumOracle()
        assert medium_result.sobol_first_order.shape == (oracle.n_inputs, oracle.n_outputs)
        assert medium_result.sobol_total_order.shape == (oracle.n_inputs, oracle.n_outputs)

    def test_first_order_bounded(self, medium_result: CharacterizationResult) -> None:
        """First-order indices should be in [-0.2, 1.2] (small estimation noise OK)."""
        assert np.all(medium_result.sobol_first_order > -0.2)
        assert np.all(medium_result.sobol_first_order < 1.2)

    def test_total_order_bounded(self, medium_result: CharacterizationResult) -> None:
        """Total-order indices should be in [-0.1, 1.2]."""
        assert np.all(medium_result.sobol_total_order > -0.1)
        assert np.all(medium_result.sobol_total_order < 1.2)

    def test_total_geq_first_order(self, medium_result: CharacterizationResult) -> None:
        """Total order >= first order (within estimation noise)."""
        diff = medium_result.sobol_total_order - medium_result.sobol_first_order
        # Allow small negative values from estimation noise
        assert np.all(diff > -0.15)

    def test_first_order_sum_leq_one(self, medium_result: CharacterizationResult) -> None:
        """Sum of first-order indices should be <= 1 (within noise)."""
        sums = medium_result.sobol_first_order.sum(axis=0)
        assert np.all(sums < 1.3)  # Allow estimation noise

    def test_deterministic_with_seed(self) -> None:
        oracle = MediumOracle()
        r1 = characterize(oracle, n_samples=2_000, n_sobol=2_000, n_pareto=1_000, seed=42)
        r2 = characterize(oracle, n_samples=2_000, n_sobol=2_000, n_pareto=1_000, seed=42)
        np.testing.assert_array_equal(r1.sobol_first_order, r2.sobol_first_order)


class TestOATEffects:
    def test_oat_shape(self, medium_result: CharacterizationResult) -> None:
        oracle = MediumOracle()
        assert medium_result.oat_effects.shape == (oracle.n_inputs, oracle.n_outputs)

    def test_oat_nonnegative(self, medium_result: CharacterizationResult) -> None:
        """OAT effects (ranges) are non-negative."""
        assert np.all(medium_result.oat_effects >= 0)


class TestParetoFront:
    def test_pareto_nonempty(self, medium_result: CharacterizationResult) -> None:
        assert medium_result.pareto_front.shape[0] > 0

    def test_pareto_output_shape(self, medium_result: CharacterizationResult) -> None:
        oracle = MediumOracle()
        assert medium_result.pareto_front.shape[1] == oracle.n_outputs

    def test_pareto_inputs_shape(self, medium_result: CharacterizationResult) -> None:
        oracle = MediumOracle()
        n_pareto = medium_result.pareto_front.shape[0]
        assert medium_result.pareto_inputs.shape == (n_pareto, oracle.n_inputs)

    def test_pareto_nondominated(self, medium_result: CharacterizationResult) -> None:
        """Every point on the Pareto front should be non-dominated."""
        oracle = MediumOracle()
        front = medium_result.pareto_front
        directions = oracle.output_directions
        n = front.shape[0]
        for i in range(min(n, 50)):  # check subset for speed
            for j in range(n):
                if i == j:
                    continue
                # Check if j dominates i
                dominates = True
                for k, d in enumerate(directions):
                    if d == "threshold":
                        continue
                    if d == "maximize":
                        if front[j, k] <= front[i, k]:
                            dominates = False
                            break
                    elif d == "minimize":
                        if front[j, k] >= front[i, k]:
                            dominates = False
                            break
                assert not dominates, f"Point {j} dominates point {i} on the Pareto front"

    def test_pareto_nontrivial(self, medium_result: CharacterizationResult) -> None:
        """Pareto front has more than 1 point (genuine trade-off)."""
        assert medium_result.pareto_front.shape[0] > 5


class TestTiming:
    def test_timing_positive(self, medium_result: CharacterizationResult) -> None:
        assert medium_result.timing_us > 0

    def test_timing_less_than_1ms(self, medium_result: CharacterizationResult) -> None:
        assert medium_result.timing_us < 1000  # < 1ms = 1000 microseconds
