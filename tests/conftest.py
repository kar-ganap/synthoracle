"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from synthoracle.oracle import Oracle
from synthoracle.oracles.medium import MediumOracle


@pytest.fixture
def medium_oracle() -> MediumOracle:
    return MediumOracle()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def random_inputs(
    oracle: Oracle, rng: np.random.Generator, n: int = 100
) -> npt.NDArray[np.float64]:
    """Generate n random valid inputs within oracle bounds."""
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, oracle.n_inputs))
