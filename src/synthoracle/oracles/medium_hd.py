"""High-dimensional embedding of Medium Oracle 1A.

12 inputs (X1-X12), 4 outputs. X1-X6 are the real inputs that drive
the system (identical to 1A). X7-X12 are noise dimensions with pure
zero effect on all outputs. The agent must discover which inputs matter.

Tests: can the agent screen and dismiss irrelevant variables in a
higher-dimensional space? From the CBM literature: mechanism bottlenecks
are most effective when k << d. 1A has k/d ≈ 1.3; HD has k/d ≈ 0.67.
"""

from __future__ import annotations

import numpy as np

from synthoracle.dag import CausalDAG, Node, NodeType
from synthoracle.oracle import Oracle
from synthoracle.oracles.medium import MediumOracle
from synthoracle.types import InputArray, OutputArray


class MediumOracleHD(Oracle):
    """12-input embedding of Medium Oracle 1A.

    X1-X6 behave identically to 1A. X7-X12 are noise dimensions
    with zero effect on all outputs. Same output directions and
    thresholds as 1A.
    """

    def __init__(self) -> None:
        self._inner = MediumOracle(variant="1A")

    @property
    def n_inputs(self) -> int:
        return 12

    @property
    def n_outputs(self) -> int:
        return 4

    @property
    def bounds(self) -> InputArray:
        return np.full((12, 2), [[0.1, 1.0]], dtype=np.float64)

    @property
    def input_names(self) -> tuple[str, ...]:
        return (
            "X1", "X2", "X3", "X4", "X5", "X6",
            "X7", "X8", "X9", "X10", "X11", "X12",
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("Y1", "Y2", "Y3", "Y4")

    @property
    def output_directions(self) -> tuple[str, ...]:
        return ("maximize", "minimize", "threshold", "maximize")

    def evaluate(self, x: InputArray) -> OutputArray:
        return self._inner.evaluate(x[:6])

    def evaluate_with_mechanisms(
        self, x: InputArray,
    ) -> tuple[OutputArray, dict[str, float]]:
        """Evaluate and return mechanism values (from inner 1A oracle)."""
        return self._inner.evaluate_with_mechanisms(x[:6])

    def adversarial_regions(self) -> list[dict[str, object]]:
        """1A adversarial regions plus noise dimension traps."""
        regions = self._inner.adversarial_regions()
        regions.append({
            "name": "noise_extreme",
            "description": "Multiple noise dims at extremes (potential false attribution)",
            "test": lambda x: (
                sum(1 for i in range(6, 12) if x[i] > 0.8 or x[i] < 0.2) >= 3
            ),
        })
        return regions

    def ground_truth(self) -> CausalDAG:
        """Ground-truth DAG: 1A edges + 6 disconnected noise input nodes."""
        base_dag = self._inner.ground_truth()
        noise_nodes = frozenset(
            Node(f"X{i}", NodeType.INPUT) for i in range(7, 13)
        )
        return CausalDAG(
            nodes=base_dag.nodes | noise_nodes,
            edges=base_dag.edges,
        )
