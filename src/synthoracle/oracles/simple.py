"""Simple oracle: smooth, no regimes, no coupling, no thresholds.

4 inputs, 2 outputs, 2 mechanisms. BO should win or tie on this oracle.
Purpose: baseline showing VR overhead isn't justified on smooth systems.
"""

from __future__ import annotations

import numpy as np

from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, Node, NodeType
from synthoracle.oracle import Oracle
from synthoracle.types import InputArray, OutputArray


class SimpleOracle(Oracle):
    """Smooth oracle with no hidden structure.

    Inputs:  X1..X4 in [0.1, 1.0]
    Outputs: Y1 (performance, maximize), Y2 (cost, minimize)

    Mechanisms:
      M_A = X1^0.8 * X2^0.6  (smooth, monotonic)
      M_B = X3^0.7 * X4^0.5  (smooth, monotonic)
      Y1 = M_A + 0.2 * M_B   (maximize)
      Y2 = 0.3 * M_A + M_B   (minimize)

    Trade-off: increasing M_A helps Y1 (coeff 1.0) but hurts Y2 (coeff 0.3).
    All inputs matter. No regime transitions, coupling, or thresholds.
    """

    @property
    def n_inputs(self) -> int:
        return 4

    @property
    def n_outputs(self) -> int:
        return 2

    @property
    def bounds(self) -> InputArray:
        return np.full((4, 2), [[0.1, 1.0]], dtype=np.float64)

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("X1", "X2", "X3", "X4")

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("Y1", "Y2")

    @property
    def output_directions(self) -> tuple[str, ...]:
        return ("maximize", "minimize")

    def evaluate(self, x: InputArray) -> OutputArray:
        x1, x2, x3, x4 = x

        m_a = x1**0.8 * x2**0.6
        m_b = x3**0.7 * x4**0.5

        y1 = m_a + 0.2 * m_b
        y2 = 0.3 * m_a + m_b

        return np.array([y1, y2], dtype=np.float64)

    def ground_truth(self) -> CausalDAG:
        """Return the ground-truth causal DAG with 8 edges, all EASY."""
        nodes = frozenset({
            Node("X1", NodeType.INPUT),
            Node("X2", NodeType.INPUT),
            Node("X3", NodeType.INPUT),
            Node("X4", NodeType.INPUT),
            Node("M_A", NodeType.MECHANISM),
            Node("M_B", NodeType.MECHANISM),
            Node("Y1", NodeType.OUTPUT),
            Node("Y2", NodeType.OUTPUT),
        })

        edges = frozenset({
            Edge("X1", "M_A", EdgeDifficulty.EASY, "smooth power law"),
            Edge("X2", "M_A", EdgeDifficulty.EASY, "smooth power law"),
            Edge("X3", "M_B", EdgeDifficulty.EASY, "smooth power law"),
            Edge("X4", "M_B", EdgeDifficulty.EASY, "smooth power law"),
            Edge("M_A", "Y1", EdgeDifficulty.EASY, "dominant term"),
            Edge("M_B", "Y1", EdgeDifficulty.EASY, "minor term"),
            Edge("M_A", "Y2", EdgeDifficulty.EASY, "minor term"),
            Edge("M_B", "Y2", EdgeDifficulty.EASY, "dominant term"),
        })

        return CausalDAG(nodes=nodes, edges=edges)
