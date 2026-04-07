"""Medium oracle: 'The Bottleneck Shift'.

6 inputs, 4 outputs, regime transitions, hidden coupling, hidden threshold.
Fully specified in docs/conceptual.md.
"""

from __future__ import annotations

import numpy as np

from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, Node, NodeType
from synthoracle.oracle import Oracle
from synthoracle.types import InputArray, OutputArray


def _sigmoid(x: float | InputArray) -> float | InputArray:
    """Standard sigmoid function."""
    result: float | InputArray = 1.0 / (1.0 + np.exp(-x))
    return result


class MediumOracle(Oracle):
    """The Bottleneck Shift oracle.

    Inputs:  X1..X6 in [0.1, 1.0]
    Outputs: Y1 (performance, maximize), Y2 (cost, minimize),
             Y3 (reliability, threshold > 0.4), Y4 (speed, maximize, saturates)

    Key features:
    - Regime transition: X1 controls whether throughput (M1) or leakage (M2) dominates
    - Hidden coupling: X4 and X6 share intermediate Z, creating unexpected correlations
    - Hidden threshold: M4 activates only when X5 > ~0.38

    Variants:
    - "1A" (default): base oracle
    - "1B": M2 formula changes from exp(-b*X3*sqrt(X1)) to exp(-b*X3/X1),
            shifting the regime boundary dramatically
    """

    def __init__(self, variant: str = "1A") -> None:
        if variant not in ("1A", "1B"):
            raise ValueError(f"Unknown variant: {variant}. Must be '1A' or '1B'.")
        self._variant = variant

    @property
    def n_inputs(self) -> int:
        return 6

    @property
    def n_outputs(self) -> int:
        return 4

    @property
    def bounds(self) -> InputArray:
        return np.full((6, 2), [[0.1, 1.0]], dtype=np.float64)

    @property
    def input_names(self) -> tuple[str, ...]:
        return ("X1", "X2", "X3", "X4", "X5", "X6")

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("Y1", "Y2", "Y3", "Y4")

    @property
    def output_directions(self) -> tuple[str, ...]:
        return ("maximize", "minimize", "threshold", "maximize")

    def evaluate(self, x: InputArray) -> OutputArray:
        y, _ = self.evaluate_with_mechanisms(x)
        return y

    def evaluate_with_mechanisms(
        self, x: InputArray,
    ) -> tuple[OutputArray, dict[str, float]]:
        """Evaluate and return intermediate mechanism values.

        Returns (Y, mechanisms) where mechanisms is a dict of all
        intermediate variables in the causal DAG.
        """
        x1, x2, x3, x4, x5, x6 = x

        # Mechanisms
        m1 = x2**1.37 * x4**0.82 * (1.0 - np.exp(-5.0 * x1))
        if self._variant == "1A":
            m2 = 0.55 * np.exp(-1.8 * x3 * np.sqrt(x1))
        else:  # 1B
            m2 = 0.55 * np.exp(-1.8 * x3 / x1)
        z = x4 / (x4 + 1.0 * x6)
        gate = float(_sigmoid(25.0 * (x5 - 0.38)))

        # Effective mechanisms (post-coupling)
        m1_eff = m1 * z
        m2_eff = m2 * (1.0 - 0.6 * z)

        # M4: split into multiplicative and additive components
        m4_mult = 1.0 + 0.4 * x6 * gate
        m4_add = 0.5 * x6 * gate

        # Outputs
        y1 = m1_eff * m4_mult + m4_add - m2_eff
        y2 = 0.85 * m2_eff + 0.35 * m1_eff / m4_mult + 0.25 * (1.0 - z)
        y3 = float(_sigmoid(8.1 * (x1 - 0.27))) * x3**0.5
        y4 = m1_eff / (1.0 + 2.0 * m1_eff)

        y = np.array([y1, y2, y3, y4], dtype=np.float64)
        mechanisms = {
            "M1": float(m1), "M2": float(m2), "Z": float(z),
            "gate": float(gate), "M1_eff": float(m1_eff),
            "M2_eff": float(m2_eff), "M4_mult": float(m4_mult),
            "M4_add": float(m4_add),
        }
        return y, mechanisms

    def adversarial_regions(self) -> list[dict[str, object]]:
        """Regions where the Medium Oracle is nonlinear or interactive."""
        return [
            {
                "name": "X5_threshold",
                "description": "Near gate sigmoid (X5 ~ 0.38)",
                "test": lambda x: 0.30 < x[4] < 0.46,
            },
            {
                "name": "X2_X4_interaction",
                "description": "Multiplicative synergy zone",
                "test": lambda x: x[1] > 0.7 and x[3] > 0.7,
            },
            {
                "name": "Z_sensitive",
                "description": "Hidden coupling most sensitive",
                "test": lambda x: 0.3 < x[3] / (x[3] + x[5]) < 0.7,
            },
            {
                "name": "M2_regime",
                "description": "Leakage-dominated regime (low X1)",
                "test": lambda x: x[0] < 0.3,
            },
        ]

    def ground_truth(self) -> CausalDAG:
        """Return the ground-truth causal DAG.

        M4 is split into M4_mult (multiplicative, modulates M1_eff in Y1)
        and M4_add (additive, direct boost to Y1). Both are gated by the
        X5 threshold and scaled by X6.

        Z enters Y2 through the denominator, creating a direct coupling
        pathway from X4/X6 to Y2.
        """
        nodes = frozenset({
            # Inputs
            Node("X1", NodeType.INPUT),
            Node("X2", NodeType.INPUT),
            Node("X3", NodeType.INPUT),
            Node("X4", NodeType.INPUT),
            Node("X5", NodeType.INPUT),
            Node("X6", NodeType.INPUT),
            # Mechanisms
            Node("M1", NodeType.MECHANISM),
            Node("M2", NodeType.MECHANISM),
            Node("Z", NodeType.MECHANISM),
            Node("M4_mult", NodeType.MECHANISM),
            Node("M4_add", NodeType.MECHANISM),
            Node("M1_eff", NodeType.MECHANISM),
            Node("M2_eff", NodeType.MECHANISM),
            # Outputs
            Node("Y1", NodeType.OUTPUT),
            Node("Y2", NodeType.OUTPUT),
            Node("Y3", NodeType.OUTPUT),
            Node("Y4", NodeType.OUTPUT),
        })

        edges = frozenset({
            # M1 = f(X1, X2, X4)
            Edge("X1", "M1", EdgeDifficulty.MEDIUM, "regime-dependent activation"),
            Edge("X2", "M1", EdgeDifficulty.EASY, "direct, strong"),
            Edge("X4", "M1", EdgeDifficulty.EASY, "direct"),
            # M2 = f(X1, X3)
            Edge("X1", "M2", EdgeDifficulty.MEDIUM, "regime-dependent"),
            Edge("X3", "M2", EdgeDifficulty.MEDIUM, "regime-dependent"),
            # Z = f(X4, X6) — hidden coupling
            Edge("X4", "Z", EdgeDifficulty.HARD, "hidden coupling"),
            Edge("X6", "Z", EdgeDifficulty.HARD, "hidden coupling"),
            # M4_mult = f(X5, X6) — multiplicative threshold
            Edge("X5", "M4_mult", EdgeDifficulty.HARD, "hidden threshold at 0.38"),
            Edge("X6", "M4_mult", EdgeDifficulty.HARD, "dual pathway"),
            # M4_add = f(X5, X6) — additive threshold
            Edge("X5", "M4_add", EdgeDifficulty.HARD, "hidden threshold at 0.38"),
            Edge("X6", "M4_add", EdgeDifficulty.HARD, "dual pathway"),
            # M1_eff = M1 * Z
            Edge("M1", "M1_eff", EdgeDifficulty.EASY, "modulated by coupling"),
            Edge("Z", "M1_eff", EdgeDifficulty.HARD, "hidden coupling"),
            # M2_eff = M2 * (1 - 0.6*Z)
            Edge("M2", "M2_eff", EdgeDifficulty.EASY, "modulated by coupling"),
            Edge("Z", "M2_eff", EdgeDifficulty.HARD, "hidden coupling"),
            # Y1 = M1_eff * M4_mult + M4_add - M2_eff
            Edge("M1_eff", "Y1", EdgeDifficulty.EASY, "performance"),
            Edge("M4_mult", "Y1", EdgeDifficulty.HARD, "multiplicative threshold"),
            Edge("M4_add", "Y1", EdgeDifficulty.HARD, "additive threshold boost"),
            Edge("M2_eff", "Y1", EdgeDifficulty.MEDIUM, "leakage penalty"),
            # Y2 = 0.85*M2_eff + 0.35*M1_eff/M4_mult + 0.25*(1-Z)
            Edge("M1_eff", "Y2", EdgeDifficulty.MEDIUM, "cost contribution"),
            Edge("M2_eff", "Y2", EdgeDifficulty.MEDIUM, "cost contribution"),
            Edge("M4_mult", "Y2", EdgeDifficulty.HARD, "threshold modulates cost"),
            Edge("Z", "Y2", EdgeDifficulty.HARD, "weak coupling increases cost"),
            # Y3 = sigmoid(8.1*(X1-0.27)) * X3^0.5
            Edge("X1", "Y3", EdgeDifficulty.MEDIUM, "reliability threshold"),
            Edge("X3", "Y3", EdgeDifficulty.MEDIUM, "reliability scaling"),
            # Y4 = M1_eff / (1 + 2.0*M1_eff)
            Edge("M1_eff", "Y4", EdgeDifficulty.EASY, "saturating speed"),
        })

        return CausalDAG(nodes=nodes, edges=edges)
