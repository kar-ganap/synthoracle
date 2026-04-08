"""Medium oracle variant 1D: 'Structural Shift'.

Same 6 inputs, 4 outputs, same mechanism types (throughput, leakage, coupling,
threshold). Structural changes from 1A:

1. X5 threshold removed — smooth monotonic + X1*X5 interaction on Y1
2. X3→Y3 becomes inverted-U (peaks at X3~0.6) instead of monotonic sqrt
3. M1_eff sign flipped on Y2 (efficiency gain, not cost)
4. X5→Y2 new edge (penalty term)
5. X3→Y4 new edge (log contribution)
6. Z coupling weakened (X6 coefficient halved)

Transfer from 1A: prior is ~50% wrong on functional forms, has 2 missing edges.
Tests whether agent can falsify wrong beliefs while leveraging correct edge existence.
"""

from __future__ import annotations

import numpy as np

from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, Node, NodeType
from synthoracle.oracles.medium import MediumOracle, _sigmoid
from synthoracle.types import InputArray, OutputArray


class MediumOracle1D(MediumOracle):
    """The Structural Shift oracle — variant 1D.

    Same mechanism topology as 1A but with altered functional forms,
    new edges, and a sign flip on Y2. Designed so the 1A prior is
    partially helpful (correct edge existence) but partially misleading
    (wrong functional forms, missing edges).
    """

    def __init__(self) -> None:
        super().__init__(variant="1A")  # base M2 formula

    def evaluate(self, x: InputArray) -> OutputArray:
        y, _ = self.evaluate_with_mechanisms(x)
        return y

    def evaluate_with_mechanisms(
        self, x: InputArray,
    ) -> tuple[OutputArray, dict[str, float]]:
        x1, x2, x3, x4, x5, x6 = x

        # M1: same as 1A
        m1 = x2**1.37 * x4**0.82 * (1.0 - np.exp(-5.0 * x1))

        # M2: same as 1A (variant 1A formula)
        m2 = 0.55 * np.exp(-1.8 * x3 * np.sqrt(x1))

        # Z coupling: weakened (0.5 instead of 1.0)
        z = x4 / (x4 + 0.5 * x6)

        # Gate: always on (no X5 threshold)
        gate = 1.0

        # Effective mechanisms
        m1_eff = m1 * z
        m2_eff = m2 * (1.0 - 0.6 * z)

        # M4: X6 * gate (gate always 1.0)
        m4_mult = 1.0 + 0.4 * x6 * gate
        m4_add = 0.5 * x6 * gate

        # Smooth X5 contribution + X1*X5 interaction
        x5_contrib = 0.3 * x5**0.8
        x1_x5_interact = 0.8 * x1 * x5

        # Y1: add smooth X5 + interaction instead of threshold-gated
        y1 = m1_eff * m4_mult + m4_add - m2_eff + x5_contrib + x1_x5_interact

        # Y2: M1_eff sign FLIPPED (efficiency gain) + X5 penalty
        y2 = 0.85 * m2_eff - 0.20 * m1_eff / m4_mult + 0.25 * (1.0 - z) + 0.15 * x5

        # Y3: inverted-U in X3 (peaks at X3~0.6)
        y3_raw = 1.0 - 4.0 * (x3 - 0.6) ** 2
        y3 = float(_sigmoid(8.1 * (x1 - 0.27))) * max(0.0, y3_raw)

        # Y4: add X3 log contribution
        y4 = m1_eff / (1.0 + 1.5 * m1_eff) + 0.12 * np.log1p(3.0 * x3)

        y = np.array([y1, y2, y3, y4], dtype=np.float64)
        mechanisms = {
            "M1": float(m1), "M2": float(m2), "Z": float(z),
            "gate": float(gate), "M1_eff": float(m1_eff),
            "M2_eff": float(m2_eff), "M4_mult": float(m4_mult),
            "M4_add": float(m4_add), "X5_contrib": float(x5_contrib),
            "X1_X5_interact": float(x1_x5_interact),
        }
        return y, mechanisms

    def adversarial_regions(self) -> list[dict[str, object]]:
        """1D-specific adversarial regions."""
        return [
            {
                "name": "X1_X5_interaction",
                "description": "New X1*X5 interaction zone",
                "test": lambda x: x[0] > 0.7 and x[4] > 0.7,
            },
            {
                "name": "X3_Y3_peak",
                "description": "Near inverted-U peak (X3 ~ 0.6)",
                "test": lambda x: 0.45 < x[2] < 0.75,
            },
            {
                "name": "X2_X4_interaction",
                "description": "Multiplicative synergy zone",
                "test": lambda x: x[1] > 0.7 and x[3] > 0.7,
            },
            {
                "name": "Z_sensitive",
                "description": "Hidden coupling (weakened in 1D)",
                "test": lambda x: 0.3 < x[3] / (x[3] + 0.5 * x[5]) < 0.7,
            },
            {
                "name": "M2_regime",
                "description": "Leakage-dominated regime (low X1)",
                "test": lambda x: x[0] < 0.3,
            },
        ]

    def ground_truth(self) -> CausalDAG:
        """Ground-truth DAG for variant 1D."""
        nodes = frozenset({
            Node("X1", NodeType.INPUT),
            Node("X2", NodeType.INPUT),
            Node("X3", NodeType.INPUT),
            Node("X4", NodeType.INPUT),
            Node("X5", NodeType.INPUT),
            Node("X6", NodeType.INPUT),
            Node("M1", NodeType.MECHANISM),
            Node("M2", NodeType.MECHANISM),
            Node("Z", NodeType.MECHANISM),
            Node("M4_mult", NodeType.MECHANISM),
            Node("M4_add", NodeType.MECHANISM),
            Node("M1_eff", NodeType.MECHANISM),
            Node("M2_eff", NodeType.MECHANISM),
            Node("Y1", NodeType.OUTPUT),
            Node("Y2", NodeType.OUTPUT),
            Node("Y3", NodeType.OUTPUT),
            Node("Y4", NodeType.OUTPUT),
        })

        edges = frozenset({
            # M1 = f(X1, X2, X4) — same as 1A
            Edge("X1", "M1", EdgeDifficulty.MEDIUM, "regime-dependent activation"),
            Edge("X2", "M1", EdgeDifficulty.EASY, "direct, strong"),
            Edge("X4", "M1", EdgeDifficulty.EASY, "direct"),
            # M2 = f(X1, X3) — same as 1A
            Edge("X1", "M2", EdgeDifficulty.MEDIUM, "regime-dependent"),
            Edge("X3", "M2", EdgeDifficulty.MEDIUM, "regime-dependent"),
            # Z = f(X4, X6) — weakened coupling
            Edge("X4", "Z", EdgeDifficulty.HARD, "hidden coupling (weakened)"),
            Edge("X6", "Z", EdgeDifficulty.HARD, "hidden coupling (weakened)"),
            # M4_mult = f(X6) — gate always on, no X5
            Edge("X6", "M4_mult", EdgeDifficulty.EASY, "always active"),
            # M4_add = f(X6) — gate always on, no X5
            Edge("X6", "M4_add", EdgeDifficulty.EASY, "always active"),
            # M1_eff, M2_eff — same as 1A
            Edge("M1", "M1_eff", EdgeDifficulty.EASY, "modulated by coupling"),
            Edge("Z", "M1_eff", EdgeDifficulty.HARD, "hidden coupling"),
            Edge("M2", "M2_eff", EdgeDifficulty.EASY, "modulated by coupling"),
            Edge("Z", "M2_eff", EdgeDifficulty.HARD, "hidden coupling"),
            # Y1 = M1_eff*M4_mult + M4_add - M2_eff + X5_smooth + X1*X5
            Edge("M1_eff", "Y1", EdgeDifficulty.EASY, "performance"),
            Edge("M4_mult", "Y1", EdgeDifficulty.EASY, "always-on modulation"),
            Edge("M4_add", "Y1", EdgeDifficulty.EASY, "always-on boost"),
            Edge("M2_eff", "Y1", EdgeDifficulty.MEDIUM, "leakage penalty"),
            Edge("X5", "Y1", EdgeDifficulty.EASY, "smooth monotonic (no threshold)"),
            Edge("X1", "Y1", EdgeDifficulty.MEDIUM, "X1*X5 interaction direct path"),
            # Y2 = 0.85*M2_eff - 0.20*M1_eff/M4_mult + 0.25*(1-Z) + 0.15*X5
            Edge("M1_eff", "Y2", EdgeDifficulty.MEDIUM, "efficiency gain (sign flip)"),
            Edge("M2_eff", "Y2", EdgeDifficulty.MEDIUM, "cost contribution"),
            Edge("M4_mult", "Y2", EdgeDifficulty.EASY, "modulates efficiency"),
            Edge("Z", "Y2", EdgeDifficulty.HARD, "coupling increases cost"),
            Edge("X5", "Y2", EdgeDifficulty.MEDIUM, "new penalty term"),
            # Y3 = sigmoid(X1) * max(0, 1-4*(X3-0.6)^2) — inverted-U
            Edge("X1", "Y3", EdgeDifficulty.MEDIUM, "reliability threshold"),
            Edge("X3", "Y3", EdgeDifficulty.MEDIUM, "inverted-U (peaks at 0.6)"),
            # Y4 = M1_eff/(1+1.5*M1_eff) + 0.12*log1p(3*X3)
            Edge("M1_eff", "Y4", EdgeDifficulty.EASY, "saturating speed"),
            Edge("X3", "Y4", EdgeDifficulty.MEDIUM, "log contribution (new)"),
        })

        return CausalDAG(nodes=nodes, edges=edges)
