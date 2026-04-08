"""Medium oracle variant 1E: 'Rewired'.

Same 6 inputs, 4 outputs, same mechanism TYPES (throughput, leakage, coupling,
threshold) but completely different input-to-mechanism WIRING:

1. M1 (throughput): X2,X4,X1 → X3,X6,X1 (inputs swapped)
2. M2 (leakage): X3,X1 → X4,X1 (X3/X4 swapped)
3. Z (coupling): X4,X6 → X2,X5 (completely different pair)
4. Gate (threshold): X5→X6 (threshold input swapped)
5. M4: X6→X5 (modulation input swapped)
6. Y3: X1,X3 → X1,X4 (X3/X4 swapped)

Transfer from 1A: prior has 3 confidently wrong edges, 4 missing edges.
81% IO edge overlap but Y2 correlation ~0 and Y4 correlation ~0.1.
Tests whether agent can unlearn confidently-held false beliefs.
"""

from __future__ import annotations

import numpy as np

from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, Node, NodeType
from synthoracle.oracle import Oracle
from synthoracle.oracles.medium import _sigmoid
from synthoracle.types import InputArray, OutputArray


class MediumOracle1E(Oracle):
    """The Rewired oracle — variant 1E.

    Same mechanism topology as 1A but with swapped input wiring.
    Designed so the 1A prior is substantially misleading: 3 false edges,
    4 missing edges, and Y2/Y4 essentially uncorrelated with 1A.

    Does NOT inherit from MediumOracle to avoid accidentally sharing
    any 1A-specific logic.
    """

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
        x1, x2, x3, x4, x5, x6 = x

        # M1: throughput from X3, X6, X1 (was X2, X4, X1)
        m1 = x3**1.37 * x6**0.82 * (1.0 - np.exp(-5.0 * x1))

        # M2: leakage from X4, X1 (was X3, X1)
        m2 = 0.55 * np.exp(-1.8 * x4 * np.sqrt(x1))

        # Z: coupling X2, X5 (was X4, X6)
        z = x2 / (x2 + 1.0 * x5)

        # Gate: threshold on X6 at ~0.38 (was X5)
        gate = float(_sigmoid(25.0 * (x6 - 0.38)))

        # Effective mechanisms
        m1_eff = m1 * z
        m2_eff = m2 * (1.0 - 0.6 * z)

        # M4: X5 * gate(X6) (was X6 * gate(X5))
        m4_mult = 1.0 + 0.4 * x5 * gate
        m4_add = 0.5 * x5 * gate

        # Outputs — same formulas, different inputs feeding in
        y1 = m1_eff * m4_mult + m4_add - m2_eff
        y2 = 0.85 * m2_eff + 0.35 * m1_eff / m4_mult + 0.25 * (1.0 - z)
        y3 = float(_sigmoid(8.1 * (x1 - 0.27))) * x4**0.5  # X4 not X3
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
        """1E-specific adversarial regions (all rewired)."""
        return [
            {
                "name": "X6_threshold",
                "description": "Gate sigmoid on X6 (was X5 in 1A)",
                "test": lambda x: 0.30 < x[5] < 0.46,
            },
            {
                "name": "X3_X6_interaction",
                "description": "M1 throughput synergy (was X2*X4 in 1A)",
                "test": lambda x: x[2] > 0.7 and x[5] > 0.7,
            },
            {
                "name": "Z_sensitive",
                "description": "X2/X5 coupling (was X4/X6 in 1A)",
                "test": lambda x: 0.3 < x[1] / (x[1] + x[4]) < 0.7,
            },
            {
                "name": "M2_regime",
                "description": "Leakage-dominated regime (low X1)",
                "test": lambda x: x[0] < 0.3,
            },
        ]

    def ground_truth(self) -> CausalDAG:
        """Ground-truth DAG for variant 1E (rewired)."""
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
            # M1 = f(X1, X3, X6) — rewired from (X1, X2, X4)
            Edge("X1", "M1", EdgeDifficulty.MEDIUM, "regime-dependent activation"),
            Edge("X3", "M1", EdgeDifficulty.EASY, "direct, strong (was X2)"),
            Edge("X6", "M1", EdgeDifficulty.EASY, "direct (was X4)"),
            # M2 = f(X1, X4) — rewired from (X1, X3)
            Edge("X1", "M2", EdgeDifficulty.MEDIUM, "regime-dependent"),
            Edge("X4", "M2", EdgeDifficulty.MEDIUM, "leakage driver (was X3)"),
            # Z = f(X2, X5) — rewired from (X4, X6)
            Edge("X2", "Z", EdgeDifficulty.HARD, "hidden coupling (was X4)"),
            Edge("X5", "Z", EdgeDifficulty.HARD, "hidden coupling (was X6)"),
            # M4_mult = f(X5, X6) — X5 modulates, X6 gates (swapped from 1A)
            Edge("X5", "M4_mult", EdgeDifficulty.HARD, "modulation (was X6)"),
            Edge("X6", "M4_mult", EdgeDifficulty.HARD, "threshold gate (was X5)"),
            # M4_add = f(X5, X6)
            Edge("X5", "M4_add", EdgeDifficulty.HARD, "modulation (was X6)"),
            Edge("X6", "M4_add", EdgeDifficulty.HARD, "threshold gate (was X5)"),
            # M1_eff, M2_eff — same topology
            Edge("M1", "M1_eff", EdgeDifficulty.EASY, "modulated by coupling"),
            Edge("Z", "M1_eff", EdgeDifficulty.HARD, "hidden coupling"),
            Edge("M2", "M2_eff", EdgeDifficulty.EASY, "modulated by coupling"),
            Edge("Z", "M2_eff", EdgeDifficulty.HARD, "hidden coupling"),
            # Y1 = M1_eff * M4_mult + M4_add - M2_eff
            Edge("M1_eff", "Y1", EdgeDifficulty.EASY, "performance"),
            Edge("M4_mult", "Y1", EdgeDifficulty.HARD, "threshold modulation"),
            Edge("M4_add", "Y1", EdgeDifficulty.HARD, "threshold boost"),
            Edge("M2_eff", "Y1", EdgeDifficulty.MEDIUM, "leakage penalty"),
            # Y2 = 0.85*M2_eff + 0.35*M1_eff/M4_mult + 0.25*(1-Z)
            Edge("M1_eff", "Y2", EdgeDifficulty.MEDIUM, "cost contribution"),
            Edge("M2_eff", "Y2", EdgeDifficulty.MEDIUM, "cost contribution"),
            Edge("M4_mult", "Y2", EdgeDifficulty.HARD, "threshold modulates cost"),
            Edge("Z", "Y2", EdgeDifficulty.HARD, "coupling cost"),
            # Y3 = sigmoid(X1) * X4^0.5 — rewired from X3
            Edge("X1", "Y3", EdgeDifficulty.MEDIUM, "reliability threshold"),
            Edge("X4", "Y3", EdgeDifficulty.MEDIUM, "reliability scaling (was X3)"),
            # Y4 = M1_eff / (1 + 2*M1_eff)
            Edge("M1_eff", "Y4", EdgeDifficulty.EASY, "saturating speed"),
        })

        return CausalDAG(nodes=nodes, edges=edges)
