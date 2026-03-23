"""Medium oracle variant 1C: structural change (adds mechanism M5).

M5 = X3^0.9 * X5^0.6 adds a new pathway from X3 and X5 to Y4.
In the base oracle, X3 does not reach Y4 and X5 does not reach Y4.
This creates a genuine structural surprise for transfer testing.
"""

from __future__ import annotations

from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, Node, NodeType
from synthoracle.oracles.medium import MediumOracle
from synthoracle.types import InputArray, OutputArray


class MediumOracle1C(MediumOracle):
    """Medium oracle with added mechanism M5.

    Y4 = M1_eff / (1 + 2.0 * M1_eff) + 0.3 * M5
    where M5 = X3^0.9 * X5^0.6

    All other outputs (Y1, Y2, Y3) are unchanged from the base oracle.
    """

    def __init__(self) -> None:
        super().__init__(variant="1A")

    def evaluate(self, x: InputArray) -> OutputArray:
        y = super().evaluate(x)
        x3, x5 = x[2], x[4]
        m5 = x3**0.9 * x5**0.6
        y[3] = y[3] + 0.3 * m5
        return y

    def adversarial_regions(self) -> list[dict[str, object]]:
        """Base regions + 1C-specific M5 coupling zone."""
        regions = super().adversarial_regions()
        regions.append({
            "name": "M5_coupling",
            "description": "1C-specific X3*X5 coupling for Y4",
            "test": lambda x: x[2] > 0.5 and x[4] > 0.5,
        })
        return regions

    def ground_truth(self) -> CausalDAG:
        """Base DAG + M5 node and 3 new edges (29 total)."""
        base_dag = super().ground_truth()
        new_nodes = base_dag.nodes | frozenset({
            Node("M5", NodeType.MECHANISM),
        })
        new_edges = base_dag.edges | frozenset({
            Edge("X3", "M5", EdgeDifficulty.MEDIUM, "new mechanism"),
            Edge("X5", "M5", EdgeDifficulty.MEDIUM, "new mechanism"),
            Edge("M5", "Y4", EdgeDifficulty.MEDIUM, "structural surprise"),
        })
        return CausalDAG(nodes=new_nodes, edges=new_edges)
