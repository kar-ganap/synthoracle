"""Causal DAG representation for synthetic oracles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    INPUT = "input"
    MECHANISM = "mechanism"
    OUTPUT = "output"


class EdgeDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class Node:
    name: str
    node_type: NodeType


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    difficulty: EdgeDifficulty
    description: str = ""


@dataclass(frozen=True)
class CausalDAG:
    """Immutable causal DAG with nodes and labeled edges.

    Ground-truth DAGs are constructed by oracles. Discovered DAGs are
    constructed by agents. Comparison via precision_recall().
    """

    nodes: frozenset[Node]
    edges: frozenset[Edge]

    def precision_recall(self, discovered: CausalDAG) -> tuple[float, float]:
        """Compare discovered edges against this (ground truth) DAG.

        Comparison is on (source, target) pairs only — difficulty and
        description metadata are ignored.

        Returns (precision, recall) where:
          precision = |discovered ∩ truth| / |discovered|  (0.0 if discovered is empty)
          recall    = |discovered ∩ truth| / |truth|       (0.0 if truth is empty)
        """
        truth_pairs = {(e.source, e.target) for e in self.edges}
        disc_pairs = {(e.source, e.target) for e in discovered.edges}

        if not truth_pairs and not disc_pairs:
            return (1.0, 1.0)

        overlap = truth_pairs & disc_pairs
        precision = len(overlap) / len(disc_pairs) if disc_pairs else 0.0
        recall = len(overlap) / len(truth_pairs) if truth_pairs else 0.0
        return (precision, recall)

    def project_to_io(self) -> CausalDAG:
        """Project DAG to input→output edges via transitive closure.

        Returns a simplified DAG containing only INPUT and OUTPUT nodes,
        with an edge (Xi, Yj) whenever there exists a directed path from
        Xi to Yj in the full DAG.
        """
        adj: dict[str, set[str]] = {n.name: set() for n in self.nodes}
        for e in self.edges:
            adj[e.source].add(e.target)

        input_nodes = {n for n in self.nodes if n.node_type == NodeType.INPUT}
        output_names = {n.name for n in self.nodes if n.node_type == NodeType.OUTPUT}

        io_edges: set[Edge] = set()
        for inp in input_nodes:
            reachable = self._reachable_from(inp.name, adj)
            for out_name in reachable & output_names:
                io_edges.add(Edge(
                    source=inp.name,
                    target=out_name,
                    difficulty=EdgeDifficulty.EASY,
                    description="projected",
                ))

        io_nodes = frozenset(
            n for n in self.nodes
            if n.node_type in (NodeType.INPUT, NodeType.OUTPUT)
        )
        return CausalDAG(nodes=io_nodes, edges=frozenset(io_edges))

    @staticmethod
    def _reachable_from(start: str, adj: dict[str, set[str]]) -> set[str]:
        """BFS to find all nodes reachable from start."""
        visited: set[str] = set()
        queue = [start]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            queue.extend(adj.get(current, set()))
        return visited
