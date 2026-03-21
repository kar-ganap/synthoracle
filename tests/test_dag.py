"""Tests for CausalDAG representation."""

from __future__ import annotations

import pytest

from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, Node, NodeType


def _make_simple_dag() -> CausalDAG:
    """A -> M -> Y, B -> M -> Y."""
    nodes = frozenset({
        Node("A", NodeType.INPUT),
        Node("B", NodeType.INPUT),
        Node("M", NodeType.MECHANISM),
        Node("Y", NodeType.OUTPUT),
    })
    edges = frozenset({
        Edge("A", "M", EdgeDifficulty.EASY),
        Edge("B", "M", EdgeDifficulty.MEDIUM),
        Edge("M", "Y", EdgeDifficulty.EASY),
    })
    return CausalDAG(nodes=nodes, edges=edges)


class TestDAGConstruction:
    def test_nodes_accessible(self) -> None:
        dag = _make_simple_dag()
        names = {n.name for n in dag.nodes}
        assert names == {"A", "B", "M", "Y"}

    def test_edges_accessible(self) -> None:
        dag = _make_simple_dag()
        assert len(dag.edges) == 3

    def test_immutability(self) -> None:
        dag = _make_simple_dag()
        with pytest.raises(AttributeError):
            dag.nodes = frozenset()  # type: ignore[misc]


class TestPrecisionRecall:
    def test_perfect_match(self) -> None:
        dag = _make_simple_dag()
        p, r = dag.precision_recall(dag)
        assert p == 1.0
        assert r == 1.0

    def test_partial_discovery(self) -> None:
        truth = _make_simple_dag()
        # Discover only A->M
        discovered = CausalDAG(
            nodes=frozenset({Node("A", NodeType.INPUT), Node("M", NodeType.MECHANISM)}),
            edges=frozenset({Edge("A", "M", EdgeDifficulty.EASY)}),
        )
        p, r = truth.precision_recall(discovered)
        assert p == 1.0  # 1/1 discovered edges are correct
        assert r == pytest.approx(1 / 3)  # 1 of 3 truth edges found

    def test_empty_discovered(self) -> None:
        truth = _make_simple_dag()
        empty = CausalDAG(nodes=frozenset(), edges=frozenset())
        p, r = truth.precision_recall(empty)
        assert p == 0.0
        assert r == 0.0

    def test_false_positive(self) -> None:
        truth = _make_simple_dag()
        # Discover A->Y (not a direct edge in truth)
        discovered = CausalDAG(
            nodes=frozenset({Node("A", NodeType.INPUT), Node("Y", NodeType.OUTPUT)}),
            edges=frozenset({Edge("A", "Y", EdgeDifficulty.EASY)}),
        )
        p, r = truth.precision_recall(discovered)
        assert p == 0.0  # A->Y is not in truth
        assert r == 0.0

    def test_both_empty(self) -> None:
        empty = CausalDAG(nodes=frozenset(), edges=frozenset())
        p, r = empty.precision_recall(empty)
        assert p == 1.0
        assert r == 1.0

    def test_ignores_metadata(self) -> None:
        """Precision/recall compares (source, target) only, not difficulty/description."""
        truth = _make_simple_dag()
        # Same edges but different difficulty labels
        discovered = CausalDAG(
            nodes=truth.nodes,
            edges=frozenset({
                Edge(e.source, e.target, EdgeDifficulty.HARD, "wrong description")
                for e in truth.edges
            }),
        )
        p, r = truth.precision_recall(discovered)
        assert p == 1.0
        assert r == 1.0


class TestProjectToIO:
    def test_simple_projection(self) -> None:
        dag = _make_simple_dag()
        projected = dag.project_to_io()
        # A->M->Y and B->M->Y should produce A->Y and B->Y
        edge_pairs = {(e.source, e.target) for e in projected.edges}
        assert edge_pairs == {("A", "Y"), ("B", "Y")}

    def test_projected_nodes_are_io_only(self) -> None:
        dag = _make_simple_dag()
        projected = dag.project_to_io()
        types = {n.node_type for n in projected.nodes}
        assert types <= {NodeType.INPUT, NodeType.OUTPUT}

    def test_no_mechanism_nodes_in_projection(self) -> None:
        dag = _make_simple_dag()
        projected = dag.project_to_io()
        names = {n.name for n in projected.nodes}
        assert "M" not in names
