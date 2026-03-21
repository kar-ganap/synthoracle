"""SynthOracle: Synthetic oracle benchmark for scientific reasoning agents."""

from synthoracle.characterize import CharacterizationResult, characterize
from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, Node, NodeType
from synthoracle.oracle import Oracle
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_1c import MediumOracle1C
from synthoracle.oracles.simple import SimpleOracle

__all__ = [
    "CausalDAG",
    "CharacterizationResult",
    "Edge",
    "EdgeDifficulty",
    "MediumOracle",
    "MediumOracle1C",
    "Node",
    "NodeType",
    "Oracle",
    "SimpleOracle",
    "characterize",
]
