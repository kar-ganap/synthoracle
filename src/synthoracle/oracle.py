"""Abstract base class for synthetic oracles."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from synthoracle.dag import CausalDAG
from synthoracle.types import InputArray, OutputArray


class Oracle(ABC):
    """Synthetic oracle with known causal structure.

    Each oracle maps inputs X to outputs Y through hidden mechanisms.
    The ground-truth causal DAG is available for evaluation.
    """

    @abstractmethod
    def evaluate(self, x: InputArray) -> OutputArray:
        """Evaluate the oracle at a single input point.

        Parameters
        ----------
        x : ndarray of shape (n_inputs,)
            Input vector within bounds.

        Returns
        -------
        y : ndarray of shape (n_outputs,)
            Output vector.
        """
        ...

    @abstractmethod
    def ground_truth(self) -> CausalDAG:
        """Return the ground-truth causal DAG."""
        ...

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        ...

    @property
    @abstractmethod
    def n_outputs(self) -> int:
        ...

    @property
    @abstractmethod
    def bounds(self) -> InputArray:
        """Input bounds, shape (n_inputs, 2). bounds[i] = [lower, upper]."""
        ...

    @property
    @abstractmethod
    def input_names(self) -> tuple[str, ...]:
        ...

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]:
        ...

    @property
    @abstractmethod
    def output_directions(self) -> tuple[str, ...]:
        """Per-output optimization direction: 'maximize', 'minimize', or 'threshold'."""
        ...

    def adversarial_regions(self) -> list[dict[str, object]]:
        """Return regions where the system is nonlinear or interactive.

        Each region is a dict with:
          - name: str — short identifier
          - description: str — what makes this region challenging
          - test: callable(x) -> bool — whether a point is in this region

        Override in subclasses with oracle-specific regions.
        """
        return []

    def evaluate_batch(self, X: InputArray) -> OutputArray:
        """Evaluate oracle at multiple input points.

        Parameters
        ----------
        X : ndarray of shape (n_points, n_inputs)

        Returns
        -------
        Y : ndarray of shape (n_points, n_outputs)
        """
        return np.array([self.evaluate(x) for x in X])
