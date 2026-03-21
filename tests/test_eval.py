"""Tests for the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from synthoracle.eval.metrics import compute_comparison, print_summary
from synthoracle.eval.plot import plot_final_hv_boxplot, plot_hv_comparison

# ---------------------------------------------------------------------------
# Fake result for testing (satisfies OptResult protocol)
# ---------------------------------------------------------------------------


@dataclass
class FakeResult:
    """Minimal result satisfying OptResult protocol."""

    X: npt.NDArray[np.float64]
    Y: npt.NDArray[np.float64]
    hypervolumes: list[float]
    pareto_X: npt.NDArray[np.float64]
    pareto_Y: npt.NDArray[np.float64]
    reference_point: npt.NDArray[np.float64]
    seed: int
    n_initial: int
    total_seconds: float


def _make_fake(
    seed: int, n_evals: int = 10, n_inputs: int = 4, n_outputs: int = 2,
    final_hv: float = 0.5,
) -> FakeResult:
    """Create a fake result with linearly increasing HV."""
    rng = np.random.default_rng(seed)
    hvs = [final_hv * (i + 1) / n_evals for i in range(n_evals)]
    return FakeResult(
        X=rng.random((n_evals, n_inputs)),
        Y=rng.random((n_evals, n_outputs)),
        hypervolumes=hvs,
        pareto_X=rng.random((3, n_inputs)),
        pareto_Y=rng.random((3, n_outputs)),
        reference_point=np.array([-0.5, -0.5]),
        seed=seed,
        n_initial=4,
        total_seconds=10.0 + seed,
    )


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_hv_curves_shape(self) -> None:
        results = {
            "A": [_make_fake(i, n_evals=10) for i in range(5)],
            "B": [_make_fake(i + 100, n_evals=10) for i in range(5)],
        }
        m = compute_comparison(results)
        assert m.hv_curves["A"].shape == (5, 10)
        assert m.hv_curves["B"].shape == (5, 10)

    def test_final_hv_shape(self) -> None:
        results = {"X": [_make_fake(i) for i in range(3)]}
        m = compute_comparison(results)
        assert m.final_hv["X"].shape == (3,)

    def test_mean_std_shape(self) -> None:
        results = {"X": [_make_fake(i, n_evals=8) for i in range(4)]}
        m = compute_comparison(results)
        assert m.hv_mean["X"].shape == (8,)
        assert m.hv_std["X"].shape == (8,)

    def test_known_values(self) -> None:
        """Two methods with known final HV."""
        results = {
            "Good": [_make_fake(i, final_hv=0.8) for i in range(3)],
            "Bad": [_make_fake(i + 10, final_hv=0.3) for i in range(3)],
        }
        m = compute_comparison(results)
        assert m.final_hv["Good"].mean() > m.final_hv["Bad"].mean()

    def test_wall_times(self) -> None:
        results = {"X": [_make_fake(i) for i in range(3)]}
        m = compute_comparison(results)
        assert m.wall_times["X"].shape == (3,)
        assert all(t > 0 for t in m.wall_times["X"])

    def test_pareto_sizes(self) -> None:
        results = {"X": [_make_fake(i) for i in range(3)]}
        m = compute_comparison(results)
        assert m.pareto_sizes["X"].shape == (3,)

    def test_method_names(self) -> None:
        results = {"BO": [_make_fake(0)], "VR": [_make_fake(1)]}
        m = compute_comparison(results)
        assert m.method_names == ["BO", "VR"]

    def test_print_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = {
            "BO": [_make_fake(i, final_hv=0.5) for i in range(3)],
            "VR": [_make_fake(i + 10, final_hv=0.4) for i in range(3)],
        }
        m = compute_comparison(results)
        print_summary(m)
        captured = capsys.readouterr()
        assert "BO" in captured.out
        assert "VR" in captured.out


# ---------------------------------------------------------------------------
# TestPlot
# ---------------------------------------------------------------------------


class TestPlot:
    def test_hv_comparison_generates_file(self, tmp_path: Path) -> None:
        results = {
            "BO": [_make_fake(i, final_hv=0.5) for i in range(3)],
            "VR": [_make_fake(i + 10, final_hv=0.4) for i in range(3)],
        }
        m = compute_comparison(results)
        out = tmp_path / "hv_comparison.png"
        plot_hv_comparison(m, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_boxplot_generates_file(self, tmp_path: Path) -> None:
        results = {
            "BO": [_make_fake(i, final_hv=0.5) for i in range(3)],
            "VR": [_make_fake(i + 10, final_hv=0.4) for i in range(3)],
        }
        m = compute_comparison(results)
        out = tmp_path / "boxplot.png"
        plot_final_hv_boxplot(m, out)
        assert out.exists()
        assert out.stat().st_size > 0
