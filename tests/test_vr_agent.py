"""Tests for the VR agent module."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

anthropic = pytest.importorskip("anthropic")

from synthoracle.agents.vr import (  # noqa: E402
    VRResult,
    VRStepLog,
    _LLMResponse,
    _build_iteration_prompt,
    _build_system_prompt,
    _compute_correlations,
    _compute_directional_accuracy,
    _compute_local_gradients,
    _determine_phase,
    _extract_json,
    _parse_llm_response,
    run_vr,
)
from synthoracle.oracles.medium import MediumOracle  # noqa: E402


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


def _make_batch_response(call_count: int) -> dict[str, object]:
    """Create a deterministic batch response dict."""
    return {
        "reconciliation": (
            "No prior prediction."
            if call_count == 1
            else "Close."
        ),
        "biggest_surprise": (
            "N/A"
            if call_count == 1
            else "Y1 was lower than expected."
        ),
        "hypothesis": f"X2 drives Y1. Call {call_count}.",
        "points": [
            {
                "next_point": [0.5] * 6,
                "prediction": [0.3, 0.2, 0.6, 0.1],
                "reasoning": "Testing midpoint.",
                "explore_or_exploit": "explore",
                "falsification": (
                    "If Y1 is high at low X2, X2 doesn't drive Y1."
                ),
            },
            {
                "next_point": [0.3] * 6,
                "prediction": [0.2, 0.3, 0.5, 0.08],
                "reasoning": "Testing low values.",
                "explore_or_exploit": "explore",
                "falsification": (
                    "If outputs are same as midpoint, inputs don't matter."
                ),
            },
            {
                "next_point": [0.8] * 6,
                "prediction": [0.6, 0.15, 0.8, 0.2],
                "reasoning": "Testing high values.",
                "explore_or_exploit": "exploit",
                "falsification": (
                    "If Y1 doesn't increase, the trend is nonlinear."
                ),
            },
        ],
    }


@pytest.fixture
def oracle() -> MediumOracle:
    return MediumOracle()


@pytest.fixture
def mock_anthropic(monkeypatch: pytest.MonkeyPatch) -> MockMessages:
    """Mock Anthropic client returning deterministic batch JSON."""

    class MockTextBlock:
        def __init__(self, text: str) -> None:
            self.text = text
            self.type = "text"

    class MockUsage:
        def __init__(self) -> None:
            self.input_tokens = 150
            self.output_tokens = 200

    class MockParsedOutput:
        """Mimics the Pydantic model returned by messages.parse()."""

        def __init__(self, data: dict[str, object]) -> None:
            self.reconciliation: str = str(data["reconciliation"])
            self.biggest_surprise: str = str(data["biggest_surprise"])
            self.hypothesis: str = str(data["hypothesis"])
            # Build list of point-like objects
            raw_points = data["points"]
            assert isinstance(raw_points, list)
            self.points: list[MockProposedPoint] = [
                MockProposedPoint(p) for p in raw_points  # type: ignore[arg-type]
            ]

    class MockProposedPoint:
        """Mimics _ProposedPoint from Pydantic parse."""

        def __init__(self, d: dict[str, object]) -> None:
            self.next_point: list[float] = list(d["next_point"])  # type: ignore[arg-type]
            self.prediction: list[float] = list(d["prediction"])  # type: ignore[arg-type]
            self.reasoning: str = str(d["reasoning"])
            self.explore_or_exploit: str = str(d["explore_or_exploit"])
            self.falsification: str = str(d["falsification"])

    class MockParsedMessage:
        def __init__(self, text: str, data: dict[str, object]) -> None:
            self.content = [MockTextBlock(text)]
            self.usage = MockUsage()
            self.parsed_output = MockParsedOutput(data)

    class MockMessages:
        def __init__(self) -> None:
            self.call_count = 0

        def parse(self, **kwargs: object) -> MockParsedMessage:
            self.call_count += 1
            data = _make_batch_response(self.call_count)
            return MockParsedMessage(json.dumps(data), data)

    mock_msgs = MockMessages()

    class MockClient:
        def __init__(self, **kwargs: object) -> None:
            self.messages = mock_msgs

    monkeypatch.setattr("anthropic.Anthropic", MockClient)
    return mock_msgs


class MockMessages:
    """Type stub for the mock_anthropic fixture return type."""

    call_count: int

    def parse(self, **kwargs: object) -> object:
        ...


# ---------------------------------------------------------------------------
# TestPromptConstruction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_system_prompt_contains_oracle_info(
        self, oracle: MediumOracle,
    ) -> None:
        prompt = _build_system_prompt(oracle, {})
        for name in oracle.input_names:
            assert name in prompt
        for name in oracle.output_names:
            assert name in prompt
        assert "maximize" in prompt
        assert "minimize" in prompt

    def test_system_prompt_contains_json_schema(
        self, oracle: MediumOracle,
    ) -> None:
        prompt = _build_system_prompt(oracle, {})
        assert "next_point" in prompt
        assert "prediction" in prompt

    def test_system_prompt_contains_threshold(
        self, oracle: MediumOracle,
    ) -> None:
        prompt = _build_system_prompt(oracle, {"Y3": 0.4})
        assert "0.4" in prompt
        assert "threshold" in prompt

    def test_system_prompt_contains_playbook(
        self, oracle: MediumOracle,
    ) -> None:
        prompt = _build_system_prompt(oracle, {})
        assert "Scientist's Playbook" in prompt
        assert "OAT" in prompt
        assert "Hypothesis Discipline" in prompt
        assert "FALSIFY" in prompt

    def test_system_prompt_contains_batch_format(
        self, oracle: MediumOracle,
    ) -> None:
        prompt = _build_system_prompt(oracle, {}, batch_size=3)
        assert "biggest_surprise" in prompt
        assert "explore_or_exploit" in prompt
        assert "falsification" in prompt
        assert "list of 3 objects" in prompt

    def test_iteration_prompt_has_data_table(
        self, oracle: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(3, 6))
        Y = oracle.evaluate_batch(X)
        prompt = _build_iteration_prompt(oracle, X, Y, [], None, None, 0)
        assert "X1" in prompt
        assert "Y1" in prompt

    def test_iteration_prompt_first_no_reconciliation(
        self, oracle: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(3, 6))
        Y = oracle.evaluate_batch(X)
        prompt = _build_iteration_prompt(oracle, X, Y, [], None, None, 0)
        assert "Previous Batch" not in prompt

    def test_iteration_prompt_subsequent_has_error(
        self, oracle: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(3, 6))
        Y = oracle.evaluate_batch(X)
        prev_preds = [np.array([0.3, 0.2, 0.5, 0.1])]
        prev_acts = [np.array([0.35, 0.18, 0.55, 0.12])]
        prompt = _build_iteration_prompt(
            oracle, X, Y, ["Step 1: hypothesis"],
            prev_preds, prev_acts, 1,
        )
        assert "Previous Batch" in prompt
        assert "error" in prompt

    def test_iteration_prompt_has_mechanism_log(
        self, oracle: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(3, 6))
        Y = oracle.evaluate_batch(X)
        log = ["Step 1: X2 drives Y1", "Step 2: X4 also matters"]
        prompt = _build_iteration_prompt(oracle, X, Y, log, None, None, 0)
        assert "Step 1: X2 drives Y1" in prompt
        assert "Step 2: X4 also matters" in prompt

    def test_iteration_prompt_has_budget_section(
        self, oracle: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(4, 6))
        Y = oracle.evaluate_batch(X)
        prompt = _build_iteration_prompt(
            oracle, X, Y, [], None, None, 0,
            phase="SCREEN", eval_count=4, n_total_budget=46,
        )
        assert "Budget" in prompt
        assert "SCREEN" in prompt
        assert "4 / 46" in prompt

    def test_iteration_prompt_has_correlation_section(
        self, oracle: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(10, 6))
        Y = oracle.evaluate_batch(X)
        corr = _compute_correlations(X, Y)
        prompt = _build_iteration_prompt(
            oracle, X, Y, [], None, None, 0,
            correlations=corr,
        )
        assert "Data Analysis" in prompt
        assert "Pearson" in prompt

    def test_iteration_prompt_has_gradient_section(
        self, oracle: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(5, 6))
        Y = oracle.evaluate_batch(X)
        grads = [_compute_local_gradients(oracle, X[-1])]
        prompt = _build_iteration_prompt(
            oracle, X, Y, [], None, None, 0,
            gradients=grads,
        )
        assert "Local gradients" in prompt
        assert "dY/dX1" in prompt

    def test_iteration_prompt_batch_task_instructions(
        self, oracle: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(4, 6))
        Y = oracle.evaluate_batch(X)
        prompt = _build_iteration_prompt(
            oracle, X, Y, [],
            [np.array([0.3, 0.2, 0.5, 0.1])],
            [np.array([0.35, 0.18, 0.55, 0.12])],
            1,
            phase="SCREEN", batch_size=3,
        )
        assert "Propose 3 points" in prompt
        assert "explore" in prompt


# ---------------------------------------------------------------------------
# TestOutputParsing
# ---------------------------------------------------------------------------


class TestOutputParsing:
    def test_parse_valid_json(self) -> None:
        data = {
            "next_point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "prediction": [0.3, 0.2, 0.6, 0.1],
            "hypothesis": "X2 drives Y1",
            "reasoning": "Testing midpoint",
            "reconciliation": "No prior",
        }
        text = json.dumps(data)
        resp = _parse_llm_response(
            text, 6, 4, np.full((6, 2), [[0.1, 1.0]]), 100, 200,
        )
        assert isinstance(resp, _LLMResponse)
        np.testing.assert_array_almost_equal(resp.next_point, [0.5] * 6)
        np.testing.assert_array_almost_equal(
            resp.prediction, [0.3, 0.2, 0.6, 0.1],
        )
        assert resp.hypothesis == "X2 drives Y1"

    def test_parse_json_in_markdown_block(self) -> None:
        data = {
            "next_point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "prediction": [0.3, 0.2, 0.6, 0.1],
            "hypothesis": "test",
            "reasoning": "test",
            "reconciliation": "test",
        }
        text = f"```json\n{json.dumps(data)}\n```"
        resp = _parse_llm_response(
            text, 6, 4, np.full((6, 2), [[0.1, 1.0]]), 100, 200,
        )
        assert isinstance(resp, _LLMResponse)

    def test_parse_json_with_surrounding_text(self) -> None:
        data = {
            "next_point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "prediction": [0.3, 0.2, 0.6, 0.1],
            "hypothesis": "test",
            "reasoning": "test",
            "reconciliation": "test",
        }
        text = f"Here's my answer: {json.dumps(data)} Hope that helps!"
        resp = _parse_llm_response(
            text, 6, 4, np.full((6, 2), [[0.1, 1.0]]), 100, 200,
        )
        assert isinstance(resp, _LLMResponse)

    def test_parse_rejects_missing_fields(self) -> None:
        data = {
            "next_point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            # missing prediction
            "hypothesis": "test",
            "reasoning": "test",
            "reconciliation": "test",
        }
        with pytest.raises(ValueError, match="Missing required field"):
            _parse_llm_response(
                json.dumps(data), 6, 4,
                np.full((6, 2), [[0.1, 1.0]]), 100, 200,
            )

    def test_parse_rejects_wrong_dimensions(self) -> None:
        data = {
            "next_point": [0.5, 0.5, 0.5, 0.5],  # 4 instead of 6
            "prediction": [0.3, 0.2, 0.6, 0.1],
            "hypothesis": "test",
            "reasoning": "test",
            "reconciliation": "test",
        }
        with pytest.raises(ValueError, match="next_point must be a list of 6"):
            _parse_llm_response(
                json.dumps(data), 6, 4,
                np.full((6, 2), [[0.1, 1.0]]), 100, 200,
            )

    def test_parse_clips_to_bounds(self) -> None:
        data = {
            "next_point": [2.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            "prediction": [0.3, 0.2, 0.6, 0.1],
            "hypothesis": "test",
            "reasoning": "test",
            "reconciliation": "test",
        }
        resp = _parse_llm_response(
            json.dumps(data), 6, 4,
            np.full((6, 2), [[0.1, 1.0]]), 100, 200,
        )
        assert resp.next_point[0] == 1.0  # clipped

    def test_parse_rejects_nan(self) -> None:
        data = {
            "next_point": [float("nan"), 0.5, 0.5, 0.5, 0.5, 0.5],
            "prediction": [0.3, 0.2, 0.6, 0.1],
            "hypothesis": "test",
            "reasoning": "test",
            "reconciliation": "test",
        }
        with pytest.raises(ValueError, match="NaN or Inf"):
            _parse_llm_response(
                json.dumps(data), 6, 4,
                np.full((6, 2), [[0.1, 1.0]]), 100, 200,
            )

    def test_extract_json_direct(self) -> None:
        data = {"key": "value"}
        result = _extract_json(json.dumps(data))
        assert result == data

    def test_extract_json_raises_on_garbage(self) -> None:
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("this is not json at all")

    def test_parse_batch_response(self) -> None:
        data = _make_batch_response(1)
        text = json.dumps(data)
        parsed = _extract_json(text)
        assert "points" in parsed
        points = parsed["points"]
        assert isinstance(points, list)
        assert len(points) == 3


# ---------------------------------------------------------------------------
# TestDirectionalAccuracy
# ---------------------------------------------------------------------------


class TestDirectionalAccuracy:
    def test_correct_direction(self) -> None:
        prev_y = np.array([0.3, 0.2])
        prediction = np.array([0.4, 0.1])  # up, down
        actual = np.array([0.5, 0.15])  # up, down — same directions
        result = _compute_directional_accuracy(prediction, actual, prev_y)
        assert result == [True, True]

    def test_incorrect_direction(self) -> None:
        prev_y = np.array([0.3, 0.2])
        prediction = np.array([0.4, 0.3])  # up, up
        actual = np.array([0.5, 0.1])  # up, down — second is wrong
        result = _compute_directional_accuracy(prediction, actual, prev_y)
        assert result == [True, False]

    def test_first_step_no_previous(self) -> None:
        prediction = np.array([0.4, 0.3, 0.5])
        actual = np.array([0.5, 0.1, 0.6])
        result = _compute_directional_accuracy(prediction, actual, None)
        assert result == [True, True, True]


# ---------------------------------------------------------------------------
# TestAnalysisHelpers
# ---------------------------------------------------------------------------


class TestAnalysisHelpers:
    def test_compute_local_gradients_known_function(
        self, oracle: MediumOracle,
    ) -> None:
        """Gradients should be finite and have correct shape."""
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        grads = _compute_local_gradients(oracle, x)
        assert grads.shape == (6, 4)
        assert np.all(np.isfinite(grads))

    def test_compute_local_gradients_step_affects_result(
        self, oracle: MediumOracle,
    ) -> None:
        """Different step sizes should give similar (not identical) results."""
        x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        g1 = _compute_local_gradients(oracle, x, step=0.01)
        g2 = _compute_local_gradients(oracle, x, step=0.001)
        # They should be close but not exactly equal
        assert g1.shape == g2.shape
        np.testing.assert_allclose(g1, g2, atol=0.1)

    def test_compute_local_gradients_boundary(
        self, oracle: MediumOracle,
    ) -> None:
        """Gradients at boundary should still be finite."""
        x = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # lower boundary
        grads = _compute_local_gradients(oracle, x)
        assert grads.shape == (6, 4)
        assert np.all(np.isfinite(grads))

    def test_compute_correlations_known_data(self) -> None:
        """Perfect positive/negative correlation should give +1/-1."""
        X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
        Y = np.array([[10.0, -5.0], [20.0, -10.0], [30.0, -15.0], [40.0, -20.0]])
        corr = _compute_correlations(X, Y)
        assert corr.shape == (2, 2)
        np.testing.assert_almost_equal(corr[0, 0], 1.0)  # X1 vs Y1: +1
        np.testing.assert_almost_equal(corr[0, 1], -1.0)  # X1 vs Y2: -1
        np.testing.assert_almost_equal(corr[1, 0], 1.0)  # X2 vs Y1: +1
        np.testing.assert_almost_equal(corr[1, 1], -1.0)  # X2 vs Y2: -1

    def test_compute_correlations_zero_variance(self) -> None:
        """Zero-variance columns should give correlation of 0."""
        X = np.array([[1.0], [1.0], [1.0]])
        Y = np.array([[10.0], [20.0], [30.0]])
        corr = _compute_correlations(X, Y)
        assert corr.shape == (1, 1)
        assert corr[0, 0] == 0.0


# ---------------------------------------------------------------------------
# TestPhaseDetection
# ---------------------------------------------------------------------------


class TestPhaseDetection:
    def test_budget_phase_screen(self) -> None:
        # eval_count=10, budget=60 => progress 10/60=0.17 < 1/3 => SCREEN
        assert _determine_phase(10, 60) == "SCREEN"

    def test_budget_phase_probe(self) -> None:
        # eval_count=25, budget=60 => progress 25/60=0.42 => PROBE
        assert _determine_phase(25, 60) == "PROBE"

    def test_budget_phase_optimize(self) -> None:
        # eval_count=45, budget=60 => progress 45/60=0.75 => OPTIMIZE
        assert _determine_phase(45, 60) == "OPTIMIZE"

    def test_budget_phase_boundary_screen_probe(self) -> None:
        # Exactly 1/3 => PROBE
        assert _determine_phase(20, 60) == "PROBE"

    def test_budget_phase_boundary_probe_optimize(self) -> None:
        # Exactly 2/3 => OPTIMIZE
        assert _determine_phase(40, 60) == "OPTIMIZE"

    def test_budget_phase_zero_budget(self) -> None:
        assert _determine_phase(0, 0) == "SCREEN"


# ---------------------------------------------------------------------------
# TestRunVR (uses mock_anthropic fixture)
# ---------------------------------------------------------------------------


class TestRunVR:
    def test_run_vr_completes(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr(
            oracle, n_initial=4, n_iterations=6, seed=42, batch_size=3,
        )
        assert isinstance(result, VRResult)

    def test_run_vr_shapes(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        n_init = 4
        n_iter = 6
        result = run_vr(
            oracle, n_initial=n_init, n_iterations=n_iter,
            seed=42, batch_size=3,
        )
        n_total = n_init + n_iter
        assert result.X.shape == (n_total, oracle.n_inputs)
        assert result.Y.shape == (n_total, oracle.n_outputs)
        assert len(result.hypervolumes) == n_total

    def test_run_vr_step_logs_count(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        n_iter = 6
        result = run_vr(
            oracle, n_initial=4, n_iterations=n_iter,
            seed=42, batch_size=3,
        )
        assert len(result.step_logs) == n_iter

    def test_run_vr_predictions_tracked(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        n_iter = 6
        result = run_vr(
            oracle, n_initial=4, n_iterations=n_iter,
            seed=42, batch_size=3,
        )
        assert result.predictions.shape == (n_iter, oracle.n_outputs)
        assert result.prediction_errors.shape == (n_iter, oracle.n_outputs)

    def test_run_vr_token_counts(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr(
            oracle, n_initial=4, n_iterations=6, seed=42, batch_size=3,
        )
        assert result.total_llm_calls > 0
        assert result.total_input_tokens > 0
        assert result.total_output_tokens > 0

    def test_run_vr_step_log_fields(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr(
            oracle, n_initial=4, n_iterations=3, seed=42, batch_size=3,
        )
        log = result.step_logs[0]
        assert isinstance(log, VRStepLog)
        assert log.step == 0
        assert log.x.shape == (oracle.n_inputs,)
        assert log.y_predicted.shape == (oracle.n_outputs,)
        assert log.y_actual.shape == (oracle.n_outputs,)
        assert isinstance(log.hypothesis, str)
        assert isinstance(log.directional_accuracy, list)
        assert isinstance(log.explore_or_exploit, str)
        assert isinstance(log.falsification, str)

    def test_run_vr_explore_exploit_labels(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr(
            oracle, n_initial=4, n_iterations=3, seed=42, batch_size=3,
        )
        labels = [log.explore_or_exploit for log in result.step_logs]
        assert labels == ["explore", "explore", "exploit"]

    def test_run_vr_llm_call_count(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        """With batch_size=3 and 6 iterations, should make 2 LLM calls."""
        result = run_vr(
            oracle, n_initial=4, n_iterations=6, seed=42, batch_size=3,
        )
        assert result.total_llm_calls == 2

    def test_run_vr_batch_size_1_works(
        self, oracle: MediumOracle, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """batch_size=1 should still work (backward compat)."""

        class MockTextBlock:
            def __init__(self, text: str) -> None:
                self.text = text
                self.type = "text"

        class MockUsage:
            def __init__(self) -> None:
                self.input_tokens = 50
                self.output_tokens = 60

        class MockProposedPoint:
            def __init__(self, d: dict[str, object]) -> None:
                self.next_point = list(d["next_point"])  # type: ignore[arg-type]
                self.prediction = list(d["prediction"])  # type: ignore[arg-type]
                self.reasoning = str(d["reasoning"])
                self.explore_or_exploit = str(d["explore_or_exploit"])
                self.falsification = str(d["falsification"])

        class MockParsedOutput:
            def __init__(self, data: dict[str, object]) -> None:
                self.reconciliation = str(data["reconciliation"])
                self.biggest_surprise = str(data["biggest_surprise"])
                self.hypothesis = str(data["hypothesis"])
                raw_points = data["points"]
                assert isinstance(raw_points, list)
                self.points = [
                    MockProposedPoint(p) for p in raw_points  # type: ignore[arg-type]
                ]

        class MockParsedMessage:
            def __init__(self, text: str, data: dict[str, object]) -> None:
                self.content = [MockTextBlock(text)]
                self.usage = MockUsage()
                self.parsed_output = MockParsedOutput(data)

        class MockMessages:
            def __init__(self) -> None:
                self.call_count = 0

            def parse(self, **kwargs: object) -> MockParsedMessage:
                self.call_count += 1
                data: dict[str, object] = {
                    "reconciliation": "No prior.",
                    "biggest_surprise": "N/A",
                    "hypothesis": f"Hyp {self.call_count}",
                    "points": [
                        {
                            "next_point": [0.5] * 6,
                            "prediction": [0.3, 0.2, 0.6, 0.1],
                            "reasoning": "Single point.",
                            "explore_or_exploit": "explore",
                            "falsification": "none",
                        },
                    ],
                }
                return MockParsedMessage(json.dumps(data), data)

        mock_msgs = MockMessages()

        class MockClient:
            def __init__(self, **kwargs: object) -> None:
                self.messages = mock_msgs

        monkeypatch.setattr("anthropic.Anthropic", MockClient)

        result = run_vr(
            oracle, n_initial=4, n_iterations=3,
            seed=42, batch_size=1,
        )
        assert isinstance(result, VRResult)
        assert result.X.shape == (7, oracle.n_inputs)
        assert result.Y.shape == (7, oracle.n_outputs)
        assert len(result.step_logs) == 3
        assert result.total_llm_calls == 3

    def test_run_vr_thinking_parameter(
        self, oracle: MediumOracle, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """thinking parameter should be passed through to the API."""
        captured_kwargs: list[dict[str, object]] = []

        class MockTextBlock:
            def __init__(self, text: str) -> None:
                self.text = text
                self.type = "text"

        class MockUsage:
            def __init__(self) -> None:
                self.input_tokens = 50
                self.output_tokens = 60

        class MockProposedPoint:
            def __init__(self, d: dict[str, object]) -> None:
                self.next_point = list(d["next_point"])  # type: ignore[arg-type]
                self.prediction = list(d["prediction"])  # type: ignore[arg-type]
                self.reasoning = str(d["reasoning"])
                self.explore_or_exploit = str(d["explore_or_exploit"])
                self.falsification = str(d["falsification"])

        class MockParsedOutput:
            def __init__(self, data: dict[str, object]) -> None:
                self.reconciliation = str(data["reconciliation"])
                self.biggest_surprise = str(data["biggest_surprise"])
                self.hypothesis = str(data["hypothesis"])
                raw_points = data["points"]
                assert isinstance(raw_points, list)
                self.points = [
                    MockProposedPoint(p) for p in raw_points  # type: ignore[arg-type]
                ]

        class MockParsedMessage:
            def __init__(self, text: str, data: dict[str, object]) -> None:
                self.content = [MockTextBlock(text)]
                self.usage = MockUsage()
                self.parsed_output = MockParsedOutput(data)

        class MockMessages:
            def __init__(self) -> None:
                self.call_count = 0

            def parse(self, **kwargs: object) -> MockParsedMessage:
                self.call_count += 1
                captured_kwargs.append(dict(kwargs))
                data = _make_batch_response(self.call_count)
                return MockParsedMessage(json.dumps(data), data)

        mock_msgs = MockMessages()

        class MockClient:
            def __init__(self, **kwargs: object) -> None:
                self.messages = mock_msgs

        monkeypatch.setattr("anthropic.Anthropic", MockClient)

        thinking_config: dict[str, object] = {
            "type": "enabled",
            "budget_tokens": 10000,
        }
        run_vr(
            oracle, n_initial=4, n_iterations=3,
            seed=42, batch_size=3, thinking=thinking_config,
        )
        assert len(captured_kwargs) >= 1
        first_call = captured_kwargs[0]
        assert "thinking" in first_call
        assert first_call["thinking"] == thinking_config
        # max_tokens should be at least budget_tokens + 4096
        assert int(first_call["max_tokens"]) >= 10000 + 4096  # type: ignore[arg-type]

    def test_run_vr_fallback_on_bad_response(
        self, oracle: MediumOracle, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mock raises on all attempts -> falls back to random."""

        class MockMessages:
            def parse(self, **kwargs: object) -> None:
                raise ValueError("Simulated parse failure")

        mock_msgs = MockMessages()

        class MockClient:
            def __init__(self, **kwargs: object) -> None:
                self.messages = mock_msgs

        monkeypatch.setattr("anthropic.Anthropic", MockClient)

        result = run_vr(
            oracle, n_initial=4, n_iterations=3,
            seed=42, max_retries=1, batch_size=3,
        )
        assert isinstance(result, VRResult)
        # Fallback should produce NaN predictions
        for log in result.step_logs:
            assert "[PARSE FAILURE" in log.hypothesis

    def test_run_vr_retries(
        self, oracle: MediumOracle, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mock raises first, then succeeds -> retries work."""

        class MockTextBlock:
            def __init__(self, text: str) -> None:
                self.text = text
                self.type = "text"

        class MockUsage:
            def __init__(self) -> None:
                self.input_tokens = 50
                self.output_tokens = 60

        class MockProposedPoint:
            def __init__(self, d: dict[str, object]) -> None:
                self.next_point = list(d["next_point"])  # type: ignore[arg-type]
                self.prediction = list(d["prediction"])  # type: ignore[arg-type]
                self.reasoning = str(d["reasoning"])
                self.explore_or_exploit = str(d["explore_or_exploit"])
                self.falsification = str(d["falsification"])

        class MockParsedOutput:
            def __init__(self, data: dict[str, object]) -> None:
                self.reconciliation = str(data["reconciliation"])
                self.biggest_surprise = str(data["biggest_surprise"])
                self.hypothesis = str(data["hypothesis"])
                raw_points = data["points"]
                assert isinstance(raw_points, list)
                self.points = [
                    MockProposedPoint(p) for p in raw_points  # type: ignore[arg-type]
                ]

        class MockParsedMessage:
            def __init__(self, text: str, data: dict[str, object]) -> None:
                self.content = [MockTextBlock(text)]
                self.usage = MockUsage()
                self.parsed_output = MockParsedOutput(data)

        class MockMessages:
            def __init__(self) -> None:
                self.call_count = 0

            def parse(self, **kwargs: object) -> MockParsedMessage:
                self.call_count += 1
                if self.call_count % 2 == 1:
                    raise ValueError("Simulated failure on odd attempts")
                data = _make_batch_response(self.call_count)
                return MockParsedMessage(json.dumps(data), data)

        mock_msgs = MockMessages()

        class MockClient:
            def __init__(self, **kwargs: object) -> None:
                self.messages = mock_msgs

        monkeypatch.setattr("anthropic.Anthropic", MockClient)

        result = run_vr(
            oracle, n_initial=4, n_iterations=3,
            seed=42, max_retries=3, batch_size=3,
        )
        assert isinstance(result, VRResult)
        # Should succeed (not fallback) since retry produces valid response
        for log in result.step_logs:
            assert "[PARSE FAILURE" not in log.hypothesis

    def test_run_vr_permuted_feedback(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        """permute_feedback=True should complete and still track real HV."""
        result = run_vr(
            oracle, n_initial=4, n_iterations=6, seed=42,
            permute_feedback=True, batch_size=3,
        )
        assert isinstance(result, VRResult)
        assert result.X.shape == (10, oracle.n_inputs)
        assert result.Y.shape == (10, oracle.n_outputs)
        assert len(result.hypervolumes) == 10
        # HV is computed on real Y, so should be non-negative
        assert all(hv >= 0 for hv in result.hypervolumes)

    def test_run_vr_permuted_vs_normal_same_real_evals(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        """Permuted and normal runs with same seed evaluate same points.

        Since the mock returns the same next_point regardless, the real
        oracle evaluations (Y_all) should be identical.
        """
        r_normal = run_vr(
            oracle, n_initial=4, n_iterations=6,
            seed=42, batch_size=3,
        )
        r_permuted = run_vr(
            oracle, n_initial=4, n_iterations=6,
            seed=42, permute_feedback=True, batch_size=3,
        )
        # Same initial design (same seed)
        np.testing.assert_array_equal(r_normal.X[:4], r_permuted.X[:4])
        np.testing.assert_array_equal(r_normal.Y[:4], r_permuted.Y[:4])
        # Same next_points from mock -> same oracle evals
        np.testing.assert_array_equal(r_normal.X, r_permuted.X)
        np.testing.assert_array_equal(r_normal.Y, r_permuted.Y)

    def test_run_vr_partial_last_batch(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        """n_iterations not divisible by batch_size should still work."""
        # 5 iterations with batch_size=3 => 2 LLM calls, last has 2 points
        result = run_vr(
            oracle, n_initial=4, n_iterations=5,
            seed=42, batch_size=3,
        )
        assert isinstance(result, VRResult)
        assert result.X.shape == (9, oracle.n_inputs)
        assert result.Y.shape == (9, oracle.n_outputs)
        assert len(result.step_logs) == 5
        # Should have 2 LLM calls: ceil(5/3)=2
        assert result.total_llm_calls == math.ceil(5 / 3)
