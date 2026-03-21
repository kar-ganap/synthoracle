"""Tests for the VR agent module."""

from __future__ import annotations

import json

import numpy as np
import pytest

anthropic = pytest.importorskip("anthropic")

from synthoracle.agents.vr import (  # noqa: E402
    VRResult,
    VRStepLog,
    _LLMResponse,
    _build_iteration_prompt,
    _build_system_prompt,
    _compute_directional_accuracy,
    _extract_json,
    _parse_llm_response,
    run_vr,
)
from synthoracle.oracles.medium import MediumOracle  # noqa: E402


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oracle() -> MediumOracle:
    return MediumOracle()


@pytest.fixture
def mock_anthropic(monkeypatch: pytest.MonkeyPatch) -> MockMessages:
    """Mock Anthropic client returning deterministic JSON."""

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
            self.hypothesis: str = str(data["hypothesis"])
            self.next_point: list[float] = list(data["next_point"])  # type: ignore[arg-type]
            self.prediction: list[float] = list(data["prediction"])  # type: ignore[arg-type]
            self.reasoning: str = str(data["reasoning"])

    class MockParsedMessage:
        def __init__(self, text: str, data: dict[str, object]) -> None:
            self.content = [MockTextBlock(text)]
            self.usage = MockUsage()
            self.parsed_output = MockParsedOutput(data)

    class MockMessages:
        def __init__(self) -> None:
            self.call_count = 0

        def _make_response(self) -> dict[str, object]:
            self.call_count += 1
            return {
                "reconciliation": (
                    "No prior prediction."
                    if self.call_count == 1
                    else "Prediction was close."
                ),
                "hypothesis": f"X2 drives Y1. Iteration {self.call_count}.",
                "next_point": [0.5] * 6,
                "prediction": [0.3, 0.2, 0.6, 0.1],
                "reasoning": "Testing midpoint.",
            }

        def create(self, **kwargs: object) -> MockParsedMessage:
            data = self._make_response()
            return MockParsedMessage(json.dumps(data), data)

        def parse(self, **kwargs: object) -> MockParsedMessage:
            data = self._make_response()
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

    def create(self, **kwargs: object) -> object:
        ...

    def parse(self, **kwargs: object) -> object:
        ...


# ---------------------------------------------------------------------------
# TestPromptConstruction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_system_prompt_contains_oracle_info(self, oracle: MediumOracle) -> None:
        prompt = _build_system_prompt(oracle, {})
        for name in oracle.input_names:
            assert name in prompt
        for name in oracle.output_names:
            assert name in prompt
        assert "maximize" in prompt
        assert "minimize" in prompt

    def test_system_prompt_contains_json_schema(self, oracle: MediumOracle) -> None:
        prompt = _build_system_prompt(oracle, {})
        assert "next_point" in prompt
        assert "prediction" in prompt

    def test_system_prompt_contains_threshold(self, oracle: MediumOracle) -> None:
        prompt = _build_system_prompt(oracle, {"Y3": 0.4})
        assert "0.4" in prompt
        assert "threshold" in prompt

    def test_iteration_prompt_has_data_table(self, oracle: MediumOracle) -> None:
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
        assert "Previous Prediction" not in prompt

    def test_iteration_prompt_subsequent_has_error(
        self, oracle: MediumOracle,
    ) -> None:
        rng = np.random.default_rng(42)
        X = rng.uniform(0.1, 1.0, size=(3, 6))
        Y = oracle.evaluate_batch(X)
        prev_pred = np.array([0.3, 0.2, 0.5, 0.1])
        prev_actual = np.array([0.35, 0.18, 0.55, 0.12])
        prompt = _build_iteration_prompt(
            oracle, X, Y, ["Step 1: hypothesis"], prev_pred, prev_actual, 1,
        )
        assert "Previous Prediction" in prompt
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
        np.testing.assert_array_almost_equal(resp.prediction, [0.3, 0.2, 0.6, 0.1])
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
                json.dumps(data), 6, 4, np.full((6, 2), [[0.1, 1.0]]), 100, 200,
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
                json.dumps(data), 6, 4, np.full((6, 2), [[0.1, 1.0]]), 100, 200,
            )

    def test_parse_clips_to_bounds(self) -> None:
        data = {
            "next_point": [2.0, 0.5, 0.5, 0.5, 0.5, 0.5],  # 2.0 > 1.0 upper bound
            "prediction": [0.3, 0.2, 0.6, 0.1],
            "hypothesis": "test",
            "reasoning": "test",
            "reconciliation": "test",
        }
        resp = _parse_llm_response(
            json.dumps(data), 6, 4, np.full((6, 2), [[0.1, 1.0]]), 100, 200,
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
                json.dumps(data), 6, 4, np.full((6, 2), [[0.1, 1.0]]), 100, 200,
            )

    def test_extract_json_direct(self) -> None:
        data = {"key": "value"}
        result = _extract_json(json.dumps(data))
        assert result == data

    def test_extract_json_raises_on_garbage(self) -> None:
        with pytest.raises(ValueError, match="Could not extract JSON"):
            _extract_json("this is not json at all")


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
# TestRunVR (uses mock_anthropic fixture)
# ---------------------------------------------------------------------------


class TestRunVR:
    def test_run_vr_completes(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr(oracle, n_initial=4, n_iterations=3, seed=42)
        assert isinstance(result, VRResult)

    def test_run_vr_shapes(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        n_init = 4
        n_iter = 3
        result = run_vr(oracle, n_initial=n_init, n_iterations=n_iter, seed=42)
        n_total = n_init + n_iter
        assert result.X.shape == (n_total, oracle.n_inputs)
        assert result.Y.shape == (n_total, oracle.n_outputs)
        assert len(result.hypervolumes) == n_total

    def test_run_vr_step_logs_count(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        n_iter = 5
        result = run_vr(oracle, n_initial=4, n_iterations=n_iter, seed=42)
        assert len(result.step_logs) == n_iter

    def test_run_vr_predictions_tracked(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        n_iter = 3
        result = run_vr(oracle, n_initial=4, n_iterations=n_iter, seed=42)
        assert result.predictions.shape == (n_iter, oracle.n_outputs)
        assert result.prediction_errors.shape == (n_iter, oracle.n_outputs)

    def test_run_vr_token_counts(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr(oracle, n_initial=4, n_iterations=3, seed=42)
        assert result.total_llm_calls > 0
        assert result.total_input_tokens > 0
        assert result.total_output_tokens > 0

    def test_run_vr_step_log_fields(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr(oracle, n_initial=4, n_iterations=2, seed=42)
        log = result.step_logs[0]
        assert isinstance(log, VRStepLog)
        assert log.step == 0
        assert log.x.shape == (oracle.n_inputs,)
        assert log.y_predicted.shape == (oracle.n_outputs,)
        assert log.y_actual.shape == (oracle.n_outputs,)
        assert isinstance(log.hypothesis, str)
        assert isinstance(log.directional_accuracy, list)

    def test_run_vr_fallback_on_bad_response(
        self, oracle: MediumOracle, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mock raises on all attempts -> falls back to random."""

        class MockMessages:
            def parse(self, **kwargs: object) -> None:
                raise ValueError("Simulated parse failure")

            def create(self, **kwargs: object) -> None:
                raise ValueError("Simulated create failure")

        mock_msgs = MockMessages()

        class MockClient:
            def __init__(self, **kwargs: object) -> None:
                self.messages = mock_msgs

        monkeypatch.setattr("anthropic.Anthropic", MockClient)

        result = run_vr(oracle, n_initial=4, n_iterations=2, seed=42, max_retries=1)
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

        class MockParsedOutput:
            def __init__(self, data: dict[str, object]) -> None:
                self.reconciliation = str(data["reconciliation"])
                self.hypothesis = str(data["hypothesis"])
                self.next_point = list(data["next_point"])  # type: ignore[arg-type]
                self.prediction = list(data["prediction"])  # type: ignore[arg-type]
                self.reasoning = str(data["reasoning"])

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
                data: dict[str, object] = {
                    "reconciliation": "Recovered.",
                    "hypothesis": "Retry worked.",
                    "next_point": [0.5] * 6,
                    "prediction": [0.3, 0.2, 0.6, 0.1],
                    "reasoning": "After retry.",
                }
                return MockParsedMessage(json.dumps(data), data)

        mock_msgs = MockMessages()

        class MockClient:
            def __init__(self, **kwargs: object) -> None:
                self.messages = mock_msgs

        monkeypatch.setattr("anthropic.Anthropic", MockClient)

        result = run_vr(oracle, n_initial=4, n_iterations=2, seed=42, max_retries=3)
        assert isinstance(result, VRResult)
        # Should succeed (not fallback) since retry produces valid response
        for log in result.step_logs:
            assert "[PARSE FAILURE" not in log.hypothesis

    def test_run_vr_permuted_feedback(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        """permute_feedback=True should complete and still track real HV."""
        result = run_vr(
            oracle, n_initial=4, n_iterations=3, seed=42,
            permute_feedback=True,
        )
        assert isinstance(result, VRResult)
        assert result.X.shape == (7, oracle.n_inputs)
        assert result.Y.shape == (7, oracle.n_outputs)
        assert len(result.hypervolumes) == 7
        # HV is computed on real Y, so should be non-negative
        assert all(hv >= 0 for hv in result.hypervolumes)

    def test_run_vr_permuted_vs_normal_same_real_evals(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        """Permuted and normal runs with same seed evaluate same points.

        Since the mock returns the same next_point regardless, the real
        oracle evaluations (Y_all) should be identical.
        """
        r_normal = run_vr(oracle, n_initial=4, n_iterations=3, seed=42)
        r_permuted = run_vr(
            oracle, n_initial=4, n_iterations=3, seed=42,
            permute_feedback=True,
        )
        # Same initial design (same seed)
        np.testing.assert_array_equal(r_normal.X[:4], r_permuted.X[:4])
        np.testing.assert_array_equal(r_normal.Y[:4], r_permuted.Y[:4])
        # Same next_points from mock → same oracle evals
        np.testing.assert_array_equal(r_normal.X, r_permuted.X)
        np.testing.assert_array_equal(r_normal.Y, r_permuted.Y)
