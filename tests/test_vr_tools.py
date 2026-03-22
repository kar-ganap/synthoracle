"""Tests for the tool-use VR agent module."""

from __future__ import annotations

import numpy as np
import pytest

anthropic = pytest.importorskip("anthropic")

from synthoracle.agents.vr_tools import (  # noqa: E402
    VRToolsResult,
    _build_analysis_tools,
    _build_oracle_tools,
    _build_tool_iteration_prompt,
    _build_tool_system_prompt,
    _execute_tool,
    _serialize_content,
    run_vr_tools,
)
from synthoracle.optim_utils import (  # noqa: E402
    compute_reference_point,
    parse_directions,
)
from synthoracle.oracles.medium import MediumOracle  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oracle() -> MediumOracle:
    return MediumOracle()


@pytest.fixture
def setup(oracle: MediumOracle) -> dict[str, object]:
    """Common setup for tool execution tests."""
    rng = np.random.default_rng(42)
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(10, oracle.n_inputs))
    Y = oracle.evaluate_batch(X)
    obj_indices, constraint_indices, signs = parse_directions(oracle)
    ref = compute_reference_point(oracle, obj_indices, signs, 42)
    return {
        "X": X,
        "Y": Y,
        "obj_indices": obj_indices,
        "constraint_indices": constraint_indices,
        "signs": signs,
        "reference_point": ref,
    }


# ---------------------------------------------------------------------------
# Mock helpers for tool-use conversation flow
# ---------------------------------------------------------------------------


class MockToolUseBlock:
    def __init__(self, tool_id: str, name: str, tool_input: dict[str, object]) -> None:
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = tool_input


class MockTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class MockUsage:
    def __init__(self) -> None:
        self.input_tokens = 200
        self.output_tokens = 300


class MockMessage:
    def __init__(self, content: list[object]) -> None:
        self.content = content
        self.usage = MockUsage()
        self.stop_reason = (
            "tool_use"
            if any(getattr(b, "type", "") == "tool_use" for b in content)
            else "end_turn"
        )


class MockMessages:
    """Mock for client.messages that simulates tool-use flow."""

    def __init__(self) -> None:
        self.call_count = 0

    def create(self, **kwargs: object) -> MockMessage:
        self.call_count += 1
        if self.call_count == 1:
            # First call: request an oat_sweep
            return MockMessage([
                MockToolUseBlock("tool_1", "oat_sweep", {
                    "input_name": "X2",
                    "n_levels": 3,
                    "base_point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    "predicted_trend": "X2 increases Y1 monotonically",
                }),
            ])
        elif self.call_count == 2:
            # After receiving sweep results: request correlation_matrix
            return MockMessage([
                MockToolUseBlock("tool_2", "correlation_matrix", {}),
            ])
        else:
            # Done: return text hypothesis
            return MockMessage([
                MockTextBlock("X2 strongly drives Y1 based on OAT sweep."),
            ])


@pytest.fixture
def mock_anthropic(monkeypatch: pytest.MonkeyPatch) -> MockMessages:
    """Mock Anthropic client returning deterministic tool-use flow."""
    mock_msgs = MockMessages()

    class MockClient:
        def __init__(self, **kwargs: object) -> None:
            self.messages = mock_msgs

    monkeypatch.setattr("anthropic.Anthropic", MockClient)
    return mock_msgs


# ---------------------------------------------------------------------------
# TestToolDefinitions
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    def test_oracle_tools_count(self, oracle: MediumOracle) -> None:
        tools = _build_oracle_tools(oracle)
        assert len(tools) == 3

    def test_analysis_tools_count(self, oracle: MediumOracle) -> None:
        tools = _build_analysis_tools(oracle)
        assert len(tools) == 5

    def test_tool_schemas_have_required_fields(self, oracle: MediumOracle) -> None:
        all_tools = _build_oracle_tools(oracle) + _build_analysis_tools(oracle)
        for tool in all_tools:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool missing 'description': {tool}"
            assert "input_schema" in tool, f"Tool missing 'input_schema': {tool}"

    def test_oracle_tool_names(self, oracle: MediumOracle) -> None:
        tools = _build_oracle_tools(oracle)
        names = {str(t["name"]) for t in tools}
        assert names == {"evaluate_point", "oat_sweep", "interaction_test"}

    def test_analysis_tool_names(self, oracle: MediumOracle) -> None:
        tools = _build_analysis_tools(oracle)
        names = {str(t["name"]) for t in tools}
        assert names == {
            "correlation_matrix",
            "sensitivity_report",
            "local_gradients",
            "current_pareto_front",
            "regression_fit",
        }


# ---------------------------------------------------------------------------
# TestToolExecution
# ---------------------------------------------------------------------------


class TestToolExecution:
    def test_evaluate_point(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "evaluate_point",
            {
                "point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "predicted_outputs": [0.5, 0.5, 0.5, 0.5],
            },
            oracle, X, Y, 10, {},
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 1
        assert new_X.shape == (1, 6)
        assert new_Y.shape == (1, 4)
        assert "outputs" in result

    def test_evaluate_point_clips_bounds(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "evaluate_point",
            {
                "point": [5.0, -1.0, 0.5, 0.5, 0.5, 0.5],
                "predicted_outputs": [0.5, 0.5, 0.5, 0.5],
            },
            oracle, X, Y, 10, {},
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 1
        # Should be clipped to [0.1, 1.0]
        assert new_X[0, 0] == pytest.approx(1.0)
        assert new_X[0, 1] == pytest.approx(0.1)

    def test_oat_sweep(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "oat_sweep",
            {
                "input_name": "X2",
                "n_levels": 5,
                "base_point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "predicted_trend": "X2 increases Y1 monotonically",
            },
            oracle, X, Y, 10, {},
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 5
        assert new_X.shape == (5, 6)
        assert new_Y.shape == (5, 4)
        assert "sweep" in result

    def test_oat_sweep_budget_exceeded(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "oat_sweep",
            {
                "input_name": "X2",
                "n_levels": 5,
                "base_point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "predicted_trend": "X2 increases Y1 monotonically",
            },
            oracle, X, Y, 2, {},  # only 2 remaining
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 0
        assert "error" in result
        assert new_X.shape[0] == 0

    def test_interaction_test(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "interaction_test",
            {
                "input_a": "X1",
                "input_b": "X2",
                "levels_a": [0.2, 0.5, 0.8],
                "levels_b": [0.3, 0.7],
                "base_point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "predicted_interaction": "X1 and X2 interact synergistically on Y1",
            },
            oracle, X, Y, 20, {},
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 6  # 3 * 2
        assert new_X.shape == (6, 6)
        assert new_Y.shape == (6, 4)
        assert "grid" in result
        assert "interaction_effects_std" in result

    def test_correlation_matrix(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "correlation_matrix", {},
            oracle, X, Y, 10, {},
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 0
        assert new_X.shape[0] == 0
        assert "correlations" in result
        corr = result["correlations"]
        assert isinstance(corr, dict)
        assert "X1" in corr

    def test_sensitivity_report(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "sensitivity_report", {},
            oracle, X, Y, 10, {},
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 0
        assert new_X.shape[0] == 0
        assert "sensitivities" in result

    def test_local_gradients(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "local_gradients",
            {"point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},
            oracle, X, Y, 10, {},
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 0
        assert new_X.shape[0] == 0
        assert "gradients" in result
        grads = result["gradients"]
        assert isinstance(grads, dict)
        assert "X1" in grads

    def test_current_pareto_front(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "current_pareto_front", {},
            oracle, X, Y, 10, {},
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 0
        assert new_X.shape[0] == 0
        assert "pareto_front" in result
        assert "hypervolume" in result
        assert "n_pareto" in result

    def test_regression_fit(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "regression_fit",
            {"inputs": ["X1", "X2"], "output": "Y1"},
            oracle, X, Y, 10, {},
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 0
        assert new_X.shape[0] == 0
        assert "coefficients" in result
        assert "r_squared" in result
        coeffs = result["coefficients"]
        assert isinstance(coeffs, dict)
        assert "X1" in coeffs
        assert "X2" in coeffs
        assert "intercept" in coeffs

    def test_evaluate_point_budget_exceeded(
        self, oracle: MediumOracle, setup: dict[str, object],
    ) -> None:
        X = setup["X"]
        Y = setup["Y"]
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        result, new_X, new_Y, cost = _execute_tool(
            "evaluate_point",
            {
                "point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "predicted_outputs": [0.5, 0.5, 0.5, 0.5],
            },
            oracle, X, Y, 0, {},  # 0 remaining
            setup["obj_indices"],  # type: ignore[arg-type]
            setup["signs"],  # type: ignore[arg-type]
            setup["constraint_indices"],  # type: ignore[arg-type]
            setup["reference_point"],  # type: ignore[arg-type]
        )
        assert cost == 0
        assert "error" in result
        assert new_X.shape[0] == 0


# ---------------------------------------------------------------------------
# TestPrompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_system_prompt_contains_oracle_info(self, oracle: MediumOracle) -> None:
        prompt = _build_tool_system_prompt(oracle, {})
        for name in oracle.input_names:
            assert name in prompt
        for name in oracle.output_names:
            assert name in prompt
        assert "maximize" in prompt
        assert "minimize" in prompt

    def test_system_prompt_contains_tool_descriptions(
        self, oracle: MediumOracle,
    ) -> None:
        prompt = _build_tool_system_prompt(oracle, {})
        assert "evaluate_point" in prompt
        assert "oat_sweep" in prompt
        assert "interaction_test" in prompt
        assert "correlation_matrix" in prompt
        assert "sensitivity_report" in prompt
        assert "local_gradients" in prompt
        assert "current_pareto_front" in prompt
        assert "regression_fit" in prompt

    def test_system_prompt_contains_playbook(self, oracle: MediumOracle) -> None:
        prompt = _build_tool_system_prompt(oracle, {})
        assert "Scientist's Playbook" in prompt
        assert "OAT" in prompt
        assert "Hypothesis Discipline" in prompt
        assert "FALSIFY" in prompt

    def test_system_prompt_threshold(self, oracle: MediumOracle) -> None:
        prompt = _build_tool_system_prompt(oracle, {"Y3": 0.4})
        assert "0.4" in prompt

    def test_iteration_prompt_budget(self) -> None:
        prompt = _build_tool_iteration_prompt(10, 72, "SCREEN", [])
        assert "10 / 72" in prompt
        assert "Remaining: 62" in prompt
        assert "SCREEN" in prompt

    def test_iteration_prompt_mechanism_log(self) -> None:
        log = ["Step 1: X2 drives Y1", "Step 2: X4 also matters"]
        prompt = _build_tool_iteration_prompt(10, 72, "PROBE", log)
        assert "Step 1: X2 drives Y1" in prompt
        assert "Step 2: X4 also matters" in prompt
        assert "PROBE" in prompt


# ---------------------------------------------------------------------------
# TestSerializeContent
# ---------------------------------------------------------------------------


class TestSerializeContent:
    def test_serialize_text_block(self) -> None:
        block = MockTextBlock("hello")
        result = _serialize_content([block])
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "hello"

    def test_serialize_tool_use_block(self) -> None:
        block = MockToolUseBlock("id1", "evaluate_point", {"point": [0.5]})
        result = _serialize_content([block])
        assert len(result) == 1
        assert result[0]["type"] == "tool_use"
        assert result[0]["id"] == "id1"
        assert result[0]["name"] == "evaluate_point"

    def test_serialize_mixed_blocks(self) -> None:
        blocks: list[object] = [
            MockTextBlock("thinking..."),
            MockToolUseBlock("id1", "oat_sweep", {"input_name": "X1"}),
        ]
        result = _serialize_content(blocks)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "tool_use"


# ---------------------------------------------------------------------------
# TestRunVRTools (uses mock_anthropic fixture)
# ---------------------------------------------------------------------------


class TestRunVRTools:
    def test_completes(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr_tools(
            oracle, n_initial=4, n_budget=7, seed=42,
            max_tool_calls_per_iteration=15,
        )
        assert isinstance(result, VRToolsResult)

    def test_budget_tracked(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr_tools(
            oracle, n_initial=4, n_budget=7, seed=42,
            max_tool_calls_per_iteration=15,
        )
        # Initial 4 + oat_sweep(3) = 7 => should stop
        assert result.eval_count == 7

    def test_shapes(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr_tools(
            oracle, n_initial=4, n_budget=7, seed=42,
            max_tool_calls_per_iteration=15,
        )
        # 4 initial + 3 from oat_sweep = 7
        assert result.X.shape == (7, oracle.n_inputs)
        assert result.Y.shape == (7, oracle.n_outputs)
        assert len(result.hypervolumes) == 7

    def test_tool_calls_logged(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr_tools(
            oracle, n_initial=4, n_budget=7, seed=42,
            max_tool_calls_per_iteration=15,
        )
        assert len(result.tool_calls) >= 1
        # Should have at least the oat_sweep and correlation_matrix
        names = [str(tc["name"]) for tc in result.tool_calls]
        assert "oat_sweep" in names

    def test_max_tool_calls_per_iteration(
        self, oracle: MediumOracle, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Respects max_tool_calls_per_iteration limit."""
        call_count = 0

        class LimitedToolMsg(MockMessages):
            def create(self, **kwargs: object) -> MockMessage:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First: free analysis tool
                    return MockMessage([
                        MockToolUseBlock("tool_1", "correlation_matrix", {}),
                    ])
                elif call_count == 2:
                    # Second: oracle tool to consume budget
                    return MockMessage([
                        MockToolUseBlock("tool_2", "evaluate_point", {
                            "point": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            "predicted_outputs": [0.5, 0.5, 0.5, 0.5],
                        }),
                    ])
                # Done: return text
                return MockMessage([
                    MockTextBlock("Done."),
                ])

        limited_msgs = LimitedToolMsg()

        class MockClient:
            def __init__(self, **kwargs: object) -> None:
                self.messages = limited_msgs

        monkeypatch.setattr("anthropic.Anthropic", MockClient)

        # n_initial=4, n_budget=5: one eval to spend.
        # The mock calls correlation_matrix (free) then evaluate_point (cost 1),
        # which fills the budget.
        result = run_vr_tools(
            oracle, n_initial=4, n_budget=5, seed=42,
            max_tool_calls_per_iteration=3,
        )
        assert isinstance(result, VRToolsResult)
        assert result.eval_count == 5
        # Should have logged 2 tool calls (correlation_matrix + evaluate_point)
        assert len(result.tool_calls) == 2

    def test_token_counts(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr_tools(
            oracle, n_initial=4, n_budget=7, seed=42,
            max_tool_calls_per_iteration=15,
        )
        assert result.total_llm_calls > 0
        assert result.total_input_tokens > 0
        assert result.total_output_tokens > 0

    def test_mechanism_log_populated(
        self, oracle: MediumOracle, mock_anthropic: MockMessages,
    ) -> None:
        result = run_vr_tools(
            oracle, n_initial=4, n_budget=7, seed=42,
            max_tool_calls_per_iteration=15,
        )
        assert len(result.mechanism_log) >= 1
