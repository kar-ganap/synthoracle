"""Verbal Regularization (VR) agent for multi-objective optimization.

Uses an LLM to hypothesize, predict, and reconcile — iteratively proposing
evaluation points while building a causal narrative of the oracle's behavior.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import anthropic
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from synthoracle.optim_utils import (
    compute_hypervolume,
    compute_reference_point,
    extract_pareto_front,
    parse_directions,
)
from synthoracle.oracle import Oracle

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VRStepLog:
    """Log of a single VR iteration."""

    step: int
    x: npt.NDArray[np.float64]  # (n_inputs,)
    y_predicted: npt.NDArray[np.float64]  # (n_outputs,)
    y_actual: npt.NDArray[np.float64]  # (n_outputs,)
    hypothesis: str
    reasoning: str
    reconciliation: str
    prediction_error: npt.NDArray[np.float64]  # y_actual - y_predicted
    directional_accuracy: list[bool]  # per-output direction correct?
    raw_response: str


@dataclass
class VRResult:
    """Full result of a VR optimization run."""

    X: npt.NDArray[np.float64]  # (n_total, n_inputs)
    Y: npt.NDArray[np.float64]  # (n_total, n_outputs)
    hypervolumes: list[float]  # length n_total
    pareto_X: npt.NDArray[np.float64]
    pareto_Y: npt.NDArray[np.float64]
    reference_point: npt.NDArray[np.float64]
    seed: int
    n_initial: int
    n_vr_iterations: int
    total_seconds: float
    step_logs: list[VRStepLog]
    predictions: npt.NDArray[np.float64]  # (n_vr_iterations, n_outputs)
    prediction_errors: npt.NDArray[np.float64]  # (n_vr_iterations, n_outputs)
    total_llm_calls: int
    total_input_tokens: int
    total_output_tokens: int


@dataclass
class _LLMResponse:
    """Parsed LLM response for a single VR iteration."""

    next_point: npt.NDArray[np.float64]
    prediction: npt.NDArray[np.float64]
    hypothesis: str
    reasoning: str
    reconciliation: str
    raw_text: str
    input_tokens: int
    output_tokens: int


class _VRResponseSchema(BaseModel):
    """Pydantic schema for structured LLM output."""

    reconciliation: str
    hypothesis: str
    next_point: list[float]
    prediction: list[float]
    reasoning: str


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_system_prompt(
    oracle: Oracle,
    thresholds: dict[str, float],
) -> str:
    """Build the system prompt describing the oracle and protocol."""
    lines: list[str] = []
    lines.append("You are a scientific optimization agent.")
    lines.append("Your goal: find inputs that optimize the outputs of a black-box system.")
    lines.append("")

    # Inputs
    lines.append("## Inputs")
    for i, name in enumerate(oracle.input_names):
        lo, hi = oracle.bounds[i]
        lines.append(f"- {name}: [{lo:.4f}, {hi:.4f}]")
    lines.append("")

    # Outputs
    lines.append("## Outputs")
    for name, direction in zip(oracle.output_names, oracle.output_directions):
        if direction == "threshold" and name in thresholds:
            lines.append(f"- {name}: {direction} (must be >= {thresholds[name]})")
        else:
            lines.append(f"- {name}: {direction}")
    lines.append("")

    # Protocol
    lines.append("## Protocol")
    lines.append("Each iteration you must:")
    lines.append("1. Reconcile: compare your previous prediction with the actual result.")
    lines.append("2. Hypothesize: state your current causal hypothesis about the system.")
    lines.append("3. Predict: predict the outputs for your proposed next input point.")
    lines.append("4. Reason: explain why you chose this point and what you expect to learn.")
    lines.append("")

    # JSON schema
    lines.append("## Output Format")
    lines.append("You must respond with a JSON object containing exactly these fields:")
    lines.append("- \"reconciliation\": string (compare prediction vs actual)")
    lines.append("- \"hypothesis\": string (your causal hypothesis)")
    lines.append(f'- "next_point": list of {oracle.n_inputs} floats (within bounds)')
    lines.append(f'- "prediction": list of {oracle.n_outputs} floats (predicted outputs)')
    lines.append("- \"reasoning\": string (why this point)")
    lines.append("")
    lines.append("Respond with JSON only.")

    return "\n".join(lines)


def _build_iteration_prompt(
    oracle: Oracle,
    X_all: npt.NDArray[np.float64],
    Y_all: npt.NDArray[np.float64],
    mechanism_log: list[str],
    prev_prediction: npt.NDArray[np.float64] | None,
    prev_actual: npt.NDArray[np.float64] | None,
    step: int,
) -> str:
    """Build the user prompt for a single VR iteration."""
    lines: list[str] = []

    # Observation table
    lines.append("## Observation Data")
    in_names = oracle.input_names
    out_names = oracle.output_names
    header = "| # | " + " | ".join(in_names) + " | " + " | ".join(out_names) + " |"
    sep = "|---| " + " | ".join(["---"] * len(in_names)) + " | "
    sep += " | ".join(["---"] * len(out_names)) + " |"
    lines.append(header)
    lines.append(sep)
    for k in range(len(X_all)):
        row = f"| {k + 1} | "
        row += " | ".join(f"{v:.4f}" for v in X_all[k]) + " | "
        row += " | ".join(f"{v:.4f}" for v in Y_all[k]) + " |"
        lines.append(row)
    lines.append("")

    # Mechanism log
    if mechanism_log:
        lines.append("## Mechanism Log")
        for entry in mechanism_log:
            lines.append(f"- {entry}")
        lines.append("")

    # Previous prediction vs actual
    if step > 0 and prev_prediction is not None and prev_actual is not None:
        lines.append("## Previous Prediction vs Actual")
        for i, name in enumerate(out_names):
            pred = prev_prediction[i]
            actual = prev_actual[i]
            error = actual - pred
            lines.append(f"- {name}: predicted={pred:.4f}, actual={actual:.4f}, error={error:.4f}")
        lines.append("")

    # Task
    lines.append("## Task")
    if step == 0:
        lines.append("This is the first iteration. Analyze the initial data,")
        lines.append("form a hypothesis about how inputs relate to outputs,")
        lines.append("propose a next input point, and predict its outputs.")
        lines.append("Set reconciliation to a brief note (no prior prediction to reconcile).")
    else:
        lines.append("Reconcile your previous prediction with the actual result.")
        lines.append("Update your hypothesis, propose a next input point, and predict outputs.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, object]:
    """Extract a JSON object from possibly noisy LLM output."""
    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return dict(result)
    except json.JSONDecodeError:
        pass

    # Try markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        try:
            result = json.loads(md_match.group(1))
            if isinstance(result, dict):
                return dict(result)
        except json.JSONDecodeError:
            pass

    # Try to find {...} with nested brace matching
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict):
                        return dict(result)
                except json.JSONDecodeError:
                    start = -1

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


def _parse_llm_response(
    text: str,
    n_inputs: int,
    n_outputs: int,
    bounds: npt.NDArray[np.float64],
    input_tokens: int,
    output_tokens: int,
) -> _LLMResponse:
    """Parse and validate an LLM response."""
    data = _extract_json(text)

    # Validate required fields
    for field in ("next_point", "prediction", "hypothesis", "reasoning", "reconciliation"):
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    next_point_raw = data["next_point"]
    prediction_raw = data["prediction"]

    if not isinstance(next_point_raw, list) or len(next_point_raw) != n_inputs:
        raise ValueError(
            f"next_point must be a list of {n_inputs} floats, "
            f"got {type(next_point_raw).__name__} of length "
            f"{len(next_point_raw) if isinstance(next_point_raw, list) else 'N/A'}"
        )

    if not isinstance(prediction_raw, list) or len(prediction_raw) != n_outputs:
        raise ValueError(
            f"prediction must be a list of {n_outputs} floats, "
            f"got {type(prediction_raw).__name__} of length "
            f"{len(prediction_raw) if isinstance(prediction_raw, list) else 'N/A'}"
        )

    next_point = np.array(next_point_raw, dtype=np.float64)
    prediction = np.array(prediction_raw, dtype=np.float64)

    # Check for NaN/Inf
    if np.any(~np.isfinite(next_point)):
        raise ValueError("next_point contains NaN or Inf values")
    if np.any(~np.isfinite(prediction)):
        raise ValueError("prediction contains NaN or Inf values")

    # Clip to bounds
    next_point = np.clip(next_point, bounds[:, 0], bounds[:, 1])

    return _LLMResponse(
        next_point=next_point,
        prediction=prediction,
        hypothesis=str(data["hypothesis"]),
        reasoning=str(data["reasoning"]),
        reconciliation=str(data["reconciliation"]),
        raw_text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ---------------------------------------------------------------------------
# Directional accuracy
# ---------------------------------------------------------------------------


def _compute_directional_accuracy(
    prediction: npt.NDArray[np.float64],
    actual: npt.NDArray[np.float64],
    prev_y: npt.NDArray[np.float64] | None,
) -> list[bool]:
    """Compute per-output directional accuracy of a prediction.

    For each output, checks whether the predicted direction of change
    (relative to previous actual) matches the actual direction.
    If prev_y is None (first step), returns all True.
    """
    if prev_y is None:
        return [True] * len(prediction)

    result: list[bool] = []
    for i in range(len(prediction)):
        pred_sign = np.sign(prediction[i] - prev_y[i])
        actual_sign = np.sign(actual[i] - prev_y[i])
        result.append(bool(pred_sign == actual_sign))
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_vr(
    oracle: Oracle,
    *,
    n_initial: int | None = None,
    n_iterations: int = 42,
    seed: int = 42,
    reference_point: npt.NDArray[np.float64] | None = None,
    thresholds: dict[str, float] | None = None,
    model: str = "claude-sonnet-4-6",
    temperature: float = 1.0,
    max_retries: int = 3,
    permute_feedback: bool = False,
) -> VRResult:
    """Run the Verbal Regularization agent.

    Parameters
    ----------
    oracle : Oracle
        The oracle to optimize.
    n_initial : int, optional
        Number of initial random evaluations. Defaults to 2 * oracle.n_inputs.
    n_iterations : int
        Number of VR iterations (LLM calls).
    seed : int
        Random seed for reproducibility.
    reference_point : ndarray, optional
        Reference point for hypervolume. Auto-computed if None.
    thresholds : dict, optional
        Threshold constraints, e.g. {"Y3": 0.4}.
    model : str
        Anthropic model identifier.
    temperature : float
        LLM sampling temperature.
    max_retries : int
        Max retries on LLM parse failure.
    permute_feedback : bool
        If True, shuffle Y rows in the observation table shown to the agent
        (column permutation ablation). HV is still computed on real Y.
        This tests whether the agent actually uses X-Y correlations.

    Returns
    -------
    VRResult
        Full optimization results with step logs and token tracking.
    """
    t0 = time.monotonic()
    client = anthropic.Anthropic()

    # Setup
    obj_indices, constraint_indices, signs = parse_directions(oracle)
    if n_initial is None:
        n_initial = 2 * oracle.n_inputs
    if thresholds is None:
        thresholds = {}
    if reference_point is None:
        reference_point = compute_reference_point(oracle, obj_indices, signs, seed)

    # Initial design (numpy random)
    rng = np.random.default_rng(seed)
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    X_all = rng.uniform(lo, hi, size=(n_initial, oracle.n_inputs))
    Y_all = oracle.evaluate_batch(X_all)

    # Compute initial hypervolumes
    hypervolumes: list[float] = []
    for k in range(1, n_initial + 1):
        hv = compute_hypervolume(
            Y_all[:k], obj_indices, signs, constraint_indices,
            thresholds, oracle.output_names, reference_point,
        )
        hypervolumes.append(hv)

    system_prompt = _build_system_prompt(oracle, thresholds)
    mechanism_log: list[str] = []
    step_logs: list[VRStepLog] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0
    prev_prediction: npt.NDArray[np.float64] | None = None
    prev_actual: npt.NDArray[np.float64] | None = None

    # Separate RNG for permutation to not affect other randomness
    permute_rng = np.random.default_rng(seed + 2000)

    for step in range(n_iterations):
        # Build observation table for the prompt
        if permute_feedback:
            # Column permutation: shuffle Y rows relative to X rows
            perm = permute_rng.permutation(len(Y_all))
            Y_shown = Y_all[perm]
            # Also permute the prev_actual shown to the agent
            prev_actual_shown = Y_shown[-1] if prev_actual is not None else None
        else:
            Y_shown = Y_all
            prev_actual_shown = prev_actual

        prompt = _build_iteration_prompt(
            oracle, X_all, Y_shown, mechanism_log,
            prev_prediction, prev_actual_shown, step,
        )

        # Call LLM with structured output (parse), fallback to manual parsing
        response: _LLMResponse | None = None
        last_text = ""
        for attempt in range(max_retries + 1):
            try:
                msg = client.messages.parse(
                    model=model,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    output_format=_VRResponseSchema,
                )
                total_llm_calls += 1
                tokens_in = msg.usage.input_tokens
                tokens_out = msg.usage.output_tokens
                total_input_tokens += tokens_in
                total_output_tokens += tokens_out

                parsed = msg.parsed_output
                if parsed is not None:
                    # Structured output succeeded
                    raw_text = msg.content[0].text  # type: ignore[union-attr]
                    last_text = raw_text
                    next_pt = np.array(parsed.next_point, dtype=np.float64)
                    pred = np.array(parsed.prediction, dtype=np.float64)

                    # Validate dimensions
                    if len(next_pt) != oracle.n_inputs:
                        raise ValueError(
                            f"next_point has {len(next_pt)} elements, "
                            f"expected {oracle.n_inputs}"
                        )
                    if len(pred) != oracle.n_outputs:
                        raise ValueError(
                            f"prediction has {len(pred)} elements, "
                            f"expected {oracle.n_outputs}"
                        )
                    if np.any(~np.isfinite(next_pt)):
                        raise ValueError("next_point contains NaN or Inf")
                    if np.any(~np.isfinite(pred)):
                        raise ValueError("prediction contains NaN or Inf")

                    next_pt = np.clip(next_pt, lo, hi)
                    response = _LLMResponse(
                        next_point=next_pt,
                        prediction=pred,
                        hypothesis=parsed.hypothesis,
                        reasoning=parsed.reasoning,
                        reconciliation=parsed.reconciliation,
                        raw_text=raw_text,
                        input_tokens=tokens_in,
                        output_tokens=tokens_out,
                    )
                    break

                # parsed_output was None — try manual text parsing
                content_block = msg.content[0]
                text = content_block.text  # type: ignore[union-attr]
                last_text = text
                response = _parse_llm_response(
                    text, oracle.n_inputs, oracle.n_outputs,
                    oracle.bounds, tokens_in, tokens_out,
                )
                break
            except (ValueError, Exception):
                if attempt == max_retries:
                    # Fallback: random point, NaN predictions
                    x_fallback = rng.uniform(lo, hi)
                    response = _LLMResponse(
                        next_point=x_fallback,
                        prediction=np.full(oracle.n_outputs, np.nan),
                        hypothesis="[PARSE FAILURE - random fallback]",
                        reasoning="[PARSE FAILURE]",
                        reconciliation="",
                        raw_text=last_text,
                        input_tokens=0,
                        output_tokens=0,
                    )

        assert response is not None  # guaranteed by retry logic

        # Evaluate oracle
        y_actual = oracle.evaluate(response.next_point)

        # Directional accuracy
        prev_y = Y_all[-1] if len(Y_all) > 0 else None
        dir_acc = _compute_directional_accuracy(response.prediction, y_actual, prev_y)

        # Log
        pred_error = y_actual - response.prediction
        step_log = VRStepLog(
            step=step,
            x=response.next_point,
            y_predicted=response.prediction,
            y_actual=y_actual,
            hypothesis=response.hypothesis,
            reasoning=response.reasoning,
            reconciliation=response.reconciliation,
            prediction_error=pred_error,
            directional_accuracy=dir_acc,
            raw_response=response.raw_text,
        )
        step_logs.append(step_log)
        mechanism_log.append(f"Step {step + 1}: {response.hypothesis}")

        # Update arrays
        X_all = np.vstack([X_all, response.next_point[np.newaxis, :]])
        Y_all = np.vstack([Y_all, y_actual[np.newaxis, :]])

        # Hypervolume
        hv = compute_hypervolume(
            Y_all, obj_indices, signs, constraint_indices,
            thresholds, oracle.output_names, reference_point,
        )
        hypervolumes.append(hv)

        # Set previous for next iteration
        prev_prediction = response.prediction
        prev_actual = y_actual

    # Extract Pareto front
    pareto_X, pareto_Y = extract_pareto_front(X_all, Y_all, oracle, thresholds)

    # Build predictions/errors arrays
    predictions = np.array([s.y_predicted for s in step_logs])
    prediction_errors = np.array([s.prediction_error for s in step_logs])

    total_seconds = time.monotonic() - t0

    return VRResult(
        X=X_all,
        Y=Y_all,
        hypervolumes=hypervolumes,
        pareto_X=pareto_X,
        pareto_Y=pareto_Y,
        reference_point=reference_point,
        seed=seed,
        n_initial=n_initial,
        n_vr_iterations=n_iterations,
        total_seconds=total_seconds,
        step_logs=step_logs,
        predictions=predictions,
        prediction_errors=prediction_errors,
        total_llm_calls=total_llm_calls,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )
