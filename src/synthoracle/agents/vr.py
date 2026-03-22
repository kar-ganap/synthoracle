"""Verbal Regularization (VR) agent for multi-objective optimization.

Uses an LLM to hypothesize, predict, and reconcile — iteratively proposing
evaluation points while building a causal narrative of the oracle's behavior.
"""

from __future__ import annotations

import json
import math
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
# Pydantic schemas
# ---------------------------------------------------------------------------


class _ProposedPoint(BaseModel):
    """Schema for a single proposed evaluation point."""

    next_point: list[float]
    prediction: list[float]
    reasoning: str
    explore_or_exploit: str
    falsification: str


class _VRBatchResponseSchema(BaseModel):
    """Pydantic schema for structured batch LLM output."""

    reconciliation: str
    biggest_surprise: str
    hypothesis: str
    points: list[_ProposedPoint]


# Keep old schema as fallback for manual JSON parsing
class _VRResponseSchema(BaseModel):
    """Pydantic schema for structured LLM output (single-point, legacy)."""

    reconciliation: str
    hypothesis: str
    next_point: list[float]
    prediction: list[float]
    reasoning: str


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
    explore_or_exploit: str
    falsification: str
    biggest_surprise: str  # agent's most informative prediction failure


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
    explore_or_exploit: str = ""
    falsification: str = ""


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _compute_local_gradients(
    oracle: Oracle, x: npt.NDArray[np.float64], step: float = 0.01,
) -> npt.NDArray[np.float64]:
    """Central finite differences. Returns (n_inputs, n_outputs)."""
    n_in = oracle.n_inputs
    n_out = oracle.n_outputs
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    grads = np.zeros((n_in, n_out), dtype=np.float64)
    for i in range(n_in):
        x_plus = x.copy()
        x_minus = x.copy()
        h = step * (hi[i] - lo[i])
        x_plus[i] = min(x[i] + h, hi[i])
        x_minus[i] = max(x[i] - h, lo[i])
        actual_h = x_plus[i] - x_minus[i]
        if actual_h < 1e-12:
            continue
        y_plus = oracle.evaluate(x_plus)
        y_minus = oracle.evaluate(x_minus)
        grads[i] = (y_plus - y_minus) / actual_h
    return grads


def _compute_correlations(
    X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Pearson correlation matrix (n_inputs, n_outputs)."""
    n_in = X.shape[1]
    n_out = Y.shape[1]
    corr = np.zeros((n_in, n_out), dtype=np.float64)
    for i in range(n_in):
        for j in range(n_out):
            x_col = X[:, i]
            y_col = Y[:, j]
            std_x = np.std(x_col)
            std_y = np.std(y_col)
            if std_x < 1e-12 or std_y < 1e-12:
                corr[i, j] = 0.0
            else:
                c = np.corrcoef(x_col, y_col)[0, 1]
                corr[i, j] = float(c) if np.isfinite(c) else 0.0
    return corr


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_system_prompt(
    oracle: Oracle,
    thresholds: dict[str, float],
    batch_size: int = 3,
) -> str:
    """Build the system prompt describing the oracle and protocol."""
    lines: list[str] = []
    lines.append("You are a scientific optimization agent.")
    lines.append(
        "Your goal: find inputs that optimize the outputs of a black-box system."
    )
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
    lines.append(
        "1. Reconcile: compare your previous prediction with the actual result."
    )
    lines.append(
        "2. Hypothesize: state your current causal hypothesis about the system."
    )
    lines.append(
        "3. Predict: predict the outputs for your proposed next input point."
    )
    lines.append(
        "4. Reason: explain why you chose this point and what you expect to learn."
    )
    lines.append("")

    # Scientist's Playbook
    lines.append("## Scientist's Playbook")
    lines.append(
        "- DON'T pigeonhole early. Before optimizing, understand the landscape."
    )
    lines.append(
        "- Vary ONE input at a time (OAT) to isolate which inputs drive "
        "which outputs."
    )
    lines.append(
        "- Unless proven otherwise, assume different inputs have different "
        "sensitivities."
    )
    lines.append(
        "- Don't assume global smoothness — test across the FULL RANGE of "
        "each input"
    )
    lines.append(
        "  (low, mid, high). Regime transitions or thresholds may occur ANYWHERE."
    )
    lines.append(
        "- Check for interactions: if one input's effect on an output changes "
        "depending on"
    )
    lines.append(
        "  another input's value, that's a coupling. Test by varying one input "
        "at different"
    )
    lines.append("  levels of the other.")
    lines.append(
        "- Prediction failures are MORE informative than correct predictions. "
        "Seek surprises."
    )
    lines.append("- NEVER evaluate a point too close to one you've already seen.")
    lines.append(
        "- These objectives CONFLICT — you cannot maximize all simultaneously. "
        "Explore the"
    )
    lines.append("  trade-off frontier, not just the maximum of one output.")
    lines.append("")

    # Hypothesis Discipline
    lines.append("## Hypothesis Discipline")
    lines.append("- State your hypothesis clearly.")
    lines.append("- State what would FALSIFY it.")
    lines.append(
        "- Design at least one point per batch to TEST a falsification condition."
    )
    lines.append(
        "- When a prediction fails, state SPECIFICALLY what surprised you and why."
    )
    lines.append("")

    # JSON schema
    n_inputs = oracle.n_inputs
    n_outputs = oracle.n_outputs
    lines.append("## Output Format")
    lines.append("Respond with a JSON object:")
    lines.append(
        '- "reconciliation": string — compare predictions vs actuals '
        "from the previous batch"
    )
    lines.append(
        '- "biggest_surprise": string — the most informative prediction '
        "failure and what it taught you"
    )
    lines.append(
        '- "hypothesis": string — your current causal model of the system'
    )
    lines.append(
        f'- "points": list of {batch_size} objects, each with:'
    )
    lines.append(
        f'  - "next_point": list of {n_inputs} floats (within bounds)'
    )
    lines.append(f'  - "prediction": list of {n_outputs} floats')
    lines.append(
        '  - "reasoning": string — why this point, what you expect to learn'
    )
    lines.append(
        '  - "explore_or_exploit": "explore" or "exploit"'
    )
    lines.append(
        '  - "falsification": string — what observation would prove '
        "your hypothesis wrong"
    )
    lines.append("")
    lines.append("Respond with JSON only.")

    return "\n".join(lines)


def _format_correlations(
    corr: npt.NDArray[np.float64],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
) -> str:
    """Format correlation matrix as a text table."""
    lines: list[str] = []
    header = "       " + "  ".join(f"{n:>6s}" for n in output_names)
    lines.append(header)
    for i, iname in enumerate(input_names):
        row = f"  {iname:>4s}" + "  ".join(f"{corr[i, j]:>6.2f}" for j in range(len(output_names)))
        lines.append(row)
    return "\n".join(lines)


def _format_gradients(
    gradients: list[npt.NDArray[np.float64]],
    input_names: tuple[str, ...],
    point_indices: list[int],
) -> str:
    """Format local gradients as text."""
    lines: list[str] = []
    for k, (grad, idx) in enumerate(zip(gradients, point_indices)):
        parts = [
            f"dY/d{iname}=[" + ",".join(f"{g:.3f}" for g in grad[i]) + "]"
            for i, iname in enumerate(input_names)
        ]
        lines.append(f"  Point {idx}: " + ", ".join(parts))
    return "\n".join(lines)


def _build_iteration_prompt(
    oracle: Oracle,
    X_all: npt.NDArray[np.float64],
    Y_all: npt.NDArray[np.float64],
    mechanism_log: list[str],
    prev_predictions: list[npt.NDArray[np.float64]] | None,
    prev_actuals: list[npt.NDArray[np.float64]] | None,
    call_idx: int,
    correlations: npt.NDArray[np.float64] | None = None,
    gradients: list[npt.NDArray[np.float64]] | None = None,
    phase: str = "SCREEN",
    eval_count: int = 0,
    n_total_budget: int = 0,
    batch_size: int = 3,
) -> str:
    """Build the user prompt for a batch VR iteration."""
    lines: list[str] = []

    # Budget / phase section
    n_remaining = n_total_budget - eval_count
    lines.append("## Budget")
    lines.append(
        f"Evaluations used: {eval_count} / {n_total_budget}. "
        f"Remaining: {n_remaining}."
    )
    lines.append(
        "Phase: SCREEN (first 1/3) -> PROBE (middle 1/3) -> OPTIMIZE (final 1/3)"
    )
    lines.append(f"Current phase: {phase}")
    lines.append("")
    lines.append(
        "SCREEN: OAT sweeps — isolate which inputs matter for which outputs."
    )
    lines.append(
        "PROBE: Test interactions and nonlinearities for important inputs."
    )
    lines.append(
        "OPTIMIZE: Exploit the learned structure to optimize the Pareto front."
    )
    lines.append("")

    # Observation table
    lines.append("## Observation Data")
    in_names = oracle.input_names
    out_names = oracle.output_names
    header = (
        "| # | "
        + " | ".join(in_names) + " | "
        + " | ".join(out_names) + " |"
    )
    sep = (
        "|---| "
        + " | ".join(["---"] * len(in_names)) + " | "
        + " | ".join(["---"] * len(out_names)) + " |"
    )
    lines.append(header)
    lines.append(sep)
    for k in range(len(X_all)):
        row = f"| {k + 1} | "
        row += " | ".join(f"{v:.4f}" for v in X_all[k]) + " | "
        row += " | ".join(f"{v:.4f}" for v in Y_all[k]) + " |"
        lines.append(row)
    lines.append("")

    # Data analysis section
    lines.append("## Data Analysis")
    if correlations is not None:
        lines.append("Input-output correlations (Pearson):")
        lines.append(_format_correlations(corr=correlations,
                                          input_names=in_names,
                                          output_names=out_names))
        lines.append("")

    if gradients is not None and len(gradients) > 0:
        lines.append("Local gradients at last evaluated points:")
        n_pts = len(X_all)
        point_indices = list(range(
            n_pts - len(gradients) + 1, n_pts + 1,
        ))
        lines.append(_format_gradients(gradients, in_names, point_indices))
        lines.append("")

    # Mechanism log
    if mechanism_log:
        lines.append("## Mechanism Log")
        for entry in mechanism_log:
            lines.append(f"- {entry}")
        lines.append("")

    # Previous prediction vs actual
    if (
        call_idx > 0
        and prev_predictions is not None
        and prev_actuals is not None
        and len(prev_predictions) > 0
    ):
        lines.append("## Previous Batch: Predictions vs Actuals")
        for pt_idx, (pred, actual) in enumerate(
            zip(prev_predictions, prev_actuals)
        ):
            lines.append(f"Point {pt_idx + 1}:")
            for i, name in enumerate(out_names):
                error = actual[i] - pred[i]
                lines.append(
                    f"  - {name}: predicted={pred[i]:.4f}, "
                    f"actual={actual[i]:.4f}, error={error:.4f}"
                )
        lines.append("")

    # Task
    lines.append("## Task")
    if call_idx == 0:
        lines.append("This is the first iteration. Analyze the initial data,")
        lines.append(
            "form a hypothesis about how inputs relate to outputs,"
        )
        lines.append(
            f"propose {batch_size} input points, and predict their outputs."
        )
        lines.append(
            "Set reconciliation and biggest_surprise to brief notes "
            "(no prior prediction to reconcile)."
        )
    else:
        lines.append(
            f"Propose {batch_size} points. For each:"
        )
        lines.append(
            "- Coordinates (within bounds, not too close to existing observations)"
        )
        lines.append("- Predicted outputs")
        lines.append("- Reasoning (what hypothesis does this test?)")
        lines.append('- Label: "explore" or "exploit"')
        lines.append(
            "- Falsification: what result would disprove your current hypothesis"
        )
        lines.append("")
        if phase == "SCREEN":
            lines.append(
                f"In SCREEN phase: at least {max(batch_size - 1, 1)} of "
                f"{batch_size} should be \"explore\" (OAT sweeps preferred)."
            )
        elif phase == "PROBE":
            lines.append(
                f"In PROBE phase: at least {max(batch_size - 1, 1)} of "
                f"{batch_size} should be \"explore\" (interaction tests)."
            )
        else:
            lines.append(
                f"In OPTIMIZE phase: at least {max(batch_size - 1, 1)} of "
                f"{batch_size} should be \"exploit\", keep 1 explore."
            )
        lines.append("")
        lines.append("State your biggest surprise from the last batch.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parsing (fallback)
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
    """Parse and validate an LLM response (single-point fallback)."""
    data = _extract_json(text)

    # Validate required fields
    for field in (
        "next_point", "prediction", "hypothesis", "reasoning", "reconciliation",
    ):
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
# Phase helpers
# ---------------------------------------------------------------------------


def _determine_phase(eval_count: int, n_total_budget: int) -> str:
    """Determine optimization phase based on progress."""
    if n_total_budget <= 0:
        return "SCREEN"
    progress = eval_count / n_total_budget
    if progress < 1 / 3:
        return "SCREEN"
    elif progress < 2 / 3:
        return "PROBE"
    else:
        return "OPTIMIZE"


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
    batch_size: int = 3,
    thinking: dict[str, object] | None = None,
    max_tokens: int = 4096,
    checkpoint_dir: str | None = None,
) -> VRResult:
    """Run the Verbal Regularization agent.

    Parameters
    ----------
    oracle : Oracle
        The oracle to optimize.
    n_initial : int, optional
        Number of initial random evaluations. Defaults to 2 * oracle.n_inputs.
    n_iterations : int
        Total number of oracle evaluations (not LLM calls).
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
    batch_size : int
        Number of points proposed per LLM call.
    thinking : dict, optional
        Extended thinking config for the Anthropic API.
    max_tokens : int
        Maximum tokens for LLM response.

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
        reference_point = compute_reference_point(
            oracle, obj_indices, signs, seed,
        )

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

    system_prompt = _build_system_prompt(oracle, thresholds, batch_size)
    mechanism_log: list[str] = []
    step_logs: list[VRStepLog] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0

    # Multi-turn conversation history
    conversation: list[dict[str, str]] = []

    # Track previous batch predictions and actuals for reconciliation
    prev_predictions: list[npt.NDArray[np.float64]] | None = None
    prev_actuals: list[npt.NDArray[np.float64]] | None = None

    # Separate RNG for permutation to not affect other randomness
    permute_rng = np.random.default_rng(seed + 2000)

    n_llm_calls = math.ceil(n_iterations / batch_size)
    eval_count = n_initial
    global_step = 0  # tracks step index for VRStepLog

    for call_idx in range(n_llm_calls):
        # How many points this batch (last batch may be smaller)
        points_this_batch = min(
            batch_size, n_iterations - (eval_count - n_initial),
        )
        if points_this_batch <= 0:
            break

        # Determine phase
        phase = _determine_phase(eval_count, n_initial + n_iterations)

        # Progress logging
        print(
            f"  [VR call {call_idx + 1}/{n_llm_calls}] "
            f"evals={eval_count}/{n_initial + n_iterations} "
            f"phase={phase} HV={hypervolumes[-1]:.4f}",
            flush=True,
        )

        # Compute analysis
        correlations = _compute_correlations(X_all, Y_all)

        # Compute gradients at last few points
        n_grad = min(batch_size, len(X_all))
        gradients = [
            _compute_local_gradients(oracle, X_all[-(n_grad - k)])
            for k in range(n_grad)
        ]

        # Build observation table for the prompt
        if permute_feedback:
            perm = permute_rng.permutation(len(Y_all))
            Y_shown = Y_all[perm]
            # Also permute the prev_actuals shown to the agent
            if prev_actuals is not None:
                # Show permuted last-batch actuals
                prev_actuals_shown: list[npt.NDArray[np.float64]] | None = [
                    Y_shown[-(len(prev_actuals) - k)]
                    for k in range(len(prev_actuals))
                ]
            else:
                prev_actuals_shown = None
        else:
            Y_shown = Y_all
            prev_actuals_shown = prev_actuals

        prompt = _build_iteration_prompt(
            oracle, X_all, Y_shown, mechanism_log,
            prev_predictions, prev_actuals_shown, call_idx,
            correlations, gradients, phase, eval_count,
            n_initial + n_iterations, batch_size,
        )

        # Build multi-turn messages
        conversation.append({"role": "user", "content": prompt})

        # Call LLM with structured output
        batch_response: list[_ProposedPoint] | None = None
        batch_hypothesis = ""
        batch_reconciliation = ""
        batch_biggest_surprise = ""
        raw_text = ""
        tokens_in = 0
        tokens_out = 0

        for attempt in range(max_retries + 1):
            try:
                parse_kwargs: dict[str, object] = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system_prompt,
                    "messages": list(conversation),
                    "temperature": temperature,
                    "output_format": _VRBatchResponseSchema,
                }
                if thinking is not None:
                    parse_kwargs["thinking"] = thinking
                    raw_budget = thinking.get("budget_tokens", 0)
                    budget_tokens = (
                        int(raw_budget)  # type: ignore[call-overload]
                        if raw_budget is not None
                        else 0
                    )
                    parse_kwargs["max_tokens"] = max(
                        max_tokens, budget_tokens + 4096,
                    )

                msg = client.messages.parse(**parse_kwargs)  # type: ignore[arg-type]
                total_llm_calls += 1
                tokens_in = msg.usage.input_tokens
                tokens_out = msg.usage.output_tokens
                total_input_tokens += tokens_in
                total_output_tokens += tokens_out

                parsed = msg.parsed_output

                # Extract text from content blocks (thinking blocks
                # may precede the text block when thinking is enabled)
                raw_text = ""
                for block in msg.content:
                    if hasattr(block, "text") and getattr(block, "type", "") == "text":
                        raw_text = block.text
                        break

                if parsed is not None:
                    batch_response = list(parsed.points)
                    batch_hypothesis = parsed.hypothesis
                    batch_reconciliation = parsed.reconciliation
                    batch_biggest_surprise = parsed.biggest_surprise
                    break

                # parsed_output was None — try manual text parsing
                text = raw_text
                raw_text = text
                data = _extract_json(text)
                batch_hypothesis = str(data.get("hypothesis", ""))
                batch_reconciliation = str(data.get("reconciliation", ""))
                batch_biggest_surprise = str(data.get("biggest_surprise", ""))
                points_raw = data.get("points", [])
                if isinstance(points_raw, list):
                    batch_response = []
                    for p in points_raw:
                        if not isinstance(p, dict):
                            continue
                        batch_response.append(
                            _ProposedPoint(
                                next_point=list(p["next_point"]),
                                prediction=list(p["prediction"]),
                                reasoning=str(p.get("reasoning", "")),
                                explore_or_exploit=str(
                                    p.get("explore_or_exploit", "explore")
                                ),
                                falsification=str(
                                    p.get("falsification", "")
                                ),
                            ),
                        )
                break
            except (ValueError, Exception):
                if attempt == max_retries:
                    # Fallback: random points
                    batch_response = []
                    for _ in range(points_this_batch):
                        x_fb = rng.uniform(lo, hi).tolist()
                        batch_response.append(
                            _ProposedPoint(
                                next_point=x_fb,
                                prediction=[float("nan")] * oracle.n_outputs,
                                reasoning="[PARSE FAILURE]",
                                explore_or_exploit="explore",
                                falsification="[PARSE FAILURE]",
                            ),
                        )
                    batch_hypothesis = "[PARSE FAILURE - random fallback]"
                    batch_reconciliation = ""
                    raw_text = ""

        assert batch_response is not None  # guaranteed by retry logic

        # Append assistant response to conversation for multi-turn context
        if raw_text:
            conversation.append({"role": "assistant", "content": raw_text})
        else:
            # Fallback case — still need assistant turn for valid alternation
            conversation.append({"role": "assistant", "content": "{}"})

        # Process each point in the batch
        batch_preds: list[npt.NDArray[np.float64]] = []
        batch_acts: list[npt.NDArray[np.float64]] = []

        for pt_idx, point in enumerate(batch_response[:points_this_batch]):
            next_pt = np.array(point.next_point, dtype=np.float64)
            pred = np.array(point.prediction, dtype=np.float64)

            # Validate dimensions
            if len(next_pt) != oracle.n_inputs:
                next_pt = rng.uniform(lo, hi)
                pred = np.full(oracle.n_outputs, np.nan)
            elif np.any(~np.isfinite(next_pt)):
                next_pt = rng.uniform(lo, hi)
                pred = np.full(oracle.n_outputs, np.nan)
            else:
                next_pt = np.clip(next_pt, lo, hi)

            if len(pred) != oracle.n_outputs:
                pred = np.full(oracle.n_outputs, np.nan)

            # Evaluate oracle
            y_actual = oracle.evaluate(next_pt)

            # Directional accuracy
            prev_y = Y_all[-1] if len(Y_all) > 0 else None
            dir_acc = _compute_directional_accuracy(pred, y_actual, prev_y)

            # Log
            pred_error = y_actual - pred
            step_log = VRStepLog(
                step=global_step,
                x=next_pt,
                y_predicted=pred,
                y_actual=y_actual,
                hypothesis=batch_hypothesis,
                reasoning=point.reasoning,
                reconciliation=batch_reconciliation,
                prediction_error=pred_error,
                directional_accuracy=dir_acc,
                raw_response=raw_text,
                explore_or_exploit=point.explore_or_exploit,
                falsification=point.falsification,
                biggest_surprise=batch_biggest_surprise,
            )
            step_logs.append(step_log)

            # Update arrays
            X_all = np.vstack([X_all, next_pt[np.newaxis, :]])
            Y_all = np.vstack([Y_all, y_actual[np.newaxis, :]])

            # Hypervolume
            hv = compute_hypervolume(
                Y_all, obj_indices, signs, constraint_indices,
                thresholds, oracle.output_names, reference_point,
            )
            hypervolumes.append(hv)

            batch_preds.append(pred)
            batch_acts.append(y_actual)
            global_step += 1
            eval_count += 1

        # Update mechanism log once per batch
        mechanism_log.append(
            f"Batch {call_idx + 1}: {batch_hypothesis}"
        )

        # Set previous batch for next iteration
        prev_predictions = batch_preds
        prev_actuals = batch_acts

        # Checkpoint: save intermediate state after each batch
        if checkpoint_dir is not None:
            import pathlib

            cp_dir = pathlib.Path(checkpoint_dir)
            cp_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cp_dir / "checkpoint.npz",
                X=X_all, Y=Y_all,
                hypervolumes=np.array(hypervolumes),
                eval_count=eval_count,
                call_idx=call_idx + 1,
                total_llm_calls=total_llm_calls,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            )
            # Save step logs as JSON for readability
            log_data = [
                {
                    "step": s.step,
                    "hypothesis": s.hypothesis,
                    "reasoning": s.reasoning,
                    "reconciliation": s.reconciliation,
                    "falsification": s.falsification,
                    "explore_or_exploit": s.explore_or_exploit,
                    "prediction_error": s.prediction_error.tolist(),
                    "x": s.x.tolist(),
                }
                for s in step_logs
            ]
            with open(cp_dir / "checkpoint_log.json", "w") as f:
                json.dump(log_data, f, indent=2)

    # Extract Pareto front
    pareto_X, pareto_Y = extract_pareto_front(
        X_all, Y_all, oracle, thresholds,
    )

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
