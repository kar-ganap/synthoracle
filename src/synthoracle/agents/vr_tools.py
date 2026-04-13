"""Tool-use Verbal Regularization (VR) agent for multi-objective optimization.

Uses Anthropic tool calling instead of batch JSON proposals. The agent decides
which experiments to run by invoking oracle tools (which cost evaluation budget)
and analysis tools (which are free).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import anthropic
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

from synthoracle.agents.vr import _compute_correlations, _compute_local_gradients, _determine_phase
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


def _make_calibration_schema(n_outputs: int, output_names: tuple[str, ...]) -> type[BaseModel]:
    """Create a CalibrationResponse schema with exact output count."""
    return type(
        "_CalibrationResponse",
        (BaseModel,),
        {
            "__annotations__": {
                "predicted_outputs": list[float],
            },
            "__doc__": (
                f"Predict exactly {n_outputs} outputs: "
                f"{', '.join(output_names)}."
            ),
            "predicted_outputs": Field(
                description=f"Exactly {n_outputs} predicted values for "
                f"{', '.join(output_names)}.",
                min_length=n_outputs,
                max_length=n_outputs,
            ),
        },
    )


class _EdgeConfidence(BaseModel):
    """A single causal edge with confidence score."""

    edge: str = Field(
        description="Causal edge in 'Xi->Yj' format (e.g. 'X2->Y1', 'X3->Y3'). "
        "For interactions use 'Xi*Xj->Yk' (e.g. 'X2*X4->Y1').",
    )
    confidence: float = Field(
        description="Confidence 0.0 (no evidence) to 1.0 (certain).",
    )
    evidence: str = Field(
        description="Brief evidence (e.g. 'OAT range=0.33', 'interaction std=0.14').",
    )


class _IterationSummary(BaseModel):
    """Structured end-of-iteration summary from the tool agent."""

    hypothesis: str = Field(
        description="Your current causal model of the system in natural language.",
    )
    new_findings: list[str] = Field(
        description="What this iteration revealed — one string per finding.",
    )
    surprises: list[str] = Field(
        description="Predictions that were wrong and what you learned from each.",
    )
    next_plan: str = Field(
        description="What to investigate next and why.",
    )
    edges: list[_EdgeConfidence] = Field(
        description="ALL causal edges you have evidence for, with confidence scores. "
        "Include every Xi->Yj edge you tested, even weak ones (confidence < 0.3). "
        "Omitting an edge means you believe it does not exist.",
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def _build_oracle_tools(oracle: Oracle) -> list[dict[str, object]]:
    """Build oracle tool definitions based on oracle metadata."""
    bounds_desc_parts: list[str] = []
    for i, name in enumerate(oracle.input_names):
        bounds_desc_parts.append(
            f"{name} in [{oracle.bounds[i, 0]:.1f}, {oracle.bounds[i, 1]:.1f}]"
        )
    point_desc = (
        f"Input point: [{', '.join(oracle.input_names)}]. "
        f"Bounds: {', '.join(bounds_desc_parts)}."
    )

    pred_desc = (
        f"Your predicted outputs [{', '.join(oracle.output_names)}]. "
        "You MUST commit to a prediction BEFORE seeing results."
    )

    return [
        {
            "name": "evaluate_point",
            "description": (
                "Evaluate the oracle at a specific input point. "
                "You must predict the outputs BEFORE seeing results. "
                "Costs 1 evaluation from your budget."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "point": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": point_desc,
                    },
                    "predicted_outputs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": pred_desc,
                    },
                },
                "required": ["point", "predicted_outputs"],
            },
        },
        {
            "name": "oat_sweep",
            "description": (
                "One-At-a-Time sweep: vary one input across its full range "
                "while holding others fixed. You must predict the TREND "
                "(direction and approximate magnitude of change). "
                "Costs n_levels evaluations."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_name": {
                        "type": "string",
                        "enum": list(oracle.input_names),
                        "description": "Which input to vary.",
                    },
                    "n_levels": {
                        "type": "integer",
                        "minimum": 3,
                        "maximum": 10,
                        "description": "Number of levels to evaluate.",
                    },
                    "base_point": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Base point (other inputs held at these values).",
                    },
                    "predicted_trend": {
                        "type": "string",
                        "description": (
                            "Free-text prediction (optional, for your reasoning notes)."
                        ),
                    },
                    "predicted_trends": {
                        "type": "object",
                        "description": (
                            "Structured per-output predictions: direction and "
                            "expected range (max - min) across the sweep."
                        ),
                        "properties": {
                            oname: {
                                "type": "object",
                                "properties": {
                                    "direction": {
                                        "type": "string",
                                        "enum": [
                                            "increase",
                                            "decrease",
                                            "flat",
                                            "nonmonotonic",
                                        ],
                                    },
                                    "magnitude": {
                                        "type": "number",
                                        "description": (
                                            "Expected output range (max - min)."
                                        ),
                                    },
                                },
                                "required": ["direction", "magnitude"],
                            }
                            for oname in oracle.output_names
                        },
                        "required": list(oracle.output_names),
                    },
                },
                "required": [
                    "input_name", "n_levels", "base_point", "predicted_trends",
                ],
            },
        },
        {
            "name": "interaction_test",
            "description": (
                "2D factorial design: vary two inputs at specified levels while "
                "holding others fixed. You must predict whether an interaction "
                "exists and its approximate size. "
                "Costs len(levels_a) * len(levels_b) evaluations."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_a": {
                        "type": "string",
                        "enum": list(oracle.input_names),
                    },
                    "input_b": {
                        "type": "string",
                        "enum": list(oracle.input_names),
                    },
                    "levels_a": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Levels for input_a.",
                    },
                    "levels_b": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Levels for input_b.",
                    },
                    "base_point": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "predicted_interaction": {
                        "type": "string",
                        "description": (
                            "Your prediction: is there an interaction between "
                            "these inputs? If so, what kind (synergistic, "
                            "antagonistic, regime-dependent)? Approximate size?"
                        ),
                    },
                },
                "required": [
                    "input_a", "input_b", "levels_a", "levels_b",
                    "base_point", "predicted_interaction",
                ],
            },
        },
    ]


def _build_analysis_tools(oracle: Oracle) -> list[dict[str, object]]:
    """Build free analysis tool definitions."""
    return [
        {
            "name": "correlation_matrix",
            "description": (
                "Pearson correlation matrix between inputs and outputs "
                "from all data. FREE (no budget cost)."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "sensitivity_report",
            "description": (
                "Estimated input sensitivities (output range per input) "
                "from all data. FREE."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "local_gradients",
            "description": (
                "Central finite-difference gradients (dOutput/dInput) "
                "at a specific point. FREE."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "point": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "required": ["point"],
            },
        },
        {
            "name": "current_pareto_front",
            "description": "Current non-dominated feasible points and hypervolume. FREE.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "regression_fit",
            "description": (
                "Fit a linear regression model to existing data for "
                "specified inputs/output. FREE."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Input names to include.",
                    },
                    "output": {
                        "type": "string",
                        "description": "Output name to predict.",
                    },
                },
                "required": ["inputs", "output"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tool(
    name: str,
    tool_input: dict[str, object],
    oracle: Oracle,
    X_all: npt.NDArray[np.float64],
    Y_all: npt.NDArray[np.float64],
    eval_budget_remaining: int,
    thresholds: dict[str, float],
    obj_indices: list[int],
    signs: npt.NDArray[np.float64],
    constraint_indices: list[int],
    reference_point: npt.NDArray[np.float64],
) -> tuple[dict[str, object], npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Execute a tool and return (result_dict, new_X_points, new_Y_points, eval_cost).

    For analysis tools, new_X_points and new_Y_points are empty arrays with 0 rows.
    eval_cost is 0 for analysis tools.
    """
    empty_X: npt.NDArray[np.float64] = np.empty((0, oracle.n_inputs), dtype=np.float64)
    empty_Y: npt.NDArray[np.float64] = np.empty((0, oracle.n_outputs), dtype=np.float64)

    # --- Oracle tools ---
    if name == "evaluate_point":
        if eval_budget_remaining < 1:
            result: dict[str, object] = {
                "error": "Budget exceeded. Use free analysis tools or produce your hypothesis.",
                "budget_remaining": 0,
            }
            return result, empty_X, empty_Y, 0

        raw_point = tool_input.get("point", [])
        assert isinstance(raw_point, list)
        point = np.array(raw_point, dtype=np.float64)
        point = np.clip(point, oracle.bounds[:, 0], oracle.bounds[:, 1])
        y = oracle.evaluate(point)

        # Compare with agent's prediction
        raw_pred = tool_input.get("predicted_outputs", [])
        pred_errors: dict[str, object] = {}
        if isinstance(raw_pred, list) and len(raw_pred) == oracle.n_outputs:
            pred = np.array(raw_pred, dtype=np.float64)
            for i, oname in enumerate(oracle.output_names):
                pred_errors[oname] = round(float(y[i] - pred[i]), 6)

        out: dict[str, object] = {
            "outputs": {n: round(float(y[i]), 6) for i, n in enumerate(oracle.output_names)},
            "prediction_errors": pred_errors,
            "cost": 1,
        }
        return out, point.reshape(1, -1), y.reshape(1, -1), 1

    if name == "oat_sweep":
        input_name = str(tool_input.get("input_name", ""))
        raw_n_levels = tool_input.get("n_levels", 5)
        n_levels = int(raw_n_levels) if raw_n_levels is not None else 5  # type: ignore[call-overload]
        raw_base = tool_input.get("base_point", [])
        assert isinstance(raw_base, list)
        base_point = np.array(raw_base, dtype=np.float64)
        base_point = np.clip(base_point, oracle.bounds[:, 0], oracle.bounds[:, 1])

        if n_levels > eval_budget_remaining:
            result = {
                "error": "Budget exceeded. Use free analysis tools or produce your hypothesis.",
                "budget_remaining": eval_budget_remaining,
            }
            return result, empty_X, empty_Y, 0

        idx = list(oracle.input_names).index(input_name)
        lo, hi = float(oracle.bounds[idx, 0]), float(oracle.bounds[idx, 1])
        levels = np.linspace(lo, hi, n_levels)

        new_X_list: list[npt.NDArray[np.float64]] = []
        new_Y_list: list[npt.NDArray[np.float64]] = []
        rows: list[dict[str, object]] = []
        for val in levels:
            pt = base_point.copy()
            pt[idx] = val
            y = oracle.evaluate(pt)
            new_X_list.append(pt)
            new_Y_list.append(y)
            row: dict[str, object] = {input_name: round(float(val), 6)}
            for j, oname in enumerate(oracle.output_names):
                row[oname] = round(float(y[j]), 6)
            rows.append(row)

        # Include agent's prediction for accountability
        predicted_trend = str(tool_input.get("predicted_trend", ""))

        # Score structured predictions if provided
        new_Y_sweep = np.array(new_Y_list, dtype=np.float64)
        predicted_trends = tool_input.get("predicted_trends", {})
        trend_scores: dict[str, object] = {}
        if isinstance(predicted_trends, dict) and predicted_trends:
            for j, oname in enumerate(oracle.output_names):
                pred = predicted_trends.get(oname, {})
                if not isinstance(pred, dict):
                    continue
                actual_range = float(
                    new_Y_sweep[:, j].max() - new_Y_sweep[:, j].min()
                )
                actual_change = float(
                    new_Y_sweep[-1, j] - new_Y_sweep[0, j]
                )
                if actual_range < 0.02:
                    actual_dir = "flat"
                elif abs(actual_change) < actual_range * 0.5:
                    actual_dir = "nonmonotonic"
                elif actual_change > 0:
                    actual_dir = "increase"
                else:
                    actual_dir = "decrease"

                pred_dir = str(pred.get("direction", "unknown"))
                pred_mag = float(pred.get("magnitude", 0.0))
                trend_scores[oname] = {
                    "direction_correct": pred_dir == actual_dir,
                    "predicted_direction": pred_dir,
                    "actual_direction": actual_dir,
                    "predicted_magnitude": round(pred_mag, 4),
                    "actual_magnitude": round(actual_range, 4),
                    "magnitude_error": round(abs(pred_mag - actual_range), 4),
                }

        sweep_result: dict[str, object] = {
            "sweep": rows,
            "varied_input": input_name,
            "n_levels": n_levels,
            "cost": n_levels,
            "your_predicted_trend": predicted_trend,
            "trend_scores": trend_scores,
            "note": "Compare actual results against your prediction above.",
        }
        new_X = np.array(new_X_list, dtype=np.float64)
        return sweep_result, new_X, new_Y_sweep, n_levels

    if name == "interaction_test":
        input_a = str(tool_input.get("input_a", ""))
        input_b = str(tool_input.get("input_b", ""))
        raw_levels_a = tool_input.get("levels_a", [])
        raw_levels_b = tool_input.get("levels_b", [])
        raw_base = tool_input.get("base_point", [])
        assert isinstance(raw_levels_a, list)
        assert isinstance(raw_levels_b, list)
        assert isinstance(raw_base, list)
        levels_a = [float(v) for v in raw_levels_a]
        levels_b = [float(v) for v in raw_levels_b]
        base_point = np.array(raw_base, dtype=np.float64)
        base_point = np.clip(base_point, oracle.bounds[:, 0], oracle.bounds[:, 1])

        cost = len(levels_a) * len(levels_b)
        if cost > eval_budget_remaining:
            result = {
                "error": "Budget exceeded. Use free analysis tools or produce your hypothesis.",
                "budget_remaining": eval_budget_remaining,
            }
            return result, empty_X, empty_Y, 0

        idx_a = list(oracle.input_names).index(input_a)
        idx_b = list(oracle.input_names).index(input_b)

        new_X_list = []
        new_Y_list = []
        grid_rows: list[dict[str, object]] = []
        for va in levels_a:
            for vb in levels_b:
                pt = base_point.copy()
                pt[idx_a] = np.clip(va, oracle.bounds[idx_a, 0], oracle.bounds[idx_a, 1])
                pt[idx_b] = np.clip(vb, oracle.bounds[idx_b, 0], oracle.bounds[idx_b, 1])
                y = oracle.evaluate(pt)
                new_X_list.append(pt)
                new_Y_list.append(y)
                row = {
                    input_a: round(float(pt[idx_a]), 6),
                    input_b: round(float(pt[idx_b]), 6),
                }
                for j, oname in enumerate(oracle.output_names):
                    row[oname] = round(float(y[j]), 6)
                grid_rows.append(row)

        # Compute simple interaction effect for each output
        new_Y_arr = np.array(new_Y_list, dtype=np.float64)
        n_a = len(levels_a)
        n_b = len(levels_b)
        Y_grid = new_Y_arr.reshape(n_a, n_b, oracle.n_outputs)
        interaction_effects: dict[str, float] = {}
        for j, oname in enumerate(oracle.output_names):
            # Interaction = observed - additive prediction
            row_means = Y_grid[:, :, j].mean(axis=1, keepdims=True)
            col_means = Y_grid[:, :, j].mean(axis=0, keepdims=True)
            grand_mean = Y_grid[:, :, j].mean()
            additive_pred = row_means + col_means - grand_mean
            interaction = Y_grid[:, :, j] - additive_pred
            interaction_effects[oname] = round(float(np.std(interaction)), 6)

        # Include agent's prediction for accountability
        predicted_interaction = str(tool_input.get("predicted_interaction", ""))

        inter_result: dict[str, object] = {
            "grid": grid_rows,
            "input_a": input_a,
            "input_b": input_b,
            "interaction_effects_std": interaction_effects,
            "cost": cost,
            "your_predicted_interaction": predicted_interaction,
            "note": "Compare actual interaction effects against your prediction above.",
        }
        new_X = np.array(new_X_list, dtype=np.float64)
        new_Y = np.array(new_Y_list, dtype=np.float64)
        return inter_result, new_X, new_Y, cost

    # --- Analysis tools (free) ---
    if name == "correlation_matrix":
        if len(X_all) < 2:
            result = {"error": "Need at least 2 data points for correlations."}
            return result, empty_X, empty_Y, 0
        corr = _compute_correlations(X_all, Y_all)
        corr_dict: dict[str, dict[str, float]] = {}
        for i, iname in enumerate(oracle.input_names):
            corr_dict[iname] = {}
            for j, oname in enumerate(oracle.output_names):
                corr_dict[iname][oname] = round(float(corr[i, j]), 4)
        result = {"correlations": corr_dict}
        return result, empty_X, empty_Y, 0

    if name == "sensitivity_report":
        if len(X_all) < 2:
            result = {"error": "Need at least 2 data points for sensitivity."}
            return result, empty_X, empty_Y, 0

        sensitivities: dict[str, dict[str, float]] = {}
        for i, iname in enumerate(oracle.input_names):
            x_col = X_all[:, i]
            x_range = float(np.ptp(x_col))
            if x_range < 1e-12:
                sensitivities[iname] = {
                    oname: 0.0 for oname in oracle.output_names
                }
                continue
            sensitivities[iname] = {}
            for j, oname in enumerate(oracle.output_names):
                y_col = Y_all[:, j]
                y_range = float(np.ptp(y_col))
                sensitivities[iname][oname] = round(y_range / x_range, 4)
        result = {"sensitivities": sensitivities}
        return result, empty_X, empty_Y, 0

    if name == "local_gradients":
        raw_point = tool_input.get("point", [])
        assert isinstance(raw_point, list)
        point = np.array(raw_point, dtype=np.float64)
        point = np.clip(point, oracle.bounds[:, 0], oracle.bounds[:, 1])
        grads = _compute_local_gradients(oracle, point)
        grad_dict: dict[str, dict[str, float]] = {}
        for i, iname in enumerate(oracle.input_names):
            grad_dict[iname] = {}
            for j, oname in enumerate(oracle.output_names):
                grad_dict[iname][oname] = round(float(grads[i, j]), 6)
        result = {"gradients": grad_dict, "point": [round(float(v), 6) for v in point]}
        return result, empty_X, empty_Y, 0

    if name == "current_pareto_front":
        pareto_X, pareto_Y = extract_pareto_front(X_all, Y_all, oracle, thresholds)
        hv = compute_hypervolume(
            Y_all, obj_indices, signs, constraint_indices,
            thresholds, oracle.output_names, reference_point,
        )
        pareto_points: list[dict[str, object]] = []
        for k in range(len(pareto_X)):
            pf_entry: dict[str, object] = {}
            for i, iname in enumerate(oracle.input_names):
                pf_entry[iname] = round(float(pareto_X[k, i]), 6)
            for j, oname in enumerate(oracle.output_names):
                pf_entry[oname] = round(float(pareto_Y[k, j]), 6)
            pareto_points.append(pf_entry)
        result = {
            "pareto_front": pareto_points,
            "n_pareto": len(pareto_X),
            "hypervolume": round(float(hv), 6),
        }
        return result, empty_X, empty_Y, 0

    if name == "regression_fit":
        raw_inputs = tool_input.get("inputs", [])
        assert isinstance(raw_inputs, list)
        input_names_req = [str(s) for s in raw_inputs]
        output_name = str(tool_input.get("output", ""))

        if len(X_all) < 2:
            result = {"error": "Need at least 2 data points for regression."}
            return result, empty_X, empty_Y, 0

        # Filter to valid input names (agent may pass interaction terms like "X1*X3")
        valid_inputs = [n for n in input_names_req if n in oracle.input_names]
        if not valid_inputs:
            result = {
                "error": f"No valid input names. Use individual inputs "
                f"({', '.join(oracle.input_names)}), not interaction terms.",
                "invalid_inputs": input_names_req,
            }
            return result, empty_X, empty_Y, 0

        # Map names to column indices
        in_indices = [list(oracle.input_names).index(n) for n in valid_inputs]
        out_idx = list(oracle.output_names).index(output_name)

        A = X_all[:, in_indices]
        # Add intercept
        A_aug = np.column_stack([A, np.ones(len(A))])
        b = Y_all[:, out_idx]
        coeffs, residuals, rank, _ = np.linalg.lstsq(A_aug, b, rcond=None)

        # R²
        y_pred = A_aug @ coeffs
        ss_res = float(np.sum((b - y_pred) ** 2))
        ss_tot = float(np.sum((b - np.mean(b)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        coeff_dict: dict[str, float] = {}
        for k, n in enumerate(valid_inputs):
            coeff_dict[n] = round(float(coeffs[k]), 6)
        coeff_dict["intercept"] = round(float(coeffs[-1]), 6)

        result = {
            "coefficients": coeff_dict,
            "r_squared": round(r_squared, 6),
            "output": output_name,
        }
        return result, empty_X, empty_Y, 0

    # Unknown tool
    result = {"error": f"Unknown tool: {name}"}
    return result, empty_X, empty_Y, 0


# ---------------------------------------------------------------------------
# Content serialization
# ---------------------------------------------------------------------------


def _serialize_content(content: object) -> list[dict[str, object]]:
    """Serialize response content blocks for conversation history."""
    serialized: list[dict[str, object]] = []
    assert isinstance(content, list)
    for block in content:
        btype = getattr(block, "type", "")
        if btype == "text":
            serialized.append({"type": "text", "text": block.text})
        elif btype == "tool_use":
            serialized.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        elif btype == "thinking":
            serialized.append({
                "type": "thinking",
                "thinking": block.thinking,
                "signature": getattr(block, "signature", ""),
            })
    return serialized


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_tool_system_prompt(oracle: Oracle, thresholds: dict[str, float]) -> str:
    """Build the system prompt describing the oracle and available tools."""
    lines: list[str] = []
    lines.append("You are a scientific optimization agent with access to experimental tools.")
    lines.append("")

    # System description
    lines.append("## System")
    lines.append("### Inputs")
    for i, name in enumerate(oracle.input_names):
        lo, hi = oracle.bounds[i]
        lines.append(f"- {name}: [{lo:.4f}, {hi:.4f}]")
    lines.append("")
    lines.append("### Outputs")
    for name, direction in zip(oracle.output_names, oracle.output_directions):
        if direction == "threshold" and name in thresholds:
            lines.append(f"- {name}: {direction} (must be >= {thresholds[name]})")
        else:
            lines.append(f"- {name}: {direction}")
    lines.append("")

    # Tools
    lines.append("## Your Tools")
    lines.append("Oracle tools (cost evaluation budget):")
    lines.append("- evaluate_point: evaluate one input configuration (cost: 1 eval)")
    lines.append("- oat_sweep: vary one input across range (cost: n_levels evals)")
    lines.append("- interaction_test: 2D factorial (cost: n_a x n_b evals)")
    lines.append("")
    lines.append("Analysis tools (FREE, no budget cost):")
    lines.append("- correlation_matrix: Pearson correlations from all data")
    lines.append("- sensitivity_report: estimated input sensitivities")
    lines.append("- local_gradients: finite-difference gradients at a point")
    lines.append("- current_pareto_front: non-dominated points and hypervolume")
    lines.append("- regression_fit: linear regression for specified inputs/output")
    lines.append("")

    # Scientist's Playbook
    lines.append("## Scientist's Playbook")
    lines.append("- DON'T pigeonhole early. Before optimizing, understand the landscape.")
    lines.append(
        "- Vary ONE input at a time (OAT) to isolate which inputs drive which outputs."
    )
    lines.append(
        "- Unless proven otherwise, assume different inputs have different sensitivities."
    )
    lines.append(
        "- Don't assume global smoothness — test across the FULL RANGE of each input"
    )
    lines.append(
        "  (low, mid, high). Regime transitions or thresholds may occur ANYWHERE."
    )
    lines.append(
        "- Check for interactions: if one input's effect on an output changes depending on"
    )
    lines.append(
        "  another input's value, that's a coupling. Test by varying one input at different"
    )
    lines.append("  levels of the other.")
    lines.append(
        "- Prediction failures are MORE informative than correct predictions. Seek surprises."
    )
    lines.append("- NEVER evaluate a point too close to one you've already seen.")
    lines.append(
        "- These objectives CONFLICT — you cannot maximize all simultaneously. Explore the"
    )
    lines.append("  trade-off frontier, not just the maximum of one output.")
    lines.append("")

    # Hypothesis Discipline
    lines.append("## Hypothesis Discipline")
    lines.append("- State your hypothesis clearly.")
    lines.append("- State what would FALSIFY it.")
    lines.append("- Design at least one experiment to TEST a falsification condition.")
    lines.append(
        "- When a prediction fails, state SPECIFICALLY what surprised you and why."
    )
    lines.append("")

    # Protocol
    lines.append("## Protocol")
    lines.append("Each iteration:")
    lines.append("1. Call any combination of tools to gather information")
    lines.append("2. Use analysis tools freely to understand your data")
    lines.append("3. When done, respond with your updated hypothesis and plan")
    lines.append("4. Budget is limited — design experiments wisely")
    lines.append("")
    lines.append("Be efficient: use oat_sweep instead of 5 separate evaluate_point calls.")
    lines.append("Use interaction_test when you suspect two inputs interact.")
    lines.append("Use analysis tools between experiments to plan your next move.")

    return "\n".join(lines)


def _build_tool_iteration_prompt(
    eval_count: int,
    n_budget: int,
    phase: str,
    mechanism_log: list[str],
) -> str:
    """Build the user prompt for each tool-use iteration."""
    remaining = n_budget - eval_count
    lines: list[str] = []

    lines.append("## Budget")
    lines.append(
        f"Oracle evaluations used: {eval_count} / {n_budget}. Remaining: {remaining}."
    )
    lines.append(f"Phase: {phase}")
    lines.append("")

    if mechanism_log:
        lines.append("## Your Mechanism Log")
        for entry in mechanism_log:
            lines.append(f"- {entry}")
        lines.append("")

    lines.append("## Task")
    lines.append(
        "Use your tools to investigate the system. Track which causal edges "
        "(Xi->Yj) you have evidence for and how confident you are in each. "
        "When done, you will be asked for a structured summary."
    )
    if phase == "SCREEN":
        lines.append(
            "In SCREEN phase: use OAT sweeps to identify which inputs matter "
            "for which outputs."
        )
    elif phase == "PROBE":
        lines.append(
            "In PROBE phase: test interactions and nonlinearities for important inputs."
        )
    else:
        lines.append(
            "In OPTIMIZE phase: exploit the learned structure to optimize "
            "the Pareto front."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VRToolsResult:
    """Results from tool-use VR agent."""

    X: npt.NDArray[np.float64]
    Y: npt.NDArray[np.float64]
    hypervolumes: list[float]
    pareto_X: npt.NDArray[np.float64]
    pareto_Y: npt.NDArray[np.float64]
    reference_point: npt.NDArray[np.float64]
    seed: int
    n_initial: int
    n_budget: int
    eval_count: int
    total_seconds: float
    mechanism_log: list[str]
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    calibration_checks: list[dict[str, object]] = field(default_factory=list)
    iteration_summaries: list[dict[str, object]] = field(default_factory=list)
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_vr_tools(
    oracle: Oracle,
    *,
    n_initial: int | None = None,
    n_budget: int = 72,
    seed: int = 42,
    reference_point: npt.NDArray[np.float64] | None = None,
    thresholds: dict[str, float] | None = None,
    model: str = "claude-opus-4-6",
    thinking: dict[str, object] | None = None,
    max_tokens: int = 16000,
    max_tool_calls_per_iteration: int = 15,
    checkpoint_dir: str | None = None,
    skip_iteration_summary: bool = False,
    prior_knowledge: str | None = None,
    calibration_interval: int = 20,
) -> VRToolsResult:
    """Run the tool-use VR agent.

    Parameters
    ----------
    oracle : Oracle
        The oracle to optimize.
    n_initial : int, optional
        Number of initial random evaluations. Defaults to 2 * oracle.n_inputs.
    n_budget : int
        Total evaluation budget (including initial).
    seed : int
        Random seed for reproducibility.
    reference_point : ndarray, optional
        Reference point for hypervolume. Auto-computed if None.
    thresholds : dict, optional
        Threshold constraints, e.g. {"Y3": 0.4}.
    model : str
        Anthropic model identifier.
    thinking : dict, optional
        Extended thinking config for the Anthropic API.
    max_tokens : int
        Maximum tokens for LLM response.
    max_tool_calls_per_iteration : int
        Maximum tool calls allowed per agent iteration.
    checkpoint_dir : str, optional
        Directory for intermediate checkpoints.

    Returns
    -------
    VRToolsResult
        Full optimization results with tool call logs and token tracking.
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

    # Initial design
    rng = np.random.default_rng(seed)
    lo = oracle.bounds[:, 0]
    hi = oracle.bounds[:, 1]
    X_all = rng.uniform(lo, hi, size=(n_initial, oracle.n_inputs))
    Y_all = oracle.evaluate_batch(X_all)

    # Initial hypervolumes
    hypervolumes: list[float] = []
    for k in range(1, n_initial + 1):
        hv = compute_hypervolume(
            Y_all[:k], obj_indices, signs, constraint_indices,
            thresholds, oracle.output_names, reference_point,
        )
        hypervolumes.append(hv)

    # Build tools
    oracle_tools = _build_oracle_tools(oracle)
    analysis_tools = _build_analysis_tools(oracle)
    all_tools = oracle_tools + analysis_tools

    system_prompt = _build_tool_system_prompt(oracle, thresholds)
    if prior_knowledge is not None:
        system_prompt += "\n\n" + prior_knowledge
        system_prompt += """

## Managing Prior Knowledge
CRITICAL: This is a DIFFERENT system. The prior is from a related system
and may be substantially wrong — wrong edges, wrong functional forms,
wrong interactions.

SCREEN FIRST, THEN COMPARE TO PRIOR:
- Start with OAT sweeps on ALL inputs, exactly as if you had no prior.
  OAT sweeps are cheap and give you ground truth for THIS system.
- After screening, compare your OAT results to the prior's claims.
  Where do they agree? Where do they disagree?
- The prior tells you what to LOOK FOR in your data, not what
  experiments to SKIP.

INTERACTION TESTS — STRICT RULE:
- Do NOT run interaction tests to verify prior interaction claims.
  The prior's interactions (e.g. X2*X4) may not exist in this system.
- Only run interaction tests when YOUR OAT data shows anomalies that
  suggest an interaction: e.g. an input's OAT range changes drastically
  at different base points, or local gradients show unexpected patterns.
- Each interaction test costs 9+ evals. That budget is better spent on
  optimization unless you have strong OAT evidence of nonadditive effects.

COMMON PRIOR FAILURES (expect these):
- Edges that exist in the prior but are absent here
- Edges absent from the prior that exist here
- Functional forms that changed (monotonic to nonmonotonic, threshold to smooth)
- Interaction pairs that shifted or disappeared entirely
- Sign reversals on specific outputs

When your OAT data contradicts the prior, TRUST YOUR DATA. Update
immediately — do not average with the prior or give it partial credit."""
    mechanism_log: list[str] = []
    tool_calls_log: list[dict[str, object]] = []
    calibration_checks: list[dict[str, object]] = []
    iteration_summaries: list[dict[str, object]] = []
    total_llm_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    conversation: list[dict[str, object]] = []
    eval_count = n_initial

    # Calibration checkpoint tracking
    next_calibration = n_initial + calibration_interval

    # Safety valve: detect iterations that add zero evals (can happen when
    # adaptive thinking consumes all output tokens, leaving no room for
    # tool_use blocks). Break out if this persists for several iterations.
    no_progress_streak = 0
    max_no_progress_iters = 3

    while eval_count < n_budget:
        eval_count_before_iter = eval_count

        # --- Calibration checkpoint ---
        if (
            calibration_interval > 0
            and eval_count >= next_calibration
            and eval_count < n_budget
        ):
            cal_point = rng.uniform(lo, hi)
            cal_point_str = [round(float(v), 4) for v in cal_point]
            cal_msg = (
                "CALIBRATION CHECK: Predict all "
                f"{oracle.n_outputs} outputs "
                f"({', '.join(oracle.output_names)}) for this point: "
                f"{cal_point_str}."
            )
            conversation.append({"role": "user", "content": cal_msg})

            cal_schema = _make_calibration_schema(
                oracle.n_outputs, oracle.output_names,
            )
            cal_kwargs: dict[str, object] = {
                "model": model,
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": list(conversation),
                "output_format": cal_schema,
            }
            if thinking is not None:
                cal_kwargs["thinking"] = thinking
                cal_kwargs["max_tokens"] = 32000
            cal_response = None
            for _cal_attempt in range(2):
                try:
                    cal_response = client.messages.parse(  # type: ignore[arg-type]
                        timeout=1200.0, **cal_kwargs,
                    )
                    break
                except Exception as cal_err:
                    if _cal_attempt == 0:
                        print(
                            f"  [CALIBRATION retry] {type(cal_err).__name__}: "
                            f"{str(cal_err)[:100]}",
                            flush=True,
                        )
                    # On second failure, leave cal_response as None
            if cal_response is not None:
                total_llm_calls += 1
                total_input_tokens += cal_response.usage.input_tokens
                total_output_tokens += cal_response.usage.output_tokens

            # Extract prediction from structured output
            cal_predicted = np.full(oracle.n_outputs, np.nan)
            cal_text = ""
            if cal_response is not None:
                for block in cal_response.content:
                    if hasattr(block, "text") and getattr(block, "type", "") == "text":
                        cal_text = block.text
                        break
                if cal_response.parsed_output is not None:
                    preds = cal_response.parsed_output.predicted_outputs
                    cal_predicted = np.array(preds, dtype=np.float64)
            conversation.append({"role": "assistant", "content": cal_text or "{}"})

            # Evaluate
            cal_actual = oracle.evaluate(cal_point)
            cal_errors = cal_actual - cal_predicted
            eval_count += 1

            # Track HV
            X_all = np.vstack([X_all, cal_point[np.newaxis, :]])
            Y_all = np.vstack([Y_all, cal_actual[np.newaxis, :]])
            hv = compute_hypervolume(
                Y_all, obj_indices, signs, constraint_indices,
                thresholds, oracle.output_names, reference_point,
            )
            hypervolumes.append(hv)

            # Log
            parse_ok = not np.any(np.isnan(cal_predicted))
            calibration_checks.append({
                "eval": eval_count,
                "point": cal_point.tolist(),
                "predicted": cal_predicted.tolist(),
                "actual": cal_actual.tolist(),
                "errors": cal_errors.tolist(),
                "max_error": float(np.nanmax(np.abs(cal_errors)))
                if parse_ok else float("nan"),
                "mae": float(np.mean(np.abs(cal_errors)))
                if parse_ok else float("nan"),
            })
            if parse_ok:
                print(
                    f"  [CALIBRATION @ eval {eval_count}] "
                    f"MAE={float(np.mean(np.abs(cal_errors))):.4f} "
                    f"max|err|={float(np.max(np.abs(cal_errors))):.4f}",
                    flush=True,
                )
            else:
                stop = getattr(cal_response, "stop_reason", "unknown")
                print(
                    f"  [CALIBRATION @ eval {eval_count}] "
                    f"parse failed — stop={stop}, "
                    f"parsed_output={cal_response.parsed_output}, "
                    f"raw: {cal_text[:200]}",
                    flush=True,
                )

            # Inform agent of result
            conversation.append({
                "role": "user",
                "content": (
                    f"Calibration result — actual outputs: "
                    f"{[round(float(v), 4) for v in cal_actual]}. "
                    f"Your errors: {[round(float(v), 4) for v in cal_errors]}. "
                    "Continue with your experiments."
                ),
            })
            next_calibration += calibration_interval

        # Phase
        phase = _determine_phase(eval_count, n_budget)

        # Build iteration prompt
        prompt = _build_tool_iteration_prompt(eval_count, n_budget, phase, mechanism_log)
        conversation.append({"role": "user", "content": prompt})

        # Tool calling loop
        tool_calls_this_iter = 0
        msg = None

        while tool_calls_this_iter < max_tool_calls_per_iteration:
            # Build API call kwargs
            create_kwargs: dict[str, object] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": list(conversation),
                "tools": all_tools,
                "tool_choice": {"type": "auto"},
            }
            if thinking is not None:
                create_kwargs["thinking"] = thinking

            t_call = time.monotonic()
            try:
                msg = client.messages.create(  # type: ignore[call-overload]
                    timeout=600.0, **create_kwargs,
                )
            except Exception as api_err:
                elapsed = time.monotonic() - t_call
                print(
                    f"    [tool call {tool_calls_this_iter + 1}] "
                    f"FAILED after {elapsed:.0f}s: {type(api_err).__name__}. "
                    f"Retrying...",
                    flush=True,
                )
                try:
                    msg = client.messages.create(  # type: ignore[call-overload]
                        timeout=1200.0, **create_kwargs,
                    )
                except Exception:
                    print(
                        f"    [tool call {tool_calls_this_iter + 1}] "
                        f"Retry also failed. Ending iteration.",
                        flush=True,
                    )
                    break

            total_llm_calls += 1
            total_input_tokens += msg.usage.input_tokens
            total_output_tokens += msg.usage.output_tokens
            call_secs = time.monotonic() - t_call

            # Progress logging with timestamp
            elapsed_total = time.monotonic() - t0
            print(
                f"    [tool call {tool_calls_this_iter + 1}] "
                f"tokens={msg.usage.input_tokens}in+{msg.usage.output_tokens}out "
                f"budget={eval_count}/{n_budget} "
                f"({call_secs:.0f}s call, {elapsed_total:.0f}s total)",
                flush=True,
            )

            # Extract tool use blocks
            tool_use_blocks = [
                b for b in msg.content if getattr(b, "type", "") == "tool_use"
            ]

            if not tool_use_blocks:
                # Model is done calling tools — extract hypothesis from text
                conversation.append({
                    "role": "assistant",
                    "content": _serialize_content(msg.content),
                })
                break

            # Execute tool calls
            conversation.append({
                "role": "assistant",
                "content": _serialize_content(msg.content),
            })
            tool_results_content: list[dict[str, object]] = []
            for tc in tool_use_blocks:
                result_dict, new_X, new_Y, cost = _execute_tool(
                    tc.name,
                    tc.input,
                    oracle,
                    X_all,
                    Y_all,
                    n_budget - eval_count,
                    thresholds,
                    obj_indices,
                    signs,
                    constraint_indices,
                    reference_point,
                )

                # Append new data
                if len(new_X) > 0:
                    X_all = np.vstack([X_all, new_X])
                    Y_all = np.vstack([Y_all, new_Y])
                    eval_count += cost
                    # Compute HV for each new point
                    for k_offset in range(cost):
                        idx_end = len(X_all) - cost + k_offset + 1
                        hv = compute_hypervolume(
                            Y_all[:idx_end],
                            obj_indices,
                            signs,
                            constraint_indices,
                            thresholds,
                            oracle.output_names,
                            reference_point,
                        )
                        hypervolumes.append(hv)

                tool_calls_log.append({
                    "name": tc.name,
                    "input": tc.input,
                    "cost": cost,
                    "result": result_dict,
                })
                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(result_dict),
                })
                tool_calls_this_iter += 1

            conversation.append({"role": "user", "content": tool_results_content})

        # Structured iteration summary via messages.parse()
        # When skip_iteration_summary=True: skip the forced articulation.
        # The agent still has all tool results in its conversation but isn't
        # required to distill them into a structured causal model. This is the
        # ablation test for whether forced articulation helps.
        summary_text = ""

        if not skip_iteration_summary:
            conversation.append({
                "role": "user",
                "content": (
                    "Provide your iteration summary. For the edges field, list EVERY "
                    "causal edge you have evidence for with confidence and evidence string."
                ),
            })
            summary_kwargs: dict[str, object] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": list(conversation),
                "output_format": _IterationSummary,
            }
            if thinking is not None:
                summary_kwargs["thinking"] = thinking
                raw_budget = thinking.get("budget_tokens", 0)
                budget_tokens = (
                    int(raw_budget) if raw_budget is not None else 0  # type: ignore[call-overload]
                )
                summary_kwargs["max_tokens"] = max(
                    max_tokens, budget_tokens + 4096,
                )
            try:
                summary_msg = client.messages.parse(  # type: ignore[arg-type]
                    timeout=600.0, **summary_kwargs,
                )
                total_llm_calls += 1
                total_input_tokens += summary_msg.usage.input_tokens
                total_output_tokens += summary_msg.usage.output_tokens
            except Exception as summary_err:
                print(
                    f"  [SUMMARY parse failed] {type(summary_err).__name__}: "
                    f"{str(summary_err)[:100]}",
                    flush=True,
                )
                summary_msg = None

            # Extract text for conversation continuity
            if summary_msg is not None:
                for block in summary_msg.content:
                    if hasattr(block, "text") and getattr(block, "type", "") == "text":
                        summary_text = block.text
                        break
            conversation.append({"role": "assistant", "content": summary_text or "{}"})

            # Store structured summary
            summary = summary_msg.parsed_output if summary_msg is not None else None
            if summary is not None:
                # Convert edges list to confidence dict for downstream analysis
                confidence_dict = {
                    e.edge: e.confidence for e in summary.edges
                }
                iteration_summaries.append({
                    "iteration": len(iteration_summaries),
                    "eval_count": eval_count,
                    "hypothesis": summary.hypothesis,
                    "new_findings": summary.new_findings,
                    "surprises": summary.surprises,
                    "next_plan": summary.next_plan,
                    "confidence": confidence_dict,
                    "edges": [
                        {"edge": e.edge, "confidence": e.confidence,
                         "evidence": e.evidence}
                        for e in summary.edges
                    ],
                })
                mechanism_log.append(summary.hypothesis)
            else:
                # Fallback: extract from tool-loop exit text
                hypothesis = ""
                if msg is not None:
                    for block in msg.content:
                        if getattr(block, "type", "") == "text":
                            hypothesis = block.text
                            break
                mechanism_log.append(f"Iteration: {hypothesis}")
        else:
            # Ablation: no structured summary. Just extract the tool-loop
            # exit text and continue. The agent isn't asked to articulate
            # a structured causal model between iterations.
            hypothesis = ""
            if msg is not None:
                for block in msg.content:
                    if getattr(block, "type", "") == "text":
                        hypothesis = block.text
                        break
            mechanism_log.append(f"Iteration (no summary): {hypothesis}")
            conversation.append({
                "role": "assistant",
                "content": hypothesis or "Continuing optimization.",
            })

        # Condense conversation to prevent unbounded context growth.
        # Keep: system prompt (implicit), iteration summaries, and a
        # brief data recap. Drop: raw tool call/result history.
        if len(conversation) > 6:
            # Build compact data summary for context retention
            y_mins = Y_all.min(axis=0)
            y_maxs = Y_all.max(axis=0)
            output_ranges = ", ".join(
                f"{oracle.output_names[j]}=[{y_mins[j]:.3f}, {y_maxs[j]:.3f}]"
                for j in range(oracle.n_outputs)
            )

            # Best values per output (accounting for direction)
            best_lines = []
            for j, (oname, direction) in enumerate(
                zip(oracle.output_names, oracle.output_directions)
            ):
                if direction == "maximize":
                    best_lines.append(f"{oname}: best={Y_all[:, j].max():.4f}")
                elif direction == "minimize":
                    best_lines.append(f"{oname}: best={Y_all[:, j].min():.4f}")

            # Recent evaluate_point results (last 5 for context)
            recent_evals = []
            for tc in reversed(tool_calls_log):
                if tc.get("name") == "evaluate_point" and len(recent_evals) < 5:
                    inp = tc.get("input", {})
                    res = tc.get("result", {})
                    if isinstance(res, dict) and "outputs" in res:
                        recent_evals.append(
                            f"  point={inp.get('point', '?')} -> "
                            f"{res['outputs']}"
                        )

            remaining = n_budget - eval_count
            condensed_content = (
                f"[Context condensed after iteration "
                f"{len(iteration_summaries)}]\n\n"
                f"## Status\n"
                f"Evals: {eval_count}/{n_budget} ({remaining} remaining). "
                f"HV: {hypervolumes[-1]:.4f}. "
                f"Pareto front: {len(Y_all)} points observed.\n\n"
                f"## Output Ranges\n{output_ranges}\n"
                f"Best values: {', '.join(best_lines)}\n\n"
                f"## Your Causal Model\n"
                f"{summary_text or mechanism_log[-1]}\n\n"
            )
            if recent_evals:
                condensed_content += (
                    f"## Recent Evaluations\n"
                    + "\n".join(recent_evals) + "\n\n"
                )
            condensed_content += (
                f"## Next Steps\n"
                f"You have {remaining} evals left. Use evaluate_point "
                f"for targeted Pareto optimization based on your causal model."
            )

            conversation = [
                {"role": "user", "content": condensed_content},
                {
                    "role": "assistant",
                    "content": (
                        "Understood. I have my causal model, the data summary, "
                        "and recent evaluations. Continuing with optimization."
                    ),
                },
            ]

        # Checkpoint
        if checkpoint_dir is not None:
            import pathlib

            cp_dir = pathlib.Path(checkpoint_dir)
            cp_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cp_dir / "checkpoint.npz",
                X=X_all,
                Y=Y_all,
                hypervolumes=np.array(hypervolumes),
                eval_count=eval_count,
                total_llm_calls=total_llm_calls,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            )

        # Progress
        elapsed_total = time.monotonic() - t0
        print(
            f"  [Iteration done] evals={eval_count}/{n_budget} "
            f"HV={hypervolumes[-1]:.4f} tools_used={tool_calls_this_iter} "
            f"({elapsed_total:.0f}s total)",
            flush=True,
        )

        # Safety valve: track no-progress iterations and bail out if stuck.
        if eval_count == eval_count_before_iter:
            no_progress_streak += 1
            print(
                f"  [NO PROGRESS] iteration added 0 evals "
                f"(streak={no_progress_streak}/{max_no_progress_iters})",
                flush=True,
            )
            if no_progress_streak >= max_no_progress_iters:
                print(
                    f"  [SAFETY VALVE] {max_no_progress_iters} consecutive "
                    f"no-progress iterations — breaking out of main loop "
                    f"(eval_count={eval_count}/{n_budget}).",
                    flush=True,
                )
                break
        else:
            no_progress_streak = 0

    # Extract Pareto front
    pareto_X, pareto_Y = extract_pareto_front(X_all, Y_all, oracle, thresholds)

    total_seconds = time.monotonic() - t0

    return VRToolsResult(
        X=X_all,
        Y=Y_all,
        hypervolumes=hypervolumes,
        pareto_X=pareto_X,
        pareto_Y=pareto_Y,
        reference_point=reference_point,
        seed=seed,
        n_initial=n_initial,
        n_budget=n_budget,
        eval_count=eval_count,
        total_seconds=total_seconds,
        mechanism_log=mechanism_log,
        tool_calls=tool_calls_log,
        calibration_checks=calibration_checks,
        iteration_summaries=iteration_summaries,
        total_llm_calls=total_llm_calls,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )
