"""Holistic within-run dynamics analysis.

Extracts per-iteration learning trajectories, edge confidence drift, tool-call
allocation patterns, and other dynamics from the existing run logs. Answers
questions the end-state rubric can't: does the agent actually learn from
feedback? Does calibration improve over time? Does prior vs fresh change
behavior in a consistent way?

This script treats the existing dataset as the final dataset — no new
experiments. Every claim in the output must cite a specific data point
produced by a function here.

Usage:
    uv run python experiments/analysis/holistic_analysis.py
    uv run python experiments/analysis/holistic_analysis.py --section A
    uv run python experiments/analysis/holistic_analysis.py --sections A,B,C

Outputs:
    experiments/analysis/results/holistic_analysis.md
    experiments/analysis/results/holistic_analysis.json
    experiments/analysis/results/holistic_plots/*.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reuse audit_all.py infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from audit_all import (  # noqa: E402
    ORACLE_CONDITIONS,
    ConditionDef,
    RunRef,
    _bo_runs_for,
)

from synthoracle.oracles.medium import MediumOracle  # noqa: E402
from synthoracle.oracles.medium_1c import MediumOracle1C  # noqa: E402
from synthoracle.oracles.medium_1d import MediumOracle1D  # noqa: E402
from synthoracle.oracles.medium_1e import MediumOracle1E  # noqa: E402
from synthoracle.oracles.medium_hd import MediumOracleHD  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
RESULTS_DIR = ANALYSIS_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "holistic_plots"
MD_PATH = RESULTS_DIR / "holistic_analysis.md"
JSON_PATH = RESULTS_DIR / "holistic_analysis.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Oracle registry for per-oracle metadata
# ---------------------------------------------------------------------------

_ORACLE_CACHE: dict[str, Any] = {}


def get_oracle(label: str) -> Any:
    """Return (cached) oracle instance for a label."""
    if label in _ORACLE_CACHE:
        return _ORACLE_CACHE[label]
    oracle: Any
    if label == "1A":
        oracle = MediumOracle(variant="1A")
    elif label == "1B":
        oracle = MediumOracle(variant="1B")
    elif label == "1C":
        oracle = MediumOracle1C()
    elif label == "1D":
        oracle = MediumOracle1D()
    elif label == "1E":
        oracle = MediumOracle1E()
    elif label == "HD":
        oracle = MediumOracleHD()
    else:
        raise ValueError(f"Unknown oracle label: {label}")
    _ORACLE_CACHE[label] = oracle
    return oracle


# ---------------------------------------------------------------------------
# Per-iteration data model
# ---------------------------------------------------------------------------


@dataclass
class IterationMetrics:
    """Metrics computed for a single iteration of a single run."""

    iteration: int  # iteration index (0-based)
    eval_count: int  # cumulative oracle evals at end of iteration
    n_tool_calls: int  # tool calls that fell inside this iteration window

    # Prediction quality metrics (from tool calls in this iteration)
    oat_sweeps_count: int
    oat_dir_accuracy: float | None  # fraction direction_correct across all OAT outputs
    oat_mag_mae: float | None  # mean magnitude_error across all OAT outputs
    eval_point_count: int
    eval_point_mae: float | None  # mean |prediction_errors| across all outputs

    # From iteration_summary itself
    n_findings: int
    n_surprises: int
    n_edges: int  # total edges in iteration_summary.edges
    n_edges_high_conf: int  # edges with confidence >= 0.5

    # Causal discovery quality at this iteration (requires GT)
    edges_tp_at_high_conf: int  # high-conf edges that are in GT IO projection
    edges_fp_at_high_conf: int  # high-conf edges that are NOT in GT
    precision_at_high_conf: float | None
    recall_at_high_conf: float | None

    # Tool call type distribution this iteration (name -> count)
    tool_call_counts: dict[str, int]


@dataclass
class RunMetrics:
    """Metrics for one full run."""

    oracle_label: str
    condition_label: str
    seed: int
    model: str
    n_initial: int
    n_iterations: int
    final_eval_count: int
    final_hv: float
    iterations: list[IterationMetrics]

    # Run-level aggregates
    total_tool_calls: int
    total_oat_sweeps: int
    total_eval_points: int
    total_interaction_tests: int
    tool_call_counts_total: dict[str, int]

    # HV trajectory (per eval count)
    hv_trajectory: list[float]


# ---------------------------------------------------------------------------
# Data loading and per-iteration extraction
# ---------------------------------------------------------------------------


def load_run(run_ref: RunRef) -> tuple[dict, dict]:
    """Load a single run's log dict + npz dict."""
    with open(run_ref.log_path) as f:
        log = json.load(f)
    npz = dict(np.load(run_ref.npz_path, allow_pickle=True))
    return log, npz


def n_initial_for_oracle(oracle_label: str) -> int:
    """Return n_initial = 2 * n_inputs (matches vr_tools default)."""
    oracle = get_oracle(oracle_label)
    return 2 * oracle.n_inputs


def ground_truth_io_pairs(oracle_label: str) -> set[tuple[str, str]]:
    """Return set of (source, target) pairs in the oracle's IO-projected GT DAG."""
    oracle = get_oracle(oracle_label)
    gt_io = oracle.ground_truth().project_to_io()
    return {(e.source, e.target) for e in gt_io.edges}


def group_tool_calls_by_iteration(
    log: dict, n_initial: int,
) -> list[list[dict]]:
    """Partition tool_calls list into per-iteration sublists.

    The logic: each iteration_summary has `eval_count` = cumulative eval count
    at end of iteration. We assign each tool call to the first iteration whose
    eval_count upper bound it does not exceed.
    """
    tool_calls = log.get("tool_calls", []) or []
    iter_summaries = log.get("iteration_summaries", []) or []

    # If no iteration summaries, put all tool calls in iteration 0 (fallback)
    if not iter_summaries:
        return [list(tool_calls)]

    iter_boundaries = [s.get("eval_count", 0) for s in iter_summaries]
    # Sorted cumulative eval count at END of each iteration
    # (monotone non-decreasing by construction)

    groups: list[list[dict]] = [[] for _ in iter_summaries]
    cumulative_cost = n_initial

    for tc in tool_calls:
        cost = int(tc.get("cost", 0) or 0)
        cumulative_cost += cost
        # Assign to the first iteration whose eval_count >= cumulative_cost
        assigned = False
        for i, boundary in enumerate(iter_boundaries):
            if cumulative_cost <= boundary:
                groups[i].append(tc)
                assigned = True
                break
        if not assigned:
            # Past the last iteration — append to last group
            groups[-1].append(tc)

    return groups


def _parse_edge_str(edge_str: str) -> tuple[str, str] | None:
    """Parse 'X1->Y2' or 'X1*X3->Y2' into (source, target). Returns None on interaction edges."""
    s = edge_str.replace(" ", "")
    parts = s.split("->")
    if len(parts) != 2:
        return None
    source, target = parts
    # Skip interaction edges for IO-projection matching
    if "*" in source:
        return None
    return (source, target)


def extract_iteration_metrics(
    log: dict, oracle_label: str, n_initial: int,
) -> list[IterationMetrics]:
    """Extract per-iteration metrics from a single run log."""
    iter_summaries = log.get("iteration_summaries", []) or []
    if not iter_summaries:
        return []

    tool_call_groups = group_tool_calls_by_iteration(log, n_initial)
    gt_pairs = ground_truth_io_pairs(oracle_label)

    iterations: list[IterationMetrics] = []
    for i, summary in enumerate(iter_summaries):
        tc_group = tool_call_groups[i] if i < len(tool_call_groups) else []

        # Tool call counts
        tc_counts: dict[str, int] = {}
        for tc in tc_group:
            name = tc.get("name", "unknown")
            tc_counts[name] = tc_counts.get(name, 0) + 1

        # OAT prediction accuracy in this iteration
        oat_dir_correct: list[bool] = []
        oat_mag_errs: list[float] = []
        for tc in tc_group:
            if tc.get("name") != "oat_sweep":
                continue
            result = tc.get("result", {})
            if not isinstance(result, dict):
                continue
            scores = result.get("trend_scores", {}) or {}
            for _oname, s in scores.items():
                if not isinstance(s, dict):
                    continue
                if "direction_correct" in s:
                    oat_dir_correct.append(bool(s["direction_correct"]))
                if "magnitude_error" in s:
                    try:
                        oat_mag_errs.append(float(s["magnitude_error"]))
                    except (TypeError, ValueError):
                        pass

        # evaluate_point prediction errors in this iteration
        ep_maes: list[float] = []
        for tc in tc_group:
            if tc.get("name") != "evaluate_point":
                continue
            result = tc.get("result", {})
            if not isinstance(result, dict):
                continue
            pred_errors = result.get("prediction_errors", {}) or {}
            if pred_errors:
                try:
                    ep_maes.append(
                        float(np.mean([abs(float(v)) for v in pred_errors.values()]))
                    )
                except (TypeError, ValueError):
                    pass

        # Edges at end of this iteration
        edges = summary.get("edges", []) or []
        n_edges = len(edges)
        high_conf_edges: list[tuple[str, str]] = []
        n_edges_high_conf = 0
        for e in edges:
            conf = e.get("confidence", 0.0)
            if not isinstance(conf, (int, float)):
                continue
            if float(conf) >= 0.5:
                n_edges_high_conf += 1
                parsed = _parse_edge_str(str(e.get("edge", "")))
                if parsed is not None:
                    high_conf_edges.append(parsed)

        tp = sum(1 for p in high_conf_edges if p in gt_pairs)
        fp = sum(1 for p in high_conf_edges if p not in gt_pairs)
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec = tp / len(gt_pairs) if gt_pairs else None

        iterations.append(IterationMetrics(
            iteration=int(summary.get("iteration", i)),
            eval_count=int(summary.get("eval_count", 0)),
            n_tool_calls=len(tc_group),
            oat_sweeps_count=sum(1 for tc in tc_group if tc.get("name") == "oat_sweep"),
            oat_dir_accuracy=(
                float(np.mean(oat_dir_correct)) if oat_dir_correct else None
            ),
            oat_mag_mae=float(np.mean(oat_mag_errs)) if oat_mag_errs else None,
            eval_point_count=sum(1 for tc in tc_group if tc.get("name") == "evaluate_point"),
            eval_point_mae=float(np.mean(ep_maes)) if ep_maes else None,
            n_findings=len(summary.get("new_findings", []) or []),
            n_surprises=len(summary.get("surprises", []) or []),
            n_edges=n_edges,
            n_edges_high_conf=n_edges_high_conf,
            edges_tp_at_high_conf=tp,
            edges_fp_at_high_conf=fp,
            precision_at_high_conf=prec,
            recall_at_high_conf=rec,
            tool_call_counts=tc_counts,
        ))

    return iterations


def extract_run_metrics(
    oracle_label: str,
    condition: ConditionDef,
    run_ref: RunRef,
) -> RunMetrics | None:
    """Extract all metrics for one run. Returns None on load failure."""
    try:
        log, npz = load_run(run_ref)
    except Exception as e:
        print(f"  WARN: failed to load {run_ref.log_path.name}: {e}")
        return None

    n_initial = n_initial_for_oracle(oracle_label)
    iterations = extract_iteration_metrics(log, oracle_label, n_initial)

    hv_trajectory = [float(h) for h in np.asarray(npz["hypervolumes"])]

    # Aggregate tool call counts
    total_counts: dict[str, int] = {}
    for tc in log.get("tool_calls", []) or []:
        name = tc.get("name", "unknown")
        total_counts[name] = total_counts.get(name, 0) + 1

    return RunMetrics(
        oracle_label=oracle_label,
        condition_label=condition.label,
        seed=run_ref.seed,
        model=condition.model,
        n_initial=n_initial,
        n_iterations=len(iterations),
        final_eval_count=int(log.get("eval_count", len(hv_trajectory))),
        final_hv=hv_trajectory[-1] if hv_trajectory else 0.0,
        iterations=iterations,
        total_tool_calls=sum(total_counts.values()),
        total_oat_sweeps=total_counts.get("oat_sweep", 0),
        total_eval_points=total_counts.get("evaluate_point", 0),
        total_interaction_tests=total_counts.get("interaction_test", 0),
        tool_call_counts_total=total_counts,
        hv_trajectory=hv_trajectory,
    )


# ---------------------------------------------------------------------------
# Run registry: load every run in the dataset
# ---------------------------------------------------------------------------


def load_all_runs() -> dict[str, dict[str, list[RunMetrics]]]:
    """Load every run in the dataset. Returns dict[oracle][condition] → [RunMetrics]."""
    all_runs: dict[str, dict[str, list[RunMetrics]]] = {}
    for oracle_label, conditions in ORACLE_CONDITIONS.items():
        all_runs[oracle_label] = {}
        for cond in conditions:
            runs = []
            for run_ref in cond.runs:
                rm = extract_run_metrics(oracle_label, cond, run_ref)
                if rm is not None:
                    runs.append(rm)
            all_runs[oracle_label][cond.label] = runs
    return all_runs


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------


class FindingsDoc:
    """Appender for holistic_analysis.md with section tracking."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []
        self.json_data: dict[str, Any] = {}

    def start(self) -> None:
        """Write header."""
        self.lines = [
            "# Holistic Within-Run Dynamics Analysis",
            "",
            "Systematic mining of per-iteration data from every run in the dataset. ",
            "Every claim in this document cites a specific data point produced by ",
            "`experiments/analysis/holistic_analysis.py`. No new experiments.",
            "",
            "**Data sources:**",
            "- `iteration_summaries` (hypothesis, edges with confidence, surprises, findings, next_plan)",
            "- `tool_calls` (grouped by iteration via cumulative cost matching against iteration eval_counts)",
            "- `calibration_checks` (per-output errors at specific eval counts)",
            "- HV trajectories (from `npz['hypervolumes']`)",
            "",
            "**Not analyzed:** per-LLM-call thinking budgets (only aggregate tokens), ",
            "ablation runs `ablation_real/permuted_seed42.npz` (no iteration_summaries — ",
            "only HV + X/Y trajectories).",
            "",
            "---",
            "",
        ]

    def append(self, text: str) -> None:
        self.lines.append(text)

    def heading(self, level: int, text: str) -> None:
        self.lines.append(f"{'#' * level} {text}")
        self.lines.append("")

    def section(self, letter: str, title: str) -> None:
        self.lines.append(f"## Section {letter} — {title}")
        self.lines.append("")

    def save_json(self, section: str, data: Any) -> None:
        """Save a section's structured data to the combined JSON."""
        self.json_data[section] = data

    def flush(self) -> None:
        """Write the markdown + JSON to disk."""
        self.path.write_text("\n".join(self.lines) + "\n")
        with open(JSON_PATH, "w") as f:
            json.dump(self.json_data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy and dataclass types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if not math.isnan(f) else None
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Section A: per-run learning trajectories
# ---------------------------------------------------------------------------


def section_a_learning_trajectories(
    all_runs: dict[str, dict[str, list[RunMetrics]]],
    doc: FindingsDoc,
) -> None:
    """Section A: does per-iteration prediction accuracy improve?

    Tests:
    - 'VR learns from feedback': per-iteration OAT direction accuracy should
      trend upward, or at least not downward, over iterations.
    - 'Surprise rate decreases': surprise count should drop as the model
      converges.
    - 'Extended budget produces more learning': HD ext should show stronger
      learning signal than HD base, not just more total iterations.
    """
    doc.section("A", "Per-run learning trajectories")
    doc.append(
        "**Question:** Does per-iteration prediction accuracy actually improve "
        "during a run? If the agent is using feedback, we expect OAT direction "
        "accuracy to trend upward and surprise counts to trend downward across "
        "iterations. If these are flat or noisy, the \"VR learns from feedback\" "
        "claim is weakened."
    )
    doc.append("")
    doc.append(
        "**Method:** For each run, group tool calls by iteration using cumulative "
        "cost matching against `iteration_summaries[i].eval_count`, then compute "
        "(a) mean `direction_correct` across all OAT outputs in the iteration, "
        "(b) mean `magnitude_error`, (c) mean |`prediction_errors[Yj]`| across "
        "`evaluate_point` calls, (d) surprise count from `iteration_summaries[i].surprises`."
    )
    doc.append("")

    # For each condition, gather per-iteration trajectories across seeds
    section_data: dict[str, Any] = {}

    for oracle_label, conditions in all_runs.items():
        for cond_label, runs in conditions.items():
            if not runs:
                continue
            cond_key = f"{oracle_label}/{cond_label}"
            cond_trajectories: list[dict[str, Any]] = []

            for run in runs:
                iters = run.iterations
                traj = {
                    "seed": run.seed,
                    "model": run.model,
                    "n_iterations": run.n_iterations,
                    "final_hv": run.final_hv,
                    "per_iter_oat_dir_acc": [
                        it.oat_dir_accuracy for it in iters
                    ],
                    "per_iter_oat_mag_mae": [
                        it.oat_mag_mae for it in iters
                    ],
                    "per_iter_eval_mae": [
                        it.eval_point_mae for it in iters
                    ],
                    "per_iter_surprise_count": [
                        it.n_surprises for it in iters
                    ],
                    "per_iter_eval_count": [
                        it.eval_count for it in iters
                    ],
                }
                cond_trajectories.append(traj)

            section_data[cond_key] = cond_trajectories

    # Summarize: for each condition, did per-iteration metrics trend in the
    # expected direction? Note: agents front-load OAT sweeps in iteration 0,
    # so "last iteration OAT accuracy" is often None. The meaningful comparison
    # is FIRST non-None vs LAST non-None across iterations.
    doc.heading(3, "A.1 Learning signal summary per condition")
    doc.append(
        "Per-condition summary. Note: agents typically **front-load OAT sweeps "
        "in iteration 0** and switch to exploitation afterward, so many iterations "
        "have `None` for OAT accuracy. All \"first/last\" comparisons use "
        "first-non-None and last-non-None values. Conditions with `n iters = 0` "
        "are old Phase 2.3 runs without structured iteration summaries (1B/1C "
        "transfer_72, 1C fresh_72)."
    )
    doc.append("")
    doc.append(
        "| Condition | n seeds | n iters (median) | OAT dir acc (fnn→lnn) | "
        "eval_point MAE (fnn→lnn) | Surprise (first→last) | seeds ↑ eval_MAE↓ | "
        "seeds ↓ surprise |"
    )
    doc.append(
        "|---|---|---|---|---|---|---|---|"
    )

    # Per-condition aggregation using first-non-None / last-non-None
    def _first_last_nn(vals: list) -> tuple[float | None, float | None]:
        """Return (first, last) non-None values, or (None, None) if none."""
        non_none = [v for v in vals if v is not None]
        if not non_none:
            return (None, None)
        return (non_none[0], non_none[-1])

    for cond_key, trajectories in section_data.items():
        if not trajectories:
            continue
        n_seeds = len(trajectories)
        n_iters_list = [t["n_iterations"] for t in trajectories]
        median_iters = int(np.median(n_iters_list))

        # OAT dir acc: first-non-None and last-non-None per seed, then mean
        oat_firsts: list[float] = []
        oat_lasts: list[float] = []
        for t in trajectories:
            fnn, lnn = _first_last_nn(t["per_iter_oat_dir_acc"])
            if fnn is not None:
                oat_firsts.append(fnn)
            if lnn is not None:
                oat_lasts.append(lnn)
        oat_s = (
            f"{float(np.mean(oat_firsts)):.2f} → {float(np.mean(oat_lasts)):.2f}"
            if oat_firsts and oat_lasts else "n/a"
        )

        # eval_point MAE: first-non-None / last-non-None per seed, then mean
        ep_firsts: list[float] = []
        ep_lasts: list[float] = []
        for t in trajectories:
            fnn, lnn = _first_last_nn(t["per_iter_eval_mae"])
            if fnn is not None:
                ep_firsts.append(fnn)
            if lnn is not None:
                ep_lasts.append(lnn)
        ep_s = (
            f"{float(np.mean(ep_firsts)):.3f} → {float(np.mean(ep_lasts)):.3f}"
            if ep_firsts and ep_lasts else "n/a"
        )

        # Surprise count: first vs last (no None filtering — surprises are always int)
        first_surp = [t["per_iter_surprise_count"][0] for t in trajectories
                      if t["per_iter_surprise_count"]]
        last_surp = [t["per_iter_surprise_count"][-1] for t in trajectories
                     if t["per_iter_surprise_count"]]
        surp_s = (
            f"{float(np.mean(first_surp)):.1f} → {float(np.mean(last_surp)):.1f}"
            if first_surp and last_surp else "n/a"
        )

        # Sign tests on eval_point MAE (should DROP over iterations)
        n_down_ep = 0
        n_counted_ep = 0
        for t in trajectories:
            fnn, lnn = _first_last_nn(t["per_iter_eval_mae"])
            if fnn is not None and lnn is not None and fnn != lnn:
                n_counted_ep += 1
                if lnn < fnn:
                    n_down_ep += 1
        ep_sign = f"{n_down_ep}/{n_counted_ep}" if n_counted_ep > 0 else "—"

        # Sign test on surprise count (should DROP)
        n_down_surp = 0
        for t in trajectories:
            vals = t["per_iter_surprise_count"]
            if len(vals) >= 2 and vals[-1] < vals[0]:
                n_down_surp += 1

        doc.append(
            f"| {cond_key} | {n_seeds} | {median_iters} | "
            f"{oat_s} | {ep_s} | {surp_s} | "
            f"{ep_sign} | {n_down_surp}/{n_seeds} |"
        )

    doc.append("")

    # Plot: 4-panel learning trajectory per seed for key conditions
    doc.heading(3, "A.2 Per-seed learning trajectories (plots)")
    doc.append(
        "4-panel trajectories for every seed of every condition. Panels: (a) OAT "
        "direction accuracy, (b) OAT magnitude MAE, (c) evaluate_point MAE, "
        "(d) surprise count. Saved to `holistic_plots/`."
    )
    doc.append("")

    plot_count = 0
    for oracle_label, conditions in all_runs.items():
        for cond_label, runs in conditions.items():
            if not runs:
                continue
            for run in runs:
                if run.n_iterations < 2:
                    continue
                plot_path = _plot_run_learning_trajectory(run, oracle_label, cond_label)
                if plot_path:
                    plot_count += 1

    doc.append(f"_Generated {plot_count} per-seed trajectory plots._")
    doc.append("")

    # A.4: Does improved prediction connect to better exploitation?
    doc.heading(3, "A.4 Does the learning loop connect to exploitation?")
    doc.append(
        "The key confound: predictions improve (eval_point MAE drops) but "
        "Section B showed within-condition prediction accuracy doesn't correlate "
        "with final HV. Is the learning loop disconnected from exploitation, "
        "or does it help in a way Section B couldn't detect?"
    )
    doc.append("")

    # Test 1: HV gain rate per iteration
    doc.heading(4, "Test 1: HV gain rate per iteration")
    doc.append(
        "If improved predictions feed into better exploitation, HV gain per eval "
        "should be **higher in later iterations** (where the model is better). "
        "Computed as (HV_end − HV_start) / n_evals for each iteration."
    )
    doc.append("")
    doc.append(
        "Note: diminishing returns near the Pareto front can also cause declining "
        "HV gain rate, so a decreasing rate is ambiguous. An INCREASING rate is "
        "strong evidence of exploitation improving."
    )
    doc.append("")

    # Collect HV gain rate per iteration for extended-budget runs
    hv_gain_data: dict[str, list[dict]] = {}

    for oracle_label, conditions in all_runs.items():
        for cond_label, runs in conditions.items():
            if not runs:
                continue
            cond_key = f"{oracle_label}/{cond_label}"
            seed_gains: list[dict] = []

            for run in runs:
                if run.n_iterations < 3:
                    continue  # need at least 3 iterations for a meaningful trajectory
                iters = run.iterations
                hv_traj = run.hv_trajectory
                n_init = run.n_initial

                gains_per_iter: list[dict] = []
                prev_eval = 0
                for it in iters:
                    eval_end = it.eval_count
                    n_evals_this_iter = eval_end - prev_eval
                    if n_evals_this_iter <= 0:
                        prev_eval = eval_end
                        continue
                    # HV at start and end of this iteration
                    hv_start = hv_traj[prev_eval - 1] if prev_eval > 0 else 0.0
                    hv_end = hv_traj[min(eval_end - 1, len(hv_traj) - 1)]
                    hv_gain = hv_end - hv_start
                    hv_gain_per_eval = hv_gain / n_evals_this_iter

                    gains_per_iter.append({
                        "iteration": it.iteration,
                        "eval_start": prev_eval,
                        "eval_end": eval_end,
                        "n_evals": n_evals_this_iter,
                        "hv_gain": round(hv_gain, 6),
                        "hv_gain_per_eval": round(hv_gain_per_eval, 6),
                    })
                    prev_eval = eval_end

                if gains_per_iter:
                    seed_gains.append({
                        "seed": run.seed,
                        "gains": gains_per_iter,
                    })

            if seed_gains:
                hv_gain_data[cond_key] = seed_gains

    # Report: for conditions with data, show whether gain rate increases or decreases
    doc.append("| Condition | Seed | Iter 0 gain/eval | Last iter gain/eval | Trend | Notes |")
    doc.append("|---|---|---|---|---|---|")

    for cond_key, seed_gains in hv_gain_data.items():
        for sg in seed_gains:
            gains = sg["gains"]
            if len(gains) < 2:
                continue
            first_gpe = gains[0]["hv_gain_per_eval"]
            last_gpe = gains[-1]["hv_gain_per_eval"]
            # Find the max gain/eval and which iteration it's in
            max_gpe = max(g["hv_gain_per_eval"] for g in gains)
            max_iter = next(g["iteration"] for g in gains
                           if g["hv_gain_per_eval"] == max_gpe)
            n_iters = len(gains)
            if last_gpe > first_gpe * 1.5:
                trend = "↑ INCREASING"
            elif last_gpe < first_gpe * 0.5:
                trend = "↓ decreasing"
            else:
                trend = "→ flat"
            doc.append(
                f"| {cond_key} | {sg['seed']} | {first_gpe:.5f} | "
                f"{last_gpe:.5f} | {trend} | max at iter {max_iter} "
                f"({max_gpe:.5f}), {n_iters} iters |"
            )
    doc.append("")

    # Test 2: Temporal coupling of prediction improvement and HV jumps
    doc.heading(4, "Test 2: Does prediction improvement precede HV breakthroughs?")
    doc.append(
        "For extended-budget runs, identify (a) the iteration where eval_point MAE "
        "first drops below 50% of its initial value ('prediction clicks'), and "
        "(b) the iteration with the largest single-iteration HV gain ('exploitation "
        "breakthrough'). If (a) precedes or coincides with (b), the learning loop "
        "connects to exploitation."
    )
    doc.append("")

    doc.append(
        "| Condition | Seed | Pred clicks (iter) | HV breakthrough (iter) | "
        "Clicks before breakthrough? |"
    )
    doc.append("|---|---|---|---|---|")

    temporal_coupling_data: list[dict] = []

    for cond_key, seed_gains in hv_gain_data.items():
        for sg in seed_gains:
            seed = sg["seed"]
            gains = sg["gains"]

            # Find the run metrics for this seed
            oracle_label = cond_key.split("/")[0]
            cond_label = cond_key.split("/")[1]
            run = None
            for r in all_runs.get(oracle_label, {}).get(cond_label, []):
                if r.seed == seed:
                    run = r
                    break
            if run is None or run.n_iterations < 3:
                continue

            # (a) Prediction clicks: first iteration where eval_point MAE < 50% of first
            ep_maes = [(it.iteration, it.eval_point_mae) for it in run.iterations
                       if it.eval_point_mae is not None]
            pred_clicks_iter = None
            if len(ep_maes) >= 2:
                first_mae = ep_maes[0][1]
                for it_idx, mae in ep_maes[1:]:
                    if mae < first_mae * 0.5:
                        pred_clicks_iter = it_idx
                        break

            # (b) HV breakthrough: iteration with largest HV gain
            if gains:
                best_gain = max(gains, key=lambda g: g["hv_gain"])
                hv_breakthrough_iter = best_gain["iteration"]
            else:
                hv_breakthrough_iter = None

            # Compare
            if pred_clicks_iter is not None and hv_breakthrough_iter is not None:
                if pred_clicks_iter <= hv_breakthrough_iter:
                    coupling = "YES — clicks first"
                else:
                    coupling = "NO — breakthrough first"
            elif pred_clicks_iter is None:
                coupling = "n/a — MAE never dropped 50%"
            else:
                coupling = "n/a"

            doc.append(
                f"| {cond_key} | {seed} | "
                f"{pred_clicks_iter if pred_clicks_iter is not None else 'n/a'} | "
                f"{hv_breakthrough_iter if hv_breakthrough_iter is not None else 'n/a'} | "
                f"{coupling} |"
            )
            temporal_coupling_data.append({
                "condition": cond_key,
                "seed": seed,
                "pred_clicks_iter": pred_clicks_iter,
                "hv_breakthrough_iter": hv_breakthrough_iter,
                "coupling": coupling,
            })

    doc.append("")

    # Test 3: Does OAT-discovered input importance guide evaluate_point targeting?
    doc.heading(4, "Test 3: Does the articulated model guide exploitation targets?")
    doc.append(
        "If the rough model from iter 0 OAT sweeps guides iter 1+ exploitation, "
        "then inputs identified as high-effect by OAT should be varied MORE in "
        "evaluate_point targets. Compute per-input: (a) OAT importance = max "
        "`actual_magnitude` across outputs from iter 0 sweeps, (b) exploitation "
        "emphasis = range of that input across iter 1+ evaluate_point X targets. "
        "Spearman correlation between (a) and (b) should be positive if model "
        "guides exploitation."
    )
    doc.append("")

    test3_data: list[dict[str, Any]] = []
    doc.append("| Condition | Seed | Spearman ρ (OAT importance vs exploit emphasis) | p-value | n inputs | Interpretation |")
    doc.append("|---|---|---|---|---|---|")

    for oracle_label, conditions in all_runs.items():
        for cond_label, runs in conditions.items():
            for run in runs:
                if run.n_iterations < 2:
                    continue

                # Load raw tool calls
                try:
                    log_name = _run_log_name(oracle_label, run.condition_label, run.seed)
                    log_path = ROOT / "experiments" / "vr_agent" / "results" / log_name
                    with open(log_path) as f:
                        log = json.load(f)
                except Exception:
                    continue

                tool_calls = log.get("tool_calls", []) or []
                n_init = run.n_initial
                oracle = get_oracle(oracle_label)
                n_inputs = oracle.n_inputs
                input_names = list(oracle.input_names)

                # Group tool calls by iteration
                tc_groups = group_tool_calls_by_iteration(log, n_init)
                if len(tc_groups) < 2:
                    continue

                # (a) OAT importance from iter 0: per-input max actual_magnitude
                oat_importance = np.zeros(n_inputs)
                for tc in tc_groups[0]:
                    if tc.get("name") != "oat_sweep":
                        continue
                    iname = tc.get("input", {}).get("input_name")
                    if iname not in input_names:
                        continue
                    idx = input_names.index(iname)
                    result = tc.get("result", {})
                    if not isinstance(result, dict):
                        continue
                    scores = result.get("trend_scores", {}) or {}
                    for _oname, s in scores.items():
                        if isinstance(s, dict) and "actual_magnitude" in s:
                            try:
                                mag = float(s["actual_magnitude"])
                                oat_importance[idx] = max(oat_importance[idx], mag)
                            except (TypeError, ValueError):
                                pass

                if oat_importance.sum() < 1e-10:
                    continue  # no OAT data in iter 0

                # (b) Exploitation emphasis from iter 1+: per-input range across
                #     evaluate_point targets
                eval_points_x: list[list[float]] = []
                for grp in tc_groups[1:]:
                    for tc in grp:
                        if tc.get("name") != "evaluate_point":
                            continue
                        pt = tc.get("input", {}).get("point", [])
                        if len(pt) == n_inputs:
                            eval_points_x.append([float(v) for v in pt])

                if len(eval_points_x) < 2:
                    continue  # need at least 2 evaluate_points for range

                eval_arr = np.array(eval_points_x)
                exploit_emphasis = eval_arr.max(axis=0) - eval_arr.min(axis=0)

                # Spearman correlation
                from scipy.stats import spearmanr
                rho, p_val = spearmanr(oat_importance, exploit_emphasis)

                cond_key = f"{oracle_label}/{cond_label}"
                if p_val < 0.1 and rho > 0.3:
                    interp = "model guides exploitation"
                elif rho < -0.1:
                    interp = "inverse — exploits LOW-effect inputs?"
                else:
                    interp = "no clear connection"

                doc.append(
                    f"| {cond_key} | {run.seed} | {rho:+.3f} | {p_val:.3f} | "
                    f"{n_inputs} | {interp} |"
                )
                test3_data.append({
                    "condition": cond_key, "seed": run.seed,
                    "rho": float(rho), "p": float(p_val),
                    "oat_importance": oat_importance.tolist(),
                    "exploit_emphasis": exploit_emphasis.tolist(),
                    "n_eval_points": len(eval_points_x),
                })

    doc.append("")

    # Test 3 summary
    if test3_data:
        pos_sig = sum(1 for d in test3_data if d["rho"] > 0.3 and d["p"] < 0.1)
        neg = sum(1 for d in test3_data if d["rho"] < -0.1)
        total = len(test3_data)
        doc.append(
            f"**Summary:** {pos_sig}/{total} seeds show significant positive "
            f"correlation (ρ > 0.3, p < 0.1) between OAT-discovered importance "
            f"and exploitation emphasis. {neg}/{total} show inverse pattern."
        )
    doc.append("")

    # Test 4: Are evaluate_point targets better than LHS on optimization objectives?
    doc.heading(4, "Test 4: Are exploitation targets better than random (LHS)?")
    doc.append(
        "Compare the mean output values of iter 1+ evaluate_point targets "
        "to the mean output values of initial LHS points, on the maximize "
        "objectives (Y1 and Y4). If evaluate_point targets are systematically "
        "better, the agent is directing exploitation to promising regions "
        "(regardless of whether it's model-guided or just following gradients)."
    )
    doc.append("")

    test4_data: list[dict[str, Any]] = []
    doc.append("| Condition | Seed | LHS mean(Y1) | Exploit mean(Y1) | LHS mean(Y4) | Exploit mean(Y4) | Y1 better? | Y4 better? |")
    doc.append("|---|---|---|---|---|---|---|---|")

    for oracle_label, conditions in all_runs.items():
        for cond_label, runs in conditions.items():
            for run in runs:
                if run.n_iterations < 2:
                    continue

                oracle = get_oracle(oracle_label)
                n_init = run.n_initial

                # Load raw data
                try:
                    log_name = _run_log_name(oracle_label, run.condition_label, run.seed)
                    log_path = ROOT / "experiments" / "vr_agent" / "results" / log_name
                    with open(log_path) as f:
                        log_data = json.load(f)
                    npz_name = log_name.replace("_log.json", ".npz")
                    npz_path = ROOT / "experiments" / "vr_agent" / "results" / npz_name
                    npz_data = dict(np.load(npz_path))
                except Exception:
                    continue

                Y_all = npz_data.get("Y")
                if Y_all is None or len(Y_all) < n_init + 2:
                    continue

                # LHS Y values (first n_initial rows)
                Y_lhs = Y_all[:n_init]

                # Evaluate_point Y values (from tool calls in iter 1+)
                tc_groups = group_tool_calls_by_iteration(log_data, n_init)
                if len(tc_groups) < 2:
                    continue

                eval_ys: list[list[float]] = []
                for grp in tc_groups[1:]:
                    for tc in grp:
                        if tc.get("name") != "evaluate_point":
                            continue
                        result = tc.get("result", {})
                        if not isinstance(result, dict):
                            continue
                        outputs = result.get("outputs", {})
                        if outputs and len(outputs) == oracle.n_outputs:
                            eval_ys.append([float(outputs.get(o, 0))
                                           for o in oracle.output_names])

                if len(eval_ys) < 2:
                    continue

                eval_Y = np.array(eval_ys)

                # Compare Y1 (maximize, index 0) and Y4 (maximize, index 3)
                lhs_y1 = float(np.mean(Y_lhs[:, 0]))
                lhs_y4 = float(np.mean(Y_lhs[:, 3]))
                exp_y1 = float(np.mean(eval_Y[:, 0]))
                exp_y4 = float(np.mean(eval_Y[:, 3]))

                y1_better = "YES" if exp_y1 > lhs_y1 else "no"
                y4_better = "YES" if exp_y4 > lhs_y4 else "no"

                cond_key = f"{oracle_label}/{cond_label}"
                doc.append(
                    f"| {cond_key} | {run.seed} | {lhs_y1:.3f} | {exp_y1:.3f} | "
                    f"{lhs_y4:.3f} | {exp_y4:.3f} | {y1_better} | {y4_better} |"
                )
                test4_data.append({
                    "condition": cond_key, "seed": run.seed,
                    "lhs_y1": lhs_y1, "exploit_y1": exp_y1,
                    "lhs_y4": lhs_y4, "exploit_y4": exp_y4,
                    "y1_better": exp_y1 > lhs_y1,
                    "y4_better": exp_y4 > lhs_y4,
                })

    doc.append("")

    # Test 4 summary
    if test4_data:
        n_y1_better = sum(1 for d in test4_data if d["y1_better"])
        n_y4_better = sum(1 for d in test4_data if d["y4_better"])
        total = len(test4_data)
        doc.append(
            f"**Summary:** {n_y1_better}/{total} seeds have exploitation targets "
            f"with higher mean Y1 than LHS. {n_y4_better}/{total} for Y4. "
            f"(Both are maximize objectives — higher = better.)"
        )
    doc.append("")

    # Summary of temporal coupling
    n_clicks_first = sum(1 for d in temporal_coupling_data
                         if "clicks first" in d["coupling"])
    n_breakthrough_first = sum(1 for d in temporal_coupling_data
                               if "breakthrough first" in d["coupling"])
    n_total = n_clicks_first + n_breakthrough_first
    if n_total > 0:
        doc.append(
            f"**Summary:** {n_clicks_first}/{n_total} seeds show prediction "
            f"improvement preceding or coinciding with the HV breakthrough. "
            f"{n_breakthrough_first}/{n_total} show the breakthrough happening "
            f"before prediction clicks."
        )
    else:
        doc.append("**Summary:** insufficient data for temporal coupling analysis.")
    doc.append("")

    # Falsification verdicts
    doc.heading(3, "A.5 Falsification verdicts (updated with A.4)")
    verdicts = _section_a_verdicts(section_data)

    # Add A.4 verdicts
    if n_total > 0:
        frac = n_clicks_first / n_total
        if frac >= 0.6:
            verdicts.append({
                "claim": "Prediction improvement precedes HV breakthroughs (learning → exploitation coupling)",
                "verdict": "PASS",
                "evidence": f"{n_clicks_first}/{n_total} seeds ({frac:.0%}) show prediction clicks before or at HV breakthrough.",
            })
        elif frac <= 0.3:
            verdicts.append({
                "claim": "Prediction improvement precedes HV breakthroughs (learning → exploitation coupling)",
                "verdict": "FALSIFIED",
                "evidence": f"Only {n_clicks_first}/{n_total} seeds ({frac:.0%}) show coupling — breakthroughs happen independent of prediction improvement.",
            })
        else:
            verdicts.append({
                "claim": "Prediction improvement precedes HV breakthroughs (learning → exploitation coupling)",
                "verdict": "QUALIFIED",
                "evidence": f"{n_clicks_first}/{n_total} seeds ({frac:.0%}) — mixed signal.",
            })

    # Add Test 3 verdict
    if test3_data:
        pos_sig = sum(1 for d in test3_data if d["rho"] > 0.3 and d["p"] < 0.1)
        total_t3 = len(test3_data)
        frac_t3 = pos_sig / total_t3 if total_t3 > 0 else 0
        if frac_t3 >= 0.5:
            verdicts.append({
                "claim": "OAT-discovered input importance guides exploitation targets (model → exploitation causal link)",
                "verdict": "PASS",
                "evidence": f"{pos_sig}/{total_t3} seeds ({frac_t3:.0%}) show significant positive Spearman ρ (> 0.3, p < 0.1) between OAT importance and exploitation emphasis.",
            })
        elif frac_t3 <= 0.1:
            verdicts.append({
                "claim": "OAT-discovered input importance guides exploitation targets (model → exploitation causal link)",
                "verdict": "FALSIFIED",
                "evidence": f"Only {pos_sig}/{total_t3} seeds ({frac_t3:.0%}) show the expected correlation.",
            })
        else:
            verdicts.append({
                "claim": "OAT-discovered input importance guides exploitation targets (model → exploitation causal link)",
                "verdict": "QUALIFIED",
                "evidence": f"{pos_sig}/{total_t3} seeds ({frac_t3:.0%}) show the expected correlation — mixed.",
            })

    # Add Test 4 verdict
    if test4_data:
        n_both_better = sum(1 for d in test4_data if d["y1_better"] and d["y4_better"])
        n_either_better = sum(1 for d in test4_data if d["y1_better"] or d["y4_better"])
        total_t4 = len(test4_data)
        frac_t4 = n_either_better / total_t4 if total_t4 > 0 else 0
        if frac_t4 >= 0.7:
            verdicts.append({
                "claim": "Exploitation targets are better than random (LHS) on maximize objectives",
                "verdict": "PASS",
                "evidence": f"{n_either_better}/{total_t4} seeds ({frac_t4:.0%}) have exploitation targets with higher mean Y1 or Y4 than LHS. {n_both_better}/{total_t4} are better on BOTH.",
            })
        else:
            verdicts.append({
                "claim": "Exploitation targets are better than random (LHS) on maximize objectives",
                "verdict": "QUALIFIED",
                "evidence": f"{n_either_better}/{total_t4} seeds ({frac_t4:.0%}) are better on at least one objective. {n_both_better}/{total_t4} on both.",
            })

    for v in verdicts:
        doc.append(f"- **{v['claim']}**: **{v['verdict']}** — {v['evidence']}")
    doc.append("")

    doc.save_json("section_a", {
        "trajectories": section_data,
        "hv_gain_data": hv_gain_data,
        "temporal_coupling": temporal_coupling_data,
        "test3_data": test3_data,
        "test4_data": test4_data,
        "verdicts": verdicts,
    })


def _plot_run_learning_trajectory(
    run: RunMetrics, oracle_label: str, cond_label: str,
) -> Path | None:
    """Save a 4-panel plot of the run's per-iteration metrics.

    Design choices for honest visualization:
    - X-axis forced to integer ticks (iterations are discrete events)
    - Y-axis forced to integer ticks for surprise count
    - Markers at every iteration; line segments ONLY between consecutive
      non-None values (gaps are visible when an iteration has no OAT/eval_point)
    - X-axis padded to show full iteration range even if data is sparse
    """
    from matplotlib.ticker import MaxNLocator

    iters = run.iterations
    if len(iters) < 2:
        return None

    iter_idx = [it.iteration for it in iters]
    n_iters = max(iter_idx) + 1 if iter_idx else 1

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    def _plot_safe(ax, x, y, title, ylabel, ylim=None, integer_y=False):
        y_clean = [v if v is not None else np.nan for v in y]
        # Plot markers at all iterations (including NaN → they won't render)
        ax.plot(x, y_clean, "o", markersize=6, color="#1f77b4", zorder=3)
        # Draw line segments only between consecutive non-NaN points
        for i in range(len(x) - 1):
            if not (np.isnan(y_clean[i]) or np.isnan(y_clean[i + 1])):
                ax.plot(
                    [x[i], x[i + 1]], [y_clean[i], y_clean[i + 1]],
                    "-", linewidth=1.5, color="#1f77b4", zorder=2,
                )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        # Force integer ticks on x-axis and pad range
        ax.set_xlim(-0.5, n_iters - 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if integer_y:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

        # Annotate how many data points are non-NaN
        n_valid = sum(1 for v in y_clean if not np.isnan(v))
        ax.text(
            0.98, 0.02, f"n={n_valid}/{len(y_clean)}",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color="gray",
        )

    _plot_safe(
        axes[0, 0], iter_idx,
        [it.oat_dir_accuracy for it in iters],
        "OAT direction accuracy", "fraction correct",
        ylim=(-0.05, 1.05),
    )
    _plot_safe(
        axes[0, 1], iter_idx,
        [it.oat_mag_mae for it in iters],
        "OAT magnitude MAE", "MAE",
    )
    _plot_safe(
        axes[1, 0], iter_idx,
        [it.eval_point_mae for it in iters],
        "evaluate_point MAE", "MAE",
    )
    _plot_safe(
        axes[1, 1], iter_idx,
        [float(it.n_surprises) for it in iters],
        "surprise count", "count",
        integer_y=True,
    )

    fig.suptitle(
        f"{oracle_label} / {cond_label} / seed {run.seed} "
        f"({run.model}, {run.n_iterations} iters, final HV={run.final_hv:.3f})",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = PLOTS_DIR / (
        f"A_{oracle_label}_{cond_label}_seed{run.seed}_learning.png"
    )
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _section_a_verdicts(section_data: dict[str, Any]) -> list[dict[str, str]]:
    """Compute falsification verdicts for Section A claims.

    Key insight from inspection: agents front-load OAT sweeps in iteration 0
    and switch to evaluate_point calls later. So "last iteration OAT accuracy"
    is often None, making OAT a weak learning signal. The primary learning
    signal is evaluate_point MAE, which is computed from per-point predictions
    the agent makes during exploitation — this spans the run more fully.
    """
    verdicts = []

    # Pool all trajectories
    all_down_ep = 0  # eval_point MAE should DROP (smaller = better prediction)
    all_ep_reduction_large = 0  # count seeds where last < 0.5 * first (big improvement)
    total_seeds_ep = 0

    all_up_oat = 0
    total_seeds_oat = 0

    all_down_surp = 0
    total_seeds_surp = 0

    for cond_key, trajectories in section_data.items():
        for t in trajectories:
            # eval_point MAE trajectory (first-non-None vs last-non-None)
            ep_vals = [v for v in t["per_iter_eval_mae"] if v is not None]
            if len(ep_vals) >= 2:
                total_seeds_ep += 1
                if ep_vals[-1] < ep_vals[0]:
                    all_down_ep += 1
                if ep_vals[-1] < 0.5 * ep_vals[0]:
                    all_ep_reduction_large += 1

            # OAT direction accuracy trajectory (first-non-None vs last-non-None)
            oat_vals = [v for v in t["per_iter_oat_dir_acc"] if v is not None]
            if len(oat_vals) >= 2:
                total_seeds_oat += 1
                if oat_vals[-1] > oat_vals[0]:
                    all_up_oat += 1

            # Surprise count trajectory (no None filtering)
            surp_vals = t["per_iter_surprise_count"]
            if len(surp_vals) >= 2:
                total_seeds_surp += 1
                if surp_vals[-1] < surp_vals[0]:
                    all_down_surp += 1

    def _classify(num: int, den: int, label_pass: str, label_fail: str) -> tuple[str, str]:
        if den == 0:
            return ("INCONCLUSIVE", "no seeds with sufficient data")
        frac = num / den
        if frac >= 0.7:
            return ("PASS", f"{num}/{den} seeds ({frac:.0%}) {label_pass}")
        if frac <= 0.3:
            return ("FALSIFIED", f"only {num}/{den} seeds ({frac:.0%}) {label_pass}")
        return ("QUALIFIED", f"{num}/{den} seeds ({frac:.0%}) {label_pass} — mixed")

    # Primary claim: evaluate_point MAE decreases over iterations
    v, ev = _classify(
        all_down_ep, total_seeds_ep,
        "show evaluate_point MAE lower in last iteration with data than first",
        "show evaluate_point MAE lower",
    )
    verdicts.append({
        "claim": "VR learns from feedback (evaluate_point MAE ↓, primary signal)",
        "verdict": v,
        "evidence": (
            f"{ev}. Additionally, {all_ep_reduction_large}/{total_seeds_ep} seeds "
            f"show a >50% MAE reduction (first iter to last iter)."
        ),
    })

    # Secondary claim: OAT direction accuracy improves (weaker because OATs are front-loaded)
    v, ev = _classify(
        all_up_oat, total_seeds_oat,
        "show OAT direction accuracy higher in last non-None iter than first",
        "show OAT direction accuracy improvement",
    )
    verdicts.append({
        "claim": "OAT direction accuracy improves over iterations (secondary; OATs are front-loaded)",
        "verdict": v,
        "evidence": ev + ". Note: many seeds only have 1 iteration with OAT sweeps, so this metric is noisier than eval_point MAE.",
    })

    # Tertiary claim: surprise count decreases
    v, ev = _classify(
        all_down_surp, total_seeds_surp,
        "show surprise count lower in last iteration than first",
        "show surprise count decrease",
    )
    verdicts.append({
        "claim": "Surprise rate decreases over iterations",
        "verdict": v,
        "evidence": ev,
    })

    return verdicts


# ---------------------------------------------------------------------------
# Section B: cross-seed dynamics correlations
# ---------------------------------------------------------------------------


def section_b_cross_seed_correlations(
    all_runs: dict[str, dict[str, list[RunMetrics]]],
    doc: FindingsDoc,
) -> None:
    """Section B: do within-run dynamics predict final outcomes?

    For each condition with n>=3, compute per-seed metrics and test whether
    late-iteration prediction accuracy, calibration learning, or surprise
    decrease rate correlate with final HV.
    """
    from scipy import stats as scipy_stats

    doc.section("B", "Cross-seed dynamics correlations")
    doc.append(
        "**Question:** Do per-seed dynamics metrics correlate with per-seed "
        "final HV within the same condition? If calibration learning predicts "
        "better HV, that's evidence the confidence-tracking mechanism works. "
        "If final recall correlates with final HV, that's evidence discovery "
        "helps exploitation (or at least that they track together)."
    )
    doc.append("")
    doc.append(
        "**Caveat:** With n=3 per condition, correlations are extremely noisy. "
        "n=10 (1A multi_seed) is the only condition with real statistical power. "
        "For n=3 conditions, individual r values should be treated as directional "
        "only. We pool across conditions where possible for more robust signals."
    )
    doc.append("")

    section_data: dict[str, Any] = {}

    # Collect per-seed metrics for all conditions with n>=3
    all_rows: list[dict[str, Any]] = []  # pooled across conditions for aggregate analysis

    doc.heading(3, "B.1 Per-condition summary")
    doc.append(
        "| Condition | n | r(HV, late_eval_MAE) | r(HV, final_recall) | "
        "r(HV, surprise_decrease) | r(HV, n_iters) |"
    )
    doc.append("|---|---|---|---|---|---|")

    for oracle_label, conditions in all_runs.items():
        for cond_label, runs in conditions.items():
            if len(runs) < 3:
                continue
            cond_key = f"{oracle_label}/{cond_label}"
            gt_n = len(ground_truth_io_pairs(oracle_label))

            seed_metrics: list[dict[str, Any]] = []
            for run in runs:
                iters = run.iterations
                if not iters:
                    continue

                # Late-iter eval_point MAE (last non-None)
                ep_vals = [it.eval_point_mae for it in iters if it.eval_point_mae is not None]
                late_ep_mae = ep_vals[-1] if ep_vals else None

                # Final recall at high confidence
                last_iter = iters[-1]
                final_recall = last_iter.recall_at_high_conf

                # Surprise decrease: (last surp - first surp)
                surp_first = iters[0].n_surprises
                surp_last = iters[-1].n_surprises
                surp_decrease = surp_first - surp_last  # positive = fewer surprises at end

                row = {
                    "seed": run.seed,
                    "final_hv": run.final_hv,
                    "late_ep_mae": late_ep_mae,
                    "final_recall": final_recall,
                    "surp_decrease": surp_decrease,
                    "n_iters": run.n_iterations,
                    "condition": cond_key,
                    "oracle": oracle_label,
                }
                seed_metrics.append(row)
                all_rows.append(row)

            section_data[cond_key] = seed_metrics

            # Compute correlations where we have enough non-None values
            def _corr(metric_key: str) -> str:
                vals = [(m["final_hv"], m[metric_key]) for m in seed_metrics
                        if m[metric_key] is not None]
                if len(vals) < 3:
                    return "n/a"
                x = [v[0] for v in vals]
                y = [v[1] for v in vals]
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    return "const"
                r, p = scipy_stats.pearsonr(x, y)
                return f"{r:+.2f} (p={p:.2f})"

            doc.append(
                f"| {cond_key} | {len(seed_metrics)} | "
                f"{_corr('late_ep_mae')} | {_corr('final_recall')} | "
                f"{_corr('surp_decrease')} | {_corr('n_iters')} |"
            )

    doc.append("")

    # Pooled analysis across ALL n>=3 conditions
    doc.heading(3, "B.2 Pooled cross-condition analysis")
    doc.append(
        "Pool all seeds from n>=3 conditions. **Warning:** pooling across "
        "conditions mixes different oracles and budgets, so correlation "
        "structure may be driven by between-condition differences rather "
        "than within-condition variation. Report for completeness but "
        "interpret cautiously."
    )
    doc.append("")

    if len(all_rows) >= 5:
        from scipy.stats import pearsonr, spearmanr

        def _pooled_corr(key: str, label: str) -> str:
            """Compute pooled Pearson + Spearman for a metric vs final_hv."""
            pairs = [(r["final_hv"], r[key]) for r in all_rows
                     if r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
            if len(pairs) < 5:
                return f"  {label}: insufficient data (n={len(pairs)})"
            x = [p[0] for p in pairs]
            y = [p[1] for p in pairs]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                return f"  {label}: constant values"
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
            return (
                f"  {label} (n={len(pairs)}): "
                f"Pearson r={r_p:+.3f} (p={p_p:.3f}), "
                f"Spearman ρ={r_s:+.3f} (p={p_s:.3f})"
            )

        doc.append("```")
        doc.append(_pooled_corr("late_ep_mae", "final_hv vs late_eval_MAE"))
        doc.append(_pooled_corr("final_recall", "final_hv vs final_recall"))
        doc.append(_pooled_corr("surp_decrease", "final_hv vs surprise_decrease"))
        doc.append(_pooled_corr("n_iters", "final_hv vs n_iterations"))
        doc.append("```")
        doc.append("")

        doc.append(
            "**Expected signs:** HV vs late_eval_MAE should be **negative** (better "
            "predictions → higher HV). HV vs final_recall should be **positive** "
            "(more discovered edges → better HV). HV vs surprise_decrease should be "
            "**positive** (more learning → higher HV). HV vs n_iters is ambiguous "
            "(more iters could mean more learning OR more failed attempts)."
        )
        doc.append("")

    # B.3 Verdicts
    doc.heading(3, "B.3 Falsification verdicts")

    verdicts = []

    # Check the most important correlation: HV vs late_eval_MAE
    ep_pairs = [(r["final_hv"], r["late_ep_mae"]) for r in all_rows
                if r["late_ep_mae"] is not None]
    if len(ep_pairs) >= 5:
        from scipy.stats import pearsonr
        x = [p[0] for p in ep_pairs]
        y = [p[1] for p in ep_pairs]
        r, p = pearsonr(x, y)
        if r < -0.2 and p < 0.1:
            verdicts.append({
                "claim": "Better prediction accuracy predicts higher HV",
                "verdict": "PASS",
                "evidence": f"Pooled Pearson r={r:.3f}, p={p:.3f}, n={len(ep_pairs)}. "
                           "Negative correlation as expected (lower MAE → higher HV).",
            })
        elif abs(r) < 0.2:
            verdicts.append({
                "claim": "Better prediction accuracy predicts higher HV",
                "verdict": "INCONCLUSIVE",
                "evidence": f"Pooled Pearson r={r:.3f}, p={p:.3f}, n={len(ep_pairs)}. "
                           "Weak or near-zero correlation — may be driven by within-condition noise "
                           "or between-condition mixing.",
            })
        else:
            verdicts.append({
                "claim": "Better prediction accuracy predicts higher HV",
                "verdict": "QUALIFIED",
                "evidence": f"Pooled Pearson r={r:.3f}, p={p:.3f}, n={len(ep_pairs)}.",
            })

    # Check recall vs HV
    recall_pairs = [(r["final_hv"], r["final_recall"]) for r in all_rows
                    if r["final_recall"] is not None]
    if len(recall_pairs) >= 5:
        from scipy.stats import pearsonr
        x = [p[0] for p in recall_pairs]
        y = [p[1] for p in recall_pairs]
        r, p = pearsonr(x, y)
        if r > 0.2 and p < 0.1:
            v = "PASS"
        elif abs(r) < 0.2:
            v = "INCONCLUSIVE"
        else:
            v = "QUALIFIED"
        verdicts.append({
            "claim": "Discovery (final recall) predicts exploitation (final HV)",
            "verdict": v,
            "evidence": f"Pooled Pearson r={r:.3f}, p={p:.3f}, n={len(recall_pairs)}.",
        })

    for vd in verdicts:
        doc.append(f"- **{vd['claim']}**: **{vd['verdict']}** — {vd['evidence']}")
    doc.append("")

    doc.save_json("section_b", {
        "per_condition": section_data,
        "pooled_n": len(all_rows),
        "verdicts": verdicts,
    })


# ---------------------------------------------------------------------------
# Section C: prior vs fresh — systematic, not just seed 42
# ---------------------------------------------------------------------------


def section_c_prior_vs_fresh(
    all_runs: dict[str, dict[str, list[RunMetrics]]],
    doc: FindingsDoc,
) -> None:
    """Section C: is the 'prior helps' effect selective-leverage, behavioral priming, or noise?

    Systematically compares prior vs fresh on 1D and 1E (n=3 each) across:
    1. Tool call type distributions (all seeds, not just seed 42)
    2. Which inputs the agent probes first (OAT ordering)
    3. Hypothesis text at iteration 0 (does prior appear?)
    4. Edge confidence at iteration 0 for GT-aligned vs variant-specific edges
    """
    doc.section("C", "Prior vs fresh: systematic, not just seed 42")
    doc.append(
        "**Question:** Is the 'prior helps' effect (1D +2%, 1E +5% at n=3) "
        "due to (a) selective leverage of correct causal claims in the prior, "
        "(b) behavioral priming that changes tool-call allocation, or "
        "(c) something else? The seed 42 spot-check (Section A of the phase 2.7 "
        "analysis) showed prior → 12 local_gradients vs fresh → 3. Is this "
        "consistent across seeds 43 and 44?"
    )
    doc.append("")

    section_data: dict[str, Any] = {}

    # Process each transfer oracle
    for oracle_label in ["1D", "1E"]:
        prior_runs = all_runs.get(oracle_label, {}).get("prior_72", [])
        fresh_runs = all_runs.get(oracle_label, {}).get("fresh_72", [])

        if not prior_runs or not fresh_runs:
            doc.append(f"_{oracle_label}: missing prior or fresh runs._")
            doc.append("")
            continue

        gt_pairs = ground_truth_io_pairs(oracle_label)
        gt_1a_pairs = ground_truth_io_pairs("1A")

        oracle_data: dict[str, Any] = {
            "prior_seeds": [], "fresh_seeds": [],
            "gt_pairs": sorted(f"{s}->{t}" for s, t in gt_pairs),
            "gt_1a_pairs": sorted(f"{s}->{t}" for s, t in gt_1a_pairs),
        }

        doc.heading(3, f"C.{oracle_label} — {oracle_label} prior vs fresh (n=3 each)")

        # C.X.1: Tool call type distribution per seed
        doc.heading(4, f"C.{oracle_label}.1 Tool call type distribution")
        doc.append(
            "Per-seed tool call counts for prior vs fresh. The seed 42 finding was: "
            "1D prior had 12 `local_gradients` vs fresh's 3. Is this consistent?"
        )
        doc.append("")

        # Collect all tool types across both conditions
        all_tool_types: set[str] = set()
        for run in prior_runs + fresh_runs:
            all_tool_types.update(run.tool_call_counts_total.keys())
        tool_types_sorted = sorted(all_tool_types)

        # Table header
        header = f"| Tool type | " + " | ".join(
            f"P s{r.seed}" for r in sorted(prior_runs, key=lambda x: x.seed)
        ) + " | " + " | ".join(
            f"F s{r.seed}" for r in sorted(fresh_runs, key=lambda x: x.seed)
        ) + " | P mean | F mean |"
        sep = "|---" * (len(prior_runs) + len(fresh_runs) + 3) + "|"
        doc.append(header)
        doc.append(sep)

        prior_sorted = sorted(prior_runs, key=lambda x: x.seed)
        fresh_sorted = sorted(fresh_runs, key=lambda x: x.seed)

        tc_comparison: dict[str, dict[str, list[int]]] = {}
        for tt in tool_types_sorted:
            p_vals = [r.tool_call_counts_total.get(tt, 0) for r in prior_sorted]
            f_vals = [r.tool_call_counts_total.get(tt, 0) for r in fresh_sorted]
            tc_comparison[tt] = {"prior": p_vals, "fresh": f_vals}
            p_cells = " | ".join(str(v) for v in p_vals)
            f_cells = " | ".join(str(v) for v in f_vals)
            p_mean = float(np.mean(p_vals))
            f_mean = float(np.mean(f_vals))
            # Bold if means differ by >50% of the larger
            max_mean = max(p_mean, f_mean, 1)
            if abs(p_mean - f_mean) / max_mean > 0.3 and max_mean > 1:
                doc.append(f"| **{tt}** | {p_cells} | {f_cells} | "
                          f"**{p_mean:.1f}** | **{f_mean:.1f}** |")
            else:
                doc.append(f"| {tt} | {p_cells} | {f_cells} | "
                          f"{p_mean:.1f} | {f_mean:.1f} |")

        doc.append("")
        oracle_data["tool_call_comparison"] = tc_comparison

        # C.X.2: First OAT targets
        doc.heading(4, f"C.{oracle_label}.2 First OAT sweep targets (input ordering)")
        doc.append(
            "The order in which the agent probes inputs via OAT sweeps reveals "
            "its exploration strategy. Does prior change which inputs are probed "
            "first?"
        )
        doc.append("")

        def _load_log_only(oracle_lbl: str, cond_lbl: str, seed: int) -> dict:
            """Load just the JSON log file (no npz needed)."""
            log_name = _run_log_name(oracle_lbl, cond_lbl, seed)
            log_path = ROOT / "experiments" / "vr_agent" / "results" / log_name
            with open(log_path) as f:
                return json.load(f)

        def _first_n_oat_targets(run: RunMetrics, n: int = 8) -> list[str]:
            """Extract first N OAT sweep input targets from the tool call sequence."""
            log = _load_log_only(oracle_label, run.condition_label, run.seed)
            targets = []
            for tc in log.get("tool_calls", []):
                if tc.get("name") == "oat_sweep":
                    targets.append(tc.get("input", {}).get("input_name", "?"))
                    if len(targets) >= n:
                        break
            return targets

        for condition_name, runs in [("prior", prior_sorted), ("fresh", fresh_sorted)]:
            for run in runs:
                try:
                    targets = _first_n_oat_targets(run, n=8)
                    doc.append(f"- **{condition_name} seed {run.seed}**: {' → '.join(targets)}")
                except Exception as e:
                    doc.append(f"- **{condition_name} seed {run.seed}**: (load error: {e})")
        doc.append("")

        # C.X.3: Hypothesis text at iteration 0
        doc.heading(4, f"C.{oracle_label}.3 Hypothesis text at iteration 0 (prior references)")
        doc.append(
            "Does the prior-equipped agent's initial hypothesis explicitly reference "
            "the prior? Searching for keywords: 'prior', 'related system', 'previous', "
            "'from the'."
        )
        doc.append("")

        prior_keywords = ["prior", "related system", "previous study", "from the",
                          "earlier", "causal model"]
        for condition_name, runs in [("prior", prior_sorted), ("fresh", fresh_sorted)]:
            for run in runs:
                if not run.iterations:
                    doc.append(f"- **{condition_name} seed {run.seed}**: no iteration summaries")
                    continue
                hyp0 = run.iterations[0]
                # We need the raw hypothesis text — it's in iteration_summaries
                try:
                    log = _load_log_only(oracle_label, run.condition_label, run.seed)
                    iters = log.get("iteration_summaries", []) or []
                    if iters:
                        hyp_text = str(iters[0].get("hypothesis", ""))
                        hyp_lower = hyp_text.lower()
                        matches = [kw for kw in prior_keywords if kw in hyp_lower]
                        mention = f"mentions: {matches}" if matches else "no prior keywords"
                        snippet = hyp_text[:150].replace("\n", " ")
                        doc.append(f"- **{condition_name} seed {run.seed}**: "
                                  f"{mention}. Snippet: \"{snippet}...\"")
                    else:
                        doc.append(f"- **{condition_name} seed {run.seed}**: no iterations")
                except Exception as e:
                    doc.append(f"- **{condition_name} seed {run.seed}**: error: {e}")
        doc.append("")

        # C.X.4: Edge confidence at iteration 0 — GT-aligned vs variant-specific
        doc.heading(4, f"C.{oracle_label}.4 Edge confidence initialization: GT-aligned vs variant-specific")
        doc.append(
            "Does the prior-equipped agent start with higher confidence on edges "
            "that are correct (shared with 1A GT) and lower confidence on edges "
            "that are wrong (1A-specific, not in this variant's GT)? This is the "
            "direct test of 'screen-first selectively leverages correct priors.'"
        )
        doc.append("")

        # Classify edges
        shared_edges = gt_pairs & gt_1a_pairs  # present in both 1A and variant
        wrong_from_1a = gt_1a_pairs - gt_pairs  # in 1A but not variant → prior should dismiss
        missing_from_1a = gt_pairs - gt_1a_pairs  # in variant but not 1A → prior doesn't know

        doc.append(f"Edge classification for {oracle_label}:")
        doc.append(f"- Shared with 1A (prior should help): {len(shared_edges)}")
        doc.append(f"- Wrong from 1A (prior should dismiss): {len(wrong_from_1a)}")
        doc.append(f"- Missing from 1A (prior doesn't know): {len(missing_from_1a)}")
        doc.append("")

        doc.append("| Condition | Seed | Mean conf (shared) | Mean conf (wrong from 1A) | "
                   "Mean conf (missing from 1A) |")
        doc.append("|---|---|---|---|---|")

        for condition_name, runs in [("prior", prior_sorted), ("fresh", fresh_sorted)]:
            for run in runs:
                if not run.iterations:
                    doc.append(f"| {condition_name} | {run.seed} | n/a | n/a | n/a |")
                    continue
                try:
                    log = _load_log_only(oracle_label, run.condition_label, run.seed)
                    iters = log.get("iteration_summaries", []) or []
                    if not iters:
                        doc.append(f"| {condition_name} | {run.seed} | n/a | n/a | n/a |")
                        continue

                    conf = iters[0].get("confidence", {}) or {}
                    # Parse edge strings and classify
                    shared_confs: list[float] = []
                    wrong_confs: list[float] = []
                    missing_confs: list[float] = []
                    for edge_str, c in conf.items():
                        if not isinstance(c, (int, float)):
                            continue
                        parsed = _parse_edge_str(str(edge_str))
                        if parsed is None:
                            continue
                        if parsed in shared_edges:
                            shared_confs.append(float(c))
                        elif parsed in wrong_from_1a:
                            wrong_confs.append(float(c))
                        elif parsed in missing_from_1a:
                            missing_confs.append(float(c))

                    def _fmt(vals: list[float]) -> str:
                        if not vals:
                            return "—"
                        return f"{float(np.mean(vals)):.2f} (n={len(vals)})"

                    doc.append(
                        f"| {condition_name} | {run.seed} | "
                        f"{_fmt(shared_confs)} | {_fmt(wrong_confs)} | "
                        f"{_fmt(missing_confs)} |"
                    )
                except Exception as e:
                    doc.append(f"| {condition_name} | {run.seed} | error | error | error |")
        doc.append("")

        # Save plots: grouped bar chart of tool-call allocation
        _plot_prior_fresh_tool_allocation(
            oracle_label, prior_sorted, fresh_sorted, tool_types_sorted,
        )

        section_data[oracle_label] = oracle_data

    # C.3: Verdicts
    doc.heading(3, "C.3 Falsification verdicts")
    verdicts = _section_c_verdicts(all_runs, section_data)
    for v in verdicts:
        doc.append(f"- **{v['claim']}**: **{v['verdict']}** — {v['evidence']}")
    doc.append("")

    doc.save_json("section_c", section_data)


def _run_log_name(oracle_label: str, condition_label: str, seed: int) -> str:
    """Reconstruct log file name from condition info."""
    # Match the naming conventions in ORACLE_CONDITIONS
    prefix_map = {
        ("1D", "prior_72"): "transfer_1d_prior",
        ("1D", "fresh_72"): "transfer_1d_fresh",
        ("1D", "prior_144"): "transfer_1d_prior_144",
        ("1E", "prior_72"): "transfer_1e_prior",
        ("1E", "fresh_72"): "transfer_1e_fresh",
        ("1E", "sonnet_prior_72"): "transfer_1e_sonnet_prior",
    }
    prefix = prefix_map.get((oracle_label, condition_label))
    if prefix:
        return f"{prefix}_seed{seed}_log.json"
    # Fallback: try common patterns
    return f"{condition_label}_seed{seed}_log.json"


def _plot_prior_fresh_tool_allocation(
    oracle_label: str,
    prior_runs: list[RunMetrics],
    fresh_runs: list[RunMetrics],
    tool_types: list[str],
) -> None:
    """Grouped bar chart: prior vs fresh tool-call counts, per seed."""
    from matplotlib.ticker import MaxNLocator

    # Only show tool types with at least 1 call in some run
    active_types = [tt for tt in tool_types
                    if any(r.tool_call_counts_total.get(tt, 0) > 0
                           for r in prior_runs + fresh_runs)]
    if not active_types:
        return

    n_types = len(active_types)
    n_seeds = max(len(prior_runs), len(fresh_runs))
    seeds_union = sorted({r.seed for r in prior_runs + fresh_runs})

    fig, axes = plt.subplots(1, len(seeds_union), figsize=(5 * len(seeds_union), 5),
                             sharey=True)
    if len(seeds_union) == 1:
        axes = [axes]

    for ax, seed in zip(axes, seeds_union):
        p_run = next((r for r in prior_runs if r.seed == seed), None)
        f_run = next((r for r in fresh_runs if r.seed == seed), None)

        x = np.arange(n_types)
        width = 0.35
        p_vals = [p_run.tool_call_counts_total.get(tt, 0) if p_run else 0
                  for tt in active_types]
        f_vals = [f_run.tool_call_counts_total.get(tt, 0) if f_run else 0
                  for tt in active_types]

        ax.bar(x - width / 2, p_vals, width, label="prior", color="#d62728", alpha=0.7)
        ax.bar(x + width / 2, f_vals, width, label="fresh", color="#1f77b4", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(active_types, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"seed {seed}", fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"{oracle_label} prior vs fresh: tool-call allocation per seed",
                 fontsize=10)
    plt.tight_layout()
    out_path = PLOTS_DIR / f"C_{oracle_label}_tool_allocation.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _section_c_verdicts(
    all_runs: dict[str, dict[str, list[RunMetrics]]],
    section_data: dict[str, Any],
) -> list[dict[str, str]]:
    """Compute falsification verdicts for Section C claims."""
    verdicts = []

    # Test 1: "Prior primes toward local gradient refinement" (seed 42 pattern)
    for oracle_label in ["1D", "1E"]:
        tc_data = section_data.get(oracle_label, {}).get("tool_call_comparison", {})
        lg_data = tc_data.get("local_gradients", {})
        if lg_data:
            p_vals = lg_data.get("prior", [])
            f_vals = lg_data.get("fresh", [])
            if p_vals and f_vals:
                p_mean = float(np.mean(p_vals))
                f_mean = float(np.mean(f_vals))
                all_prior_higher = all(p > f for p, f in zip(p_vals, f_vals))
                if p_mean > f_mean * 1.5 and all_prior_higher:
                    v = "PASS"
                    ev = (
                        f"{oracle_label} prior uses {p_mean:.1f} local_gradients "
                        f"vs fresh {f_mean:.1f} — consistently higher across all "
                        f"seeds ({p_vals} vs {f_vals})"
                    )
                elif p_mean > f_mean:
                    v = "QUALIFIED"
                    ev = (
                        f"{oracle_label} prior uses {p_mean:.1f} local_gradients "
                        f"vs fresh {f_mean:.1f} — directionally higher but not "
                        f"consistent across seeds ({p_vals} vs {f_vals})"
                    )
                else:
                    v = "FALSIFIED"
                    ev = (
                        f"{oracle_label} prior uses {p_mean:.1f} local_gradients "
                        f"vs fresh {f_mean:.1f} — seed 42 pattern does NOT generalize"
                    )
                verdicts.append({
                    "claim": f"Prior primes toward local_gradients on {oracle_label}",
                    "verdict": v,
                    "evidence": ev,
                })

    # Test 2: "Screen-first selectively leverages correct priors"
    # Check if prior starts with higher confidence on shared edges than fresh does
    # (We'd need the actual confidence data from the iteration summaries which
    #  is in the table output — extract from the data we collected)
    verdicts.append({
        "claim": "Screen-first selectively leverages correct priors (edge confidence initialization)",
        "verdict": "SEE TABLE C.{1D,1E}.4",
        "evidence": "Inspect the edge confidence initialization tables above. "
                    "If prior-run shared-edge confidence > fresh-run shared-edge "
                    "confidence at iteration 0, the selective-leverage claim has "
                    "support. If they're similar, the prior isn't being used for "
                    "edge-specific initialization.",
    })

    # Test 3: "Bimodal variance on 1D is real"
    d_prior = all_runs.get("1D", {}).get("prior_72", [])
    d_fresh = all_runs.get("1D", {}).get("fresh_72", [])
    if len(d_prior) == 3 and len(d_fresh) == 3:
        d_prior_sorted = sorted(d_prior, key=lambda r: r.seed)
        d_fresh_sorted = sorted(d_fresh, key=lambda r: r.seed)
        diffs = [p.final_hv - f.final_hv
                 for p, f in zip(d_prior_sorted, d_fresh_sorted)]
        signs = [("+" if d > 0 else "−") for d in diffs]
        verdicts.append({
            "claim": "1D prior-fresh bimodality is real (not just seed 42 outlier)",
            "verdict": "QUALIFIED" if len(set(signs)) > 1 else "FALSIFIED",
            "evidence": (
                f"Paired diffs (prior − fresh): {[f'{d:+.4f}' for d in diffs]}. "
                f"Signs: {signs}. {'Mixed signs confirm bimodality' if len(set(signs)) > 1 else 'All same sign — not bimodal'}."
            ),
        })

    return verdicts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


SECTIONS: dict[str, Callable[[dict, FindingsDoc], None]] = {
    "A": section_a_learning_trajectories,
    "B": section_b_cross_seed_correlations,
    "C": section_c_prior_vs_fresh,
    # D-I to be added incrementally
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sections", type=str, default=None,
        help="Comma-separated list of sections to run (e.g., 'A,B,C'). Default: all.",
    )
    args = parser.parse_args()

    print("Loading all runs from dataset...")
    all_runs = load_all_runs()
    total_runs = sum(
        len(runs) for conds in all_runs.values() for runs in conds.values()
    )
    print(f"  Loaded {total_runs} runs across "
          f"{sum(len(c) for c in all_runs.values())} conditions "
          f"× {len(all_runs)} oracles.")

    # Figure out which sections to run
    if args.sections:
        sections_to_run = [s.strip() for s in args.sections.split(",")]
    else:
        sections_to_run = list(SECTIONS.keys())

    doc = FindingsDoc(MD_PATH)
    doc.start()

    for section_letter in sections_to_run:
        if section_letter not in SECTIONS:
            print(f"  WARN: unknown section {section_letter}, skipping")
            continue
        print(f"\n=== Running Section {section_letter} ===")
        SECTIONS[section_letter](all_runs, doc)

    doc.flush()
    print(f"\nWrote:")
    print(f"  {MD_PATH}")
    print(f"  {JSON_PATH}")
    print(f"  {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
