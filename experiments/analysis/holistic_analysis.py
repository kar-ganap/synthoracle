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

    # Falsification verdicts
    doc.heading(3, "A.3 Falsification verdicts")
    verdicts = _section_a_verdicts(section_data)
    for v in verdicts:
        doc.append(f"- **{v['claim']}**: **{v['verdict']}** — {v['evidence']}")
    doc.append("")

    doc.save_json("section_a", {
        "trajectories": section_data,
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
# Main
# ---------------------------------------------------------------------------


SECTIONS: dict[str, Callable[[dict, FindingsDoc], None]] = {
    "A": section_a_learning_trajectories,
    # B-I to be added incrementally
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
