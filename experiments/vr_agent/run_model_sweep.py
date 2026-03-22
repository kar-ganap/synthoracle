"""Phase B: Model capability sweep — Sonnet 4.6 and Haiku 4.5 on both agents.

Tests whether cheaper models produce comparable scientific reasoning to Opus 4.6.
Runs 4 configurations on Medium 1A (seed 42, 72 eval budget), then prints
a comparison table against existing Opus baselines.

Usage:
    uv run --extra vr python experiments/vr_agent/run_model_sweep.py
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from synthoracle.agents.vr import VRResult, run_vr
from synthoracle.agents.vr_tools import VRToolsResult, run_vr_tools
from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, NodeType
from synthoracle.oracles.medium import MediumOracle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BUDGET = 72  # tool agent: total oracle evals
N_ITERATIONS = 60  # batch agent: VR-proposed evals (+ 12 initial = 72 total)
BATCH_SIZE = 3
EDGE_THRESHOLD = 0.05  # output range threshold for declaring an edge

# (label, agent_type, model_id, thinking_config)
RUNS: list[tuple[str, str, str, dict[str, object] | None]] = [
    ("sonnet_tool", "tool", "claude-sonnet-4-6", {"type": "adaptive"}),
    ("sonnet_batch", "batch", "claude-sonnet-4-6", {"type": "adaptive"}),
    ("haiku_tool", "tool", "claude-haiku-4-5-20251001", None),
    ("haiku_batch", "batch", "claude-haiku-4-5-20251001", None),
]

# Opus baselines (already saved from Phase A)
OPUS_TOOL_NPZ = RESULTS_DIR / "tools_test_seed42.npz"
OPUS_TOOL_LOG = RESULTS_DIR / "tools_test_seed42_log.json"
OPUS_BATCH_NPZ = RESULTS_DIR / "opus_test_seed42.npz"
OPUS_BATCH_LOG = RESULTS_DIR / "opus_test_seed42_log.json"

# Per-model pricing: ($/M input tokens, $/M output tokens)
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}

# BO baseline (10-seed mean) for HV plot reference
BO_RESULTS = Path("experiments/comparison/results")
BO_1A_PATHS = [BO_RESULTS / f"bo_seed{s}.npz" for s in range(42, 52)]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    label: str
    agent_type: str
    model: str
    final_hv: float
    n_evals: int
    edge_precision: float
    edge_recall: float
    n_mechanisms: int  # unique IO edges discovered
    directional_accuracy: float
    wall_time: float
    input_tokens: int
    output_tokens: int
    llm_calls: int
    cost_usd: float
    hv_curve: list[float]
    failed: bool = False
    error_msg: str = ""


# ---------------------------------------------------------------------------
# Edge extraction helpers (adapted from compute_metrics.py)
# ---------------------------------------------------------------------------


def _replay_oat_sweep(
    oracle: MediumOracle,
    input_name: str,
    n_levels: int,
    base_point: list[float],
) -> npt.NDArray[np.float64]:
    idx = list(oracle.input_names).index(input_name)
    lo, hi = float(oracle.bounds[idx, 0]), float(oracle.bounds[idx, 1])
    levels = np.linspace(lo, hi, n_levels)
    base = np.clip(np.array(base_point, dtype=np.float64),
                   oracle.bounds[:, 0], oracle.bounds[:, 1])
    Y_list = []
    for val in levels:
        pt = base.copy()
        pt[idx] = val
        Y_list.append(oracle.evaluate(pt))
    return np.array(Y_list, dtype=np.float64)


def _replay_interaction_test(
    oracle: MediumOracle,
    input_a: str,
    input_b: str,
    levels_a: list[float],
    levels_b: list[float],
    base_point: list[float],
) -> dict[str, float]:
    idx_a = list(oracle.input_names).index(input_a)
    idx_b = list(oracle.input_names).index(input_b)
    base = np.clip(np.array(base_point, dtype=np.float64),
                   oracle.bounds[:, 0], oracle.bounds[:, 1])
    Y_list = []
    for va in levels_a:
        for vb in levels_b:
            pt = base.copy()
            pt[idx_a] = np.clip(va, oracle.bounds[idx_a, 0], oracle.bounds[idx_a, 1])
            pt[idx_b] = np.clip(vb, oracle.bounds[idx_b, 0], oracle.bounds[idx_b, 1])
            Y_list.append(oracle.evaluate(pt))
    Y_arr = np.array(Y_list, dtype=np.float64)
    n_a, n_b = len(levels_a), len(levels_b)
    Y_grid = Y_arr.reshape(n_a, n_b, oracle.n_outputs)
    effects: dict[str, float] = {}
    for j, oname in enumerate(oracle.output_names):
        row_means = Y_grid[:, :, j].mean(axis=1, keepdims=True)
        col_means = Y_grid[:, :, j].mean(axis=0, keepdims=True)
        grand_mean = Y_grid[:, :, j].mean()
        additive_pred = row_means + col_means - grand_mean
        interaction = Y_grid[:, :, j] - additive_pred
        effects[oname] = float(np.std(interaction))
    return effects


def extract_edges_from_tool_calls(
    oracle: MediumOracle,
    tool_calls: list[dict[str, Any]],
) -> frozenset[Edge]:
    discovered: set[tuple[str, str]] = set()
    for tc in tool_calls:
        name = tc.get("name", "")
        inp = tc.get("input", {})
        if name == "oat_sweep":
            Y_sweep = _replay_oat_sweep(
                oracle, inp["input_name"],
                int(inp.get("n_levels", 5)), inp["base_point"],
            )
            for j, oname in enumerate(oracle.output_names):
                if float(Y_sweep[:, j].max() - Y_sweep[:, j].min()) > EDGE_THRESHOLD:
                    discovered.add((inp["input_name"], oname))
        elif name == "interaction_test":
            effects = _replay_interaction_test(
                oracle, inp["input_a"], inp["input_b"],
                [float(v) for v in inp["levels_a"]],
                [float(v) for v in inp["levels_b"]],
                inp["base_point"],
            )
            for oname, std_val in effects.items():
                if std_val > EDGE_THRESHOLD:
                    discovered.add((inp["input_a"], oname))
                    discovered.add((inp["input_b"], oname))
    return frozenset(Edge(s, t, EdgeDifficulty.EASY, "discovered") for s, t in discovered)


def extract_edges_from_xy(
    oracle: MediumOracle,
    X: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
) -> frozenset[Edge]:
    discovered: set[tuple[str, str]] = set()
    for i in range(oracle.n_inputs):
        lo, hi = float(oracle.bounds[i, 0]), float(oracle.bounds[i, 1])
        bin_edges = np.linspace(lo, hi, 6)
        for j in range(oracle.n_outputs):
            bin_means = []
            for b in range(5):
                if b == 4:
                    mask = (X[:, i] >= bin_edges[b]) & (X[:, i] <= bin_edges[b + 1])
                else:
                    mask = (X[:, i] >= bin_edges[b]) & (X[:, i] < bin_edges[b + 1])
                if mask.sum() > 0:
                    bin_means.append(float(Y[mask, j].mean()))
            if len(bin_means) >= 2 and (max(bin_means) - min(bin_means)) > EDGE_THRESHOLD:
                discovered.add((oracle.input_names[i], oracle.output_names[j]))
    return frozenset(Edge(s, t, EdgeDifficulty.EASY, "discovered") for s, t in discovered)


def compute_edge_pr(
    oracle: MediumOracle,
    disc_edges: frozenset[Edge],
) -> tuple[float, float]:
    gt_io = oracle.ground_truth().project_to_io()
    io_nodes = frozenset(
        n for n in oracle.ground_truth().nodes
        if n.node_type in (NodeType.INPUT, NodeType.OUTPUT)
    )
    discovered_dag = CausalDAG(nodes=io_nodes, edges=disc_edges)
    return gt_io.precision_recall(discovered_dag)


# ---------------------------------------------------------------------------
# Directional accuracy helpers
# ---------------------------------------------------------------------------


def tool_directional_accuracy(tool_calls: list[dict[str, Any]]) -> float:
    """Compute directional accuracy from evaluate_point prediction errors."""
    correct = 0
    total = 0
    for tc in tool_calls:
        if tc.get("name") != "evaluate_point":
            continue
        pred_errors = tc.get("result", {}).get("prediction_errors", {})
        if not pred_errors:
            # Try the input-level predicted_outputs
            continue
        for _oname, err in pred_errors.items():
            # A prediction error near 0 is directionally accurate
            # We count it as correct if the sign matches or error is small
            total += 1
            if abs(float(err)) < 0.1:
                correct += 1
    return correct / total if total > 0 else 0.0


def batch_directional_accuracy(step_logs: list[dict[str, Any]]) -> float:
    """Compute overall directional accuracy from batch step logs."""
    flat: list[bool] = []
    for s in step_logs:
        da = s.get("directional_accuracy", [])
        flat.extend(da)
    return sum(flat) / len(flat) if flat else 0.0


# ---------------------------------------------------------------------------
# Run a single experiment
# ---------------------------------------------------------------------------


def run_single(
    label: str,
    agent_type: str,
    model: str,
    thinking: dict[str, object] | None,
    oracle: MediumOracle,
) -> RunSummary:
    """Run one agent configuration and save results."""
    ckpt = str(RESULTS_DIR / f"sweep_{label}_ckpt")
    pricing = PRICING.get(model, (0.0, 0.0))

    print(f"\n{'=' * 70}")
    print(f"  Running: {label} ({model})")
    print(f"  Agent: {agent_type}, Thinking: {thinking}")
    print(f"{'=' * 70}\n")

    if agent_type == "tool":
        result = run_vr_tools(
            oracle, n_budget=N_BUDGET, seed=SEED,
            thresholds={"Y3": 0.4}, model=model, thinking=thinking,
            max_tokens=16000, max_tool_calls_per_iteration=15,
            checkpoint_dir=ckpt,
        )
        return _summarize_tool_result(label, model, result, oracle, pricing)
    else:
        result = run_vr(
            oracle, n_iterations=N_ITERATIONS, batch_size=BATCH_SIZE,
            seed=SEED, thresholds={"Y3": 0.4}, model=model,
            thinking=thinking, max_tokens=16000, checkpoint_dir=ckpt,
        )
        return _summarize_batch_result(label, model, result, oracle, pricing)


def _summarize_tool_result(
    label: str, model: str, result: VRToolsResult,
    oracle: MediumOracle, pricing: tuple[float, float],
) -> RunSummary:
    """Extract metrics from a tool-use run and save artifacts."""
    # Save npz
    np.savez(
        RESULTS_DIR / f"sweep_{label}_seed42.npz",
        X=result.X, Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
    )
    # Save json log
    with open(RESULTS_DIR / f"sweep_{label}_seed42_log.json", "w") as f:
        json.dump({
            "tool_calls": result.tool_calls,
            "mechanism_log": result.mechanism_log,
            "calibration_checks": result.calibration_checks,
            "iteration_summaries": result.iteration_summaries,
            "eval_count": result.eval_count,
            "total_llm_calls": result.total_llm_calls,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
        }, f, indent=2)

    # Edge precision/recall
    disc_edges = extract_edges_from_tool_calls(oracle, result.tool_calls)
    precision, recall = compute_edge_pr(oracle, disc_edges)

    # Directional accuracy from evaluate_point calls in tool_calls
    dir_acc = tool_directional_accuracy(result.tool_calls)

    cost = (result.total_input_tokens * pricing[0] / 1e6
            + result.total_output_tokens * pricing[1] / 1e6)

    return RunSummary(
        label=label, agent_type="tool", model=model,
        final_hv=result.hypervolumes[-1],
        n_evals=result.eval_count,
        edge_precision=precision, edge_recall=recall,
        n_mechanisms=len(disc_edges),
        directional_accuracy=dir_acc,
        wall_time=result.total_seconds,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
        llm_calls=result.total_llm_calls,
        cost_usd=cost,
        hv_curve=list(result.hypervolumes),
    )


def _summarize_batch_result(
    label: str, model: str, result: VRResult,
    oracle: MediumOracle, pricing: tuple[float, float],
) -> RunSummary:
    """Extract metrics from a batch run and save artifacts."""
    # Save npz
    np.savez(
        RESULTS_DIR / f"sweep_{label}_seed42.npz",
        X=result.X, Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
        predictions=result.predictions,
        prediction_errors=result.prediction_errors,
    )
    # Save json log
    log_data = [
        {
            "step": s.step,
            "hypothesis": s.hypothesis,
            "reasoning": s.reasoning,
            "reconciliation": s.reconciliation,
            "falsification": s.falsification,
            "explore_or_exploit": s.explore_or_exploit,
            "biggest_surprise": s.biggest_surprise,
            "prediction_error": s.prediction_error.tolist(),
            "directional_accuracy": s.directional_accuracy,
            "x": s.x.tolist(),
        }
        for s in result.step_logs
    ]
    with open(RESULTS_DIR / f"sweep_{label}_seed42_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

    # Edge precision/recall from X/Y data
    disc_edges = extract_edges_from_xy(oracle, result.X, result.Y)
    precision, recall = compute_edge_pr(oracle, disc_edges)

    # Directional accuracy from step logs
    dir_acc_flat = [a for s in result.step_logs for a in s.directional_accuracy]
    dir_acc = sum(dir_acc_flat) / len(dir_acc_flat) if dir_acc_flat else 0.0

    cost = (result.total_input_tokens * pricing[0] / 1e6
            + result.total_output_tokens * pricing[1] / 1e6)

    return RunSummary(
        label=label, agent_type="batch", model=model,
        final_hv=result.hypervolumes[-1],
        n_evals=result.n_initial + result.n_vr_iterations,
        edge_precision=precision, edge_recall=recall,
        n_mechanisms=len(disc_edges),
        directional_accuracy=dir_acc,
        wall_time=result.total_seconds,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
        llm_calls=result.total_llm_calls,
        cost_usd=cost,
        hv_curve=list(result.hypervolumes),
    )


# ---------------------------------------------------------------------------
# Load Opus baselines
# ---------------------------------------------------------------------------


def load_opus_baselines(oracle: MediumOracle) -> list[RunSummary]:
    """Load existing Opus tool + batch results as RunSummary objects."""
    summaries: list[RunSummary] = []
    opus_pricing = PRICING["claude-opus-4-6"]

    # Tool Opus
    if OPUS_TOOL_NPZ.exists() and OPUS_TOOL_LOG.exists():
        data = np.load(OPUS_TOOL_NPZ)
        with open(OPUS_TOOL_LOG) as f:
            log = json.load(f)
        tool_calls = log.get("tool_calls", [])
        disc_edges = extract_edges_from_tool_calls(oracle, tool_calls)
        precision, recall = compute_edge_pr(oracle, disc_edges)
        dir_acc = tool_directional_accuracy(tool_calls)
        in_tok = int(log.get("total_input_tokens", 0))
        out_tok = int(log.get("total_output_tokens", 0))
        cost = in_tok * opus_pricing[0] / 1e6 + out_tok * opus_pricing[1] / 1e6
        summaries.append(RunSummary(
            label="opus_tool", agent_type="tool", model="claude-opus-4-6",
            final_hv=float(data["hypervolumes"][-1]),
            n_evals=int(log.get("eval_count", len(data["hypervolumes"]))),
            edge_precision=precision, edge_recall=recall,
            n_mechanisms=len(disc_edges),
            directional_accuracy=dir_acc,
            wall_time=0.0,  # not in log
            input_tokens=in_tok, output_tokens=out_tok,
            llm_calls=int(log.get("total_llm_calls", 0)),
            cost_usd=cost,
            hv_curve=data["hypervolumes"].tolist(),
        ))

    # Batch Opus
    if OPUS_BATCH_NPZ.exists() and OPUS_BATCH_LOG.exists():
        data = np.load(OPUS_BATCH_NPZ)
        with open(OPUS_BATCH_LOG) as f:
            log = json.load(f)
        # log is a list of step dicts for the batch agent
        step_logs = log if isinstance(log, list) else []
        disc_edges = extract_edges_from_xy(oracle, data["X"], data["Y"])
        precision, recall = compute_edge_pr(oracle, disc_edges)
        dir_acc = batch_directional_accuracy(step_logs)
        # Token counts not in the batch log file — estimate from known run
        summaries.append(RunSummary(
            label="opus_batch", agent_type="batch", model="claude-opus-4-6",
            final_hv=float(data["hypervolumes"][-1]),
            n_evals=len(data["hypervolumes"]),
            edge_precision=precision, edge_recall=recall,
            n_mechanisms=len(disc_edges),
            directional_accuracy=dir_acc,
            wall_time=0.0,
            input_tokens=0, output_tokens=0,
            llm_calls=0,
            cost_usd=0.0,
            hv_curve=data["hypervolumes"].tolist(),
        ))

    return summaries


# ---------------------------------------------------------------------------
# BO baseline
# ---------------------------------------------------------------------------


def load_bo_mean_hv() -> npt.NDArray[np.float64] | None:
    """Load BO 10-seed mean HV for 1A."""
    hvs = []
    for p in BO_1A_PATHS:
        if p.exists():
            hvs.append(np.load(p)["hypervolumes"])
    if not hvs:
        return None
    max_len = max(len(h) for h in hvs)
    padded = [np.pad(h, (0, max_len - len(h)), constant_values=h[-1]) for h in hvs]
    return np.mean(padded, axis=0)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def print_comparison_table(summaries: list[RunSummary]) -> None:
    """Print a formatted comparison table."""
    # Find Opus tool/batch HVs as reference
    opus_tool_hv = next(
        (s.final_hv for s in summaries if s.label == "opus_tool"), None)
    opus_batch_hv = next(
        (s.final_hv for s in summaries if s.label == "opus_batch"), None)

    print(f"\n{'=' * 120}")
    print(f"  Phase B: Model Capability Sweep — Comparison Table")
    print(f"{'=' * 120}")

    header = (
        f"  {'Run':<16} {'Final HV':>10} {'%Opus':>7} {'Evals':>6} "
        f"{'Edge P':>7} {'Edge R':>7} {'Mechs':>6} {'Dir Acc':>8} "
        f"{'Time(s)':>8} {'In tok':>10} {'Out tok':>10} {'Cost':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for s in summaries:
        # Determine reference HV
        if s.agent_type == "tool":
            ref_hv = opus_tool_hv
        else:
            ref_hv = opus_batch_hv
        pct = f"{s.final_hv / ref_hv * 100:.1f}%" if ref_hv and ref_hv > 0 else "---"

        cost_str = f"${s.cost_usd:.2f}" if s.cost_usd > 0 else "---"
        time_str = f"{s.wall_time:.0f}" if s.wall_time > 0 else "---"
        in_tok_str = f"{s.input_tokens:,}" if s.input_tokens > 0 else "---"
        out_tok_str = f"{s.output_tokens:,}" if s.output_tokens > 0 else "---"

        print(
            f"  {s.label:<16} {s.final_hv:>10.6f} {pct:>7} {s.n_evals:>6} "
            f"{s.edge_precision:>7.3f} {s.edge_recall:>7.3f} {s.n_mechanisms:>6} "
            f"{s.directional_accuracy:>7.1%} "
            f"{time_str:>8} {in_tok_str:>10} {out_tok_str:>10} {cost_str:>8}"
        )

        if s.failed:
            print(f"    ERROR: {s.error_msg[:80]}")

    print()

    # Go/no-go assessment
    print("  Go/No-Go Assessment:")
    for s in summaries:
        if s.label.startswith("opus"):
            continue
        if s.agent_type == "tool":
            ref = opus_tool_hv
        else:
            ref = opus_batch_hv
        if ref is None or ref <= 0:
            print(f"    {s.label}: No Opus baseline to compare")
            continue
        pct = s.final_hv / ref
        if "sonnet" in s.label:
            viable = pct >= 0.85 and s.n_mechanisms >= 5
            verdict = "VIABLE" if viable else "NOT VIABLE"
            print(f"    {s.label}: HV={pct:.1%} of Opus, {s.n_mechanisms} mechanisms "
                  f"-> {verdict} (threshold: >=85% HV, >=5 mechanisms)")
        elif "haiku" in s.label:
            viable = pct >= 0.70 and not s.failed
            verdict = "VIABLE" if viable else "NOT VIABLE"
            print(f"    {s.label}: HV={pct:.1%} of Opus, failed={s.failed} "
                  f"-> {verdict} (threshold: >=70% HV, coherent execution)")


# ---------------------------------------------------------------------------
# HV comparison plot
# ---------------------------------------------------------------------------


def plot_hv_comparison(summaries: list[RunSummary]) -> None:
    """Plot HV curves for all runs on one figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Color/style scheme: model -> color, agent -> linestyle
    model_colors = {
        "claude-opus-4-6": "#1f77b4",
        "claude-sonnet-4-6": "#ff7f0e",
        "claude-haiku-4-5-20251001": "#2ca02c",
    }
    agent_styles = {"tool": "-", "batch": "--"}

    for s in summaries:
        if s.failed:
            continue
        color = model_colors.get(s.model, "gray")
        ls = agent_styles.get(s.agent_type, "-")
        model_short = s.model.split("-")[1].capitalize()
        evals = list(range(1, len(s.hv_curve) + 1))
        ax.plot(evals, s.hv_curve, color=color, linestyle=ls,
                linewidth=1.5, label=f"{model_short} {s.agent_type}")

    # BO reference
    bo_mean = load_bo_mean_hv()
    if bo_mean is not None:
        evals_bo = list(range(1, len(bo_mean) + 1))
        ax.plot(evals_bo, bo_mean, color="gray", linestyle=":",
                linewidth=1.5, alpha=0.7, label="BO (10-seed mean)")

    ax.axvline(x=12, color="gray", linestyle="--", alpha=0.2, label="Initial")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Phase B: Model Capability Sweep — Medium 1A")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_sweep_hv.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  HV plot saved: {RESULTS_DIR / 'model_sweep_hv.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    oracle = MediumOracle()
    new_summaries: list[RunSummary] = []

    for label, agent_type, model, thinking in RUNS:
        try:
            summary = run_single(label, agent_type, model, thinking, oracle)
            new_summaries.append(summary)
            print(f"\n  {label} complete: HV={summary.final_hv:.6f}, "
                  f"cost=${summary.cost_usd:.2f}, time={summary.wall_time:.0f}s")
        except Exception:
            print(f"\n  {label} FAILED:")
            traceback.print_exc()
            new_summaries.append(RunSummary(
                label=label, agent_type=agent_type, model=model,
                final_hv=0.0, n_evals=0,
                edge_precision=0.0, edge_recall=0.0, n_mechanisms=0,
                directional_accuracy=0.0, wall_time=0.0,
                input_tokens=0, output_tokens=0, llm_calls=0, cost_usd=0.0,
                hv_curve=[], failed=True,
                error_msg=traceback.format_exc()[-200:],
            ))

    # Load Opus baselines
    opus_summaries = load_opus_baselines(oracle)

    # Combine: Opus first, then new runs
    all_summaries = opus_summaries + new_summaries

    print_comparison_table(all_summaries)
    plot_hv_comparison(all_summaries)

    # Print mechanism logs for qualitative review
    for s in new_summaries:
        if s.failed:
            continue
        log_path = RESULTS_DIR / f"sweep_{s.label}_seed42_log.json"
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            if s.agent_type == "tool":
                mech_log = log.get("mechanism_log", [])
                if mech_log:
                    print(f"\n  Mechanism log ({s.label}, {len(mech_log)} entries):")
                    for i, m in enumerate(mech_log):
                        print(f"    [{i + 1}] {m[:120]}")
            else:
                steps = log if isinstance(log, list) else []
                if steps:
                    print(f"\n  Hypothesis log ({s.label}, {len(steps)} steps):")
                    for step in steps[:5]:
                        print(f"    [Step {step.get('step', '?')}] "
                              f"{step.get('hypothesis', '')[:120]}")
                    if len(steps) > 5:
                        print(f"    ... ({len(steps) - 5} more steps)")


if __name__ == "__main__":
    main()
