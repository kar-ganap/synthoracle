"""Compute all analysis metrics from existing experiment data.

No API calls. Oracle evals are <1ms so replaying sweeps is free.

Usage:
    uv run python experiments/analysis/compute_metrics.py

Outputs:
    experiments/analysis/reference_hvs.json          -- reference HVs for 1A/1B/1C
    experiments/analysis/results/hv_vs_budget.png     -- HV convergence: multi-panel
    experiments/analysis/results/prediction_accuracy.png -- rolling pred error/accuracy
    stdout: all tables
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from synthoracle.characterize import characterize
from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, Node, NodeType
from synthoracle.optim_utils import compute_hypervolume, compute_reference_point, parse_directions
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_1c import MediumOracle1C
from synthoracle.oracles.medium_1d import MediumOracle1D
from synthoracle.oracles.medium_1e import MediumOracle1E

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent  # synthoracle/
VR_RESULTS = ROOT / "experiments" / "vr_agent" / "results"
COMP_RESULTS = ROOT / "experiments" / "comparison" / "results"
BO_RESULTS = ROOT / "experiments" / "bo_baseline" / "results"
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
OUT_DIR = ANALYSIS_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Oracle configs
# ---------------------------------------------------------------------------

ORACLE_CONFIGS: list[
    tuple[str, MediumOracle | MediumOracle1C | MediumOracle1D | MediumOracle1E, dict[str, float]]
] = [
    ("1A", MediumOracle(variant="1A"), {"Y3": 0.4}),
    ("1B", MediumOracle(variant="1B"), {"Y3": 0.4}),
    ("1C", MediumOracle1C(), {"Y3": 0.4}),
    ("1D", MediumOracle1D(), {"Y3": 0.4}),
    ("1E", MediumOracle1E(), {"Y3": 0.4}),
]

# ---------------------------------------------------------------------------
# Run inventory
# ---------------------------------------------------------------------------

# Tool-use VR runs: label -> (oracle_label, log_path, npz_path)
TOOL_USE_RUNS: dict[str, tuple[str, Path, Path]] = {
    "1A tools": ("1A", VR_RESULTS / "tools_test_seed42_log.json", VR_RESULTS / "tools_test_seed42.npz"),
    "1B transfer": ("1B", VR_RESULTS / "transfer_medium_1b_seed42_log.json", VR_RESULTS / "transfer_medium_1b_seed42.npz"),
    "1B fresh": ("1B", VR_RESULTS / "no_transfer_1b_seed42_log.json", VR_RESULTS / "no_transfer_1b_seed42.npz"),
    "1C transfer": ("1C", VR_RESULTS / "transfer_1c_seed42_log.json", VR_RESULTS / "transfer_1c_seed42.npz"),
    "1C fresh": ("1C", VR_RESULTS / "no_transfer_1c_seed42_log.json", VR_RESULTS / "no_transfer_1c_seed42.npz"),
    # Phase B: Model sweep tool runs
    "Sonnet tool": ("1A", VR_RESULTS / "sweep_sonnet_tool_seed42_log.json", VR_RESULTS / "sweep_sonnet_tool_seed42.npz"),
    "Haiku tool": ("1A", VR_RESULTS / "sweep_haiku_tool_seed42_log.json", VR_RESULTS / "sweep_haiku_tool_seed42.npz"),
    # Head-to-head (structured iteration summaries)
    "Opus h2h": ("1A", VR_RESULTS / "opus_h2h_seed42_log.json", VR_RESULTS / "opus_h2h_seed42.npz"),
    "Sonnet h2h": ("1A", VR_RESULTS / "sonnet_h2h_seed42_log.json", VR_RESULTS / "sonnet_h2h_seed42.npz"),
    # n=6 ablation
    "Opus n=6": ("1A", VR_RESULTS / "opus_n6_seed42_log.json", VR_RESULTS / "opus_n6_seed42.npz"),
    # Extended budget
    "1A ext 144": ("1A", VR_RESULTS / "1a_extended_144_seed42_log.json", VR_RESULTS / "1a_extended_144_seed42.npz"),
    # Transfer 1D
    "1D prior": ("1D", VR_RESULTS / "transfer_1d_prior_seed42_log.json", VR_RESULTS / "transfer_1d_prior_seed42.npz"),
    "1D fresh": ("1D", VR_RESULTS / "transfer_1d_fresh_seed42_log.json", VR_RESULTS / "transfer_1d_fresh_seed42.npz"),
    "1D prior 144": ("1D", VR_RESULTS / "transfer_1d_prior_144_seed42_log.json", VR_RESULTS / "transfer_1d_prior_144_seed42.npz"),
    # Transfer 1E
    "1E prior": ("1E", VR_RESULTS / "transfer_1e_prior_seed42_log.json", VR_RESULTS / "transfer_1e_prior_seed42.npz"),
    "1E fresh": ("1E", VR_RESULTS / "transfer_1e_fresh_seed42_log.json", VR_RESULTS / "transfer_1e_fresh_seed42.npz"),
}

# Batch VR run (Phase 2.2): label -> (oracle_label, log_path, npz_path)
BATCH_RUNS: dict[str, tuple[str, Path, Path]] = {
    "1A batch": ("1A", VR_RESULTS / "opus_test_seed42_log.json", VR_RESULTS / "opus_test_seed42.npz"),
    # Phase B: Model sweep batch runs
    "Sonnet batch": ("1A", VR_RESULTS / "sweep_sonnet_batch_seed42_log.json", VR_RESULTS / "sweep_sonnet_batch_seed42.npz"),
    "Haiku batch": ("1A", VR_RESULTS / "sweep_haiku_batch_seed42_log.json", VR_RESULTS / "sweep_haiku_batch_seed42.npz"),
}

# V1 VR run (Phase 2.0 Sonnet): label -> (oracle_label, log_path, npz_path)
V1_RUNS: dict[str, tuple[str, Path, Path]] = {
    "1A v1": ("1A", VR_RESULTS / "medium_1a_seed42_log.json", VR_RESULTS / "medium_1a_seed42.npz"),
}

# Ablation runs: label -> (oracle_label, npz_path)
ABLATION_RUNS: dict[str, tuple[str, Path]] = {
    "Ablation real": ("1A", VR_RESULTS / "ablation_real_seed42.npz"),
    "Ablation permuted": ("1A", VR_RESULTS / "ablation_permuted_seed42.npz"),
}

# BO baseline runs: label -> (oracle_label, npz_path or list of npz_paths)
# For 1A we have 10 seeds; for 1B/1C we have single seeds
BO_RUNS_1A_MULTI: list[Path] = [COMP_RESULTS / f"bo_seed{s}.npz" for s in range(42, 52)]
BO_RUNS_SINGLE: dict[str, tuple[str, Path]] = {
    "BO 1A (Phase 1.2)": ("1A", BO_RESULTS / "medium_1a_seed42.npz"),
    "BO 1B (Phase 1.2)": ("1B", BO_RESULTS / "medium_1b_seed42.npz"),
    "BO 1C (Phase 1.2)": ("1C", BO_RESULTS / "medium_1c_seed42.npz"),
}

EDGE_THRESHOLD = 0.05  # output range threshold for declaring an edge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_oracle(label: str) -> MediumOracle | MediumOracle1C | MediumOracle1D | MediumOracle1E:
    """Get oracle instance by label."""
    for lbl, oracle, _ in ORACLE_CONFIGS:
        if lbl == label:
            return oracle
    raise ValueError(f"Unknown label: {label}")


def _get_thresholds(label: str) -> dict[str, float]:
    """Get constraint thresholds by oracle label."""
    for lbl, _, thresholds in ORACLE_CONFIGS:
        if lbl == label:
            return thresholds
    raise ValueError(f"Unknown label: {label}")


def _load_tool_calls(log_path: Path) -> list[dict[str, Any]]:
    """Load tool calls from a tool-use log file."""
    with open(log_path) as f:
        log = json.load(f)
    if isinstance(log, dict):
        return log.get("tool_calls", [])
    return []


def _load_step_logs(log_path: Path) -> list[dict[str, Any]]:
    """Load step logs from a batch/v1 log file (list of step entries)."""
    with open(log_path) as f:
        log = json.load(f)
    if isinstance(log, list):
        return log
    return []


def _replay_oat_sweep(
    oracle: MediumOracle | MediumOracle1C,
    input_name: str,
    n_levels: int,
    base_point: list[float],
) -> npt.NDArray[np.float64]:
    """Replay an OAT sweep and return Y array of shape (n_levels, n_outputs)."""
    idx = list(oracle.input_names).index(input_name)
    lo, hi = float(oracle.bounds[idx, 0]), float(oracle.bounds[idx, 1])
    levels = np.linspace(lo, hi, n_levels)

    base = np.array(base_point, dtype=np.float64)
    base = np.clip(base, oracle.bounds[:, 0], oracle.bounds[:, 1])

    Y_list: list[npt.NDArray[np.float64]] = []
    for val in levels:
        pt = base.copy()
        pt[idx] = val
        Y_list.append(oracle.evaluate(pt))
    return np.array(Y_list, dtype=np.float64)


def _replay_interaction_test(
    oracle: MediumOracle | MediumOracle1C,
    input_a: str,
    input_b: str,
    levels_a: list[float],
    levels_b: list[float],
    base_point: list[float],
) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
    """Replay an interaction test; return (Y_grid_flat, interaction_effects_std)."""
    idx_a = list(oracle.input_names).index(input_a)
    idx_b = list(oracle.input_names).index(input_b)
    base = np.array(base_point, dtype=np.float64)
    base = np.clip(base, oracle.bounds[:, 0], oracle.bounds[:, 1])

    Y_list: list[npt.NDArray[np.float64]] = []
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

    return Y_arr, effects


def _extract_edges_from_xy_data(
    oracle: MediumOracle | MediumOracle1C,
    X: npt.NDArray[np.float64],
    Y: npt.NDArray[np.float64],
) -> frozenset[Edge]:
    """Extract IO edges from arbitrary X/Y data using partial correlations.

    For each input, compute the range of each output conditional on that
    input varying (approximate OAT from nearest-neighbor pairs). Uses
    the same EDGE_THRESHOLD as tool-based extraction for apples-to-apples.
    """
    discovered_pairs: set[tuple[str, str]] = set()
    n_inputs = oracle.n_inputs
    n_outputs = oracle.n_outputs

    for i in range(n_inputs):
        # Bin the input into 5 equal-width bins
        lo, hi = float(oracle.bounds[i, 0]), float(oracle.bounds[i, 1])
        bin_edges = np.linspace(lo, hi, 6)

        for j in range(n_outputs):
            # Compute mean output per bin
            bin_means = []
            for b in range(5):
                mask = (X[:, i] >= bin_edges[b]) & (X[:, i] < bin_edges[b + 1])
                if b == 4:  # include upper bound in last bin
                    mask = (X[:, i] >= bin_edges[b]) & (X[:, i] <= bin_edges[b + 1])
                if mask.sum() > 0:
                    bin_means.append(float(Y[mask, j].mean()))

            if len(bin_means) >= 2:
                out_range = max(bin_means) - min(bin_means)
                if out_range > EDGE_THRESHOLD:
                    discovered_pairs.add((oracle.input_names[i], oracle.output_names[j]))

    return frozenset(
        Edge(src, tgt, EdgeDifficulty.EASY, "discovered")
        for src, tgt in discovered_pairs
    )


def _extract_edges_from_tool_calls(
    oracle: MediumOracle | MediumOracle1C,
    tool_calls: list[dict[str, Any]],
) -> frozenset[Edge]:
    """Extract discovered IO edges by replaying OAT sweeps from tool calls."""
    discovered_pairs: set[tuple[str, str]] = set()

    for tc in tool_calls:
        name = tc.get("name", "")
        inp = tc.get("input", {})

        if name == "oat_sweep":
            input_name = inp["input_name"]
            n_levels = int(inp.get("n_levels", 5))
            base_point = inp["base_point"]
            Y_sweep = _replay_oat_sweep(oracle, input_name, n_levels, base_point)

            for j, oname in enumerate(oracle.output_names):
                out_range = float(Y_sweep[:, j].max() - Y_sweep[:, j].min())
                if out_range > EDGE_THRESHOLD:
                    discovered_pairs.add((input_name, oname))

        elif name == "interaction_test":
            input_a = inp["input_a"]
            input_b = inp["input_b"]
            levels_a = [float(v) for v in inp["levels_a"]]
            levels_b = [float(v) for v in inp["levels_b"]]
            base_point = inp["base_point"]
            _, effects = _replay_interaction_test(
                oracle, input_a, input_b, levels_a, levels_b, base_point,
            )
            for oname, std_val in effects.items():
                if std_val > EDGE_THRESHOLD:
                    discovered_pairs.add((input_a, oname))
                    discovered_pairs.add((input_b, oname))

    return frozenset(
        Edge(src, tgt, EdgeDifficulty.EASY, "discovered")
        for src, tgt in discovered_pairs
    )


def _discovery_timeline_from_tool_calls(
    oracle: MediumOracle | MediumOracle1C,
    tool_calls: list[dict[str, Any]],
    initial_evals: int = 12,
) -> list[tuple[str, int]]:
    """Track when each IO edge is first discovered. Returns sorted (mechanism, eval_count) pairs."""
    cumulative_cost = initial_evals
    discovered_at: dict[str, int] = {}

    for tc in tool_calls:
        name = tc.get("name", "")
        inp = tc.get("input", {})
        cost = int(tc.get("cost", 0))

        if name == "oat_sweep":
            input_name = inp["input_name"]
            n_levels = int(inp.get("n_levels", 5))
            base_point = inp["base_point"]
            Y_sweep = _replay_oat_sweep(oracle, input_name, n_levels, base_point)

            for j, oname in enumerate(oracle.output_names):
                out_range = float(Y_sweep[:, j].max() - Y_sweep[:, j].min())
                if out_range > EDGE_THRESHOLD:
                    key = f"OAT: {input_name} -> {oname} (range={out_range:.3f})"
                    if key not in discovered_at:
                        discovered_at[key] = cumulative_cost + cost

        elif name == "interaction_test":
            input_a = inp["input_a"]
            input_b = inp["input_b"]
            levels_a = [float(v) for v in inp["levels_a"]]
            levels_b = [float(v) for v in inp["levels_b"]]
            base_point = inp["base_point"]
            _, effects = _replay_interaction_test(
                oracle, input_a, input_b, levels_a, levels_b, base_point,
            )
            for oname, std_val in effects.items():
                if std_val > EDGE_THRESHOLD:
                    key = f"Interaction: {input_a}*{input_b} -> {oname} (std={std_val:.3f})"
                    if key not in discovered_at:
                        discovered_at[key] = cumulative_cost + cost

        cumulative_cost += cost

    return sorted(discovered_at.items(), key=lambda x: x[1])


def _load_bo_1a_mean() -> npt.NDArray[np.float64] | None:
    """Load BO 10-seed mean HV trajectory for 1A. Returns None if no data."""
    bo_hvs_list: list[npt.NDArray[np.float64]] = []
    for p in BO_RUNS_1A_MULTI:
        if p.exists():
            bo_hvs_list.append(np.load(p)["hypervolumes"])
    if not bo_hvs_list:
        return None
    max_len = max(len(h) for h in bo_hvs_list)
    padded = []
    for h in bo_hvs_list:
        if len(h) < max_len:
            padded.append(np.pad(h, (0, max_len - len(h)), constant_values=h[-1]))
        else:
            padded.append(h)
    return np.mean(padded, axis=0)


def _load_bo_1a_stats() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Load BO 10-seed mean and std HV for 1A. Returns None if no data."""
    bo_hvs_list: list[npt.NDArray[np.float64]] = []
    for p in BO_RUNS_1A_MULTI:
        if p.exists():
            bo_hvs_list.append(np.load(p)["hypervolumes"])
    if not bo_hvs_list:
        return None
    max_len = max(len(h) for h in bo_hvs_list)
    padded = []
    for h in bo_hvs_list:
        if len(h) < max_len:
            padded.append(np.pad(h, (0, max_len - len(h)), constant_values=h[-1]))
        else:
            padded.append(h)
    matrix = np.array(padded)
    return matrix.mean(axis=0), matrix.std(axis=0)


def rolling_mean(arr: npt.NDArray[np.float64], w: int) -> npt.NDArray[np.float64]:
    """Compute rolling mean with minimum periods=1."""
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(len(arr)):
        start = max(0, i - w + 1)
        out[i] = np.mean(arr[start:i + 1])
    return out


# ======================================================================
# 1. Reference Hypervolumes
# ======================================================================

def compute_reference_hvs() -> dict[str, float]:
    """Characterize each oracle and compute reference HVs. Cached to file."""
    print("=" * 70)
    print("  1. Reference Hypervolumes")
    print("=" * 70)

    cache_path = ANALYSIS_DIR / "reference_hvs.json"
    if cache_path.exists():
        with open(cache_path) as f:
            ref_hvs = json.load(f)
        # Verify all labels present
        if all(lbl in ref_hvs for lbl, _, _ in ORACLE_CONFIGS):
            print(f"\n  Loaded from cache: {cache_path}")
            for lbl in ref_hvs:
                print(f"    {lbl}: {ref_hvs[lbl]:.6f}")
            return ref_hvs

    ref_hvs: dict[str, float] = {}

    for label, oracle, thresholds in ORACLE_CONFIGS:
        print(f"\n  Characterizing Medium {label} (n_pareto=200,000) ...")
        char_result = characterize(
            oracle, n_pareto=200_000, thresholds=thresholds, seed=42,
        )

        obj_indices, constraint_indices, signs = parse_directions(oracle)
        ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

        hv = compute_hypervolume(
            char_result.pareto_front,
            obj_indices,
            signs,
            constraint_indices,
            thresholds,
            oracle.output_names,
            ref_point,
        )
        ref_hvs[label] = float(hv)
        print(f"    Reference point: {ref_point}")
        print(f"    Pareto front size: {char_result.pareto_front.shape[0]}")
        print(f"    Reference HV: {hv:.6f}")

    with open(cache_path, "w") as f:
        json.dump(ref_hvs, f, indent=2)
    print(f"\n  Saved to {cache_path}")

    return ref_hvs


# ======================================================================
# 2. Edge Precision/Recall
# ======================================================================

def compute_edge_precision_recall() -> None:
    """Compute edge precision/recall for all tool-use runs.

    BO and batch/v1 runs produce no causal model, so they are listed as N/A.
    """
    print("\n" + "=" * 70)
    print("  2. Edge Precision / Recall")
    print("=" * 70)

    header = f"  {'Run':<22} {'Precision':>10} {'Recall':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'GT edges':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Tool-use runs: extract edges from OAT sweep data
    for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
        if not log_path.exists():
            print(f"  {run_name:<22} (file not found: {log_path.name})")
            continue

        oracle = _get_oracle(label)
        gt_io = oracle.ground_truth().project_to_io()
        gt_pairs = {(e.source, e.target) for e in gt_io.edges}

        tool_calls = _load_tool_calls(log_path)
        disc_edges = _extract_edges_from_tool_calls(oracle, tool_calls)
        disc_pairs = {(e.source, e.target) for e in disc_edges}

        io_nodes = frozenset(
            n for n in oracle.ground_truth().nodes
            if n.node_type in (NodeType.INPUT, NodeType.OUTPUT)
        )
        discovered_dag = CausalDAG(nodes=io_nodes, edges=disc_edges)
        precision, recall = gt_io.precision_recall(discovered_dag)

        tp = len(gt_pairs & disc_pairs)
        fp = len(disc_pairs - gt_pairs)
        fn = len(gt_pairs - disc_pairs)

        print(f"  {run_name:<22} {precision:>10.3f} {recall:>10.3f} {tp:>5} {fp:>5} {fn:>5} {len(gt_pairs):>10}")

        if fn > 0:
            missed = gt_pairs - disc_pairs
            print(f"    Missed: {', '.join(f'{s}->{t}' for s, t in sorted(missed))}")
        if fp > 0:
            false_pos = disc_pairs - gt_pairs
            print(f"    False+: {', '.join(f'{s}->{t}' for s, t in sorted(false_pos))}")

    # Batch/v1 runs: extract edges from X/Y data (binned sensitivity)
    for run_name, (label, _log_path, npz_path) in {**BATCH_RUNS, **V1_RUNS}.items():
        if not npz_path.exists():
            print(f"  {run_name:<22} (npz not found)")
            continue

        oracle = _get_oracle(label)
        gt_io = oracle.ground_truth().project_to_io()
        gt_pairs = {(e.source, e.target) for e in gt_io.edges}

        data = np.load(npz_path)
        disc_edges = _extract_edges_from_xy_data(oracle, data["X"], data["Y"])
        disc_pairs = {(e.source, e.target) for e in disc_edges}

        io_nodes = frozenset(
            n for n in oracle.ground_truth().nodes
            if n.node_type in (NodeType.INPUT, NodeType.OUTPUT)
        )
        discovered_dag = CausalDAG(nodes=io_nodes, edges=disc_edges)
        precision, recall = gt_io.precision_recall(discovered_dag)

        tp = len(gt_pairs & disc_pairs)
        fp = len(disc_pairs - gt_pairs)
        fn = len(gt_pairs - disc_pairs)

        print(f"  {run_name:<22} {precision:>10.3f} {recall:>10.3f} {tp:>5} {fp:>5} {fn:>5} {len(gt_pairs):>10}")
        if fn > 0:
            missed = gt_pairs - disc_pairs
            print(f"    Missed: {', '.join(f'{s}->{t}' for s, t in sorted(missed))}")
        if fp > 0:
            false_pos = disc_pairs - gt_pairs
            print(f"    False+: {', '.join(f'{s}->{t}' for s, t in sorted(false_pos))}")

    # BO runs
    bo_labels = ["BO 1A (10-seed)", "BO 1A (Phase 1.2)", "BO 1B (Phase 1.2)", "BO 1C (Phase 1.2)"]
    for bo_label in bo_labels:
        print(f"  {bo_label:<22} {'N/A':>10} {'N/A':>10}   {'(BO produces no causal model)':>5}")

    # Ablation runs
    for run_name in ABLATION_RUNS:
        print(f"  {run_name:<22} {'N/A':>10} {'N/A':>10}   {'(ablation -- no causal model)':>5}")


# ======================================================================
# 3. Mechanism Discovery Timeline
# ======================================================================

def compute_mechanism_timeline() -> None:
    """Track when each causal mechanism is first discovered, for all tool-use runs.

    Also checks batch run hypothesis text for mechanism mentions.
    """
    print("\n" + "=" * 70)
    print("  3. Mechanism Discovery Timeline")
    print("=" * 70)

    # --- Tool-use runs ---
    for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
        print(f"\n  --- {run_name} (oracle={label}) ---")
        if not log_path.exists():
            print("  (log file not found)")
            continue

        oracle = _get_oracle(label)
        tool_calls = _load_tool_calls(log_path)
        timeline = _discovery_timeline_from_tool_calls(oracle, tool_calls)

        if not timeline:
            print("  No discoveries found.")
            continue

        print(f"  {'Eval #':>8}  {'Mechanism'}")
        print(f"  {'-' * 60}")
        for mechanism, eval_count in timeline:
            print(f"  {eval_count:>8}  {mechanism}")

        total_cost = 12 + sum(int(tc.get("cost", 0)) for tc in tool_calls)
        print(f"  Total discoveries: {len(timeline)}, total evals: {total_cost}")

    # --- Batch run: extract mechanism mentions from hypothesis text ---
    for run_name, (label, log_path, _npz_path) in BATCH_RUNS.items():
        print(f"\n  --- {run_name} (oracle={label}, from hypothesis text) ---")
        if not log_path.exists():
            print("  (log file not found)")
            continue

        oracle = _get_oracle(label)
        step_logs = _load_step_logs(log_path)
        _print_hypothesis_mechanism_mentions(oracle, step_logs, run_name)

    # --- V1 run: same format ---
    for run_name, (label, log_path, _npz_path) in V1_RUNS.items():
        print(f"\n  --- {run_name} (oracle={label}, from hypothesis text) ---")
        if not log_path.exists():
            print("  (log file not found)")
            continue

        oracle = _get_oracle(label)
        step_logs = _load_step_logs(log_path)
        _print_hypothesis_mechanism_mentions(oracle, step_logs, run_name)


def _print_hypothesis_mechanism_mentions(
    oracle: MediumOracle | MediumOracle1C,
    step_logs: list[dict[str, Any]],
    run_name: str,
) -> None:
    """Scan hypothesis text for first mention of each input->output relationship."""
    # Build search patterns: for each (input, output) pair, look for mentions
    input_names = list(oracle.input_names)
    output_names = list(oracle.output_names)

    # Track first step where Xi is mentioned in context with Yj
    first_mention: dict[tuple[str, str], int] = {}

    for entry in step_logs:
        step = entry.get("step", -1)
        hypothesis = entry.get("hypothesis", "")
        # Also check reasoning and reconciliation for mechanism mentions
        reasoning = entry.get("reasoning", "")
        text = f"{hypothesis} {reasoning}"

        for iname in input_names:
            if iname not in text:
                continue
            for oname in output_names:
                if oname not in text:
                    continue
                pair = (iname, oname)
                if pair not in first_mention:
                    first_mention[pair] = step

    if not first_mention:
        print("  No specific Xi->Yj mentions found in hypotheses.")
        return

    # Get ground truth IO edges
    gt_io = oracle.ground_truth().project_to_io()
    gt_pairs = {(e.source, e.target) for e in gt_io.edges}

    sorted_mentions = sorted(first_mention.items(), key=lambda x: x[1])
    print(f"  {'Step':>6}  {'Edge':<15} {'In GT?':>8}")
    print(f"  {'-' * 35}")
    for (iname, oname), step in sorted_mentions:
        in_gt = "yes" if (iname, oname) in gt_pairs else "NO"
        print(f"  {step:>6}  {iname}->{oname:<10} {in_gt:>8}")

    # Coverage summary
    gt_mentioned = {p for p in first_mention if p in gt_pairs}
    print(f"  GT coverage: {len(gt_mentioned)}/{len(gt_pairs)} edges mentioned")


# ======================================================================
# 4. Budget-Normalized HV
# ======================================================================

def compute_budget_normalized_hv(ref_hvs: dict[str, float]) -> None:
    """Find eval count to reach 50%/75%/90% of reference HV for each run."""
    print("\n" + "=" * 70)
    print("  4. Budget-Normalized HV (evals to reach X% of reference)")
    print("=" * 70)

    # Collect all runs: name -> (oracle_label, hvs)
    runs: dict[str, tuple[str, npt.NDArray[np.float64]]] = {}

    # BO 10-seed mean on 1A
    bo_mean = _load_bo_1a_mean()
    if bo_mean is not None:
        runs["BO 1A (10-seed mean)"] = ("1A", bo_mean)

    # BO Phase 1.2 single seeds
    for name, (label, npz_path) in BO_RUNS_SINGLE.items():
        if npz_path.exists():
            runs[name] = (label, np.load(npz_path)["hypervolumes"])

    # Tool-use VR runs
    for name, (label, _log_path, npz_path) in TOOL_USE_RUNS.items():
        vr_name = f"VR {name}"
        if npz_path.exists():
            runs[vr_name] = (label, np.load(npz_path)["hypervolumes"])

    # Batch VR run
    for name, (label, _log_path, npz_path) in BATCH_RUNS.items():
        vr_name = f"VR {name}"
        if npz_path.exists():
            runs[vr_name] = (label, np.load(npz_path)["hypervolumes"])

    # V1 VR run
    for name, (label, _log_path, npz_path) in V1_RUNS.items():
        vr_name = f"VR {name}"
        if npz_path.exists():
            runs[vr_name] = (label, np.load(npz_path)["hypervolumes"])

    # Ablation runs
    for name, (label, npz_path) in ABLATION_RUNS.items():
        if npz_path.exists():
            runs[name] = (label, np.load(npz_path)["hypervolumes"])

    thresholds_pct = [0.50, 0.75, 0.90]

    # Print grouped by oracle
    for oracle_label in ("1A", "1B", "1C"):
        oracle_runs = {k: v for k, v in runs.items() if v[0] == oracle_label}
        if not oracle_runs:
            continue

        ref_hv = ref_hvs.get(oracle_label, 0.0)
        print(f"\n  Oracle {oracle_label} (ref HV = {ref_hv:.4f})")
        print(f"  {'Run':<28} {'#evals':>7}", end="")
        for t in thresholds_pct:
            print(f"  {f'{t:.0%}':>8}", end="")
        print()
        print("  " + "-" * 60)

        for name, (_, hvs) in oracle_runs.items():
            print(f"  {name:<28} {len(hvs):>7}", end="")
            for t in thresholds_pct:
                target = t * ref_hv
                indices = np.where(hvs >= target)[0]
                if len(indices) > 0:
                    print(f"  {indices[0] + 1:>8}", end="")
                else:
                    print(f"  {'---':>8}", end="")
            print()


# ======================================================================
# 5. HV vs Budget Plot (multi-panel: one per oracle)
# ======================================================================

def plot_hv_vs_budget(ref_hvs: dict[str, float]) -> None:
    """Plot HV vs budget: 3 subplots (1A, 1B, 1C), each with BO + VR variants."""
    print("\n" + "=" * 70)
    print("  5. HV vs Budget Plot (3 panels)")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Colors for different run types
    colors = {
        "bo": "#1f77b4",
        "bo_phase12": "#aec7e8",
        "tools": "#d62728",
        "batch": "#ff7f0e",
        "v1": "#9467bd",
        "transfer": "#2ca02c",
        "fresh": "#8c564b",
        "ablation_real": "#e377c2",
        "ablation_permuted": "#7f7f7f",
        "sonnet_tool": "#e6550d",
        "sonnet_batch": "#fdae6b",
        "haiku_tool": "#31a354",
        "haiku_batch": "#a1d99b",
    }

    # --- Panel 0: Oracle 1A ---
    ax = axes[0]
    ax.set_title("Oracle 1A")

    # BO 10-seed with band
    bo_stats = _load_bo_1a_stats()
    if bo_stats is not None:
        bo_mean, bo_std = bo_stats
        bo_evals = np.arange(1, len(bo_mean) + 1)
        ax.plot(bo_evals, bo_mean, color=colors["bo"], linewidth=2, label="BO (10-seed mean)")
        ax.fill_between(bo_evals, bo_mean - bo_std, bo_mean + bo_std,
                        alpha=0.15, color=colors["bo"])

    # BO Phase 1.2 single seed
    bo_1a_p12 = BO_RESULTS / "medium_1a_seed42.npz"
    if bo_1a_p12.exists():
        hvs = np.load(bo_1a_p12)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["bo_phase12"],
                linewidth=1.5, linestyle="--", label="BO (Phase 1.2)")

    # VR tools 1A
    tools_npz = VR_RESULTS / "tools_test_seed42.npz"
    if tools_npz.exists():
        hvs = np.load(tools_npz)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["tools"],
                linewidth=2, label="VR tools")

    # VR batch 1A
    batch_npz = VR_RESULTS / "opus_test_seed42.npz"
    if batch_npz.exists():
        hvs = np.load(batch_npz)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["batch"],
                linewidth=2, label="VR batch (Opus)")

    # VR v1 1A
    v1_npz = VR_RESULTS / "medium_1a_seed42.npz"
    if v1_npz.exists():
        hvs = np.load(v1_npz)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["v1"],
                linewidth=2, label="VR v1 (Sonnet)")

    # Ablation real
    abl_real = VR_RESULTS / "ablation_real_seed42.npz"
    if abl_real.exists():
        hvs = np.load(abl_real)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["ablation_real"],
                linewidth=1.5, linestyle=":", label="Ablation (real)")

    # Ablation permuted
    abl_perm = VR_RESULTS / "ablation_permuted_seed42.npz"
    if abl_perm.exists():
        hvs = np.load(abl_perm)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["ablation_permuted"],
                linewidth=1.5, linestyle=":", label="Ablation (permuted)")

    # Phase B: Model sweep runs
    sweep_runs = [
        ("sweep_sonnet_tool_seed42.npz", colors["sonnet_tool"], "-", "Sonnet tool"),
        ("sweep_sonnet_batch_seed42.npz", colors["sonnet_batch"], "--", "Sonnet batch"),
        ("sweep_haiku_tool_seed42.npz", colors["haiku_tool"], "-", "Haiku tool"),
        ("sweep_haiku_batch_seed42.npz", colors["haiku_batch"], "--", "Haiku batch"),
    ]
    for fname, color, ls, lbl in sweep_runs:
        sweep_npz = VR_RESULTS / fname
        if sweep_npz.exists():
            hvs = np.load(sweep_npz)["hypervolumes"]
            ax.plot(np.arange(1, len(hvs) + 1), hvs, color=color,
                    linewidth=1.5, linestyle=ls, label=lbl)

    # Reference HV line
    if "1A" in ref_hvs:
        ax.axhline(y=ref_hvs["1A"], color="gray", linestyle="--", alpha=0.4, label=f"Ref HV ({ref_hvs['1A']:.3f})")

    ax.set_xlabel("Oracle Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- Panel 1: Oracle 1B ---
    ax = axes[1]
    ax.set_title("Oracle 1B")

    # BO Phase 1.2
    bo_1b = BO_RESULTS / "medium_1b_seed42.npz"
    if bo_1b.exists():
        hvs = np.load(bo_1b)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["bo"],
                linewidth=2, label="BO (Phase 1.2)")

    # VR transfer 1B
    tr_1b = VR_RESULTS / "transfer_medium_1b_seed42.npz"
    if tr_1b.exists():
        hvs = np.load(tr_1b)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["transfer"],
                linewidth=2, label="VR transfer")

    # VR fresh 1B
    fr_1b = VR_RESULTS / "no_transfer_1b_seed42.npz"
    if fr_1b.exists():
        hvs = np.load(fr_1b)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["fresh"],
                linewidth=2, label="VR fresh")

    if "1B" in ref_hvs:
        ax.axhline(y=ref_hvs["1B"], color="gray", linestyle="--", alpha=0.4, label=f"Ref HV ({ref_hvs['1B']:.3f})")

    ax.set_xlabel("Oracle Evaluations")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Oracle 1C ---
    ax = axes[2]
    ax.set_title("Oracle 1C")

    # BO Phase 1.2
    bo_1c = BO_RESULTS / "medium_1c_seed42.npz"
    if bo_1c.exists():
        hvs = np.load(bo_1c)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["bo"],
                linewidth=2, label="BO (Phase 1.2)")

    # VR transfer 1C
    tr_1c = VR_RESULTS / "transfer_1c_seed42.npz"
    if tr_1c.exists():
        hvs = np.load(tr_1c)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["transfer"],
                linewidth=2, label="VR transfer")

    # VR fresh 1C
    fr_1c = VR_RESULTS / "no_transfer_1c_seed42.npz"
    if fr_1c.exists():
        hvs = np.load(fr_1c)["hypervolumes"]
        ax.plot(np.arange(1, len(hvs) + 1), hvs, color=colors["fresh"],
                linewidth=2, label="VR fresh")

    if "1C" in ref_hvs:
        ax.axhline(y=ref_hvs["1C"], color="gray", linestyle="--", alpha=0.4, label=f"Ref HV ({ref_hvs['1C']:.3f})")

    ax.set_xlabel("Oracle Evaluations")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("HV vs Budget: BO Baseline vs VR Variants", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = OUT_DIR / "hv_vs_budget.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out_path}")


# ======================================================================
# 6. Prediction Accuracy
# ======================================================================

def compute_prediction_accuracy() -> None:
    """Compute prediction accuracy for all runs that have prediction data.

    - Batch/v1 runs: per-step prediction_error and directional_accuracy from step logs.
    - Tool-use runs: extract predicted_outputs from evaluate_point tool calls,
      replay oracle to get actual, compute MAE and directional accuracy.
    """
    print("\n" + "=" * 70)
    print("  6. Prediction Accuracy")
    print("=" * 70)

    all_results: list[tuple[str, float, float, int]] = []  # (name, mae, dir_acc, count)

    # --- Batch and v1 runs (have per-step prediction_error and directional_accuracy) ---
    for run_name, (label, log_path, npz_path) in {**BATCH_RUNS, **V1_RUNS}.items():
        print(f"\n  --- {run_name} (step-level predictions) ---")
        if not log_path.exists():
            print("  (log file not found)")
            continue

        step_logs = _load_step_logs(log_path)
        if not step_logs:
            print("  (empty or unexpected format)")
            continue

        steps: list[int] = []
        pred_errors: list[list[float]] = []
        dir_accs: list[list[bool]] = []

        for entry in step_logs:
            pe = entry.get("prediction_error")
            da = entry.get("directional_accuracy")
            if pe is not None and da is not None:
                steps.append(entry["step"])
                pred_errors.append(pe)
                dir_accs.append(da)

        if not steps:
            print("  No prediction data found in log.")
            continue

        errors = np.array(pred_errors)
        accuracies = np.array(dir_accs, dtype=float)
        abs_errors = np.abs(errors)
        output_names = ("Y1", "Y2", "Y3", "Y4")

        print(f"  Steps with predictions: {len(steps)}")
        print(f"\n  {'Output':<8} {'MAE':>8} {'Dir Acc':>10}")
        print(f"  {'-' * 30}")
        for j, oname in enumerate(output_names):
            mae = float(abs_errors[:, j].mean())
            acc = float(accuracies[:, j].mean())
            print(f"  {oname:<8} {mae:>8.4f} {acc:>10.1%}")

        overall_mae = float(abs_errors.mean())
        overall_acc = float(accuracies.mean())
        print(f"  {'Overall':<8} {overall_mae:>8.4f} {overall_acc:>10.1%}")
        all_results.append((run_name, overall_mae, overall_acc, len(steps)))

    # --- Tool-use runs (extract from evaluate_point calls) ---
    for run_name, (label, log_path, npz_path) in TOOL_USE_RUNS.items():
        print(f"\n  --- {run_name} (evaluate_point predictions) ---")
        if not log_path.exists():
            print("  (log file not found)")
            continue

        oracle = _get_oracle(label)
        tool_calls = _load_tool_calls(log_path)

        # Extract evaluate_point calls with predicted_outputs
        predictions: list[list[float]] = []
        actuals: list[npt.NDArray[np.float64]] = []

        for tc in tool_calls:
            if tc.get("name") != "evaluate_point":
                continue
            inp = tc.get("input", {})
            pred = inp.get("predicted_outputs")
            point = inp.get("point")
            if pred is None or point is None:
                continue

            actual = oracle.evaluate(np.array(point, dtype=np.float64))
            predictions.append(pred)
            actuals.append(actual)

        if not predictions:
            print("  No evaluate_point calls with predictions found.")
            all_results.append((run_name, float("nan"), float("nan"), 0))
            continue

        pred_arr = np.array(predictions)
        actual_arr = np.array(actuals)
        errors = pred_arr - actual_arr
        abs_errors = np.abs(errors)
        output_names = list(oracle.output_names)

        # Directional accuracy: prediction correctly indicates direction of change
        # relative to mean of actuals (simplified: just check sign agreement of error)
        # More meaningful: for each point, did the prediction correctly rank outputs
        # relative to the running mean? Use simpler metric: |pred - actual| and
        # whether pred and actual are on the same side of the running mean of actuals.
        # Simplest robust approach: directional accuracy = sign(pred - prev_actual) == sign(actual - prev_actual)
        # But we don't have prev_actual in sequence. Use the raw MAE and skip directional for tool-use.

        print(f"  evaluate_point calls with predictions: {len(predictions)}")
        print(f"\n  {'Output':<8} {'MAE':>8}")
        print(f"  {'-' * 20}")
        for j, oname in enumerate(output_names):
            mae = float(abs_errors[:, j].mean())
            print(f"  {oname:<8} {mae:>8.4f}")

        overall_mae = float(abs_errors.mean())
        print(f"  {'Overall':<8} {overall_mae:>8.4f}")

        # Also extract directional accuracy from OAT sweep predictions
        sweep_correct = 0
        sweep_total = 0
        for tc in tool_calls:
            if tc.get("name") != "oat_sweep":
                continue
            inp = tc.get("input", {})
            predicted_trend = str(inp.get("predicted_trend", "")).lower()
            input_name = inp.get("input_name", "")
            n_levels = int(inp.get("n_levels", 5))
            base_point = inp.get("base_point", [])
            if not base_point or not input_name:
                continue

            # Replay sweep to get actual directions
            Y_sweep = _replay_oat_sweep(oracle, input_name, n_levels, base_point)
            for j, oname in enumerate(oracle.output_names):
                actual_change = float(Y_sweep[-1, j] - Y_sweep[0, j])
                if abs(actual_change) < 0.02:
                    actual_dir = "flat"
                elif actual_change > 0:
                    actual_dir = "increase"
                else:
                    actual_dir = "decrease"

                # Parse predicted direction from text
                oname_lower = oname.lower()
                # Look for "Y1: increases" or "Y1 increases" patterns
                pred_dir = "unknown"
                for pattern in [f"{oname_lower}: increase", f"{oname_lower} increase",
                                f"{oname_lower}: rise", f"{oname_lower} rise",
                                f"{oname_lower}: positive", f"{oname_lower}: monot"]:
                    if pattern in predicted_trend:
                        pred_dir = "increase"
                        break
                for pattern in [f"{oname_lower}: decrease", f"{oname_lower} decrease",
                                f"{oname_lower}: drop", f"{oname_lower}: negative"]:
                    if pattern in predicted_trend:
                        pred_dir = "decrease"
                        break
                for pattern in [f"{oname_lower}: flat", f"{oname_lower}: roughly flat",
                                f"{oname_lower}: no effect", f"{oname_lower}: negligible"]:
                    if pattern in predicted_trend:
                        pred_dir = "flat"
                        break

                if pred_dir != "unknown":
                    sweep_total += 1
                    if pred_dir == actual_dir:
                        sweep_correct += 1

        if sweep_total > 0:
            sweep_acc = sweep_correct / sweep_total
            print(f"\n  OAT sweep directional accuracy: {sweep_correct}/{sweep_total} = {sweep_acc:.1%}")
            all_results.append((run_name, overall_mae, sweep_acc, len(predictions) + sweep_total))
        else:
            all_results.append((run_name, overall_mae, float("nan"), len(predictions)))

    # --- Summary table ---
    print(f"\n  {'='*60}")
    print(f"  Prediction Accuracy Summary")
    print(f"  {'='*60}")
    print(f"  {'Run':<22} {'#Preds':>8} {'MAE':>8} {'Dir Acc':>10}")
    print(f"  {'-' * 52}")
    for name, mae, dir_acc, count in all_results:
        mae_str = f"{mae:.4f}" if not np.isnan(mae) else "N/A"
        acc_str = f"{dir_acc:.1%}" if not np.isnan(dir_acc) else "N/A"
        print(f"  {name:<22} {count:>8} {mae_str:>8} {acc_str:>10}")


def plot_prediction_accuracy() -> None:
    """Plot rolling prediction accuracy for batch and v1 runs."""
    print("\n" + "=" * 70)
    print("  6b. Prediction Accuracy Plot")
    print("=" * 70)

    # Collect runs with per-step prediction data
    plot_runs: list[tuple[str, list[int], npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []

    for run_name, (_label, log_path, _npz_path) in {**BATCH_RUNS, **V1_RUNS}.items():
        if not log_path.exists():
            continue
        step_logs = _load_step_logs(log_path)
        if not step_logs:
            continue

        steps: list[int] = []
        pred_errors: list[list[float]] = []
        dir_accs: list[list[bool]] = []

        for entry in step_logs:
            pe = entry.get("prediction_error")
            da = entry.get("directional_accuracy")
            if pe is not None and da is not None:
                steps.append(entry["step"])
                pred_errors.append(pe)
                dir_accs.append(da)

        if steps:
            plot_runs.append((
                run_name,
                steps,
                np.abs(np.array(pred_errors)),
                np.array(dir_accs, dtype=float),
            ))

    if not plot_runs:
        print("  No runs with per-step prediction data found.")
        return

    n_runs = len(plot_runs)
    fig, axes = plt.subplots(n_runs, 2, figsize=(14, 5 * n_runs), squeeze=False)
    output_names = ("Y1", "Y2", "Y3", "Y4")
    output_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for row, (run_name, steps, abs_errors, accuracies) in enumerate(plot_runs):
        n_steps = len(steps)
        window = max(5, n_steps // 10)
        step_arr = np.array(steps) + 1  # 1-indexed

        # Left: rolling MAE per output
        ax_mae = axes[row, 0]
        for j, (oname, color) in enumerate(zip(output_names, output_colors)):
            roll = rolling_mean(abs_errors[:, j], window)
            ax_mae.plot(step_arr, roll, color=color, linewidth=1.5, label=oname)
        ax_mae.set_ylabel("Rolling Mean |Prediction Error|")
        ax_mae.set_title(f"{run_name}: Prediction Error (window={window})")
        ax_mae.legend(fontsize=8)
        ax_mae.grid(True, alpha=0.3)

        # Right: rolling directional accuracy per output
        ax_acc = axes[row, 1]
        for j, (oname, color) in enumerate(zip(output_names, output_colors)):
            roll = rolling_mean(accuracies[:, j], window)
            ax_acc.plot(step_arr, roll, color=color, linewidth=1.5, label=oname)
        overall_roll = rolling_mean(accuracies.mean(axis=1), window)
        ax_acc.plot(step_arr, overall_roll, "k--", linewidth=2, label="Overall")
        ax_acc.set_ylabel("Rolling Directional Accuracy")
        ax_acc.set_title(f"{run_name}: Directional Accuracy (window={window})")
        ax_acc.set_ylim(-0.05, 1.05)
        ax_acc.legend(fontsize=8)
        ax_acc.grid(True, alpha=0.3)

        if row == n_runs - 1:
            ax_mae.set_xlabel("Step")
            ax_acc.set_xlabel("Step")

    plt.tight_layout()
    out_path = OUT_DIR / "prediction_accuracy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out_path}")


# ======================================================================
# 7. Surprise Analysis
# ======================================================================

SURPRISE_THRESHOLD = 0.05  # max |pred_error| per output to flag as surprise


def _compute_tool_surprises(
    oracle: MediumOracle | MediumOracle1C,
    tool_calls: list[dict[str, Any]],
    initial_evals: int = 12,
) -> list[dict[str, Any]]:
    """Extract surprise events from tool-use agent runs.

    Returns list of surprise dicts with: eval_count, tool_name, type, magnitude.
    """
    surprises: list[dict[str, Any]] = []
    cumulative_cost = initial_evals

    for tc in tool_calls:
        name = tc.get("name", "")
        inp = tc.get("input", {})
        result = tc.get("result", {})
        cost = int(tc.get("cost", 0))

        if name == "evaluate_point":
            # Check prediction errors from result (new format) or replay
            pred_errors = result.get("prediction_errors", {}) if result else {}
            if not pred_errors:
                # Fall back: replay from input if result not saved
                raw_pred = inp.get("predicted_outputs", [])
                raw_point = inp.get("point", [])
                if raw_pred and raw_point:
                    actual = oracle.evaluate(np.array(raw_point, dtype=np.float64))
                    pred = np.array(raw_pred, dtype=np.float64)
                    if len(pred) == oracle.n_outputs:
                        pred_errors = {
                            oname: float(actual[i] - pred[i])
                            for i, oname in enumerate(oracle.output_names)
                        }

            if pred_errors:
                max_err = max(abs(float(v)) for v in pred_errors.values())
                if max_err > SURPRISE_THRESHOLD:
                    surprises.append({
                        "eval": cumulative_cost + cost,
                        "tool": "evaluate_point",
                        "type": "prediction_error",
                        "magnitude": round(max_err, 4),
                        "details": pred_errors,
                    })

        elif name == "oat_sweep":
            # Try structured trend_scores first (new format), then free-text
            trend_scores = (result.get("trend_scores", {})
                            if isinstance(result, dict) else {})
            if trend_scores:
                # Structured scoring — each output has direction_correct flag
                mismatches: list[str] = []
                mag_errors: list[float] = []
                for oname, score in trend_scores.items():
                    if not isinstance(score, dict):
                        continue
                    if not score.get("direction_correct", True):
                        mismatches.append(
                            f"{oname}: predicted {score.get('predicted_direction')}, "
                            f"actual {score.get('actual_direction')}"
                        )
                    mag_errors.append(float(score.get("magnitude_error", 0.0)))

                if mismatches:
                    surprises.append({
                        "eval": cumulative_cost + cost,
                        "tool": "oat_sweep",
                        "type": "trend_mismatch",
                        "magnitude": len(mismatches),
                        "details": mismatches,
                        "mean_mag_error": round(
                            sum(mag_errors) / len(mag_errors), 4
                        ) if mag_errors else 0.0,
                    })
            else:
                # Fall back to free-text parsing (old format)
                predicted_trend = str(inp.get("predicted_trend", "")).lower()
                input_name = inp.get("input_name", "")
                n_levels = int(inp.get("n_levels", 5))
                base_point = inp.get("base_point", [])
                if predicted_trend and input_name and base_point:
                    Y_sweep = _replay_oat_sweep(
                        oracle, input_name, n_levels, base_point,
                    )
                    mismatches = []
                    for j, oname in enumerate(oracle.output_names):
                        actual_change = float(Y_sweep[-1, j] - Y_sweep[0, j])
                        if abs(actual_change) < 0.02:
                            actual_dir = "flat"
                        elif actual_change > 0:
                            actual_dir = "increase"
                        else:
                            actual_dir = "decrease"

                        oname_lower = oname.lower()
                        pred_dir = "unknown"
                        for pat in [f"{oname_lower}: increase",
                                    f"{oname_lower} increase",
                                    f"{oname_lower}: rise",
                                    f"{oname_lower}: positive"]:
                            if pat in predicted_trend:
                                pred_dir = "increase"
                                break
                        for pat in [f"{oname_lower}: decrease",
                                    f"{oname_lower} decrease",
                                    f"{oname_lower}: drop",
                                    f"{oname_lower}: negative"]:
                            if pat in predicted_trend:
                                pred_dir = "decrease"
                                break
                        for pat in [f"{oname_lower}: flat",
                                    f"{oname_lower}: no effect",
                                    f"{oname_lower}: negligible"]:
                            if pat in predicted_trend:
                                pred_dir = "flat"
                                break

                        if pred_dir != "unknown" and pred_dir != actual_dir:
                            mismatches.append(
                                f"{oname}: predicted {pred_dir}, actual {actual_dir}"
                            )

                    if mismatches:
                        surprises.append({
                            "eval": cumulative_cost + cost,
                            "tool": "oat_sweep",
                            "type": "trend_mismatch",
                            "magnitude": len(mismatches),
                            "details": mismatches,
                        })

        cumulative_cost += cost

    return surprises


def _compute_batch_surprises(
    step_logs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract surprise events from batch agent step logs.

    Returns list of surprise dicts with: step, magnitude, biggest_surprise text.
    """
    surprises: list[dict[str, Any]] = []

    for entry in step_logs:
        step = entry.get("step", -1)
        pred_error = entry.get("prediction_error")
        biggest_surprise = entry.get("biggest_surprise", "")

        if pred_error is not None:
            abs_errors = [abs(float(e)) for e in pred_error]
            max_err = max(abs_errors) if abs_errors else 0.0
            if max_err > SURPRISE_THRESHOLD:
                surprises.append({
                    "step": step,
                    "type": "prediction_error",
                    "magnitude": round(max_err, 4),
                    "biggest_surprise": biggest_surprise[:200] if biggest_surprise else "",
                })

    return surprises


def compute_surprise_analysis() -> None:
    """Analyze surprise patterns across all runs.

    Measures: surprise frequency, surprise rate by phase (early/mid/late),
    whether surprise rate decreases over time (evidence of learning).
    """
    print("\n" + "=" * 70)
    print("  7. Surprise Analysis")
    print("=" * 70)

    # Summary table header
    summary: list[tuple[str, int, int, float, float, float]] = []
    # (name, n_events, n_total, rate_early, rate_mid, rate_late)

    # --- Tool-use runs ---
    for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
        if not log_path.exists():
            continue

        oracle = _get_oracle(label)
        tool_calls = _load_tool_calls(log_path)
        surprises = _compute_tool_surprises(oracle, tool_calls)

        # Count total prediction-bearing tool calls (evaluate_point + oat_sweep with predictions)
        pred_calls = [tc for tc in tool_calls
                      if tc.get("name") in ("evaluate_point", "oat_sweep")
                      and (tc.get("input", {}).get("predicted_outputs")
                           or tc.get("input", {}).get("predicted_trend"))]
        n_pred = len(pred_calls)

        if not surprises:
            summary.append((run_name, 0, n_pred, 0.0, 0.0, 0.0))
            continue

        # Bin by phase: early (first third of evals), mid, late
        max_eval = max(s["eval"] for s in surprises)
        third = max_eval / 3
        early = [s for s in surprises if s["eval"] <= third]
        mid = [s for s in surprises if third < s["eval"] <= 2 * third]
        late = [s for s in surprises if s["eval"] > 2 * third]

        # Rate = surprise events per phase (normalized by phase count if possible)
        n_early = max(1, sum(1 for tc in tool_calls if int(tc.get("cost", 0)) > 0
                             and _cumcost(tool_calls, tc, 12) <= third))
        n_mid = max(1, sum(1 for tc in tool_calls if int(tc.get("cost", 0)) > 0
                           and third < _cumcost(tool_calls, tc, 12) <= 2 * third))
        n_late = max(1, sum(1 for tc in tool_calls if int(tc.get("cost", 0)) > 0
                            and _cumcost(tool_calls, tc, 12) > 2 * third))

        summary.append((
            run_name, len(surprises), n_pred,
            len(early) / n_early, len(mid) / n_mid, len(late) / n_late,
        ))

    # --- Batch runs ---
    for run_name, (label, log_path, _npz_path) in {**BATCH_RUNS, **V1_RUNS}.items():
        if not log_path.exists():
            continue

        step_logs = _load_step_logs(log_path)
        if not step_logs:
            continue

        surprises = _compute_batch_surprises(step_logs)
        n_total = len(step_logs)

        if not surprises:
            summary.append((run_name, 0, n_total, 0.0, 0.0, 0.0))
            continue

        # Bin by phase thirds
        third = n_total / 3
        early = [s for s in surprises if s["step"] < third]
        mid = [s for s in surprises if third <= s["step"] < 2 * third]
        late = [s for s in surprises if s["step"] >= 2 * third]

        n_early = max(1, int(third))
        n_mid = max(1, int(third))
        n_late = max(1, n_total - 2 * int(third))

        summary.append((
            run_name, len(surprises), n_total,
            len(early) / n_early, len(mid) / n_mid, len(late) / n_late,
        ))

    # Print summary table
    print(f"\n  {'Run':<22} {'Surprises':>10} {'Total':>7} {'Early':>8} {'Mid':>8} {'Late':>8} {'Trend':>10}")
    print("  " + "-" * 80)

    for name, n_surp, n_total, r_early, r_mid, r_late in summary:
        # Determine trend
        if r_early > 0 and r_late > 0:
            if r_late < r_early * 0.7:
                trend = "LEARNING"
            elif r_late > r_early * 1.3:
                trend = "DEGRADING"
            else:
                trend = "FLAT"
        elif r_early == 0 and r_late == 0:
            trend = "NO DATA"
        else:
            trend = "---"

        print(
            f"  {name:<22} {n_surp:>10} {n_total:>7} "
            f"{r_early:>8.2f} {r_mid:>8.2f} {r_late:>8.2f} {trend:>10}"
        )

    # Print detailed surprise logs for batch runs that have biggest_surprise text
    for run_name, (label, log_path, _npz_path) in BATCH_RUNS.items():
        if not log_path.exists():
            continue
        step_logs = _load_step_logs(log_path)
        surprises = _compute_batch_surprises(step_logs)
        notable = [s for s in surprises if s.get("biggest_surprise")]
        if notable:
            print(f"\n  --- {run_name}: Notable surprises ---")
            for s in notable[:10]:
                print(f"    Step {s['step']:>3} (mag={s['magnitude']:.3f}): "
                      f"{s['biggest_surprise'][:100]}")
            if len(notable) > 10:
                print(f"    ... ({len(notable) - 10} more)")


def _cumcost(
    tool_calls: list[dict[str, Any]], tc: dict[str, Any], initial: int,
) -> float:
    """Compute cumulative eval cost up to and including a given tool call."""
    total = initial
    for t in tool_calls:
        total += int(t.get("cost", 0))
        if t is tc:
            return float(total)
    return float(total)


def compute_calibration_analysis() -> None:
    """Analyze calibration checkpoint data from tool-use runs."""
    print("\n" + "=" * 70)
    print("  7c. Calibration Checkpoints")
    print("=" * 70)

    found_any = False

    for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
        if not log_path.exists():
            continue

        with open(log_path) as f:
            log = json.load(f)

        cal_checks = log.get("calibration_checks", []) if isinstance(log, dict) else []
        if not cal_checks:
            continue

        found_any = True
        print(f"\n  --- {run_name} ({len(cal_checks)} checkpoints) ---")
        print(f"  {'Eval':>6} {'MAE':>8} {'Max|err|':>10} {'Per-output errors'}")
        print(f"  {'-' * 60}")

        for check in cal_checks:
            eval_n = check.get("eval", "?")
            mae = check.get("mae", float("nan"))
            max_err = check.get("max_error", float("nan"))
            errors = check.get("errors", [])
            if math.isnan(mae):
                print(f"  {eval_n:>6} {'FAILED':>8} {'---':>10} (parse failure)")
            else:
                err_str = ", ".join(f"{e:.4f}" for e in errors)
                print(f"  {eval_n:>6} {mae:>8.4f} {max_err:>10.4f} [{err_str}]")

        # Trend assessment — only from successful checkpoints
        valid_maes = [
            c.get("mae", float("nan")) for c in cal_checks
            if not math.isnan(c.get("mae", float("nan")))
        ]
        if len(valid_maes) >= 2:
            first_mae = valid_maes[0]
            last_mae = valid_maes[-1]
            if last_mae < first_mae * 0.7:
                trend = "LEARNING (MAE decreased)"
            elif last_mae > first_mae * 1.3:
                trend = "DEGRADING (MAE increased)"
            else:
                trend = "STABLE"
            print(f"  Trend: {trend} ({first_mae:.4f} -> {last_mae:.4f})")
        elif len(valid_maes) == 1:
            print(f"  Trend: single checkpoint (MAE={valid_maes[0]:.4f})")

    if not found_any:
        print("\n  No calibration checkpoint data found in any run.")
        print("  (Calibration checkpoints are available in runs using the updated agent.)")


def plot_surprise_trajectory() -> None:
    """Plot surprise magnitude over time overlaid with HV curve for 1A runs."""
    print("\n" + "=" * 70)
    print("  7b. Surprise Trajectory Plot")
    print("=" * 70)

    # Collect 1A runs with surprise data
    plot_data: list[tuple[str, list[float], list[float], list[float]]] = []
    # (name, eval_indices, surprise_magnitudes, hv_curve)

    # Batch runs
    for run_name, (label, log_path, npz_path) in {**BATCH_RUNS, **V1_RUNS}.items():
        if label != "1A" or not log_path.exists() or not npz_path.exists():
            continue
        step_logs = _load_step_logs(log_path)
        surprises = _compute_batch_surprises(step_logs)
        if not surprises:
            continue
        data = np.load(npz_path)
        hvs = data["hypervolumes"].tolist()
        # Map step index to eval index (step 0 = eval n_initial+1)
        n_initial = 12
        evals = [float(n_initial + s["step"] + 1) for s in surprises]
        mags = [s["magnitude"] for s in surprises]
        plot_data.append((run_name, evals, mags, hvs))

    # Tool runs
    for run_name, (label, log_path, npz_path) in TOOL_USE_RUNS.items():
        if label != "1A" or not log_path.exists() or not npz_path.exists():
            continue
        oracle = _get_oracle(label)
        tool_calls = _load_tool_calls(log_path)
        surprises = _compute_tool_surprises(oracle, tool_calls)
        if not surprises:
            continue
        data = np.load(npz_path)
        hvs = data["hypervolumes"].tolist()
        evals = [float(s["eval"]) for s in surprises]
        mags = [float(s["magnitude"]) if isinstance(s["magnitude"], (int, float)) else 1.0
                for s in surprises]
        plot_data.append((run_name, evals, mags, hvs))

    if not plot_data:
        print("  No 1A runs with surprise data found.")
        return

    n_runs = len(plot_data)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, 4 * n_runs), squeeze=False)

    for row, (name, evals_s, mags, hvs) in enumerate(plot_data):
        ax = axes[row, 0]

        # HV curve on primary axis
        hv_evals = np.arange(1, len(hvs) + 1)
        ax.plot(hv_evals, hvs, color="#1f77b4", linewidth=1.5, label="HV")
        ax.set_ylabel("Hypervolume", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")

        # Surprise markers on secondary axis
        ax2 = ax.twinx()
        ax2.scatter(evals_s, mags, color="#d62728", s=30, alpha=0.7,
                    label="Surprise", zorder=5)
        ax2.set_ylabel("Surprise magnitude", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        ax.set_title(f"{name}: Surprises vs HV")
        ax.set_xlabel("Oracle Evaluations")
        ax.axvline(x=12, color="gray", linestyle="--", alpha=0.2)
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / "surprise_trajectory.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out_path}")


# ======================================================================
# 8. Causal Model Convergence
# ======================================================================

CONFIDENCE_THRESHOLD = 0.5  # edges with confidence >= this are "claimed"


def compute_causal_model_convergence() -> None:
    """Track per-iteration edge precision/recall from structured iteration summaries.

    For each tool-use run with iteration_summaries, extract the confidence dict,
    threshold at CONFIDENCE_THRESHOLD, and score against ground truth IO edges.
    """
    print("\n" + "=" * 70)
    print("  8. Causal Model Convergence")
    print("=" * 70)

    found_any = False

    for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
        if not log_path.exists():
            continue

        with open(log_path) as f:
            log = json.load(f)

        summaries = log.get("iteration_summaries", []) if isinstance(log, dict) else []
        if not summaries:
            continue

        found_any = True
        oracle = _get_oracle(label)
        gt_io = oracle.ground_truth().project_to_io()
        gt_pairs = {(e.source, e.target) for e in gt_io.edges}

        print(f"\n  --- {run_name} ({len(summaries)} iterations) ---")
        print(f"  {'Iter':>5} {'Evals':>6} {'Claimed':>8} {'TP':>4} {'FP':>4} "
              f"{'Prec':>6} {'Recall':>7} {'#Edges':>7}")
        print(f"  {'-' * 55}")

        for s in summaries:
            confidence = s.get("confidence", {})
            # Threshold to get claimed edges
            claimed = set()
            for edge_str, conf in confidence.items():
                if not isinstance(conf, (int, float)):
                    continue
                if conf >= CONFIDENCE_THRESHOLD:
                    # Parse "Xi->Yj" format
                    parts = edge_str.replace(" ", "").split("->")
                    if len(parts) == 2:
                        claimed.add((parts[0], parts[1]))
                    # Also handle "Xi*Xj->Yk" interaction format
                    elif "*" in edge_str and "->" in edge_str:
                        interaction_part, target = edge_str.split("->")
                        for src in interaction_part.split("*"):
                            claimed.add((src.strip(), target.strip()))

            tp = len(gt_pairs & claimed)
            fp = len(claimed - gt_pairs)
            prec = tp / len(claimed) if claimed else 0.0
            rec = tp / len(gt_pairs) if gt_pairs else 0.0

            print(
                f"  {s.get('iteration', '?'):>5} {s.get('eval_count', '?'):>6} "
                f"{len(claimed):>8} {tp:>4} {fp:>4} "
                f"{prec:>6.3f} {rec:>7.3f} {len(confidence):>7}"
            )

        # Confidence calibration: bin all edges by confidence, check GT fraction
        print(f"\n  Confidence calibration:")
        bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
        # Use last iteration's confidence
        if summaries:
            last_conf = summaries[-1].get("confidence", {})
            for lo_b, hi_b in bins:
                edges_in_bin = []
                for edge_str, conf in last_conf.items():
                    if not isinstance(conf, (int, float)):
                        continue
                    if lo_b <= conf < hi_b:
                        parts = edge_str.replace(" ", "").split("->")
                        if len(parts) == 2:
                            is_true = (parts[0], parts[1]) in gt_pairs
                            edges_in_bin.append(is_true)
                if edges_in_bin:
                    frac_true = sum(edges_in_bin) / len(edges_in_bin)
                    print(f"    [{lo_b:.2f}, {hi_b:.2f}): {len(edges_in_bin)} edges, "
                          f"{frac_true:.0%} true")
                else:
                    print(f"    [{lo_b:.2f}, {hi_b:.2f}): no edges")

    if not found_any:
        print("\n  No iteration summary data found in any run.")
        print("  (Iteration summaries require the updated structured agent.)")


def plot_causal_convergence() -> None:
    """Plot per-iteration precision/recall for runs with iteration summaries."""
    print("\n" + "=" * 70)
    print("  8b. Causal Convergence Plot")
    print("=" * 70)

    plot_data: list[tuple[str, list[int], list[float], list[float]]] = []

    for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
        if not log_path.exists():
            continue

        with open(log_path) as f:
            log = json.load(f)

        summaries = log.get("iteration_summaries", []) if isinstance(log, dict) else []
        if not summaries:
            continue

        oracle = _get_oracle(label)
        gt_io = oracle.ground_truth().project_to_io()
        gt_pairs = {(e.source, e.target) for e in gt_io.edges}

        evals_list: list[int] = []
        prec_list: list[float] = []
        rec_list: list[float] = []

        for s in summaries:
            confidence = s.get("confidence", {})
            claimed = set()
            for edge_str, conf in confidence.items():
                if not isinstance(conf, (int, float)):
                    continue
                if conf >= CONFIDENCE_THRESHOLD:
                    parts = edge_str.replace(" ", "").split("->")
                    if len(parts) == 2:
                        claimed.add((parts[0], parts[1]))

            tp = len(gt_pairs & claimed)
            prec = tp / len(claimed) if claimed else 0.0
            rec = tp / len(gt_pairs) if gt_pairs else 0.0

            evals_list.append(int(s.get("eval_count", 0)))
            prec_list.append(prec)
            rec_list.append(rec)

        plot_data.append((run_name, evals_list, prec_list, rec_list))

    if not plot_data:
        print("  No data for causal convergence plot.")
        return

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    for name, evals_l, prec_l, rec_l in plot_data:
        ax_p.plot(evals_l, prec_l, "o-", linewidth=1.5, markersize=6, label=name)
        ax_r.plot(evals_l, rec_l, "o-", linewidth=1.5, markersize=6, label=name)

    ax_p.set_xlabel("Oracle Evaluations")
    ax_p.set_ylabel("Precision")
    ax_p.set_title("Causal Model Precision (claimed edges)")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.legend(fontsize=7)
    ax_p.grid(True, alpha=0.3)

    ax_r.set_xlabel("Oracle Evaluations")
    ax_r.set_ylabel("Recall")
    ax_r.set_title("Causal Model Recall (GT edges found)")
    ax_r.set_ylim(-0.05, 1.05)
    ax_r.legend(fontsize=7)
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "causal_convergence.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out_path}")


# ======================================================================
# 8c. Adversarial Prediction Analysis
# ======================================================================


def compute_adversarial_prediction_analysis() -> None:
    """Compare prediction errors in adversarial vs non-adversarial regions.

    Uses oracle.adversarial_regions() so the analysis is oracle-driven,
    not hardcoded to any specific oracle variant.
    """
    print("\n" + "=" * 70)
    print("  8c. Adversarial Prediction Analysis")
    print("=" * 70)

    found_any = False

    for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
        if not log_path.exists():
            continue

        oracle = _get_oracle(label)
        regions = oracle.adversarial_regions()
        if not regions:
            continue

        tool_calls = _load_tool_calls(log_path)
        eval_pts = [tc for tc in tool_calls if tc.get("name") == "evaluate_point"]
        if not eval_pts:
            continue

        # Collect (point, max_error) pairs with region labels
        region_errors: dict[str, list[float]] = {r["name"]: [] for r in regions}
        region_errors["non-adversarial"] = []

        for tc in eval_pts:
            inp = tc.get("input", {})
            result = tc.get("result", {})
            point = inp.get("point", [])
            if not point:
                continue

            # Get prediction error
            pred_errors = result.get("prediction_errors", {}) if isinstance(result, dict) else {}
            if not pred_errors:
                # Replay from input
                raw_pred = inp.get("predicted_outputs", [])
                if raw_pred and len(raw_pred) == oracle.n_outputs:
                    actual = oracle.evaluate(np.array(point, dtype=np.float64))
                    pred = np.array(raw_pred, dtype=np.float64)
                    pred_errors = {
                        oname: float(actual[i] - pred[i])
                        for i, oname in enumerate(oracle.output_names)
                    }
            if not pred_errors:
                continue

            max_err = max(abs(float(v)) for v in pred_errors.values())
            x = np.array(point, dtype=np.float64)

            # Classify into regions
            in_any = False
            for r in regions:
                test_fn = r["test"]
                try:
                    if test_fn(x):
                        region_errors[str(r["name"])].append(max_err)
                        in_any = True
                except (IndexError, ZeroDivisionError):
                    pass
            if not in_any:
                region_errors["non-adversarial"].append(max_err)

        # Print if we have data
        has_data = any(len(errs) > 0 for errs in region_errors.values())
        if not has_data:
            continue

        found_any = True
        print(f"\n  --- {run_name} ---")
        print(f"  {'Region':<25} {'#Pts':>5} {'MAE':>8} {'Max|err|':>10}")
        print(f"  {'-' * 52}")

        for rname in ["non-adversarial"] + [r["name"] for r in regions]:
            errs = region_errors.get(str(rname), [])
            if not errs:
                continue
            mae = sum(errs) / len(errs)
            max_e = max(errs)
            print(f"  {rname:<25} {len(errs):>5} {mae:>8.4f} {max_e:>10.4f}")

    if not found_any:
        print("\n  No evaluate_point data with adversarial regions found.")


# ======================================================================
# 8d. Confidence Calibration Over Time
# ======================================================================


def compute_calibration_over_time() -> None:
    """Track confidence calibration per iteration.

    For each iteration, bin edges by confidence and compute fraction true.
    Calibration error = mean |bucket_midpoint - fraction_true|.
    """
    print("\n" + "=" * 70)
    print("  8d. Confidence Calibration Over Time")
    print("=" * 70)

    cal_bins = [(0.0, 0.33, "Low"), (0.33, 0.67, "Med"), (0.67, 1.01, "High")]
    found_any = False

    # Collect data for plot
    plot_data: list[tuple[str, list[int], list[float]]] = []

    for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
        if not log_path.exists():
            continue

        with open(log_path) as f:
            log = json.load(f)
        summaries = log.get("iteration_summaries", []) if isinstance(log, dict) else []
        if not summaries:
            continue

        oracle = _get_oracle(label)
        gt_io = oracle.ground_truth().project_to_io()
        gt_pairs = {(e.source, e.target) for e in gt_io.edges}

        if not found_any:
            print(f"\n  {'Run':<22} {'Iter':>5} {'Evals':>6} {'CalErr':>7} "
                  f"{'Low':>8} {'Med':>8} {'High':>8}")
            print(f"  {'-' * 70}")
            found_any = True

        evals_list: list[int] = []
        cal_errors: list[float] = []

        for s in summaries:
            confidence = s.get("confidence", {})
            if not confidence:
                continue

            bin_results: list[tuple[float, float | None]] = []  # (midpoint, frac_true)
            bin_strs: list[str] = []

            for lo_b, hi_b, bin_name in cal_bins:
                midpoint = (lo_b + hi_b) / 2
                edges_in_bin: list[bool] = []
                for edge_str, conf in confidence.items():
                    if not isinstance(conf, (int, float)):
                        continue
                    if lo_b <= conf < hi_b:
                        parts = edge_str.replace(" ", "").split("->")
                        if len(parts) == 2:
                            edges_in_bin.append((parts[0], parts[1]) in gt_pairs)

                if edges_in_bin:
                    frac = sum(edges_in_bin) / len(edges_in_bin)
                    bin_results.append((midpoint, frac))
                    bin_strs.append(f"{frac:.0%}({len(edges_in_bin)})")
                else:
                    bin_results.append((midpoint, None))
                    bin_strs.append("---")

            # Calibration error: mean |midpoint - frac_true| for bins with data
            valid = [(m, f) for m, f in bin_results if f is not None]
            cal_err = (sum(abs(m - f) for m, f in valid) / len(valid)
                       if valid else float("nan"))

            evals_list.append(int(s.get("eval_count", 0)))
            cal_errors.append(cal_err)

            print(f"  {run_name:<22} {s.get('iteration', '?'):>5} "
                  f"{s.get('eval_count', '?'):>6} {cal_err:>7.3f} "
                  f"{bin_strs[0]:>8} {bin_strs[1]:>8} {bin_strs[2]:>8}")

        if evals_list:
            plot_data.append((run_name, evals_list, cal_errors))

    if not found_any:
        print("\n  No iteration summary data for calibration tracking.")
        return

    # Plot
    if plot_data:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for name, evals_l, errors in plot_data:
            ax.plot(evals_l, errors, "o-", linewidth=2, markersize=6, label=name)
        ax.set_xlabel("Oracle Evaluations")
        ax.set_ylabel("Calibration Error (lower = better)")
        ax.set_title("Confidence Calibration Over Time")
        ax.set_ylim(-0.05, 0.6)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, label="Perfect")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = OUT_DIR / "calibration_over_time.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved to {out_path}")


# ======================================================================
# 9. Information-Theoretic Analysis (T6)
# ======================================================================


def _compute_sobol_indices_cached(
    oracle: MediumOracle | MediumOracle1C,
    label: str,
) -> npt.NDArray[np.float64]:
    """Get total-order Sobol indices, computing and caching if needed.

    Returns shape (n_inputs, n_outputs).
    """
    cache_path = ANALYSIS_DIR / f"sobol_total_{label}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    from synthoracle.characterize import compute_sobol_indices
    rng = np.random.default_rng(42)
    _, st = compute_sobol_indices(oracle, 50_000, rng)
    np.save(cache_path, st)
    return st


def _parse_edge_to_inputs(edge_str: str) -> tuple[list[str], str]:
    """Parse 'Xi->Yj' or 'Xi*Xj->Yk' to (input_names, output_name)."""
    edge_str = edge_str.replace(" ", "")
    if "->" not in edge_str:
        return [], ""
    src_part, tgt = edge_str.split("->", 1)
    # Handle interaction: "X2*X4" -> ["X2", "X4"]
    inputs = [s.strip() for s in src_part.split("*") if s.strip()]
    return inputs, tgt.strip()


def _compute_agent_info_capture(
    confidence: dict[str, object],
    sobol_total: npt.NDArray[np.float64],
    oracle: MediumOracle | MediumOracle1C,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, float]:
    """Compute information capture per output from agent's edge confidence.

    Returns dict mapping output name to fraction of Sobol sensitivity explained
    by the agent's claimed inputs (confidence >= threshold).
    """
    input_names = list(oracle.input_names)
    output_names = list(oracle.output_names)

    # Build claimed input sets per output
    claimed_per_output: dict[str, set[str]] = {o: set() for o in output_names}
    for edge_str, conf in confidence.items():
        if not isinstance(conf, (int, float)) or conf < threshold:
            continue
        inputs, output = _parse_edge_to_inputs(str(edge_str))
        if output in claimed_per_output:
            claimed_per_output[output].update(inputs)

    # Compute capture ratio per output
    capture: dict[str, float] = {}
    for j, oname in enumerate(output_names):
        total_sobol = float(sobol_total[:, j].sum())
        if total_sobol < 1e-10:
            capture[oname] = 1.0  # output has no sensitivity → trivially captured
            continue
        claimed_sobol = 0.0
        for i, iname in enumerate(input_names):
            if iname in claimed_per_output[oname]:
                claimed_sobol += float(sobol_total[i, j])
        capture[oname] = min(1.0, claimed_sobol / total_sobol)

    return capture


def _compute_mechanism_sufficiency(
    oracle: MediumOracle,
    n_samples: int = 100_000,
) -> dict[str, float]:
    """Compute mechanism sufficiency: how much of Y variance is explained by M.

    For each output, fits a nonparametric regression (binned means) of Y on M
    and computes R². For Y1, Y2, Y4 this should be ~1.0 (fully determined by M).
    For Y3 it should be < 1.0 (bypasses mechanism layer).
    """
    rng = np.random.default_rng(42)
    lo, hi = oracle.bounds[:, 0], oracle.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(n_samples, oracle.n_inputs))

    Y_list = []
    M_list = []
    for i in range(n_samples):
        y, m = oracle.evaluate_with_mechanisms(X[i])
        Y_list.append(y)
        M_list.append([m["M1_eff"], m["M2_eff"], m["Z"],
                        m["M4_mult"], m["M4_add"], m["gate"]])

    Y_arr = np.array(Y_list)
    M_arr = np.array(M_list)

    # For each output, compute R² of a polynomial regression on M
    sufficiency: dict[str, float] = {}
    for j, oname in enumerate(oracle.output_names):
        y = Y_arr[:, j]
        # OLS with M columns + pairwise cross terms
        # Build feature matrix: M columns + pairwise products
        n_m = M_arr.shape[1]
        features = [M_arr]
        for a in range(n_m):
            for b in range(a, n_m):
                features.append((M_arr[:, a] * M_arr[:, b]).reshape(-1, 1))
        feat = np.hstack(features)
        feat = np.hstack([feat, np.ones((n_samples, 1))])  # intercept

        # OLS
        coeffs, residuals, _, _ = np.linalg.lstsq(feat, y, rcond=None)
        y_pred = feat @ coeffs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        sufficiency[oname] = round(r2, 6)

    return sufficiency


def compute_information_analysis() -> None:
    """T6: Information-theoretic evaluation of mechanism discovery.

    Computes:
    1. Oracle mechanism sufficiency: R²(M→Y) per output
    2. Agent information capture: Sobol-based proxy per iteration
    3. End-of-run capture for past runs (from edge P/R data)
    """
    print("\n" + "=" * 70)
    print("  9. Information-Theoretic Analysis (T6)")
    print("=" * 70)

    # --- 9a. Oracle mechanism sufficiency ---
    print("\n  9a. Mechanism Sufficiency: R²(M → Y)")
    print(f"  {'Oracle':<10} {'Y1':>8} {'Y2':>8} {'Y3':>8} {'Y4':>8} {'Mean':>8}")
    print(f"  {'-' * 50}")

    for label, oracle, _ in ORACLE_CONFIGS:
        if not hasattr(oracle, "evaluate_with_mechanisms"):
            print(f"  {label:<10} (no mechanism exposure)")
            continue
        suff = _compute_mechanism_sufficiency(oracle)  # type: ignore[arg-type]
        vals = [suff.get(o, 0.0) for o in oracle.output_names]
        mean_r2 = sum(vals) / len(vals)
        print(f"  {label:<10} {vals[0]:>8.4f} {vals[1]:>8.4f} "
              f"{vals[2]:>8.4f} {vals[3]:>8.4f} {mean_r2:>8.4f}")

    # --- 9b. Agent information capture ---
    print("\n  9b. Agent Information Capture (Sobol-based)")

    for oracle_label in ("1A", "1B", "1C"):
        oracle = _get_oracle(oracle_label)
        sobol_total = _compute_sobol_indices_cached(oracle, oracle_label)

        # Runs with iteration summaries (per-iteration trajectory)
        found_iter = False
        for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
            if label != oracle_label or not log_path.exists():
                continue
            with open(log_path) as f:
                log = json.load(f)
            summaries = log.get("iteration_summaries", []) if isinstance(log, dict) else []
            if not summaries:
                continue

            if not found_iter:
                print(f"\n  Oracle {oracle_label} — per-iteration capture:")
                print(f"  {'Run':<22} {'Iter':>5} {'Evals':>6} "
                      f"{'Y1':>6} {'Y2':>6} {'Y3':>6} {'Y4':>6} {'Mean':>6}")
                print(f"  {'-' * 65}")
                found_iter = True

            for s in summaries:
                conf = s.get("confidence", {})
                capture = _compute_agent_info_capture(conf, sobol_total, oracle)
                vals = [capture.get(o, 0.0) for o in oracle.output_names]
                mean_c = sum(vals) / len(vals)
                print(f"  {run_name:<22} {s.get('iteration', '?'):>5} "
                      f"{s.get('eval_count', '?'):>6} "
                      f"{vals[0]:>6.3f} {vals[1]:>6.3f} "
                      f"{vals[2]:>6.3f} {vals[3]:>6.3f} {mean_c:>6.3f}")

        # Runs without iteration summaries — use end-of-run edge data
        for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
            if label != oracle_label or not log_path.exists():
                continue
            with open(log_path) as f:
                log = json.load(f)
            # Skip if already handled above
            if (isinstance(log, dict) and log.get("iteration_summaries")):
                continue
            # Build confidence from discovered edges (all confidence = 1.0)
            tool_calls = _load_tool_calls(log_path)
            disc_edges = _extract_edges_from_tool_calls(oracle, tool_calls)
            conf = {f"{e.source}->{e.target}": 1.0 for e in disc_edges}
            capture = _compute_agent_info_capture(conf, sobol_total, oracle)
            vals = [capture.get(o, 0.0) for o in oracle.output_names]
            mean_c = sum(vals) / len(vals)
            print(f"  {run_name:<22} {'end':>5} {'---':>6} "
                  f"{vals[0]:>6.3f} {vals[1]:>6.3f} "
                  f"{vals[2]:>6.3f} {vals[3]:>6.3f} {mean_c:>6.3f} (from edges)")


def plot_information_capture() -> None:
    """Plot information capture trajectory for runs with iteration summaries."""
    print("\n" + "=" * 70)
    print("  9b. Information Capture Plot")
    print("=" * 70)

    plot_data: list[tuple[str, list[int], list[float]]] = []

    for run_name, (label, log_path, _npz_path) in TOOL_USE_RUNS.items():
        if not log_path.exists():
            continue
        oracle = _get_oracle(label)
        sobol_total = _compute_sobol_indices_cached(oracle, label)

        with open(log_path) as f:
            log = json.load(f)
        summaries = log.get("iteration_summaries", []) if isinstance(log, dict) else []
        if not summaries:
            continue

        evals_list: list[int] = []
        captures: list[float] = []
        for s in summaries:
            conf = s.get("confidence", {})
            capture = _compute_agent_info_capture(conf, sobol_total, oracle)
            mean_c = sum(capture.values()) / len(capture) if capture else 0.0
            evals_list.append(int(s.get("eval_count", 0)))
            captures.append(mean_c)

        plot_data.append((run_name, evals_list, captures))

    if not plot_data:
        print("  No iteration summary data for information capture plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name, evals_l, caps in plot_data:
        ax.plot(evals_l, caps, "o-", linewidth=2, markersize=6, label=name)

    ax.set_xlabel("Oracle Evaluations")
    ax.set_ylabel("Information Capture (Sobol fraction)")
    ax.set_title("T6: Agent Information Capture Over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, label="Perfect capture")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "information_capture.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out_path}")


# ======================================================================
# 10. Multi-Seed Aggregate Analysis
# ======================================================================

MULTI_SEED_DIR = VR_RESULTS / "multi_seed"


def _load_multi_seed_data() -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Load all multi-seed results. Returns list of (seed, log_dict, npz_dict)."""
    results = []
    for log_path in sorted(MULTI_SEED_DIR.glob("seed*_log.json")):
        seed_str = log_path.stem.replace("_log", "").replace("seed", "")
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        npz_path = MULTI_SEED_DIR / f"seed{seed}.npz"
        if not npz_path.exists():
            continue
        with open(log_path) as f:
            log = json.load(f)
        data = dict(np.load(npz_path))
        results.append((seed, log, data))
    return results


def compute_multi_seed_aggregate() -> None:
    """Section 10: Aggregate analysis over all multi-seed runs."""
    print("\n" + "=" * 70)
    print("  10. Multi-Seed Aggregate Analysis (Medium 1A)")
    print("=" * 70)

    seeds_data = _load_multi_seed_data()
    if not seeds_data:
        print("\n  No multi-seed data found.")
        return

    oracle = MediumOracle()
    gt_io = oracle.ground_truth().project_to_io()
    gt_pairs = {(e.source, e.target) for e in gt_io.edges}
    sobol_total = _compute_sobol_indices_cached(oracle, "1A")
    regions = oracle.adversarial_regions()

    n_seeds = len(seeds_data)
    print(f"\n  Seeds loaded: {n_seeds}")

    # --- 10a. HV ---
    final_hvs = [float(d["hypervolumes"][-1]) for _, _, d in seeds_data]
    print(f"\n  10a. Final Hypervolume")
    print(f"  VR tool agent: {np.mean(final_hvs):.4f} +/- {np.std(final_hvs):.4f}")
    print(f"    range: [{np.min(final_hvs):.4f}, {np.max(final_hvs):.4f}]")

    bo_stats = _load_bo_1a_stats()
    if bo_stats is not None:
        bo_mean_final = float(bo_stats[0][-1])
        bo_std_final = float(bo_stats[1][-1])
        print(f"  BO baseline:   {bo_mean_final:.4f} +/- {bo_std_final:.4f}")
        # Simple comparison
        vr_mean = np.mean(final_hvs)
        print(f"  VR/BO ratio:   {vr_mean / bo_mean_final:.1%}")

    # --- 10b. Edge P/R ---
    precisions = []
    recalls = []
    missed_counts: dict[str, int] = {}
    for seed, log, data in seeds_data:
        tool_calls = log.get("tool_calls", [])
        disc_edges = _extract_edges_from_tool_calls(oracle, tool_calls)
        disc_pairs = {(e.source, e.target) for e in disc_edges}
        io_nodes = frozenset(
            n for n in oracle.ground_truth().nodes
            if n.node_type in (NodeType.INPUT, NodeType.OUTPUT)
        )
        dag = CausalDAG(nodes=io_nodes, edges=disc_edges)
        p, r = gt_io.precision_recall(dag)
        precisions.append(p)
        recalls.append(r)
        for src, tgt in gt_pairs - disc_pairs:
            key = f"{src}->{tgt}"
            missed_counts[key] = missed_counts.get(key, 0) + 1

    print(f"\n  10b. Edge Precision / Recall")
    print(f"  Precision: {np.mean(precisions):.3f} +/- {np.std(precisions):.3f}")
    print(f"  Recall:    {np.mean(recalls):.3f} +/- {np.std(recalls):.3f}")
    if missed_counts:
        print(f"  Most missed edges:")
        for edge, count in sorted(missed_counts.items(), key=lambda x: -x[1]):
            print(f"    {edge}: missed in {count}/{n_seeds} seeds")

    # --- 10c. OAT Prediction Accuracy ---
    dir_accs = []
    mag_errs = []
    for seed, log, data in seeds_data:
        tc_list = log.get("tool_calls", [])
        seed_correct = 0
        seed_total = 0
        seed_mag_errs: list[float] = []
        for tc in tc_list:
            if tc.get("name") != "oat_sweep":
                continue
            result = tc.get("result", {})
            if not isinstance(result, dict):
                continue
            scores = result.get("trend_scores", {})
            for oname, score in scores.items():
                if not isinstance(score, dict):
                    continue
                seed_total += 1
                if score.get("direction_correct"):
                    seed_correct += 1
                seed_mag_errs.append(float(score.get("magnitude_error", 0.0)))
        if seed_total > 0:
            dir_accs.append(seed_correct / seed_total)
        if seed_mag_errs:
            mag_errs.append(np.mean(seed_mag_errs))

    print(f"\n  10c. OAT Prediction Accuracy")
    if dir_accs:
        print(f"  Direction accuracy: {np.mean(dir_accs):.1%} +/- {np.std(dir_accs):.1%}")
    if mag_errs:
        print(f"  Magnitude error:   {np.mean(mag_errs):.4f} +/- {np.std(mag_errs):.4f}")

    # --- 10d. Calibration ---
    cal_first = []
    cal_last = []
    learning_count = 0
    for seed, log, data in seeds_data:
        checks = log.get("calibration_checks", [])
        valid = [c for c in checks if not math.isnan(c.get("mae", float("nan")))]
        if len(valid) >= 1:
            cal_first.append(valid[0]["mae"])
        if len(valid) >= 2:
            cal_last.append(valid[-1]["mae"])
            if valid[-1]["mae"] < valid[0]["mae"] * 0.8:
                learning_count += 1

    print(f"\n  10d. Calibration Checkpoints")
    if cal_first:
        print(f"  First checkpoint MAE: {np.mean(cal_first):.4f} +/- {np.std(cal_first):.4f}")
    if cal_last:
        print(f"  Last checkpoint MAE:  {np.mean(cal_last):.4f} +/- {np.std(cal_last):.4f}")
        print(f"  Seeds showing learning (last < 80% of first): "
              f"{learning_count}/{len(cal_last)}")

    # --- 10e. Surprise Rate ---
    surprise_rates: list[float] = []
    for seed, log, data in seeds_data:
        tc_list = log.get("tool_calls", [])
        eval_pts = [tc for tc in tc_list if tc.get("name") == "evaluate_point"]
        if not eval_pts:
            continue
        n_surprises = 0
        for tc in eval_pts:
            result = tc.get("result", {})
            if not isinstance(result, dict):
                continue
            pred_errors = result.get("prediction_errors", {})
            if pred_errors:
                max_err = max(abs(float(v)) for v in pred_errors.values())
                if max_err > SURPRISE_THRESHOLD:
                    n_surprises += 1
        surprise_rates.append(n_surprises / len(eval_pts))

    print(f"\n  10e. Surprise Rate (evaluate_point, threshold={SURPRISE_THRESHOLD})")
    if surprise_rates:
        print(f"  Mean surprise rate: {np.mean(surprise_rates):.1%} +/- "
              f"{np.std(surprise_rates):.1%}")

    # --- 10f. Adversarial Prediction ---
    if regions:
        adv_errors: dict[str, list[float]] = {r["name"]: [] for r in regions}
        adv_errors["non-adversarial"] = []
        for seed, log, data in seeds_data:
            for tc in log.get("tool_calls", []):
                if tc.get("name") != "evaluate_point":
                    continue
                inp = tc.get("input", {})
                result = tc.get("result", {})
                point = inp.get("point", [])
                if not point:
                    continue
                pred_errors = result.get("prediction_errors", {}) if isinstance(result, dict) else {}
                if not pred_errors:
                    continue
                max_err = max(abs(float(v)) for v in pred_errors.values())
                x = np.array(point, dtype=np.float64)
                in_any = False
                for r in regions:
                    try:
                        if r["test"](x):
                            adv_errors[str(r["name"])].append(max_err)
                            in_any = True
                    except (IndexError, ZeroDivisionError):
                        pass
                if not in_any:
                    adv_errors["non-adversarial"].append(max_err)

        print(f"\n  10f. Adversarial Prediction (pooled across seeds)")
        print(f"  {'Region':<25} {'#Pts':>5} {'MAE':>8} {'Max|err|':>10}")
        print(f"  {'-' * 52}")
        for rname in ["non-adversarial"] + [str(r["name"]) for r in regions]:
            errs = adv_errors.get(rname, [])
            if errs:
                print(f"  {rname:<25} {len(errs):>5} {np.mean(errs):>8.4f} "
                      f"{np.max(errs):>10.4f}")

    # --- 10g. Information Capture ---
    captures: list[float] = []
    for seed, log, data in seeds_data:
        tc_list = log.get("tool_calls", [])
        disc_edges = _extract_edges_from_tool_calls(oracle, tc_list)
        conf = {f"{e.source}->{e.target}": 1.0 for e in disc_edges}
        capture = _compute_agent_info_capture(conf, sobol_total, oracle)
        captures.append(np.mean(list(capture.values())))

    print(f"\n  10g. Information Capture (Sobol proxy)")
    if captures:
        print(f"  Mean capture: {np.mean(captures):.4f} +/- {np.std(captures):.4f}")

    # --- 10h. Cost ---
    costs = []
    for seed, log, data in seeds_data:
        it = int(log.get("total_input_tokens", 0))
        ot = int(log.get("total_output_tokens", 0))
        costs.append(it * 5.0 / 1e6 + ot * 25.0 / 1e6)

    print(f"\n  10h. Cost")
    print(f"  Per seed: ${np.mean(costs):.2f} +/- ${np.std(costs):.2f}")
    print(f"  Total:    ${np.sum(costs):.2f}")

    # --- 10i. Per-seed detail table ---
    print(f"\n  10i. Per-Seed Detail")
    print(f"  {'Seed':>6} {'HV':>8} {'Prec':>6} {'Rec':>6} "
          f"{'OAT%':>6} {'Cal1':>6} {'CalN':>6} {'Cost':>7}")
    print(f"  {'-' * 55}")
    for i, (seed, log, data) in enumerate(seeds_data):
        hv = final_hvs[i]
        p = precisions[i]
        r = recalls[i]
        da = f"{dir_accs[i]:.0%}" if i < len(dir_accs) else "---"
        checks = log.get("calibration_checks", [])
        valid = [c for c in checks if not math.isnan(c.get("mae", float("nan")))]
        c1 = f"{valid[0]['mae']:.3f}" if valid else "---"
        cn = f"{valid[-1]['mae']:.3f}" if len(valid) >= 2 else c1
        cost = costs[i]
        print(f"  {seed:>6} {hv:>8.4f} {p:>6.3f} {r:>6.3f} "
              f"{da:>6} {c1:>6} {cn:>6} ${cost:>6.2f}")


def plot_multi_seed_summary() -> None:
    """2x2 summary plot for multi-seed results."""
    print("\n" + "=" * 70)
    print("  10b. Multi-Seed Summary Plot")
    print("=" * 70)

    seeds_data = _load_multi_seed_data()
    if not seeds_data:
        print("  No multi-seed data.")
        return

    oracle = MediumOracle()
    gt_io = oracle.ground_truth().project_to_io()
    gt_pairs = {(e.source, e.target) for e in gt_io.edges}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: HV boxplot VR vs BO ---
    ax = axes[0, 0]
    final_hvs = [float(d["hypervolumes"][-1]) for _, _, d in seeds_data]
    box_data = [final_hvs]
    labels = ["VR tool"]
    bo_hvs = []
    for s in range(42, 52):
        bo_path = COMP_RESULTS / f"bo_seed{s}.npz"
        if bo_path.exists():
            bo_hvs.append(float(np.load(bo_path)["hypervolumes"][-1]))
    if bo_hvs:
        box_data.append(bo_hvs)
        labels.append("BO")
    ax.boxplot(box_data, labels=labels)
    ax.set_ylabel("Final Hypervolume")
    ax.set_title("Final HV Distribution (10 seeds)")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Edge recall per seed ---
    ax = axes[0, 1]
    recalls = []
    for seed, log, data in seeds_data:
        disc_edges = _extract_edges_from_tool_calls(oracle, log.get("tool_calls", []))
        disc_pairs = {(e.source, e.target) for e in disc_edges}
        recalls.append(len(gt_pairs & disc_pairs) / len(gt_pairs))
    ax.bar(range(len(seeds_data)), recalls, color="#2ca02c", alpha=0.7)
    ax.set_xticks(range(len(seeds_data)))
    ax.set_xticklabels([str(s) for s, _, _ in seeds_data], fontsize=8)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Edge Recall")
    ax.set_title("Causal Edge Recall per Seed")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(recalls), color="black", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Calibration trajectories ---
    ax = axes[1, 0]
    for seed, log, data in seeds_data:
        checks = log.get("calibration_checks", [])
        valid = [c for c in checks if not math.isnan(c.get("mae", float("nan")))]
        if len(valid) >= 2:
            evals = [c["eval"] for c in valid]
            maes = [c["mae"] for c in valid]
            ax.plot(evals, maes, "o-", alpha=0.5, markersize=4)
    ax.set_xlabel("Oracle Evaluations")
    ax.set_ylabel("Calibration MAE")
    ax.set_title("Calibration Trajectories (all seeds)")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: OAT direction accuracy per seed ---
    ax = axes[1, 1]
    dir_accs = []
    for seed, log, data in seeds_data:
        correct = 0
        total = 0
        for tc in log.get("tool_calls", []):
            if tc.get("name") != "oat_sweep":
                continue
            scores = tc.get("result", {}).get("trend_scores", {})
            for score in scores.values():
                if isinstance(score, dict):
                    total += 1
                    if score.get("direction_correct"):
                        correct += 1
        dir_accs.append(correct / total if total > 0 else 0)
    ax.bar(range(len(seeds_data)), dir_accs, color="#ff7f0e", alpha=0.7)
    ax.set_xticks(range(len(seeds_data)))
    ax.set_xticklabels([str(s) for s, _, _ in seeds_data], fontsize=8)
    ax.set_xlabel("Seed")
    ax.set_ylabel("OAT Direction Accuracy")
    ax.set_title("OAT Prediction Accuracy per Seed")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=np.mean(dir_accs), color="black", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Medium 1A: Opus Tool Agent (10 seeds)", fontsize=14)
    plt.tight_layout()
    out_path = OUT_DIR / "multi_seed_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out_path}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    print("SynthOracle Analysis: Compute All Metrics")
    print("=" * 70)
    print()

    # 1. Reference HVs (cached)
    ref_hvs = compute_reference_hvs()

    # 2. Edge Precision/Recall (all tool-use runs)
    compute_edge_precision_recall()

    # 3. Mechanism Discovery Timeline (all runs)
    compute_mechanism_timeline()

    # 4. Budget-Normalized HV (all runs)
    compute_budget_normalized_hv(ref_hvs)

    # 5. HV vs Budget plot (3 panels)
    plot_hv_vs_budget(ref_hvs)

    # 6. Prediction Accuracy (all runs with predictions)
    compute_prediction_accuracy()
    plot_prediction_accuracy()

    # 7. Surprise Analysis
    compute_surprise_analysis()
    compute_calibration_analysis()
    plot_surprise_trajectory()

    # 8. Causal Model Convergence
    compute_causal_model_convergence()
    plot_causal_convergence()
    compute_adversarial_prediction_analysis()
    compute_calibration_over_time()

    # 9. Information-Theoretic Analysis (T6)
    compute_information_analysis()
    plot_information_capture()

    # 10. Multi-Seed Aggregate
    compute_multi_seed_aggregate()
    plot_multi_seed_summary()

    print("\n" + "=" * 70)
    print("  Done. All metrics computed for all runs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
