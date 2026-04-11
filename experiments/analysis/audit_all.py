"""Cross-oracle audit: extract intrinsic + observed metrics for all 6 oracles.

Produces `experiments/analysis/results/audit_data.json` — a structured artifact
that build_rubric.py consumes to produce the difficulty scaling rubric.

Design principles:
- Honest n_seeds tracking — never average across different n
- Durable metrics first; ephemeral metrics (cost, wall time) stored but
  filtered out of the rubric
- Reuses helpers from compute_metrics.py (no upstream refactor)
- Standalone — can be re-run after new experiments land

Usage:
    uv run python experiments/analysis/audit_all.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Reuse helpers from compute_metrics + characterize
sys.path.insert(0, str(Path(__file__).parent))
from compute_metrics import (  # noqa: E402
    EDGE_THRESHOLD,
    ORACLE_CONFIGS,
    _compute_agent_info_capture,
    _compute_mechanism_sufficiency,
    _compute_sobol_indices_cached,
    _extract_edges_from_tool_calls,
)

from synthoracle.characterize import compute_sobol_indices  # noqa: E402
from synthoracle.dag import CausalDAG, NodeType  # noqa: E402
from synthoracle.oracle import Oracle  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
VR_RESULTS = ROOT / "experiments" / "vr_agent" / "results"
COMP_RESULTS = ROOT / "experiments" / "comparison" / "results"
BO_RESULTS = ROOT / "experiments" / "bo_baseline" / "results"
ANALYSIS_DIR = ROOT / "experiments" / "analysis"
OUT_DIR = ANALYSIS_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pricing for cost extraction (kept in audit_data, filtered from rubric)
PRICING = {
    "opus": (5.0, 25.0),
    "sonnet": (3.0, 15.0),
    "haiku": (0.80, 4.0),
}

# Sobol effective-input threshold
SOBOL_DEFF_THRESHOLD = 0.05
SOBOL_NOISE_THRESHOLD = 0.02

# Sample efficiency thresholds (% of reference HV)
SE_THRESHOLDS = (0.50, 0.75, 0.90)


# ---------------------------------------------------------------------------
# Run registry: maps each (oracle, condition) → list of run files
# ---------------------------------------------------------------------------


@dataclass
class RunRef:
    """A single run file pair."""

    seed: int
    log_path: Path
    npz_path: Path


@dataclass
class ConditionDef:
    """A run condition: a labeled set of seeds with a budget and model."""

    label: str
    budget: int
    model: str  # "opus", "sonnet", "haiku"
    runs: list[RunRef] = field(default_factory=list)
    notes: str = ""


def _condition(
    label: str,
    budget: int,
    model: str,
    seeds: list[int],
    log_template: str,
    npz_template: str,
    notes: str = "",
) -> ConditionDef:
    """Build a condition by templating log/npz paths over seeds."""
    runs = []
    for s in seeds:
        log_p = VR_RESULTS / log_template.format(seed=s)
        npz_p = VR_RESULTS / npz_template.format(seed=s)
        if log_p.exists() and npz_p.exists():
            runs.append(RunRef(seed=s, log_path=log_p, npz_path=npz_p))
    return ConditionDef(
        label=label, budget=budget, model=model, runs=runs, notes=notes,
    )


# Conditions per oracle
ORACLE_CONDITIONS: dict[str, list[ConditionDef]] = {
    "1A": [
        _condition(
            "multi_seed_72",
            72,
            "opus",
            list(range(42, 52)),
            "multi_seed/seed{seed}_log.json",
            "multi_seed/seed{seed}.npz",
            notes="10-seed multi-seed Opus tool agent",
        ),
        _condition(
            "extended_144",
            144,
            "opus",
            [42, 43, 44, 45],
            "1a_extended_144_seed{seed}_log.json",
            "1a_extended_144_seed{seed}.npz",
            notes="4-seed extended budget Opus",
        ),
    ],
    "1B": [
        _condition(
            "transfer_72",
            72,
            "opus",
            [42],
            "transfer_medium_1b_seed{seed}_log.json",
            "transfer_medium_1b_seed{seed}.npz",
            notes="n=1 transfer from 1A prior",
        ),
        _condition(
            "fresh_72",
            72,
            "opus",
            [42],
            "no_transfer_1b_seed{seed}_log.json",
            "no_transfer_1b_seed{seed}.npz",
            notes="n=1 fresh (no prior)",
        ),
    ],
    "1C": [
        _condition(
            "transfer_72",
            72,
            "opus",
            [42],
            "transfer_1c_seed{seed}_log.json",
            "transfer_1c_seed{seed}.npz",
            notes="n=1 transfer from 1A prior",
        ),
        _condition(
            "fresh_72",
            72,
            "opus",
            [42],
            "no_transfer_1c_seed{seed}_log.json",
            "no_transfer_1c_seed{seed}.npz",
            notes="n=1 fresh (no prior)",
        ),
    ],
    "1D": [
        _condition(
            "prior_72",
            72,
            "opus",
            [42, 43, 44],  # extended to n=3 by run_transfer_1d_1e_n3.py
            "transfer_1d_prior_seed{seed}_log.json",
            "transfer_1d_prior_seed{seed}.npz",
            notes="3-seed prior, screen-first protocol",
        ),
        _condition(
            "fresh_72",
            72,
            "opus",
            [42, 43, 44],  # extended to n=3
            "transfer_1d_fresh_seed{seed}_log.json",
            "transfer_1d_fresh_seed{seed}.npz",
            notes="3-seed fresh",
        ),
        _condition(
            "prior_144",
            144,
            "opus",
            [42],
            "transfer_1d_prior_144_seed{seed}_log.json",
            "transfer_1d_prior_144_seed{seed}.npz",
            notes="n=1 prior at extended budget",
        ),
    ],
    "1E": [
        _condition(
            "prior_72",
            72,
            "opus",
            [42, 43, 44],  # extended to n=3
            "transfer_1e_prior_seed{seed}_log.json",
            "transfer_1e_prior_seed{seed}.npz",
            notes="3-seed prior (Opus), screen-first",
        ),
        _condition(
            "fresh_72",
            72,
            "opus",
            [42, 43, 44],  # extended to n=3
            "transfer_1e_fresh_seed{seed}_log.json",
            "transfer_1e_fresh_seed{seed}.npz",
            notes="3-seed fresh",
        ),
        _condition(
            "sonnet_prior_72",
            72,
            "sonnet",
            [42],
            "transfer_1e_sonnet_prior_seed{seed}_log.json",
            "transfer_1e_sonnet_prior_seed{seed}.npz",
            notes="n=1 Sonnet prior (model generality)",
        ),
    ],
    "HD": [
        _condition(
            "base_72",
            72,
            "opus",
            [42, 43, 44],
            "hd_vr_seed{seed}_log.json",
            "hd_vr_seed{seed}.npz",
            notes="3-seed Opus, 12 inputs (6 noise)",
        ),
        _condition(
            "extended_144",
            144,
            "opus",
            [42, 43, 44],
            "hd_vr_ext_seed{seed}_log.json",
            "hd_vr_ext_seed{seed}.npz",
            notes="3-seed Opus, 144 budget",
        ),
        _condition(
            "sonnet_72",
            72,
            "sonnet",
            [42, 43, 44],
            "hd_vr_sonnet_seed{seed}_log.json",
            "hd_vr_sonnet_seed{seed}.npz",
            notes="3-seed Sonnet (model generality)",
        ),
    ],
}


def _bo_runs_for(oracle_label: str) -> list[Path]:
    """Return BO baseline npz paths for a given oracle.

    Prefers the new 144-eval BO baselines (`bo_{oracle}_n144_seed{N}.npz`)
    when available — they extend further than the original 54-66 eval
    baselines, fixing the crossover_eval comparison length mismatch with
    extended-budget VR runs.
    """
    # Map oracle label → (default paths, n144 prefix)
    n144_files: dict[str, list[Path]] = {
        "1A": [VR_RESULTS / f"bo_1a_n144_seed{s}.npz" for s in [42, 43, 44]],
        "1D": [VR_RESULTS / f"bo_1d_n144_seed{s}.npz" for s in [42, 43, 44]],
        "1E": [VR_RESULTS / f"bo_1e_n144_seed{s}.npz" for s in [42, 43, 44]],
        "HD": [VR_RESULTS / f"bo_hd_n144_seed{s}.npz" for s in [42, 43, 44]],
    }
    if oracle_label in n144_files:
        existing_n144 = [p for p in n144_files[oracle_label] if p.exists()]
        if existing_n144:
            return existing_n144  # prefer longer curves

    # Fallback: original BO baselines
    if oracle_label == "1A":
        return [COMP_RESULTS / f"bo_seed{s}.npz" for s in range(42, 52)
                if (COMP_RESULTS / f"bo_seed{s}.npz").exists()]
    if oracle_label == "1B":
        return [BO_RESULTS / "medium_1b_seed42.npz"] if (
            BO_RESULTS / "medium_1b_seed42.npz").exists() else []
    if oracle_label == "1C":
        return [BO_RESULTS / "medium_1c_seed42.npz"] if (
            BO_RESULTS / "medium_1c_seed42.npz").exists() else []
    if oracle_label == "1D":
        return [VR_RESULTS / f"bo_1d_seed{s}.npz" for s in [42, 43]
                if (VR_RESULTS / f"bo_1d_seed{s}.npz").exists()]
    if oracle_label == "1E":
        return [VR_RESULTS / f"bo_1e_seed{s}.npz" for s in [42, 43, 44, 45]
                if (VR_RESULTS / f"bo_1e_seed{s}.npz").exists()]
    if oracle_label == "HD":
        return [VR_RESULTS / f"bo_hd_seed{s}.npz" for s in [42, 43, 44]
                if (VR_RESULTS / f"bo_hd_seed{s}.npz").exists()]
    return []


# ---------------------------------------------------------------------------
# Intrinsic characterization
# ---------------------------------------------------------------------------


def _dag_depth(dag: CausalDAG) -> int:
    """Compute longest input→output path length in the DAG."""
    # Build adjacency
    adj: dict[str, list[str]] = {}
    for e in dag.edges:
        adj.setdefault(e.source, []).append(e.target)

    inputs = [n.name for n in dag.nodes if n.node_type == NodeType.INPUT]
    outputs = {n.name for n in dag.nodes if n.node_type == NodeType.OUTPUT}

    max_depth = 0
    for start in inputs:
        # BFS to find longest path to any output
        stack: list[tuple[str, int]] = [(start, 0)]
        seen: dict[str, int] = {}
        while stack:
            node, depth = stack.pop()
            if depth > seen.get(node, -1):
                seen[node] = depth
            else:
                continue
            for nxt in adj.get(node, []):
                stack.append((nxt, depth + 1))
                if nxt in outputs:
                    max_depth = max(max_depth, depth + 1)
    return max_depth


def _sobol_first_order_cached(
    oracle: Oracle, label: str,
) -> np.ndarray:
    """Compute and cache Sobol first-order indices (n_inputs, n_outputs)."""
    cache_path = ANALYSIS_DIR / f"sobol_first_{label}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    print(f"  Computing Sobol first-order for {label} ...")
    rng = np.random.default_rng(42)
    s1, _ = compute_sobol_indices(oracle, 50_000, rng)
    np.save(cache_path, s1)
    return s1


def compute_intrinsics(
    label: str, oracle: Oracle, ref_hv: float,
) -> dict[str, Any]:
    """Compute all intrinsic properties for one oracle."""
    dag = oracle.ground_truth()
    n_inputs = oracle.n_inputs
    n_outputs = oracle.n_outputs

    # Mechanism count from DAG
    mech_nodes = [n for n in dag.nodes if n.node_type == NodeType.MECHANISM]
    k = len(mech_nodes)

    # DAG depth
    depth = _dag_depth(dag)

    # Sobol total (already cached)
    sobol_total = _compute_sobol_indices_cached(oracle, label)
    # Sobol first-order (compute + cache if missing)
    sobol_first = _sobol_first_order_cached(oracle, label)

    # d_eff: inputs with mean(total Sobol) > threshold
    mean_sobol_total_per_input = sobol_total.mean(axis=1)
    d_eff = int(np.sum(mean_sobol_total_per_input > SOBOL_DEFF_THRESHOLD))

    # Noise dim count: inputs with mean(total Sobol) < noise threshold
    noise_dim_count = int(np.sum(mean_sobol_total_per_input < SOBOL_NOISE_THRESHOLD))

    # Interaction fraction: Σ(total - first) / Σ total
    sum_total = float(sobol_total.sum())
    sum_first = float(sobol_first.sum())
    interaction_fraction = (
        max(0.0, sum_total - sum_first) / sum_total if sum_total > 0 else 0.0
    )

    # Adversarial regions
    try:
        adv_regions = oracle.adversarial_regions()
        adv_count = len(adv_regions)
    except (AttributeError, NotImplementedError):
        adv_count = 0

    # Mechanism sufficiency R²(M→Y) — uses _compute_mechanism_sufficiency
    # This works only on oracles with evaluate_with_mechanisms; use a try/except
    try:
        suff = _compute_mechanism_sufficiency(oracle)  # type: ignore[arg-type]
        sufficiency = {k_: float(v) for k_, v in suff.items()}
        sufficiency_mean = float(np.mean(list(suff.values())))
    except (AttributeError, NotImplementedError, KeyError) as e:
        print(f"  WARN: mechanism sufficiency failed for {label}: {e}")
        sufficiency = {}
        sufficiency_mean = float("nan")

    # Top-2 input contributors per output (from total Sobol)
    top_inputs_per_output: dict[str, list[tuple[str, float]]] = {}
    input_names = list(oracle.input_names)
    for j, oname in enumerate(oracle.output_names):
        col = sobol_total[:, j]
        # Sort indices by descending Sobol
        order = np.argsort(-col)
        top = [
            (input_names[i], float(col[i]))
            for i in order[:2] if col[i] > 0.01
        ]
        top_inputs_per_output[oname] = top

    return {
        "label": label,
        "d": n_inputs,
        "n_outputs": n_outputs,
        "k_mechanisms": k,
        "k_over_d": k / n_inputs,
        "d_eff": d_eff,
        "k_over_d_eff": k / d_eff if d_eff > 0 else float("inf"),
        "dag_depth": depth,
        "interaction_fraction": interaction_fraction,
        "adversarial_region_count": adv_count,
        "noise_dim_count": noise_dim_count,
        "reference_hv": ref_hv,
        "mechanism_sufficiency": {
            "per_output": sufficiency,
            "mean": sufficiency_mean,
        },
        "top_sobol_inputs_per_output": top_inputs_per_output,
        "input_names": input_names,
        "output_names": list(oracle.output_names),
        "output_directions": list(oracle.output_directions),
    }


# ---------------------------------------------------------------------------
# Prior quality vs 1A (for transfer variants)
# ---------------------------------------------------------------------------


def compute_prior_quality_vs_1A(
    variant_label: str, variant_oracle: Oracle, oracle_1a: Oracle,
) -> dict[str, Any]:
    """Compare a variant's IO projection to 1A's. Reports edge overlap and Y-correlation."""
    gt_variant = variant_oracle.ground_truth().project_to_io()
    gt_1a = oracle_1a.ground_truth().project_to_io()

    pairs_variant = {(e.source, e.target) for e in gt_variant.edges}
    pairs_1a = {(e.source, e.target) for e in gt_1a.edges}

    shared = pairs_variant & pairs_1a
    wrong_in_prior = pairs_1a - pairs_variant  # 1A says yes, variant says no → must unlearn
    missing_from_prior = pairs_variant - pairs_1a  # variant says yes, 1A doesn't → must discover

    # Y-correlation: shared LHS sample of common 6 inputs
    # Only meaningful if both oracles have the same input dimensionality
    y_corr_per_output: dict[str, float] = {}
    if variant_oracle.n_inputs == oracle_1a.n_inputs:
        rng = np.random.default_rng(7)
        n = 200
        lo = oracle_1a.bounds[:, 0]
        hi = oracle_1a.bounds[:, 1]
        X = rng.uniform(lo, hi, size=(n, oracle_1a.n_inputs))
        Y_1a = oracle_1a.evaluate_batch(X)
        Y_var = variant_oracle.evaluate_batch(X)
        for j, oname in enumerate(oracle_1a.output_names):
            a = Y_1a[:, j]
            b = Y_var[:, j]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                y_corr_per_output[oname] = float("nan")
            else:
                corr = np.corrcoef(a, b)[0, 1]
                y_corr_per_output[oname] = float(corr)
    else:
        # HD has different n_inputs (12 vs 6); not directly comparable on shared X
        y_corr_per_output = {o: float("nan") for o in oracle_1a.output_names}

    return {
        "variant": variant_label,
        "edges_in_1a_prior": len(pairs_1a),
        "edges_in_variant": len(pairs_variant),
        "edges_shared": len(shared),
        "edges_wrong_in_prior": len(wrong_in_prior),  # must unlearn
        "edges_missing_from_prior": len(missing_from_prior),  # must discover
        "wrong_edge_list": sorted(f"{s}->{t}" for s, t in wrong_in_prior),
        "missing_edge_list": sorted(f"{s}->{t}" for s, t in missing_from_prior),
        "y_correlation": y_corr_per_output,
    }


# ---------------------------------------------------------------------------
# Observed metric extractors
# ---------------------------------------------------------------------------


def _load_run(run: RunRef) -> tuple[dict, dict]:
    """Load a single run's log dict + npz dict."""
    with open(run.log_path) as f:
        log = json.load(f)
    npz = dict(np.load(run.npz_path, allow_pickle=True))
    return log, npz


def extract_optim(
    cond: ConditionDef, ref_hv: float, bo_paths: list[Path],
) -> dict[str, Any]:
    """Extract HV statistics + sample efficiency vs BO."""
    if not cond.runs:
        return {"n": 0, "note": "no runs"}

    vr_finals: list[float] = []
    vr_curves: list[np.ndarray] = []
    for run in cond.runs:
        _, npz = _load_run(run)
        hvs = np.asarray(npz["hypervolumes"], dtype=np.float64)
        vr_finals.append(float(hvs[-1]))
        vr_curves.append(hvs)

    bo_curves: list[np.ndarray] = []
    bo_finals: list[float] = []
    for p in bo_paths:
        d = np.load(p)
        hvs = np.asarray(d["hypervolumes"], dtype=np.float64)
        bo_curves.append(hvs)
        bo_finals.append(float(hvs[-1]))

    # HV as fraction of reference HV
    vr_fracs = [v / ref_hv for v in vr_finals]
    bo_fracs = [v / ref_hv for v in bo_finals]

    # Sample efficiency: evals to reach X% of ref HV (per curve, then average)
    def _evals_to_threshold(curves: list[np.ndarray], frac: float) -> Optional[float]:
        target = frac * ref_hv
        evals = []
        for hv in curves:
            idx = np.argmax(hv >= target) if (hv >= target).any() else None
            if idx is not None and hv[idx] >= target:
                evals.append(int(idx) + 1)  # 1-indexed
        if not evals:
            return None
        return float(np.mean(evals))

    sample_eff_vr = {
        f"{int(t*100)}pct": _evals_to_threshold(vr_curves, t)
        for t in SE_THRESHOLDS
    }
    sample_eff_bo = {
        f"{int(t*100)}pct": _evals_to_threshold(bo_curves, t)
        for t in SE_THRESHOLDS
    }

    # Crossover eval: first eval where mean(VR) reaches BO's final HV.
    # Uses BO end-point as a fixed target, robust to VR/BO curve length mismatch.
    crossover = None
    seeds_crossing = 0
    if vr_curves and bo_curves:
        bo_final_mean = float(np.mean(bo_finals))
        # Find earliest eval where VR mean >= BO final
        max_vr_len = max(len(c) for c in vr_curves)
        # Pad all VR curves to the longest length (with their final HV)
        vr_padded = np.full((len(vr_curves), max_vr_len), np.nan)
        for i, c in enumerate(vr_curves):
            vr_padded[i, :len(c)] = c
            vr_padded[i, len(c):] = c[-1]
        vr_mean = np.nanmean(vr_padded, axis=0)
        ge = vr_mean >= bo_final_mean
        if ge.any():
            crossover = int(np.argmax(ge)) + 1
        # Per-seed: how many VR seeds reach BO final by their end?
        for hv in vr_curves:
            if hv[-1] >= bo_final_mean:
                seeds_crossing += 1

    return {
        "n_seeds_vr": len(cond.runs),
        "n_seeds_bo": len(bo_paths),
        "vr_final_hv_mean": float(np.mean(vr_finals)),
        "vr_final_hv_std": float(np.std(vr_finals)),
        "vr_final_hv_min": float(np.min(vr_finals)),
        "vr_final_hv_max": float(np.max(vr_finals)),
        "vr_final_hv_frac_ref_mean": float(np.mean(vr_fracs)),
        "vr_final_hv_frac_ref_std": float(np.std(vr_fracs)),
        "bo_final_hv_mean": float(np.mean(bo_finals)) if bo_finals else None,
        "bo_final_hv_std": float(np.std(bo_finals)) if bo_finals else None,
        "bo_final_hv_frac_ref_mean": float(np.mean(bo_fracs)) if bo_fracs else None,
        "vr_over_bo": (
            float(np.mean(vr_finals)) / float(np.mean(bo_finals))
            if bo_finals else None
        ),
        "sample_efficiency_vr": sample_eff_vr,
        "sample_efficiency_bo": sample_eff_bo,
        "crossover_eval": crossover,
        "seeds_crossing_bo": seeds_crossing,
    }


def extract_edge_pr(
    cond: ConditionDef, oracle: Oracle,
) -> dict[str, Any]:
    """Edge precision/recall using IO projection of ground truth."""
    if not cond.runs:
        return {"n": 0}

    gt_io = oracle.ground_truth().project_to_io()
    gt_pairs = {(e.source, e.target) for e in gt_io.edges}
    # Edges by difficulty (from IO projection)
    edges_by_difficulty: dict[str, set[tuple[str, str]]] = {
        "EASY": set(), "MEDIUM": set(), "HARD": set(),
    }
    for e in gt_io.edges:
        edges_by_difficulty[e.difficulty.name].add((e.source, e.target))

    precisions: list[float] = []
    recalls: list[float] = []
    missed_counts: dict[str, int] = {}
    found_by_difficulty: dict[str, list[int]] = {
        "EASY": [], "MEDIUM": [], "HARD": [],
    }

    for run in cond.runs:
        log, _ = _load_run(run)
        tool_calls = log.get("tool_calls", [])
        try:
            disc_edges = _extract_edges_from_tool_calls(oracle, tool_calls)  # type: ignore[arg-type]
        except Exception as e:
            print(f"  WARN: edge extraction failed for {run.log_path.name}: {e}")
            continue
        disc_pairs = {(e_.source, e_.target) for e_ in disc_edges}
        tp = len(gt_pairs & disc_pairs)
        fp = len(disc_pairs - gt_pairs)
        fn = len(gt_pairs - disc_pairs)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / len(gt_pairs) if gt_pairs else 1.0
        precisions.append(prec)
        recalls.append(rec)
        for s, t in gt_pairs - disc_pairs:
            key = f"{s}->{t}"
            missed_counts[key] = missed_counts.get(key, 0) + 1
        for diff, edges in edges_by_difficulty.items():
            found_by_difficulty[diff].append(len(edges & disc_pairs))

    if not precisions:
        return {"n": 0}

    hardest_missed = max(missed_counts.items(), key=lambda kv: kv[1])[0] if missed_counts else None

    return {
        "n_seeds": len(precisions),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "n_gt_edges": len(gt_pairs),
        "hardest_missed_edge": hardest_missed,
        "missed_counts": dict(sorted(missed_counts.items(), key=lambda kv: -kv[1])),
        "edges_by_difficulty_in_gt": {k: len(v) for k, v in edges_by_difficulty.items()},
        "edges_found_by_difficulty_mean": {
            k: float(np.mean(v)) if v else 0.0 for k, v in found_by_difficulty.items()
        },
    }


def extract_info_capture(
    cond: ConditionDef, oracle: Oracle, sobol_total: np.ndarray,
) -> dict[str, Any]:
    """Sobol-weighted recall (info capture) per output, end-of-run."""
    if not cond.runs:
        return {"n": 0}

    captures_per_run: list[dict[str, float]] = []
    for run in cond.runs:
        log, _ = _load_run(run)
        # Build confidence dict from final iteration_summary if available;
        # else from discovered edges (fallback at confidence 1.0)
        summaries = log.get("iteration_summaries", []) or []
        if summaries:
            last = summaries[-1]
            conf_raw = last.get("confidence", {}) or {}
            conf = {str(k): float(v) for k, v in conf_raw.items()
                    if isinstance(v, (int, float))}
        else:
            tool_calls = log.get("tool_calls", [])
            try:
                disc_edges = _extract_edges_from_tool_calls(oracle, tool_calls)  # type: ignore[arg-type]
                conf = {f"{e.source}->{e.target}": 1.0 for e in disc_edges}
            except Exception:
                continue

        try:
            cap = _compute_agent_info_capture(  # type: ignore[arg-type]
                conf, sobol_total, oracle,
            )
            captures_per_run.append({k: float(v) for k, v in cap.items()})
        except Exception as e:
            print(f"  WARN: info capture failed for {run.log_path.name}: {e}")

    if not captures_per_run:
        return {"n": 0}

    # Aggregate per output
    output_names = list(oracle.output_names)
    per_output_mean: dict[str, float] = {}
    per_output_std: dict[str, float] = {}
    for o in output_names:
        vals = [c.get(o, 0.0) for c in captures_per_run]
        per_output_mean[o] = float(np.mean(vals))
        per_output_std[o] = float(np.std(vals))

    means_across_outputs = [
        float(np.mean(list(c.values()))) for c in captures_per_run
    ]

    return {
        "n_seeds": len(captures_per_run),
        "per_output_mean": per_output_mean,
        "per_output_std": per_output_std,
        "mean_across_outputs": float(np.mean(means_across_outputs)),
        "std_across_outputs": float(np.std(means_across_outputs)),
    }


def extract_oat_accuracy(cond: ConditionDef) -> dict[str, Any]:
    """OAT direction accuracy + magnitude MAE from trend_scores."""
    if not cond.runs:
        return {"n": 0}

    dir_accs: list[float] = []
    mag_errs: list[float] = []
    n_pred_per_seed: list[int] = []

    for run in cond.runs:
        log, _ = _load_run(run)
        correct = 0
        total = 0
        seed_mags: list[float] = []
        for tc in log.get("tool_calls", []):
            if tc.get("name") != "oat_sweep":
                continue
            result = tc.get("result", {})
            if not isinstance(result, dict):
                continue
            scores = result.get("trend_scores", {}) or {}
            for _, s in scores.items():
                if not isinstance(s, dict):
                    continue
                total += 1
                if s.get("direction_correct"):
                    correct += 1
                seed_mags.append(float(s.get("magnitude_error", 0.0)))
        if total > 0:
            dir_accs.append(correct / total)
            n_pred_per_seed.append(total)
        if seed_mags:
            mag_errs.append(float(np.mean(seed_mags)))

    if not dir_accs:
        return {"n": 0}

    return {
        "n_seeds": len(dir_accs),
        "direction_accuracy_mean": float(np.mean(dir_accs)),
        "direction_accuracy_std": float(np.std(dir_accs)),
        "magnitude_mae_mean": float(np.mean(mag_errs)) if mag_errs else None,
        "magnitude_mae_std": float(np.std(mag_errs)) if mag_errs else None,
        "predictions_per_seed_mean": float(np.mean(n_pred_per_seed)),
    }


def extract_calibration(cond: ConditionDef) -> dict[str, Any]:
    """Calibration MAE first/last + learning fraction."""
    if not cond.runs:
        return {"n": 0}

    firsts: list[float] = []
    lasts: list[float] = []
    n_checks_per_seed: list[int] = []
    learning_count = 0
    has_calibration_count = 0

    for run in cond.runs:
        log, _ = _load_run(run)
        checks = log.get("calibration_checks", []) or []
        valid = []
        for c in checks:
            if isinstance(c, dict):
                mae = c.get("mae")
                if mae is not None and not (isinstance(mae, float) and math.isnan(mae)):
                    valid.append(float(mae))
        n_checks_per_seed.append(len(valid))
        if not valid:
            continue
        has_calibration_count += 1
        firsts.append(valid[0])
        if len(valid) >= 2:
            lasts.append(valid[-1])
            if valid[-1] < 0.8 * valid[0]:
                learning_count += 1

    if has_calibration_count == 0:
        return {"n_seeds_with_data": 0, "n_seeds": len(cond.runs)}

    return {
        "n_seeds": len(cond.runs),
        "n_seeds_with_data": has_calibration_count,
        "median_checkpoints_per_seed": float(np.median(n_checks_per_seed)),
        "first_mae_mean": float(np.mean(firsts)) if firsts else None,
        "last_mae_mean": float(np.mean(lasts)) if lasts else None,
        "learning_fraction": (
            learning_count / len(lasts) if lasts else None
        ),
        "n_seeds_showing_learning": learning_count,
        "n_seeds_with_two_checkpoints": len(lasts),
    }


def extract_adversarial(cond: ConditionDef, oracle: Oracle) -> dict[str, Any]:
    """Prediction error in adversarial regions vs non-adversarial."""
    if not cond.runs:
        return {"n": 0}

    try:
        regions = oracle.adversarial_regions()
    except (AttributeError, NotImplementedError):
        return {"n": 0, "note": "no adversarial regions defined"}

    adv_errors: dict[str, list[float]] = {str(r["name"]): [] for r in regions}
    adv_errors["non-adversarial"] = []

    for run in cond.runs:
        log, _ = _load_run(run)
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
            try:
                max_err = max(abs(float(v)) for v in pred_errors.values())
            except (TypeError, ValueError):
                continue
            x = np.array(point, dtype=np.float64)
            in_any = False
            for r in regions:
                try:
                    test_fn = r.get("test")  # type: ignore[union-attr]
                    if test_fn is not None and callable(test_fn) and test_fn(x):
                        adv_errors[str(r["name"])].append(max_err)
                        in_any = True
                except (IndexError, ZeroDivisionError):
                    pass
            if not in_any:
                adv_errors["non-adversarial"].append(max_err)

    region_summary: dict[str, dict[str, float]] = {}
    for rname, errs in adv_errors.items():
        if errs:
            region_summary[rname] = {
                "n_points": len(errs),
                "mae": float(np.mean(errs)),
                "max_err": float(np.max(errs)),
            }

    # Adv/non-adv ratio (max region MAE / non-adversarial MAE)
    non_adv = region_summary.get("non-adversarial", {})
    non_adv_mae = non_adv.get("mae", 0.0)
    non_adv_n = non_adv.get("n_points", 0)
    adv_only = {k: v for k, v in region_summary.items() if k != "non-adversarial"}
    max_adv_mae = max((v["mae"] for v in adv_only.values()), default=0.0)
    if adv_only and non_adv_mae > 0 and non_adv_n > 0:
        adv_ratio = max_adv_mae / non_adv_mae
    else:
        adv_ratio = None  # no non-adversarial baseline OR no adv regions hit

    return {
        "n_seeds": len(cond.runs),
        "regions": region_summary,
        "max_adv_over_non_adv_ratio": adv_ratio,
        "max_adv_mae": float(max_adv_mae) if adv_only else None,
        "non_adv_n_points": non_adv_n,
        "adv_total_n_points": sum(v.get("n_points", 0) for v in adv_only.values()),
    }


def extract_screening(
    cond: ConditionDef, oracle: Oracle, sobol_total: np.ndarray,
) -> dict[str, Any]:
    """Tool-call evals on low-Sobol inputs (HD specifically; generalizable)."""
    if not cond.runs:
        return {"n": 0}

    input_names = list(oracle.input_names)
    mean_sobol_per_input = sobol_total.mean(axis=1)
    low_sobol_inputs = {
        input_names[i] for i in range(oracle.n_inputs)
        if mean_sobol_per_input[i] < SOBOL_NOISE_THRESHOLD
    }

    if not low_sobol_inputs:
        return {"n_seeds": len(cond.runs), "low_sobol_inputs": [], "note": "no zero-effect inputs"}

    total_oat = 0
    noise_oat = 0
    total_evals = 0
    noise_evals = 0
    max_noise_edge_conf = 0.0

    for run in cond.runs:
        log, _ = _load_run(run)
        for tc in log.get("tool_calls", []):
            if tc.get("name") == "oat_sweep":
                iname = tc.get("input", {}).get("input_name")
                n_lev = int(tc.get("input", {}).get("n_levels", 5))
                total_oat += 1
                total_evals += n_lev
                if iname in low_sobol_inputs:
                    noise_oat += 1
                    noise_evals += n_lev
            elif tc.get("name") == "interaction_test":
                ia = tc.get("input", {}).get("input_a")
                ib = tc.get("input", {}).get("input_b")
                la = len(tc.get("input", {}).get("levels_a", []))
                lb = len(tc.get("input", {}).get("levels_b", []))
                cells = la * lb
                total_evals += cells
                if ia in low_sobol_inputs or ib in low_sobol_inputs:
                    noise_evals += cells

        # Max noise-edge confidence in iteration_summaries
        for s in log.get("iteration_summaries", []) or []:
            for e in s.get("edges", []) or []:
                edge_str = str(e.get("edge", ""))
                conf = e.get("confidence", 0.0)
                if not isinstance(conf, (int, float)):
                    continue
                # Edge string like "X7->Y1" — check if source is a low-Sobol input
                src = edge_str.split("->")[0] if "->" in edge_str else ""
                if src in low_sobol_inputs:
                    max_noise_edge_conf = max(max_noise_edge_conf, float(conf))

    return {
        "n_seeds": len(cond.runs),
        "low_sobol_inputs": sorted(low_sobol_inputs),
        "oat_total": total_oat,
        "oat_noise": noise_oat,
        "noise_oat_fraction": noise_oat / total_oat if total_oat > 0 else 0.0,
        "total_evals_in_tool_calls": total_evals,
        "noise_evals_in_tool_calls": noise_evals,
        "noise_eval_fraction": noise_evals / total_evals if total_evals > 0 else 0.0,
        "max_noise_edge_confidence": max_noise_edge_conf,
    }


def extract_cost(cond: ConditionDef) -> dict[str, Any]:
    """Token counts + dollar cost (ephemeral; stored but filtered from rubric)."""
    if not cond.runs:
        return {"n": 0}

    pricing = PRICING.get(cond.model, PRICING["opus"])
    costs: list[float] = []
    in_tokens: list[int] = []
    out_tokens: list[int] = []
    iters: list[int] = []

    for run in cond.runs:
        log, _ = _load_run(run)
        it = int(log.get("total_input_tokens", 0))
        ot = int(log.get("total_output_tokens", 0))
        in_tokens.append(it)
        out_tokens.append(ot)
        cost = it * pricing[0] / 1e6 + ot * pricing[1] / 1e6
        costs.append(cost)
        iters.append(len(log.get("iteration_summaries", []) or []))

    return {
        "n_seeds": len(cond.runs),
        "cost_per_seed_mean": float(np.mean(costs)) if costs else None,
        "cost_per_seed_std": float(np.std(costs)) if costs else None,
        "total_cost": float(np.sum(costs)) if costs else None,
        "input_tokens_mean": float(np.mean(in_tokens)) if in_tokens else None,
        "output_tokens_mean": float(np.mean(out_tokens)) if out_tokens else None,
        "iters_per_seed_median": float(np.median(iters)) if iters else None,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def build_audit_data() -> dict[str, Any]:
    """Build the full audit data dict for all 6 oracles."""
    print("=" * 70)
    print("  Cross-Oracle Audit")
    print("=" * 70)

    # Reference HVs from cache (already populated)
    with open(ANALYSIS_DIR / "reference_hvs.json") as f:
        ref_hvs = json.load(f)

    # 1A oracle for prior-quality comparisons
    oracle_1a = next(o for lbl, o, _ in ORACLE_CONFIGS if lbl == "1A")

    audit: dict[str, Any] = {}

    for label, oracle, _thresholds in ORACLE_CONFIGS:
        print(f"\n{'=' * 70}")
        print(f"  Auditing {label}")
        print(f"{'=' * 70}")

        ref_hv = ref_hvs[label]
        intrinsics = compute_intrinsics(label, oracle, ref_hv)
        sobol_total = _compute_sobol_indices_cached(oracle, label)

        # Prior quality (only for transfer variants — not 1A or HD)
        prior_quality = None
        if label not in ("1A", "HD"):
            prior_quality = compute_prior_quality_vs_1A(label, oracle, oracle_1a)

        # Per-condition observed metrics
        bo_paths = _bo_runs_for(label)
        conditions_data: dict[str, Any] = {}
        for cond in ORACLE_CONDITIONS.get(label, []):
            print(f"  → condition {cond.label} (n={len(cond.runs)} seeds)")
            cond_data = {
                "label": cond.label,
                "budget": cond.budget,
                "model": cond.model,
                "n_seeds": len(cond.runs),
                "notes": cond.notes,
                "optimization": extract_optim(cond, ref_hv, bo_paths),
                "edge_pr": extract_edge_pr(cond, oracle),
                "info_capture": extract_info_capture(cond, oracle, sobol_total),
                "oat_accuracy": extract_oat_accuracy(cond),
                "calibration": extract_calibration(cond),
                "adversarial": extract_adversarial(cond, oracle),
                "screening": extract_screening(cond, oracle, sobol_total),
                "cost": extract_cost(cond),
            }
            conditions_data[cond.label] = cond_data

        audit[label] = {
            "intrinsics": intrinsics,
            "prior_quality_vs_1A": prior_quality,
            "conditions": conditions_data,
        }

    return audit


def main() -> None:
    audit = build_audit_data()

    out_path = OUT_DIR / "audit_data.json"
    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)

    # Print compact summary
    print(f"\n{'=' * 70}")
    print(f"  Summary")
    print(f"{'=' * 70}")
    for label in ["1A", "1B", "1C", "1D", "1E", "HD"]:
        d = audit[label]
        intr = d["intrinsics"]
        n_conds = len(d["conditions"])
        suff_mean = intr["mechanism_sufficiency"]["mean"]
        suff_str = f"{suff_mean:.3f}" if not math.isnan(suff_mean) else "n/a"
        print(
            f"  {label:<3}: d={intr['d']:>2}, k={intr['k_mechanisms']:>2}, "
            f"k/d={intr['k_over_d']:.2f}, "
            f"d_eff={intr['d_eff']:>2}, "
            f"interactions={intr['interaction_fraction']:.2f}, "
            f"suff={suff_str}, "
            f"conditions={n_conds}"
        )

    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
