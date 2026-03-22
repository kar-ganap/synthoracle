"""Clean Opus vs Sonnet tool agent head-to-head with full diagnostics.

Both runs use the updated agent with:
- Structured OAT predictions (per-output direction + magnitude)
- Calibration checkpoints every 20 evals
- Tool result saving for offline surprise analysis

Usage:
    ANTHROPIC_API_KEY=... uv run --extra vr python experiments/vr_agent/run_head_to_head.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from synthoracle.agents.vr_tools import VRToolsResult, run_vr_tools
from synthoracle.dag import CausalDAG, Edge, EdgeDifficulty, NodeType
from synthoracle.oracles.medium import MediumOracle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BUDGET = 72

RUNS: list[tuple[str, str, dict[str, object] | None]] = [
    ("opus_h2h", "claude-opus-4-6", {"type": "adaptive"}),
    ("sonnet_h2h", "claude-sonnet-4-6", {"type": "adaptive"}),
]

PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}

EDGE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Edge extraction (from compute_metrics.py)
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
    base = np.clip(
        np.array(base_point, dtype=np.float64),
        oracle.bounds[:, 0], oracle.bounds[:, 1],
    )
    Y_list = []
    for val in levels:
        pt = base.copy()
        pt[idx] = val
        Y_list.append(oracle.evaluate(pt))
    return np.array(Y_list, dtype=np.float64)


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
            # Simplified — just check if component edges are discovered
            result = tc.get("result", {})
            if isinstance(result, dict):
                effects = result.get("interaction_effects_std", {})
                for oname, std_val in effects.items():
                    if isinstance(std_val, (int, float)) and std_val > EDGE_THRESHOLD:
                        discovered.add((inp.get("input_a", ""), oname))
                        discovered.add((inp.get("input_b", ""), oname))
    return frozenset(
        Edge(s, t, EdgeDifficulty.EASY, "discovered") for s, t in discovered
    )


# ---------------------------------------------------------------------------
# Run + save
# ---------------------------------------------------------------------------


def run_single(
    label: str, model: str, thinking: dict[str, object] | None,
    oracle: MediumOracle,
) -> VRToolsResult:
    ckpt = str(RESULTS_DIR / f"{label}_ckpt")
    print(f"\n{'=' * 70}")
    print(f"  Running: {label} ({model})")
    print(f"{'=' * 70}\n")

    return run_vr_tools(
        oracle, n_budget=N_BUDGET, seed=SEED,
        thresholds={"Y3": 0.4}, model=model, thinking=thinking,
        max_tokens=16000, max_tool_calls_per_iteration=15,
        checkpoint_dir=ckpt, calibration_interval=20,
    )


def save_result(
    label: str, result: VRToolsResult, output_dir: Path,
) -> None:
    np.savez(
        output_dir / f"{label}_seed42.npz",
        X=result.X, Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
    )
    with open(output_dir / f"{label}_seed42_log.json", "w") as f:
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


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def print_comparison(
    results: dict[str, VRToolsResult], oracle: MediumOracle,
) -> None:
    gt_io = oracle.ground_truth().project_to_io()
    gt_pairs = {(e.source, e.target) for e in gt_io.edges}

    print(f"\n{'=' * 100}")
    print(f"  Opus vs Sonnet Tool Agent — Head-to-Head Comparison")
    print(f"{'=' * 100}")

    # --- HV ---
    print(f"\n  1. Hypervolume")
    print(f"  {'Run':<16} {'Final HV':>10} {'HV @ 36':>10} {'HV @ 54':>10}")
    print(f"  {'-' * 50}")
    for label, r in results.items():
        hv_36 = r.hypervolumes[35] if len(r.hypervolumes) > 35 else 0.0
        hv_54 = r.hypervolumes[53] if len(r.hypervolumes) > 53 else 0.0
        print(f"  {label:<16} {r.hypervolumes[-1]:>10.6f} {hv_36:>10.6f} {hv_54:>10.6f}")

    # --- Edge P/R ---
    print(f"\n  2. Edge Precision / Recall")
    print(f"  {'Run':<16} {'Precision':>10} {'Recall':>10} {'Edges':>7} {'Missed':>30}")
    print(f"  {'-' * 75}")
    for label, r in results.items():
        disc_edges = extract_edges_from_tool_calls(oracle, r.tool_calls)
        disc_pairs = {(e.source, e.target) for e in disc_edges}
        io_nodes = frozenset(
            n for n in oracle.ground_truth().nodes
            if n.node_type in (NodeType.INPUT, NodeType.OUTPUT)
        )
        dag = CausalDAG(nodes=io_nodes, edges=disc_edges)
        p, rec = gt_io.precision_recall(dag)
        missed = gt_pairs - disc_pairs
        missed_str = ", ".join(f"{s}->{t}" for s, t in sorted(missed)) or "none"
        print(f"  {label:<16} {p:>10.3f} {rec:>10.3f} {len(disc_edges):>7} {missed_str:>30}")

    # --- Tool usage ---
    print(f"\n  3. Tool Usage")
    print(f"  {'Run':<16} {'Total':>7} {'OAT':>5} {'Interact':>9} {'Eval Pt':>8} "
          f"{'Analysis':>9}")
    print(f"  {'-' * 58}")
    for label, r in results.items():
        counts: dict[str, int] = {}
        for tc in r.tool_calls:
            n = str(tc["name"])
            counts[n] = counts.get(n, 0) + 1
        oat = counts.get("oat_sweep", 0)
        inter = counts.get("interaction_test", 0)
        evpt = counts.get("evaluate_point", 0)
        analysis = sum(v for k, v in counts.items()
                       if k not in ("oat_sweep", "interaction_test", "evaluate_point"))
        print(f"  {label:<16} {len(r.tool_calls):>7} {oat:>5} {inter:>9} {evpt:>8} "
              f"{analysis:>9}")

    # --- Structured OAT prediction accuracy ---
    print(f"\n  4. OAT Prediction Accuracy (structured)")
    print(f"  {'Run':<16} {'Sweeps':>7} {'Dir Correct':>12} {'Dir Acc':>8} "
          f"{'Mean Mag Err':>13}")
    print(f"  {'-' * 60}")
    for label, r in results.items():
        dir_correct = 0
        dir_total = 0
        mag_errors: list[float] = []
        for tc in r.tool_calls:
            if tc.get("name") != "oat_sweep":
                continue
            result = tc.get("result", {})
            if not isinstance(result, dict):
                continue
            scores = result.get("trend_scores", {})
            if not scores:
                continue
            for oname, score in scores.items():
                if not isinstance(score, dict):
                    continue
                dir_total += 1
                if score.get("direction_correct"):
                    dir_correct += 1
                mag_errors.append(float(score.get("magnitude_error", 0.0)))
        dir_acc = dir_correct / dir_total if dir_total > 0 else 0.0
        mean_mag = sum(mag_errors) / len(mag_errors) if mag_errors else 0.0
        print(f"  {label:<16} {dir_total // 4:>7} {dir_correct}/{dir_total:>10} "
              f"{dir_acc:>8.1%} {mean_mag:>13.4f}")

    # --- Calibration checkpoints ---
    print(f"\n  5. Calibration Checkpoints")
    for label, r in results.items():
        if not r.calibration_checks:
            print(f"  {label}: no calibration data")
            continue
        print(f"\n  {label}:")
        print(f"    {'Eval':>6} {'MAE':>8} {'Max|err|':>10} "
              f"{'Per-output errors'}")
        print(f"    {'-' * 60}")
        for check in r.calibration_checks:
            eval_n = check.get("eval", "?")
            mae = check.get("mae", float("nan"))
            max_err = check.get("max_error", float("nan"))
            errors = check.get("errors", [])
            err_str = ", ".join(f"{e:.4f}" for e in errors)
            print(f"    {eval_n:>6} {mae:>8.4f} {max_err:>10.4f} [{err_str}]")

        first_mae = r.calibration_checks[0].get("mae", 0)
        last_mae = r.calibration_checks[-1].get("mae", 0)
        if isinstance(first_mae, (int, float)) and isinstance(last_mae, (int, float)):
            if last_mae < first_mae * 0.7:
                trend = "LEARNING"
            elif last_mae > first_mae * 1.3:
                trend = "DEGRADING"
            else:
                trend = "STABLE"
            print(f"    Trend: {trend} ({first_mae:.4f} -> {last_mae:.4f})")

    # --- Surprise events ---
    print(f"\n  6. Surprise Events (evaluate_point |error| > 0.05)")
    for label, r in results.items():
        surprises = []
        for tc in r.tool_calls:
            if tc.get("name") != "evaluate_point":
                continue
            result = tc.get("result", {})
            if not isinstance(result, dict):
                continue
            pred_errors = result.get("prediction_errors", {})
            if pred_errors:
                max_err = max(abs(float(v)) for v in pred_errors.values())
                if max_err > 0.05:
                    surprises.append((max_err, pred_errors))
        print(f"  {label}: {len(surprises)} surprises out of "
              f"{sum(1 for tc in r.tool_calls if tc.get('name') == 'evaluate_point')} "
              f"evaluate_point calls")
        for mag, errs in surprises[:5]:
            print(f"    max|err|={mag:.4f}: {errs}")

    # --- Cost ---
    print(f"\n  7. Cost")
    print(f"  {'Run':<16} {'LLM calls':>10} {'In tok':>12} {'Out tok':>12} "
          f"{'Cost':>8} {'Time(s)':>8}")
    print(f"  {'-' * 70}")
    for label, r in results.items():
        pricing = PRICING.get(
            next(m for l, m, _ in RUNS if l == label), (0, 0),
        )
        cost = (r.total_input_tokens * pricing[0] / 1e6
                + r.total_output_tokens * pricing[1] / 1e6)
        print(f"  {label:<16} {r.total_llm_calls:>10} {r.total_input_tokens:>12,} "
              f"{r.total_output_tokens:>12,} ${cost:>7.2f} {r.total_seconds:>8.0f}")

    # --- Mechanism log ---
    print(f"\n  8. Mechanism Logs")
    for label, r in results.items():
        print(f"\n  {label} ({len(r.mechanism_log)} entries):")
        for i, m in enumerate(r.mechanism_log):
            print(f"    [{i + 1}] {m[:120]}")


def plot_comparison(
    results: dict[str, VRToolsResult],
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {"opus_h2h": "#1f77b4", "sonnet_h2h": "#ff7f0e"}

    for label, r in results.items():
        evals = list(range(1, len(r.hypervolumes) + 1))
        model_short = "Opus" if "opus" in label else "Sonnet"
        ax.plot(evals, r.hypervolumes, color=colors.get(label, "gray"),
                linewidth=2, label=f"{model_short} tool")

        # Mark calibration checkpoints
        for check in r.calibration_checks:
            ev = check.get("eval", 0)
            if 0 < ev <= len(r.hypervolumes):
                ax.axvline(x=ev, color=colors.get(label, "gray"),
                           linestyle=":", alpha=0.3)

    ax.axvline(x=12, color="gray", linestyle="--", alpha=0.2, label="Initial")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Opus vs Sonnet Tool Agent — Head-to-Head (Full Diagnostics)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "h2h_opus_sonnet_hv.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  HV plot saved: {RESULTS_DIR / 'h2h_opus_sonnet_hv.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    oracle = MediumOracle()
    results: dict[str, VRToolsResult] = {}

    for label, model, thinking in RUNS:
        try:
            r = run_single(label, model, thinking, oracle)
            results[label] = r
            save_result(label, r, RESULTS_DIR)
            pricing = PRICING.get(model, (0, 0))
            cost = (r.total_input_tokens * pricing[0] / 1e6
                    + r.total_output_tokens * pricing[1] / 1e6)
            print(f"\n  {label} complete: HV={r.hypervolumes[-1]:.6f}, "
                  f"cost=${cost:.2f}, time={r.total_seconds:.0f}s")
        except Exception:
            print(f"\n  {label} FAILED:")
            traceback.print_exc()

    if results:
        print_comparison(results, oracle)
        plot_comparison(results)


if __name__ == "__main__":
    main()
