"""Full diagnostic analysis for HD oracle multi-seed runs.

Standalone because compute_metrics.py is tightly coupled to 1A/1B/1C/1D/1E
assumptions (multi_seed dir, sobol cache, surprise-trajectory plots). This
script focuses on what the HD oracle uniquely tests: can the agent screen
irrelevant inputs in a higher-dimensional space?

Loads:
    experiments/vr_agent/results/hd_vr_seed{42,43,44}.npz / _log.json    (72)
    experiments/vr_agent/results/hd_vr_ext_seed{42,43,44}.npz / _log.json (144)
    experiments/vr_agent/results/bo_hd_seed{42,43,44}.npz

Outputs:
    stdout: HV, edge P/R, OAT accuracy, calibration, screening, adversarial
    experiments/analysis/results/hd_hv_envelope.png
    experiments/analysis/results/hd_ext_hv_envelope.png

Usage:
    uv run python experiments/analysis/analyze_hd.py          # 72-budget runs
    uv run python experiments/analysis/analyze_hd.py ext      # 144-budget runs
    uv run python experiments/analysis/analyze_hd.py both     # both + comparison
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

# Reuse helpers from compute_metrics
sys.path.insert(0, str(Path(__file__).parent))
from compute_metrics import (  # noqa: E402
    _extract_edges_from_tool_calls,
)

from synthoracle.dag import CausalDAG, NodeType  # noqa: E402
from synthoracle.oracles.medium_hd import MediumOracleHD  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
VR_RESULTS = ROOT / "experiments" / "vr_agent" / "results"
OUT_DIR = ROOT / "experiments" / "analysis" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
NOISE_INPUTS = ("X7", "X8", "X9", "X10", "X11", "X12")
PRICING = (5.0, 25.0)  # Opus 4.6: $/M (input, output)
SURPRISE_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_vr_seeds(
    prefix: str = "hd_vr",
) -> list[tuple[int, dict[str, Any], dict[str, np.ndarray]]]:
    """Load all HD VR seed data: (seed, log, npz_dict).

    prefix='hd_vr' loads 72-budget runs; 'hd_vr_ext' loads 144-budget.
    """
    out = []
    for seed in SEEDS:
        npz_path = VR_RESULTS / f"{prefix}_seed{seed}.npz"
        log_path = VR_RESULTS / f"{prefix}_seed{seed}_log.json"
        if not npz_path.exists() or not log_path.exists():
            print(f"  WARN: missing {prefix} seed {seed}")
            continue
        with open(log_path) as f:
            log = json.load(f)
        data = dict(np.load(npz_path))
        out.append((seed, log, data))
    return out


def load_bo_seeds() -> list[tuple[int, dict[str, np.ndarray]]]:
    """Load all BO HD seed data."""
    out = []
    for seed in SEEDS:
        p = VR_RESULTS / f"bo_hd_seed{seed}.npz"
        if not p.exists():
            continue
        out.append((seed, dict(np.load(p))))
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_hv(
    vr_seeds: list[tuple[int, dict, dict]],
    bo_seeds: list[tuple[int, dict]],
) -> None:
    print("\n" + "=" * 70)
    print("  1. Final Hypervolume")
    print("=" * 70)

    vr_final = [float(d["hypervolumes"][-1]) for _, _, d in vr_seeds]
    bo_final = [float(d["hypervolumes"][-1]) for _, d in bo_seeds]

    print(f"\n  VR HD ({len(vr_final)} seeds):")
    print(f"    HV = {np.mean(vr_final):.4f} +/- {np.std(vr_final):.4f}")
    print(f"    range: [{np.min(vr_final):.4f}, {np.max(vr_final):.4f}]")
    print(f"  BO HD ({len(bo_final)} seeds):")
    print(f"    HV = {np.mean(bo_final):.4f} +/- {np.std(bo_final):.4f}")
    print(f"    range: [{np.min(bo_final):.4f}, {np.max(bo_final):.4f}]")
    if bo_final:
        ratio = np.mean(vr_final) / np.mean(bo_final)
        print(f"  VR/BO ratio: {ratio:.1%}")


def section_edge_pr(
    oracle: MediumOracleHD,
    vr_seeds: list[tuple[int, dict, dict]],
) -> tuple[list[float], list[float]]:
    """Edge precision/recall per seed + aggregate. Noise edges count as FP."""
    print("\n" + "=" * 70)
    print("  2. Edge Precision / Recall (vs ground-truth IO projection)")
    print("=" * 70)

    gt_io = oracle.ground_truth().project_to_io()
    gt_pairs = {(e.source, e.target) for e in gt_io.edges}
    io_nodes = frozenset(
        n for n in oracle.ground_truth().nodes
        if n.node_type in (NodeType.INPUT, NodeType.OUTPUT)
    )

    print(f"  Ground-truth IO edges: {len(gt_pairs)}")
    print(f"  {'seed':>5} {'prec':>6} {'rec':>6} {'TP':>4} {'FP':>4} "
          f"{'FN':>4} {'FP_noise':>9}")

    precisions, recalls = [], []
    all_missed: dict[str, int] = {}
    all_fp: dict[str, int] = {}
    for seed, log, _ in vr_seeds:
        tool_calls = log.get("tool_calls", [])
        disc_edges = _extract_edges_from_tool_calls(oracle, tool_calls)  # type: ignore[arg-type]
        disc_pairs = {(e.source, e.target) for e in disc_edges}
        dag = CausalDAG(nodes=io_nodes, edges=disc_edges)
        p, r = gt_io.precision_recall(dag)

        tp = len(gt_pairs & disc_pairs)
        fp_all = disc_pairs - gt_pairs
        fn_all = gt_pairs - disc_pairs
        fp = len(fp_all)
        fn = len(fn_all)
        fp_noise = sum(1 for s, _ in fp_all if s in NOISE_INPUTS)

        precisions.append(p)
        recalls.append(r)
        print(f"  {seed:>5} {p:>6.3f} {r:>6.3f} {tp:>4} {fp:>4} {fn:>4} "
              f"{fp_noise:>9}")

        for s, t in fn_all:
            all_missed[f"{s}->{t}"] = all_missed.get(f"{s}->{t}", 0) + 1
        for s, t in fp_all:
            all_fp[f"{s}->{t}"] = all_fp.get(f"{s}->{t}", 0) + 1

    print(f"\n  Aggregate:")
    print(f"    Precision: {np.mean(precisions):.3f} +/- {np.std(precisions):.3f}")
    print(f"    Recall:    {np.mean(recalls):.3f} +/- {np.std(recalls):.3f}")

    if all_missed:
        print(f"\n  Most missed (real) edges:")
        for edge, c in sorted(all_missed.items(), key=lambda x: -x[1]):
            print(f"    {edge}: missed in {c}/{len(vr_seeds)} seeds")
    if all_fp:
        print(f"\n  False positive edges:")
        for edge, c in sorted(all_fp.items(), key=lambda x: -x[1]):
            tag = " (noise)" if edge.split("->")[0] in NOISE_INPUTS else ""
            print(f"    {edge}: {c}/{len(vr_seeds)} seeds{tag}")
    else:
        print(f"\n  False positives: zero across all seeds.")

    return precisions, recalls


def section_screening(
    vr_seeds: list[tuple[int, dict, dict]],
) -> None:
    """Count OAT sweeps / interactions on real vs noise inputs."""
    print("\n" + "=" * 70)
    print("  3. Screening Efficiency")
    print("=" * 70)

    print(f"  {'seed':>5} {'oat_real':>9} {'oat_noise':>10} "
          f"{'int_real':>9} {'int_noise':>10} {'total_evals':>12}")

    total_oat_real = 0
    total_oat_noise = 0
    total_int_real = 0
    total_int_noise = 0
    total_evals_real = 0
    total_evals_noise = 0

    for seed, log, _ in vr_seeds:
        tcs = log.get("tool_calls", [])
        oat_real = oat_noise = 0
        int_real = int_noise = 0
        evals_real = evals_noise = 0

        for tc in tcs:
            if tc.get("name") == "oat_sweep":
                iname = tc.get("input", {}).get("input_name")
                if iname in NOISE_INPUTS:
                    oat_noise += 1
                    evals_noise += int(tc.get("input", {}).get("n_levels", 5))
                else:
                    oat_real += 1
                    evals_real += int(tc.get("input", {}).get("n_levels", 5))
            elif tc.get("name") == "interaction_test":
                ia = tc.get("input", {}).get("input_a")
                ib = tc.get("input", {}).get("input_b")
                noise_hit = (ia in NOISE_INPUTS) or (ib in NOISE_INPUTS)
                la = len(tc.get("input", {}).get("levels_a", []))
                lb = len(tc.get("input", {}).get("levels_b", []))
                if noise_hit:
                    int_noise += 1
                    evals_noise += la * lb
                else:
                    int_real += 1
                    evals_real += la * lb

        total_oat_real += oat_real
        total_oat_noise += oat_noise
        total_int_real += int_real
        total_int_noise += int_noise
        total_evals_real += evals_real
        total_evals_noise += evals_noise

        print(f"  {seed:>5} {oat_real:>9} {oat_noise:>10} "
              f"{int_real:>9} {int_noise:>10} "
              f"{evals_real + evals_noise:>12}")

    tool_total = total_oat_real + total_oat_noise + total_int_real + total_int_noise
    eval_total = total_evals_real + total_evals_noise
    print(f"\n  Aggregate over {len(vr_seeds)} seeds:")
    print(f"    OAT sweeps:        real={total_oat_real} noise={total_oat_noise}")
    print(f"    Interaction tests: real={total_int_real} noise={total_int_noise}")
    if tool_total > 0:
        noise_frac = (total_oat_noise + total_int_noise) / tool_total
        print(f"    Noise tool-call fraction: {noise_frac:.1%}")
    if eval_total > 0:
        eval_noise_frac = total_evals_noise / eval_total
        print(f"    Tool-call evals on noise: "
              f"{total_evals_noise}/{eval_total} ({eval_noise_frac:.1%})")

    print("\n  (Note: 1/12 of inputs would be sampled at random = 50.0% on noise;\n"
          "   the agent allocates 0% => perfect screening.)")


def section_oat_accuracy(
    vr_seeds: list[tuple[int, dict, dict]],
) -> None:
    """Direction + magnitude accuracy on OAT predictions (on real inputs only)."""
    print("\n" + "=" * 70)
    print("  4. OAT Prediction Accuracy")
    print("=" * 70)

    dir_accs = []
    mag_errs = []
    print(f"  {'seed':>5} {'n_pred':>7} {'dir_acc':>8} {'mag_MAE':>9}")
    for seed, log, _ in vr_seeds:
        correct = 0
        total = 0
        seed_mag: list[float] = []
        for tc in log.get("tool_calls", []):
            if tc.get("name") != "oat_sweep":
                continue
            result = tc.get("result", {})
            if not isinstance(result, dict):
                continue
            scores = result.get("trend_scores", {}) or {}
            for _oname, score in scores.items():
                if not isinstance(score, dict):
                    continue
                total += 1
                if score.get("direction_correct"):
                    correct += 1
                seed_mag.append(float(score.get("magnitude_error", 0.0)))

        if total > 0:
            da = correct / total
            dir_accs.append(da)
        else:
            da = float("nan")
        if seed_mag:
            me = float(np.mean(seed_mag))
            mag_errs.append(me)
        else:
            me = float("nan")
        print(f"  {seed:>5} {total:>7} {da:>8.1%} {me:>9.4f}")

    if dir_accs:
        print(f"\n  Aggregate direction accuracy: "
              f"{np.mean(dir_accs):.1%} +/- {np.std(dir_accs):.1%}")
    if mag_errs:
        print(f"  Aggregate magnitude MAE:      "
              f"{np.mean(mag_errs):.4f} +/- {np.std(mag_errs):.4f}")


def section_calibration(
    vr_seeds: list[tuple[int, dict, dict]],
) -> None:
    print("\n" + "=" * 70)
    print("  5. Calibration over time (structured checkpoints)")
    print("=" * 70)

    print(f"  {'seed':>5} {'n_ckpt':>7} {'first_MAE':>11} {'last_MAE':>10} "
          f"{'learn?':>7}")

    firsts, lasts = [], []
    learning_count = 0
    for seed, log, _ in vr_seeds:
        checks = log.get("calibration_checks", []) or []
        valid = [c for c in checks
                 if isinstance(c, dict) and not math.isnan(
                     c.get("mae", float("nan")))]
        if not valid:
            print(f"  {seed:>5} {0:>7} {'---':>11} {'---':>10} {'---':>7}")
            continue
        first_mae = float(valid[0]["mae"])
        last_mae = float(valid[-1]["mae"])
        firsts.append(first_mae)
        lasts.append(last_mae)
        learned = "yes" if last_mae < first_mae * 0.8 else "no"
        if learned == "yes":
            learning_count += 1
        print(f"  {seed:>5} {len(valid):>7} {first_mae:>11.4f} "
              f"{last_mae:>10.4f} {learned:>7}")

    if firsts:
        print(f"\n  First checkpoint MAE: "
              f"{np.mean(firsts):.4f} +/- {np.std(firsts):.4f}")
    if lasts:
        print(f"  Last checkpoint MAE:  "
              f"{np.mean(lasts):.4f} +/- {np.std(lasts):.4f}")
    if lasts:
        print(f"  Seeds showing learning (last < 80% of first): "
              f"{learning_count}/{len(lasts)}")


def section_adversarial(
    oracle: MediumOracleHD,
    vr_seeds: list[tuple[int, dict, dict]],
) -> None:
    """Prediction error in adversarial regions."""
    print("\n" + "=" * 70)
    print("  6. Adversarial Region Prediction Error")
    print("=" * 70)

    regions = oracle.adversarial_regions()
    adv_errors: dict[str, list[float]] = {str(r["name"]): [] for r in regions}
    adv_errors["non-adversarial"] = []

    for _seed, log, _ in vr_seeds:
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
                    if r["test"](x):  # type: ignore[operator]
                        adv_errors[str(r["name"])].append(max_err)
                        in_any = True
                except (IndexError, ZeroDivisionError):
                    pass
            if not in_any:
                adv_errors["non-adversarial"].append(max_err)

    print(f"  {'Region':<25} {'#Pts':>5} {'MAE':>8} {'Max|err|':>10}")
    print(f"  {'-' * 52}")
    for rname in ["non-adversarial"] + [str(r["name"]) for r in regions]:
        errs = adv_errors.get(rname, [])
        if errs:
            print(f"  {rname:<25} {len(errs):>5} {np.mean(errs):>8.4f} "
                  f"{np.max(errs):>10.4f}")


def section_noise_edge_confidence(
    vr_seeds: list[tuple[int, dict, dict]],
) -> None:
    """Across seeds, check what confidences the agent assigned to noise edges."""
    print("\n" + "=" * 70)
    print("  7. Noise Edge Confidence (final iteration summaries)")
    print("=" * 70)

    for seed, log, _ in vr_seeds:
        summaries = log.get("iteration_summaries", []) or []
        if not summaries:
            print(f"  seed {seed}: no iteration summaries")
            continue
        last = summaries[-1]
        edges = last.get("edges", []) or []
        noise = [(e["edge"], float(e["confidence"])) for e in edges
                 if any(n in e["edge"] for n in NOISE_INPUTS)]
        real_set = [e for e in edges
                    if not any(n in e["edge"] for n in NOISE_INPUTS)]

        print(f"\n  seed {seed}: {len(real_set)} real edges, {len(noise)} noise edges listed")
        if noise:
            max_conf = max(c for _, c in noise)
            mean_conf = float(np.mean([c for _, c in noise]))
            high_conf = [e for e in noise if e[1] > 0.3]
            print(f"    noise confidences: mean={mean_conf:.3f} max={max_conf:.3f}")
            if high_conf:
                print(f"    >0.3 confidence: {len(high_conf)} edges")
                for name, conf in sorted(high_conf, key=lambda x: -x[1]):
                    print(f"      {name}: {conf:.2f}")
            else:
                print(f"    all noise edges at <= 0.3 confidence (explicit dismissal)")


def section_cost(vr_seeds: list[tuple[int, dict, dict]]) -> None:
    print("\n" + "=" * 70)
    print("  8. Cost")
    print("=" * 70)

    print(f"  {'seed':>5} {'input_tok':>12} {'output_tok':>12} {'cost':>8} "
          f"{'iters':>6}")
    costs = []
    for seed, log, _ in vr_seeds:
        it = int(log.get("total_input_tokens", 0))
        ot = int(log.get("total_output_tokens", 0))
        n_iters = len(log.get("iteration_summaries", []) or [])
        cost = it * PRICING[0] / 1e6 + ot * PRICING[1] / 1e6
        costs.append(cost)
        print(f"  {seed:>5} {it:>12,} {ot:>12,} ${cost:>6.2f} {n_iters:>6}")

    print(f"\n  Per seed: ${np.mean(costs):.2f} +/- ${np.std(costs):.2f}")
    print(f"  Total:    ${np.sum(costs):.2f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_hv_envelope(
    vr_seeds: list[tuple[int, dict, dict]],
    bo_seeds: list[tuple[int, dict]],
    out_name: str = "hd_hv_envelope.png",
    title: str = "HD Oracle (12 inputs, 6 noise): Opus VR vs BO",
) -> None:
    """HV vs evaluations: VR mean +/- std, BO mean +/- std, individual seed curves."""
    _fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # VR
    if vr_seeds:
        max_len = max(len(d["hypervolumes"]) for _, _, d in vr_seeds)
        vr_matrix = np.full((len(vr_seeds), max_len), np.nan)
        for i, (_, _, d) in enumerate(vr_seeds):
            hvs = d["hypervolumes"]
            vr_matrix[i, :len(hvs)] = hvs
            vr_matrix[i, len(hvs):] = hvs[-1]
        vr_mean = np.nanmean(vr_matrix, axis=0)
        vr_std = np.nanstd(vr_matrix, axis=0)
        evals = np.arange(1, max_len + 1)
        ax.plot(evals, vr_mean, color="#d62728", linewidth=2,
                label=f"VR tool (mean, n={len(vr_seeds)})")
        ax.fill_between(evals, vr_mean - vr_std, vr_mean + vr_std,
                        alpha=0.2, color="#d62728")
        for _, _, d in vr_seeds:
            ax.plot(range(1, len(d["hypervolumes"]) + 1), d["hypervolumes"],
                    color="#d62728", alpha=0.25, linewidth=0.6)

    # BO
    if bo_seeds:
        max_len = max(len(d["hypervolumes"]) for _, d in bo_seeds)
        bo_matrix = np.full((len(bo_seeds), max_len), np.nan)
        for i, (_, d) in enumerate(bo_seeds):
            hvs = d["hypervolumes"]
            bo_matrix[i, :len(hvs)] = hvs
            bo_matrix[i, len(hvs):] = hvs[-1]
        bo_mean = np.nanmean(bo_matrix, axis=0)
        bo_std = np.nanstd(bo_matrix, axis=0)
        evals = np.arange(1, max_len + 1)
        ax.plot(evals, bo_mean, color="#1f77b4", linewidth=2,
                label=f"BO (mean, n={len(bo_seeds)})")
        ax.fill_between(evals, bo_mean - bo_std, bo_mean + bo_std,
                        alpha=0.15, color="#1f77b4")

    ax.axvline(x=24, color="gray", linestyle="--", alpha=0.4,
               label="n_initial=24")
    ax.set_xlabel("Oracle Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  HV envelope saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_full_analysis(
    oracle: MediumOracleHD,
    vr_seeds: list[tuple[int, dict, dict]],
    bo_seeds: list[tuple[int, dict]],
    label: str,
    out_name: str,
) -> None:
    """Run all diagnostic sections for a given VR seed set."""
    print("\n" + "#" * 70)
    print(f"#  {label}")
    print("#" * 70)
    print(f"\n  Loaded {len(vr_seeds)} VR seeds, {len(bo_seeds)} BO seeds")

    if not vr_seeds:
        print("  No VR seeds — skipping.")
        return

    section_hv(vr_seeds, bo_seeds)
    section_edge_pr(oracle, vr_seeds)
    section_screening(vr_seeds)
    section_oat_accuracy(vr_seeds)
    section_calibration(vr_seeds)
    section_adversarial(oracle, vr_seeds)
    section_noise_edge_confidence(vr_seeds)
    section_cost(vr_seeds)

    plot_hv_envelope(
        vr_seeds, bo_seeds,
        out_name=out_name,
        title=f"HD Oracle: Opus VR vs BO — {label}",
    )


def _compare_72_vs_144(
    vr72: list[tuple[int, dict, dict]],
    vr144: list[tuple[int, dict, dict]],
) -> None:
    """Side-by-side comparison of 72 vs 144 budget runs."""
    print("\n" + "=" * 70)
    print("  Comparison: HD 72 vs HD 144 (extended budget)")
    print("=" * 70)

    hv72 = [float(d["hypervolumes"][-1]) for _, _, d in vr72]
    hv144 = [float(d["hypervolumes"][-1]) for _, _, d in vr144]
    print(f"\n  HV @ 72:  {np.mean(hv72):.4f} +/- {np.std(hv72):.4f} "
          f"(n={len(hv72)})")
    print(f"  HV @ 144: {np.mean(hv144):.4f} +/- {np.std(hv144):.4f} "
          f"(n={len(hv144)})")

    cost72 = []
    cost144 = []
    for _, log, _ in vr72:
        cost72.append(log.get("total_input_tokens", 0) * 5.0 / 1e6
                      + log.get("total_output_tokens", 0) * 25.0 / 1e6)
    for _, log, _ in vr144:
        cost144.append(log.get("total_input_tokens", 0) * 5.0 / 1e6
                       + log.get("total_output_tokens", 0) * 25.0 / 1e6)
    print(f"\n  Cost @ 72:  ${np.mean(cost72):.2f} +/- ${np.std(cost72):.2f}")
    print(f"  Cost @ 144: ${np.mean(cost144):.2f} +/- ${np.std(cost144):.2f}")


def main() -> None:
    print("=" * 70)
    print("  HD Oracle — Full Diagnostic Analysis")
    print("=" * 70)

    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode not in ("base", "ext", "both"):
        print(f"  Usage: analyze_hd.py [base|ext|both]")
        sys.exit(1)

    oracle = MediumOracleHD()
    bo_seeds = load_bo_seeds()

    if mode in ("base", "both"):
        vr72 = load_vr_seeds(prefix="hd_vr")
        _run_full_analysis(
            oracle, vr72, bo_seeds,
            label="HD 72-budget (base)",
            out_name="hd_hv_envelope.png",
        )

    if mode in ("ext", "both"):
        vr144 = load_vr_seeds(prefix="hd_vr_ext")
        _run_full_analysis(
            oracle, vr144, bo_seeds,
            label="HD 144-budget (extended)",
            out_name="hd_ext_hv_envelope.png",
        )

    if mode == "both":
        vr72_loaded = load_vr_seeds(prefix="hd_vr")
        vr144_loaded = load_vr_seeds(prefix="hd_vr_ext")
        if vr72_loaded and vr144_loaded:
            _compare_72_vs_144(vr72_loaded, vr144_loaded)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
