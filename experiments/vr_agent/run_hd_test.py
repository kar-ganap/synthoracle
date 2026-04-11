"""High-dimensional oracle test: 12 inputs (6 real + 6 noise).

Tests whether the VR agent can screen and dismiss irrelevant variables.
Runs VR (multiple seeds) and BO (3 seeds) for comparison.

Usage:
    # BO baseline (free)
    uv run --extra bo python experiments/vr_agent/run_hd_test.py bo

    # VR agent (API required) — runs seeds 42, 43, 44 by default
    source .env && uv run --extra vr python experiments/vr_agent/run_hd_test.py vr

    # Both
    source .env && uv run --extra bo --extra vr python experiments/vr_agent/run_hd_test.py all

    # Extended budget (144 evals, no calibration) — all three seeds
    source .env && uv run --extra vr python experiments/vr_agent/run_hd_test.py vr-ext

    # Extended budget — single seed (for sanity check)
    source .env && uv run --extra vr python experiments/vr_agent/run_hd_test.py vr-ext 42
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracles.medium_hd import MediumOracleHD

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VR_SEEDS = [42, 43, 44]
VR_EXT_SEEDS = [42, 43, 44]
N_BUDGET_VR = 72
N_BUDGET_VR_EXT = 144
N_ITERATIONS_BO = 42
THRESHOLDS = {"Y3": 0.4}
NOISE_INPUTS = ("X7", "X8", "X9", "X10", "X11", "X12")


def run_bo_hd() -> None:
    """Run BO baseline on HD oracle (3 seeds)."""
    from synthoracle.baselines.bo import run_bo

    oracle = MediumOracleHD()
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    hvs_all = []
    for seed in [42, 43, 44]:
        out = RESULTS_DIR / f"bo_hd_seed{seed}.npz"
        if out.exists():
            hv = float(np.load(out)["hypervolumes"][-1])
            hvs_all.append(hv)
            print(f"  seed {seed} (existing): HV={hv:.4f}")
            continue

        print(f"  Running BO HD seed {seed}...", flush=True)
        r = run_bo(
            oracle, n_iterations=N_ITERATIONS_BO, seed=seed,
            reference_point=ref_point, thresholds=THRESHOLDS,
        )
        np.savez(
            out, X=r.X, Y=r.Y,
            hypervolumes=np.array(r.hypervolumes),
            reference_point=ref_point,
        )
        hvs_all.append(r.hypervolumes[-1])
        print(f"  seed {seed}: HV={hvs_all[-1]:.4f}")

    print(f"\nBO HD ({len(hvs_all)} seeds): "
          f"{np.mean(hvs_all):.4f} +/- {np.std(hvs_all):.4f}")


def _seed_screening_stats(tool_calls: list[dict]) -> dict[str, object]:
    """Count OAT sweeps on real vs noise inputs."""
    oat_sweeps = [tc for tc in tool_calls if tc["name"] == "oat_sweep"]
    noise = [tc for tc in oat_sweeps
             if tc["input"].get("input_name") in NOISE_INPUTS]
    return {
        "oat_total": len(oat_sweeps),
        "oat_real": len(oat_sweeps) - len(noise),
        "oat_noise": len(noise),
    }


def _seed_edge_stats(iteration_summaries: list[dict]) -> dict[str, object]:
    """Count real vs noise edges in the final iteration summary."""
    if not iteration_summaries:
        return {"real_edges": 0, "noise_edges": 0, "noise_edge_names": []}
    edges = iteration_summaries[-1].get("edges", [])
    noise_edges = [e for e in edges
                   if any(n in e["edge"] for n in NOISE_INPUTS)]
    real_edges = [e for e in edges if e not in noise_edges]
    return {
        "real_edges": len(real_edges),
        "noise_edges": len(noise_edges),
        "noise_edge_names": [
            (e["edge"], float(e["confidence"])) for e in noise_edges
        ],
    }


PRICING = {
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}


def _run_vr_seed(
    seed: int,
    oracle: MediumOracleHD,
    ref_point: np.ndarray,
    n_budget: int = N_BUDGET_VR,
    file_prefix: str = "hd_vr",
    calibration_interval: int = 20,
    max_tokens: int = 16000,
    model: str = "claude-opus-4-6",
) -> dict | None:
    """Run one VR seed, save to disk, return summary dict. None on failure."""
    from synthoracle.agents.vr_tools import run_vr_tools

    out_npz = RESULTS_DIR / f"{file_prefix}_seed{seed}.npz"
    out_log = RESULTS_DIR / f"{file_prefix}_seed{seed}_log.json"

    pricing = PRICING.get(model, (5.0, 25.0))

    if out_npz.exists() and out_log.exists():
        data = np.load(out_npz)
        with open(out_log) as f:
            log = json.load(f)
        hv = float(data["hypervolumes"][-1])
        cost = (log.get("total_input_tokens", 0) * pricing[0] / 1e6
                + log.get("total_output_tokens", 0) * pricing[1] / 1e6)
        print(f"  Seed {seed}: already on disk, HV={hv:.4f} (cost=${cost:.2f})")
        return {
            "seed": seed,
            "hv": hv,
            "cost": cost,
            "eval_count": int(log.get("eval_count", len(data["hypervolumes"]))),
            "n_iters": len(log.get("iteration_summaries", [])),
            "hypervolumes": data["hypervolumes"].tolist(),
            **_seed_screening_stats(log.get("tool_calls", [])),
            **_seed_edge_stats(log.get("iteration_summaries", [])),
        }

    print(f"\n  Running VR HD seed {seed} ({n_budget} evals, 12 inputs, "
          f"{model})...", flush=True)
    try:
        r = run_vr_tools(
            oracle, n_budget=n_budget, seed=seed,
            thresholds=THRESHOLDS, reference_point=ref_point,
            model=model, thinking={"type": "adaptive"},
            max_tokens=max_tokens, max_tool_calls_per_iteration=15,
            checkpoint_dir=str(RESULTS_DIR / f"{file_prefix}_seed{seed}_ckpt"),
            calibration_interval=calibration_interval,
        )
    except Exception:
        print(f"  Seed {seed} FAILED:")
        traceback.print_exc()
        return None

    np.savez(
        out_npz, X=r.X, Y=r.Y,
        hypervolumes=np.array(r.hypervolumes),
        pareto_X=r.pareto_X, pareto_Y=r.pareto_Y,
        reference_point=r.reference_point,
    )
    with open(out_log, "w") as f:
        json.dump({
            "tool_calls": r.tool_calls,
            "mechanism_log": r.mechanism_log,
            "calibration_checks": r.calibration_checks,
            "iteration_summaries": r.iteration_summaries,
            "eval_count": r.eval_count,
            "total_llm_calls": r.total_llm_calls,
            "total_input_tokens": r.total_input_tokens,
            "total_output_tokens": r.total_output_tokens,
        }, f, indent=2)

    cost = (r.total_input_tokens * pricing[0] / 1e6
            + r.total_output_tokens * pricing[1] / 1e6)
    print(f"  Seed {seed}: HV={r.hypervolumes[-1]:.4f} cost=${cost:.2f} "
          f"evals={r.eval_count} iters={len(r.iteration_summaries)}")

    return {
        "seed": seed,
        "hv": float(r.hypervolumes[-1]),
        "cost": cost,
        "eval_count": r.eval_count,
        "n_iters": len(r.iteration_summaries),
        "hypervolumes": list(r.hypervolumes),
        **_seed_screening_stats(r.tool_calls),
        **_seed_edge_stats(r.iteration_summaries),
    }


def _print_vr_aggregate(
    summaries: list[dict],
    tag: str,
) -> None:
    """Shared aggregate printout for VR HD runs."""
    hvs = [s["hv"] for s in summaries]
    costs = [s["cost"] for s in summaries]
    oat_totals = [s["oat_total"] for s in summaries]
    oat_noise = [s["oat_noise"] for s in summaries]
    noise_edges = [s["noise_edges"] for s in summaries]
    real_edges = [s["real_edges"] for s in summaries]

    print(f"\n{'=' * 60}")
    print(f"  {tag} aggregate ({len(summaries)} seeds)")
    print(f"{'=' * 60}")
    print(f"  Final HV: {np.mean(hvs):.4f} +/- {np.std(hvs):.4f} "
          f"(min={np.min(hvs):.4f}, max={np.max(hvs):.4f})")
    print(f"  Total cost: ${sum(costs):.2f} "
          f"(mean ${np.mean(costs):.2f}/seed)")

    print(f"\n  Per-seed summary:")
    print(f"  {'seed':>5} {'HV':>8} {'evals':>6} {'iters':>5} "
          f"{'oat_real':>9} {'oat_noise':>10} {'real_edges':>11} "
          f"{'noise_edges':>12}")
    for s in summaries:
        print(f"  {s['seed']:>5} {s['hv']:>8.4f} {s['eval_count']:>6} "
              f"{s['n_iters']:>5} {s['oat_real']:>9} {s['oat_noise']:>10} "
              f"{s['real_edges']:>11} {s['noise_edges']:>12}")

    print(f"\n  Screening efficiency (across seeds):")
    print(f"    OAT total: {sum(oat_totals)} "
          f"(real {sum(oat_totals) - sum(oat_noise)}, "
          f"noise {sum(oat_noise)})")
    if sum(oat_totals):
        frac = sum(oat_noise) / sum(oat_totals)
        print(f"    Noise sweep fraction: {frac:.1%}")

    print(f"\n  Edge confidence (final iteration, across seeds):")
    print(f"    Real edges (mean):  {np.mean(real_edges):.1f}")
    print(f"    Noise edges (mean): {np.mean(noise_edges):.2f}")
    for s in summaries:
        bad = s.get("noise_edge_names", [])
        if bad:
            print(f"    seed {s['seed']} noise edges:")
            for name, conf in sorted(bad, key=lambda x: -x[1]):
                print(f"      {name}: {conf:.2f}")

    # BO comparison
    bo_hvs = []
    for seed in [42, 43, 44]:
        p = RESULTS_DIR / f"bo_hd_seed{seed}.npz"
        if p.exists():
            bo_hvs.append(float(np.load(p)["hypervolumes"][-1]))
    if bo_hvs:
        bo_mean = float(np.mean(bo_hvs))
        bo_std = float(np.std(bo_hvs))
        vr_mean = float(np.mean(hvs))
        print(f"\n  Comparison vs BO HD ({len(bo_hvs)} seeds):")
        print(f"    VR: {vr_mean:.4f} +/- {np.std(hvs):.4f}")
        print(f"    BO: {bo_mean:.4f} +/- {bo_std:.4f}")
        print(f"    VR/BO: {vr_mean / bo_mean:.1%}")


def run_vr_hd(
    seeds: list[int] = VR_SEEDS,
    n_budget: int = N_BUDGET_VR,
    file_prefix: str = "hd_vr",
    calibration_interval: int = 20,
    tag: str = "VR HD",
    max_tokens: int = 16000,
    model: str = "claude-opus-4-6",
) -> None:
    """Run VR agent on HD oracle across seeds, print aggregate stats."""
    oracle = MediumOracleHD()
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    summaries: list[dict] = []
    for seed in seeds:
        s = _run_vr_seed(
            seed, oracle, ref_point,
            n_budget=n_budget,
            file_prefix=file_prefix,
            calibration_interval=calibration_interval,
            max_tokens=max_tokens,
            model=model,
        )
        if s is not None:
            summaries.append(s)

    if not summaries:
        print("  No successful seeds.")
        return

    _print_vr_aggregate(summaries, tag)


def main() -> None:
    valid_modes = ("bo", "vr", "all", "vr-ext", "vr-sonnet", "vr-haiku")
    if len(sys.argv) < 2 or len(sys.argv) > 3 or sys.argv[1] not in valid_modes:
        print("Usage: python run_hd_test.py "
              "[bo|vr|all|vr-ext|vr-sonnet|vr-haiku] [seed]")
        print("  vr-ext:        run all extended seeds (144 evals, Opus)")
        print("  vr-ext <seed>: run a single extended seed")
        print("  vr-sonnet:     run Sonnet HD seeds 42-44 (72 evals)")
        print("  vr-haiku:      run Haiku HD seeds 42-44 (72 evals)")
        sys.exit(1)

    mode = sys.argv[1]

    if mode in ("bo", "all"):
        print("=" * 60)
        print("  BO baseline: HD oracle (3 seeds)")
        print("=" * 60)
        run_bo_hd()

    if mode in ("vr", "all"):
        print("\n" + "=" * 60)
        print(f"  VR agent: HD oracle ({len(VR_SEEDS)} seeds, 72 evals)")
        print("=" * 60)
        run_vr_hd()

    if mode == "vr-sonnet":
        print("\n" + "=" * 60)
        print(f"  VR agent: HD oracle Sonnet ({len(VR_SEEDS)} seeds, 72 evals)")
        print("=" * 60)
        run_vr_hd(
            seeds=VR_SEEDS,
            n_budget=N_BUDGET_VR,
            file_prefix="hd_vr_sonnet",
            calibration_interval=20,
            tag="VR HD Sonnet",
            model="claude-sonnet-4-6",
        )

    if mode == "vr-haiku":
        print("\n" + "=" * 60)
        print(f"  VR agent: HD oracle Haiku ({len(VR_SEEDS)} seeds, 72 evals)")
        print("  (Phase 2.3 pilot on 1A: iteration_summaries failed;")
        print("   expect partial rubric data — edge P/R + tool-call allocation)")
        print("=" * 60)
        run_vr_hd(
            seeds=VR_SEEDS,
            n_budget=N_BUDGET_VR,
            file_prefix="hd_vr_haiku",
            calibration_interval=20,
            tag="VR HD Haiku",
            model="claude-haiku-4-5-20251001",
        )

    if mode == "vr-ext":
        if len(sys.argv) == 3:
            try:
                single_seed = int(sys.argv[2])
            except ValueError:
                print(f"  Invalid seed: {sys.argv[2]}")
                sys.exit(1)
            seeds = [single_seed]
            label = f"1 seed ({single_seed})"
        else:
            seeds = VR_EXT_SEEDS
            label = f"{len(seeds)} seeds"
        print("\n" + "=" * 60)
        print(f"  VR agent: HD oracle extended ({label}, 144 evals, "
              f"calibration every 20, max_tokens=24000)")
        print("=" * 60)
        run_vr_hd(
            seeds=seeds,
            n_budget=N_BUDGET_VR_EXT,
            file_prefix="hd_vr_ext",
            calibration_interval=20,
            tag="VR HD ext",
            max_tokens=24000,
        )


if __name__ == "__main__":
    main()
