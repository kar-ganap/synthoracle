"""High-dimensional oracle test: 12 inputs (6 real + 6 noise).

Tests whether the VR agent can screen and dismiss irrelevant variables.
Runs VR (1 seed) and BO (3 seeds) for comparison.

Usage:
    # BO baseline (free)
    uv run --extra bo python experiments/vr_agent/run_hd_test.py bo

    # VR agent (API required)
    source .env && uv run --extra vr python experiments/vr_agent/run_hd_test.py vr

    # Both
    source .env && uv run --extra bo --extra vr python experiments/vr_agent/run_hd_test.py all
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

SEED = 42
N_BUDGET_VR = 72
N_ITERATIONS_BO = 42
THRESHOLDS = {"Y3": 0.4}


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


def run_vr_hd() -> None:
    """Run VR agent on HD oracle (1 seed, full diagnostics)."""
    from synthoracle.agents.vr_tools import run_vr_tools

    oracle = MediumOracleHD()
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    out_npz = RESULTS_DIR / "hd_vr_seed42.npz"
    out_log = RESULTS_DIR / "hd_vr_seed42_log.json"

    if out_npz.exists():
        hv = float(np.load(out_npz)["hypervolumes"][-1])
        print(f"  Already exists: HV={hv:.4f}")
        return

    print("  Running VR HD seed 42 (72 evals, 12 inputs)...", flush=True)
    try:
        r = run_vr_tools(
            oracle, n_budget=N_BUDGET_VR, seed=SEED,
            thresholds=THRESHOLDS, reference_point=ref_point,
            model="claude-opus-4-6", thinking={"type": "adaptive"},
            max_tokens=16000, max_tool_calls_per_iteration=15,
            checkpoint_dir=str(RESULTS_DIR / "hd_vr_ckpt"),
            calibration_interval=20,
        )
    except Exception:
        print("  FAILED:")
        traceback.print_exc()
        return

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

    cost = (r.total_input_tokens * 5.0 / 1e6
            + r.total_output_tokens * 25.0 / 1e6)
    hvs = r.hypervolumes

    # Analyze screening efficiency
    oat_sweeps = [tc for tc in r.tool_calls if tc["name"] == "oat_sweep"]
    noise_sweeps = [tc for tc in oat_sweeps
                    if tc["input"]["input_name"] in
                    ("X7", "X8", "X9", "X10", "X11", "X12")]
    real_sweeps = [tc for tc in oat_sweeps
                   if tc["input"]["input_name"] not in
                   ("X7", "X8", "X9", "X10", "X11", "X12")]

    print(f"\n  Results:")
    print(f"  Final HV: {hvs[-1]:.4f}")
    print(f"  Cost: ${cost:.2f}")
    print(f"  Evals: {r.eval_count}, Iters: {len(r.iteration_summaries)}")
    print(f"\n  Screening efficiency:")
    print(f"  OAT sweeps total: {len(oat_sweeps)}")
    print(f"    Real inputs (X1-X6): {len(real_sweeps)}")
    print(f"    Noise inputs (X7-X12): {len(noise_sweeps)}")
    print(f"    Noise sweep fraction: {len(noise_sweeps)/len(oat_sweeps):.0%}"
          if oat_sweeps else "    No OAT sweeps")

    # Check edge confidence for noise dimensions
    if r.iteration_summaries:
        last = r.iteration_summaries[-1]
        edges = last.get("edges", [])
        noise_edges = [e for e in edges
                       if any(f"X{i}" in e["edge"]
                              for i in range(7, 13))]
        real_edges = [e for e in edges
                      if not any(f"X{i}" in e["edge"]
                                 for i in range(7, 13))]
        print(f"\n  Edge confidence (final iteration):")
        print(f"    Real edges: {len(real_edges)}")
        print(f"    Noise edges: {len(noise_edges)}")
        if noise_edges:
            for e in sorted(noise_edges, key=lambda x: -x["confidence"]):
                print(f"      {e['edge']}: {e['confidence']:.2f}")

    # Load BO for comparison
    bo_hvs = []
    for seed in [42, 43, 44]:
        p = RESULTS_DIR / f"bo_hd_seed{seed}.npz"
        if p.exists():
            bo_hvs.append(float(np.load(p)["hypervolumes"][-1]))
    if bo_hvs:
        bo_mean = np.mean(bo_hvs)
        print(f"\n  Comparison:")
        print(f"  VR: {hvs[-1]:.4f}")
        print(f"  BO: {bo_mean:.4f}")
        print(f"  VR/BO: {hvs[-1]/bo_mean:.1%}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("bo", "vr", "all"):
        print("Usage: python run_hd_test.py [bo|vr|all]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode in ("bo", "all"):
        print("=" * 60)
        print("  BO baseline: HD oracle (3 seeds)")
        print("=" * 60)
        run_bo_hd()

    if mode in ("vr", "all"):
        print("\n" + "=" * 60)
        print("  VR agent: HD oracle (1 seed)")
        print("=" * 60)
        run_vr_hd()


if __name__ == "__main__":
    main()
