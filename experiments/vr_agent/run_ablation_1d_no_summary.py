"""Ablation: 1D fresh without forced iteration summary.

Tests whether the summary penalty (35% on 1A) or benefit (14% on HD)
transfers to 1D — the oracle with correct topology but wrong functional
forms (inverted-U on X3→Y3, removed X5 threshold, Y2 sign flip).

Prediction: summary hurts on 1D by ~10-25% (wrong functional forms
get cemented, but topology is correct so no real-edge dismissals).

Comparison: 1D fresh VR with summary (n=3): HV = 1.186 ± 0.074.

Usage:
    source .env && uv run --extra vr python experiments/vr_agent/run_ablation_1d_no_summary.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from synthoracle.agents.vr_tools import run_vr_tools
from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracles.medium_1d import MediumOracle1D

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
N_BUDGET = 72
MODEL = "claude-opus-4-6"
THINKING: dict[str, object] = {"type": "adaptive"}
THRESHOLDS = {"Y3": 0.4}
PRICING = (5.0, 25.0)


def main() -> None:
    oracle = MediumOracle1D()
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    print("=" * 60)
    print("  Ablation: 1D fresh without forced iteration summary")
    print("  Oracle: 1D (Structural Shift), Budget: 72, Model: Opus")
    print("=" * 60)

    results = []
    for seed in SEEDS:
        out_npz = RESULTS_DIR / f"ablation_1d_no_summary_seed{seed}.npz"
        out_log = RESULTS_DIR / f"ablation_1d_no_summary_seed{seed}_log.json"

        if out_npz.exists() and out_log.exists():
            data = np.load(out_npz)
            hv = float(data["hypervolumes"][-1])
            print(f"  Seed {seed}: already on disk (HV={hv:.4f})")
            results.append({"seed": seed, "hv": hv})
            continue

        print(f"\n  Running 1D ablation seed {seed} (no summary)...", flush=True)
        try:
            r = run_vr_tools(
                oracle,
                n_budget=N_BUDGET,
                seed=seed,
                thresholds=THRESHOLDS,
                reference_point=ref_point,
                model=MODEL,
                thinking=THINKING,
                max_tokens=16000,
                max_tool_calls_per_iteration=15,
                checkpoint_dir=str(RESULTS_DIR / f"ablation_1d_no_summary_seed{seed}_ckpt"),
                calibration_interval=20,
                skip_iteration_summary=True,
            )
        except Exception:
            print(f"  Seed {seed} FAILED:")
            traceback.print_exc()
            continue

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

        cost = (r.total_input_tokens * PRICING[0] / 1e6
                + r.total_output_tokens * PRICING[1] / 1e6)
        print(f"  Seed {seed}: HV={r.hypervolumes[-1]:.4f}, "
              f"cost=${cost:.2f}, evals={r.eval_count}, "
              f"time={r.total_seconds:.0f}s")
        results.append({"seed": seed, "hv": float(r.hypervolumes[-1]), "cost": cost})

    if not results:
        print("\n  No successful runs.")
        return

    hvs = [r["hv"] for r in results]
    print(f"\n{'=' * 60}")
    print(f"  1D ablation aggregate (n={len(results)})")
    print(f"{'=' * 60}")
    print(f"  HV: {np.mean(hvs):.4f} ± {np.std(hvs):.4f}")

    # Compare to 1D fresh VR with summary
    vr_hvs = []
    for s in [42, 43, 44]:
        p = RESULTS_DIR / f"transfer_1d_fresh_seed{s}.npz"
        if p.exists():
            vr_hvs.append(float(np.load(p)["hypervolumes"][-1]))
    if vr_hvs:
        print(f"\n  Comparison:")
        print(f"  1D fresh VR with summary (n={len(vr_hvs)}): {np.mean(vr_hvs):.4f}")
        print(f"  1D ablation no summary (n={len(results)}):  {np.mean(hvs):.4f}")
        diff = float(np.mean(hvs)) - float(np.mean(vr_hvs))
        pct = diff / float(np.mean(vr_hvs)) * 100
        print(f"  Difference: {diff:+.4f} ({pct:+.1f}%)")

    # BO comparison
    bo_hvs = []
    for s in [42, 43]:
        p = RESULTS_DIR / f"bo_1d_seed{s}.npz"
        if p.exists():
            bo_hvs.append(float(np.load(p)["hypervolumes"][-1]))
    if bo_hvs:
        print(f"  BO 1D (n={len(bo_hvs)}): {np.mean(bo_hvs):.4f}")
        print(f"  Ablation / BO: {np.mean(hvs)/np.mean(bo_hvs):.1%}")


if __name__ == "__main__":
    main()
