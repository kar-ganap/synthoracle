"""Sonnet 1A: with summary AND without summary (n=3 each).

Completes the 2×2 matrix (model × summary) on 1A to test whether
the summary-hurts direction is model-general.

Usage:
    source .env && uv run --extra vr python experiments/vr_agent/run_sonnet_1a_matrix.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from synthoracle.agents.vr_tools import run_vr_tools
from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracles.medium import MediumOracle

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
N_BUDGET = 72
MODEL = "claude-sonnet-4-6"
THINKING: dict[str, object] = {"type": "adaptive"}
THRESHOLDS = {"Y3": 0.4}
PRICING = (3.0, 15.0)


def run_condition(label: str, seed: int, ref_point, oracle, skip_summary: bool) -> dict | None:
    out_npz = RESULTS_DIR / f"{label}_seed{seed}.npz"
    out_log = RESULTS_DIR / f"{label}_seed{seed}_log.json"

    if out_npz.exists() and out_log.exists():
        data = np.load(out_npz)
        hv = float(data["hypervolumes"][-1])
        print(f"  {label} seed {seed}: already on disk (HV={hv:.4f})")
        return {"seed": seed, "hv": hv}

    print(f"\n  Running {label} seed {seed}...", flush=True)
    try:
        r = run_vr_tools(
            oracle, n_budget=N_BUDGET, seed=seed,
            thresholds=THRESHOLDS, reference_point=ref_point,
            model=MODEL, thinking=THINKING,
            max_tokens=16000, max_tool_calls_per_iteration=15,
            checkpoint_dir=str(RESULTS_DIR / f"{label}_seed{seed}_ckpt"),
            calibration_interval=20,
            skip_iteration_summary=skip_summary,
        )
    except Exception:
        print(f"  {label} seed {seed} FAILED:")
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

    cost = (r.total_input_tokens * PRICING[0] / 1e6
            + r.total_output_tokens * PRICING[1] / 1e6)
    print(f"  {label} seed {seed}: HV={r.hypervolumes[-1]:.4f}, "
          f"cost=${cost:.2f}, evals={r.eval_count}, "
          f"time={r.total_seconds:.0f}s")
    return {"seed": seed, "hv": float(r.hypervolumes[-1]), "cost": cost}


def main() -> None:
    oracle = MediumOracle(variant="1A")
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    print("=" * 60)
    print("  Sonnet 1A: with summary + without summary (n=3 each)")
    print("  Completes the 2×2 matrix (model × summary)")
    print("=" * 60)

    conditions = [
        ("sonnet_1a_with_summary", False),    # skip_summary=False → WITH summary
        ("sonnet_1a_no_summary", True),        # skip_summary=True → WITHOUT summary
    ]

    all_results: dict[str, list[dict]] = {}
    for label, skip in conditions:
        print(f"\n{'='*60}")
        print(f"  Condition: {label} (skip_summary={skip})")
        print(f"{'='*60}")
        results = []
        for seed in SEEDS:
            r = run_condition(label, seed, ref_point, oracle, skip)
            if r is not None:
                results.append(r)
        all_results[label] = results

    # Summary
    print(f"\n{'='*60}")
    print(f"  Sonnet 1A 2-condition comparison")
    print(f"{'='*60}")
    for label, results in all_results.items():
        if results:
            hvs = [r["hv"] for r in results]
            print(f"  {label} (n={len(results)}): {np.mean(hvs):.4f} ± {np.std(hvs):.4f}")

    # Full 2×2 matrix
    with_sum = all_results.get("sonnet_1a_with_summary", [])
    no_sum = all_results.get("sonnet_1a_no_summary", [])
    if with_sum and no_sum:
        w = np.mean([r["hv"] for r in with_sum])
        n = np.mean([r["hv"] for r in no_sum])
        diff = n - w
        pct = diff / w * 100 if w > 0 else 0
        print(f"\n  Summary effect on Sonnet 1A: {diff:+.4f} ({pct:+.1f}%)")
        if diff > 0.005:
            print(f"  → Summary HURTS Sonnet on 1A (removing it helps)")
        elif diff < -0.005:
            print(f"  → Summary HELPS Sonnet on 1A")
        else:
            print(f"  → Neutral")

    print(f"\n  Full 2×2 matrix:")
    print(f"  {'':>25} {'With summary':>15} {'No summary':>15} {'Effect':>10}")
    print(f"  {'Opus 1A':>25} {'0.193':>15} {'0.259':>15} {'-35%':>10}")
    if with_sum and no_sum:
        w_s = f"{np.mean([r['hv'] for r in with_sum]):.3f}"
        n_s = f"{np.mean([r['hv'] for r in no_sum]):.3f}"
        print(f"  {'Sonnet 1A':>25} {w_s:>15} {n_s:>15} {f'{pct:+.0f}%':>10}")
    print(f"\n  {'Opus HD':>25} {'0.257':>15} {'0.222':>15} {'+16%':>10}")
    print(f"  {'Sonnet HD':>25} {'0.254':>15} {'0.188':>15} {'+35%':>10}")


if __name__ == "__main__":
    main()
