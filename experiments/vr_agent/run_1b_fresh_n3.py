"""1B fresh multi-seed (n=3) for Rule 8 parity with 1D.

Runs VR on MediumOracle(variant='1B') with no prior knowledge, seeds 42, 43, 44.
Seed 42 already exists (from Phase 2.3 pilot) and loads from disk.

Motivation: Rule 8 in the difficulty rubric uses 1B and 1D as low-R²(M→Y)
data points to test whether low mechanism sufficiency predicts poor VR/BO
ratio. With 1D at n=3 and 1B at n=1, the statistical asymmetry is a
reviewer magnet. This script brings 1B fresh to n=3.

Note: 1B fresh (no prior) is the right condition for this test. 1B prior
is a useless transfer test because 1B has identical IO topology to 1A
(18/18 edges shared, 0 wrong, 0 missing — only Y2 functional change).

Usage:
    source .env && uv run --extra vr python experiments/vr_agent/run_1b_fresh_n3.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from synthoracle.agents.vr_tools import run_vr_tools
from synthoracle.oracles.medium import MediumOracle

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
N_BUDGET = 72
MODEL = "claude-opus-4-6"
THINKING: dict[str, object] = {"type": "adaptive"}
THRESHOLDS = {"Y3": 0.4}
PRICING = (5.0, 25.0)


def run_one(seed: int) -> dict | None:
    """Run one seed of 1B fresh. Returns summary dict or None if failed."""
    out_npz = RESULTS_DIR / f"no_transfer_1b_seed{seed}.npz"
    out_log = RESULTS_DIR / f"no_transfer_1b_seed{seed}_log.json"
    ckpt = RESULTS_DIR / f"no_transfer_1b_seed{seed}_ckpt"

    if out_npz.exists() and out_log.exists():
        with open(out_log) as f:
            log = json.load(f)
        data = np.load(out_npz)
        hv = float(data["hypervolumes"][-1])
        cost = (log.get("total_input_tokens", 0) * PRICING[0] / 1e6
                + log.get("total_output_tokens", 0) * PRICING[1] / 1e6)
        print(f"  seed {seed}: already on disk (HV={hv:.4f}, cost=${cost:.2f})")
        return {"seed": seed, "hv": hv, "cost": cost, "from_disk": True}

    print(f"\n  Running 1B fresh seed {seed} (72 evals, no prior)...",
          flush=True)
    oracle = MediumOracle(variant="1B")
    try:
        result = run_vr_tools(
            oracle,
            n_budget=N_BUDGET,
            seed=seed,
            thresholds=THRESHOLDS,
            model=MODEL,
            thinking=THINKING,
            max_tokens=16000,
            max_tool_calls_per_iteration=15,
            checkpoint_dir=str(ckpt),
            calibration_interval=20,
            prior_knowledge=None,  # fresh — no 1A prior
        )
    except Exception:
        print(f"  seed {seed} FAILED:")
        traceback.print_exc()
        return None

    np.savez(
        out_npz, X=result.X, Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
    )
    with open(out_log, "w") as f:
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

    cost = (result.total_input_tokens * PRICING[0] / 1e6
            + result.total_output_tokens * PRICING[1] / 1e6)
    print(f"  seed {seed}: HV={result.hypervolumes[-1]:.4f}, "
          f"cost=${cost:.2f}, time={result.total_seconds:.0f}s")
    return {
        "seed": seed,
        "hv": float(result.hypervolumes[-1]),
        "cost": cost,
        "from_disk": False,
    }


def main() -> None:
    print("=" * 60)
    print("  1B fresh multi-seed (n=3) for Rule 8 parity with 1D")
    print("=" * 60)
    print(f"Oracle: MediumOracle(variant='1B')")
    print(f"Seeds: {SEEDS}")
    print(f"Budget: {N_BUDGET} evals per seed")

    results = []
    for seed in SEEDS:
        r = run_one(seed)
        if r is not None:
            results.append(r)

    if not results:
        print("\n  No successful runs.")
        return

    hvs = [r["hv"] for r in results]
    cost_new = sum(r["cost"] for r in results if not r.get("from_disk"))

    print(f"\n{'=' * 60}")
    print(f"  1B fresh aggregate (n={len(results)})")
    print(f"{'=' * 60}")
    print(f"  HV mean: {np.mean(hvs):.4f}")
    print(f"  HV std:  {np.std(hvs):.4f}")
    print(f"  HV range: [{np.min(hvs):.4f}, {np.max(hvs):.4f}]")
    print(f"  New spend: ${cost_new:.2f}")

    # Reference BO 1B for VR/BO
    bo_path = Path("experiments/bo_baseline/results/medium_1b_seed42.npz")
    if bo_path.exists():
        bo = np.load(bo_path)
        bo_hv = float(bo["hypervolumes"][-1])
        vr_over_bo = float(np.mean(hvs)) / bo_hv
        print(f"\n  vs BO 1B (n=1): BO={bo_hv:.4f}, "
              f"VR/BO = {vr_over_bo:.3f}")


if __name__ == "__main__":
    main()
