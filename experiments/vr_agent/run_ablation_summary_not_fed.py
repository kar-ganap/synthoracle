"""Ablation: summary produced but NOT fed back into context.

The agent still produces the structured iteration summary (calls
messages.parse with the Pydantic schema) — so the cognitive work of
articulation happens. But the summary is NOT embedded in the condensed
conversation context. The next iteration gets status + output ranges +
recent evaluations but NOT "## Your Causal Model".

Tests whether the problem with VR is:
(a) The ACT of producing the summary (cognitive cost / thinking displacement)
(b) The PERSISTENCE of the summary in context (belief anchoring)

If this condition ≈ ablation (no summary): cementing is the problem.
If this condition ≈ VR (full summary): cognitive cost is the problem.

Usage:
    source .env && uv run --extra vr python experiments/vr_agent/run_ablation_summary_not_fed.py
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
MODEL = "claude-opus-4-6"
THINKING: dict[str, object] = {"type": "adaptive"}
THRESHOLDS = {"Y3": 0.4}
PRICING = (5.0, 25.0)


def main() -> None:
    oracle = MediumOracle(variant="1A")
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    print("=" * 60)
    print("  Ablation: summary produced but NOT fed back into context")
    print("  Oracle: 1A, Budget: 72, Model: Opus")
    print("  Agent produces structured summary → logged but NOT in condensed context")
    print("=" * 60)

    results = []
    for seed in SEEDS:
        out_npz = RESULTS_DIR / f"ablation_summary_not_fed_seed{seed}.npz"
        out_log = RESULTS_DIR / f"ablation_summary_not_fed_seed{seed}_log.json"

        if out_npz.exists() and out_log.exists():
            data = np.load(out_npz)
            hv = float(data["hypervolumes"][-1])
            print(f"  Seed {seed}: already on disk (HV={hv:.4f})")
            results.append({"seed": seed, "hv": hv})
            continue

        print(f"\n  Running summary-not-fed seed {seed}...", flush=True)
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
                checkpoint_dir=str(RESULTS_DIR / f"ablation_summary_not_fed_seed{seed}_ckpt"),
                calibration_interval=20,
                skip_iteration_summary=False,   # DO produce summary
                summary_in_context=False,        # but DON'T feed it back
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
              f"iters={len(r.iteration_summaries)}, "
              f"time={r.total_seconds:.0f}s")
        results.append({"seed": seed, "hv": float(r.hypervolumes[-1]), "cost": cost})

    if not results:
        print("\n  No successful runs.")
        return

    hvs = [r["hv"] for r in results]
    print(f"\n{'=' * 60}")
    print(f"  Summary-not-fed aggregate (n={len(results)})")
    print(f"{'=' * 60}")
    print(f"  HV: {np.mean(hvs):.4f} ± {np.std(hvs):.4f}")

    # 3-way comparison
    print(f"\n  3-way comparison on 1A:")
    print(f"  VR with summary (n=10):      0.1926 ± 0.028")
    print(f"  Summary not fed back (n={len(results)}): {np.mean(hvs):.4f} ± {np.std(hvs):.4f}")
    print(f"  Ablation no summary (n=5):   0.2586 ± 0.014")
    print(f"  BO @ 144:                    0.2811")


if __name__ == "__main__":
    main()
