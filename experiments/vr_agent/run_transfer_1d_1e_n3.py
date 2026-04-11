"""Transfer tests at n=3: 1D and 1E with/without 1A prior, seeds 42-44.

Extends the original Phase 2.5 pilot (run_transfer_1d_1e.py, n=1) to n=3
per condition for the difficulty rubric. Reuses the same prior extraction
and screen-first protocol as the original.

Skips runs already on disk (so seed 42 of each condition loads for free).

Usage:
    source .env && uv run --extra vr python experiments/vr_agent/run_transfer_1d_1e_n3.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from synthoracle.agents.vr_tools import VRToolsResult, run_vr_tools
from synthoracle.oracle import Oracle
from synthoracle.oracles.medium_1d import MediumOracle1D
from synthoracle.oracles.medium_1e import MediumOracle1E

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
N_BUDGET = 72
MODEL = "claude-opus-4-6"
THINKING: dict[str, object] = {"type": "adaptive"}
THRESHOLDS = {"Y3": 0.4}
PRICING = (5.0, 25.0)

PRIOR_SOURCE = Path("experiments/vr_agent/results/multi_seed/seed42_log.json")


# ---------------------------------------------------------------------------
# Prior extraction (same as original Phase 2.5 script)
# ---------------------------------------------------------------------------


def extract_prior_knowledge() -> str:
    """Extract transferable causal model from 1A seed 42."""
    with open(PRIOR_SOURCE) as f:
        log = json.load(f)

    last = log["iteration_summaries"][-1]
    hypothesis = last["hypothesis"]
    edges = last.get("edges", [])

    lines = [
        "## Prior Causal Model (from a related system)",
        "",
        "A previous study on a similar system produced the following causal model.",
        "This system may share some structure but will have differences.",
        "",
        "### Hypothesis",
        hypothesis,
        "",
        "### Discovered Edges (with confidence 0-1 and evidence)",
    ]
    for e in sorted(edges, key=lambda x: -x["confidence"]):
        lines.append(
            f"- {e['edge']}: confidence={e['confidence']:.2f} — {e['evidence']}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-seed run
# ---------------------------------------------------------------------------


def run_one(
    label: str, oracle: Oracle, seed: int, prior: str | None,
) -> dict | None:
    """Run one seed of a condition. Returns summary dict or None if failed."""
    out_npz = RESULTS_DIR / f"{label}_seed{seed}.npz"
    out_log = RESULTS_DIR / f"{label}_seed{seed}_log.json"
    ckpt = RESULTS_DIR / f"{label}_seed{seed}_ckpt"

    if out_npz.exists() and out_log.exists():
        with open(out_log) as f:
            log = json.load(f)
        d = np.load(out_npz)
        hv = float(d["hypervolumes"][-1])
        cost = (log.get("total_input_tokens", 0) * PRICING[0] / 1e6
                + log.get("total_output_tokens", 0) * PRICING[1] / 1e6)
        print(f"  {label} seed {seed}: already on disk (HV={hv:.4f}, "
              f"cost=${cost:.2f})")
        return {"seed": seed, "hv": hv, "cost": cost, "loaded_from_disk": True}

    print(f"\n  Running {label} seed {seed} ...", flush=True)
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
            prior_knowledge=prior,
        )
    except Exception:
        print(f"  {label} seed {seed} FAILED:")
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
    print(f"  {label} seed {seed}: HV={result.hypervolumes[-1]:.4f}, "
          f"cost=${cost:.2f}, time={result.total_seconds:.0f}s")

    return {
        "seed": seed,
        "hv": float(result.hypervolumes[-1]),
        "cost": cost,
        "loaded_from_disk": False,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    o1d = MediumOracle1D()
    o1e = MediumOracle1E()
    prior = extract_prior_knowledge()

    print(f"Prior knowledge: {len(prior)} chars from {PRIOR_SOURCE}")
    print(f"Seeds to run: {SEEDS}")

    conditions: list[tuple[str, Oracle, str | None]] = [
        ("transfer_1d_prior", o1d, prior),
        ("transfer_1d_fresh", o1d, None),
        ("transfer_1e_prior", o1e, prior),
        ("transfer_1e_fresh", o1e, None),
    ]

    results: dict[str, list[dict]] = {}
    for label, oracle, prior_knowledge in conditions:
        print(f"\n{'=' * 60}")
        print(f"  Condition: {label}")
        print(f"{'=' * 60}")
        seed_results = []
        for seed in SEEDS:
            r = run_one(label, oracle, seed, prior_knowledge)
            if r is not None:
                seed_results.append(r)
        results[label] = seed_results

    # Aggregate
    print(f"\n{'=' * 60}")
    print(f"  Multi-seed transfer aggregate")
    print(f"{'=' * 60}")
    print(f"  {'condition':<22} {'n':>3} {'HV mean':>10} {'HV std':>10} "
          f"{'cost total':>12}")
    for label, seed_results in results.items():
        if not seed_results:
            print(f"  {label:<22} 0   no successful runs")
            continue
        hvs = [r["hv"] for r in seed_results]
        cost_new = sum(r["cost"] for r in seed_results
                       if not r.get("loaded_from_disk"))
        print(f"  {label:<22} {len(hvs):>3} {np.mean(hvs):>10.4f} "
              f"{np.std(hvs):>10.4f} ${cost_new:>11.2f}")

    # Prior penalty per variant
    print(f"\n  Prior penalty (n=3 mean):")
    for variant in ["1d", "1e"]:
        prior_label = f"transfer_{variant}_prior"
        fresh_label = f"transfer_{variant}_fresh"
        prior_runs = results.get(prior_label, [])
        fresh_runs = results.get(fresh_label, [])
        if prior_runs and fresh_runs:
            hv_p = np.mean([r["hv"] for r in prior_runs])
            hv_f = np.mean([r["hv"] for r in fresh_runs])
            penalty = (hv_f - hv_p) / hv_f if hv_f > 0 else 0.0
            print(f"    {variant.upper()}: prior {hv_p:.4f} vs fresh {hv_f:.4f} "
                  f"→ penalty {penalty * 100:+.1f}%")


if __name__ == "__main__":
    main()
