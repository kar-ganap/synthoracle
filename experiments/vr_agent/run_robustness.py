"""Robustness runs to close reviewer gaps.

1. BO baseline on 1E (3 more seeds: 43-45, seed 42 already done)
2. 1A extended budget (3 more seeds: 43-45 at 144 evals, no calibration)
3. Sonnet transfer on 1E (1 seed, with prior, to show results aren't Opus-specific)

Usage:
    # BO baselines (free, no API key needed)
    uv run --extra bo python experiments/vr_agent/run_robustness.py bo

    # 1A extended seeds (API required)
    source .env && uv run --extra vr python experiments/vr_agent/run_robustness.py extended

    # Sonnet 1E transfer (API required)
    source .env && uv run --extra vr python experiments/vr_agent/run_robustness.py sonnet_transfer
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np

from synthoracle.optim_utils import compute_reference_point, parse_directions

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_prior_knowledge() -> str:
    """Extract transferable causal model from 1A seed 42."""
    with open(RESULTS_DIR / "multi_seed" / "seed42_log.json") as f:
        log = json.load(f)
    last = log["iteration_summaries"][-1]
    lines = [
        "## Prior Causal Model (from a related system)",
        "",
        "A previous study on a similar system produced the following causal model.",
        "This system may share some structure but will have differences.",
        "",
        "### Hypothesis",
        last["hypothesis"],
        "",
        "### Discovered Edges",
    ]
    for e in sorted(last.get("edges", []), key=lambda x: -x["confidence"]):
        lines.append(
            f"- {e['edge']}: confidence={e['confidence']:.2f} — {e['evidence']}"
        )
    return "\n".join(lines)


def run_bo_1e() -> None:
    """Run 3 more BO seeds on 1E."""
    from synthoracle.baselines.bo import run_bo
    from synthoracle.oracles.medium_1e import MediumOracle1E

    oracle = MediumOracle1E()
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    all_hvs = []

    # Load existing seed 42
    p42 = RESULTS_DIR / "bo_1e_seed42.npz"
    if p42.exists():
        all_hvs.append(float(np.load(p42)["hypervolumes"][-1]))
        print(f"  seed 42 (existing): HV={all_hvs[-1]:.4f}")

    for seed in [43, 44, 45]:
        out = RESULTS_DIR / f"bo_1e_seed{seed}.npz"
        if out.exists():
            hv = float(np.load(out)["hypervolumes"][-1])
            all_hvs.append(hv)
            print(f"  seed {seed} (existing): HV={hv:.4f}")
            continue

        print(f"  Running BO 1E seed {seed}...", flush=True)
        r = run_bo(
            oracle, n_iterations=42, seed=seed,
            reference_point=ref_point, thresholds={"Y3": 0.4},
        )
        np.savez(
            out, X=r.X, Y=r.Y,
            hypervolumes=np.array(r.hypervolumes),
            reference_point=ref_point,
        )
        all_hvs.append(r.hypervolumes[-1])
        print(f"  seed {seed}: HV={all_hvs[-1]:.4f}")

    print(f"\nBO 1E ({len(all_hvs)} seeds): "
          f"{np.mean(all_hvs):.4f} +/- {np.std(all_hvs):.4f}")


def run_extended_1a() -> None:
    """Run 3 more seeds of 1A at 144 budget."""
    from synthoracle.agents.vr_tools import run_vr_tools
    from synthoracle.oracles.medium import MediumOracle

    oracle = MediumOracle()
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    bo_mean = np.mean([
        float(np.load(f"experiments/comparison/results/bo_seed{s}.npz")["hypervolumes"][-1])
        for s in range(42, 52)
    ])

    all_hvs = []

    # Load existing seed 42
    p42 = RESULTS_DIR / "1a_extended_144_seed42.npz"
    if p42.exists():
        hv = float(np.load(p42)["hypervolumes"][-1])
        all_hvs.append(hv)
        print(f"  seed 42 (existing): HV={hv:.4f}")

    for seed in [43, 44, 45]:
        out_npz = RESULTS_DIR / f"1a_extended_144_seed{seed}.npz"
        out_log = RESULTS_DIR / f"1a_extended_144_seed{seed}_log.json"
        if out_npz.exists():
            hv = float(np.load(out_npz)["hypervolumes"][-1])
            all_hvs.append(hv)
            print(f"  seed {seed} (existing): HV={hv:.4f}")
            continue

        print(f"\n  Running 1A extended seed {seed} (144 evals)...", flush=True)
        try:
            r = run_vr_tools(
                oracle, n_budget=144, seed=seed,
                thresholds={"Y3": 0.4}, reference_point=ref_point,
                model="claude-opus-4-6", thinking={"type": "adaptive"},
                max_tokens=16000, max_tool_calls_per_iteration=15,
                checkpoint_dir=str(RESULTS_DIR / f"1a_extended_144_seed{seed}_ckpt"),
                calibration_interval=0,
            )
        except Exception:
            print(f"  FAILED on seed {seed}:")
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

        cost = (r.total_input_tokens * 5.0 / 1e6
                + r.total_output_tokens * 25.0 / 1e6)
        hvs = r.hypervolumes
        all_hvs.append(hvs[-1])
        crossed = any(hv >= bo_mean for hv in hvs)
        cross_eval = next((i + 1 for i, hv in enumerate(hvs) if hv >= bo_mean), None)
        print(f"  seed {seed}: HV={hvs[-1]:.4f}, cost=${cost:.2f}, "
              f"crossed BO={'yes @ eval ' + str(cross_eval) if crossed else 'no'}")

    print(f"\n1A extended ({len(all_hvs)} seeds): "
          f"{np.mean(all_hvs):.4f} +/- {np.std(all_hvs):.4f}")
    print(f"BO mean: {bo_mean:.4f}")


def run_sonnet_transfer_1e() -> None:
    """Run Sonnet on 1E with prior to show results aren't Opus-specific."""
    from synthoracle.agents.vr_tools import run_vr_tools
    from synthoracle.oracles.medium_1e import MediumOracle1E

    oracle = MediumOracle1E()
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)
    prior = extract_prior_knowledge()

    out_npz = RESULTS_DIR / "transfer_1e_sonnet_prior_seed42.npz"
    out_log = RESULTS_DIR / "transfer_1e_sonnet_prior_seed42_log.json"

    if out_npz.exists():
        hv = float(np.load(out_npz)["hypervolumes"][-1])
        print(f"  Already exists: HV={hv:.4f}")
        return

    print("  Running Sonnet on 1E with prior (72 evals)...", flush=True)
    try:
        r = run_vr_tools(
            oracle, n_budget=72, seed=42,
            thresholds={"Y3": 0.4}, reference_point=ref_point,
            model="claude-sonnet-4-6", thinking={"type": "adaptive"},
            max_tokens=16000, max_tool_calls_per_iteration=15,
            checkpoint_dir=str(RESULTS_DIR / "transfer_1e_sonnet_prior_ckpt"),
            calibration_interval=20, prior_knowledge=prior,
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

    cost = (r.total_input_tokens * 3.0 / 1e6
            + r.total_output_tokens * 15.0 / 1e6)
    bo_hv = float(np.load(RESULTS_DIR / "bo_1e_seed42.npz")["hypervolumes"][-1])
    opus_hv = float(np.load(RESULTS_DIR / "transfer_1e_prior_seed42.npz")["hypervolumes"][-1])
    print(f"  Sonnet 1E prior: HV={r.hypervolumes[-1]:.4f}")
    print(f"  Opus 1E prior:   HV={opus_hv:.4f}")
    print(f"  BO 1E:           HV={bo_hv:.4f}")
    print(f"  Cost: ${cost:.2f}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("bo", "extended", "sonnet_transfer", "all"):
        print("Usage: python run_robustness.py [bo|extended|sonnet_transfer|all]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode in ("bo", "all"):
        print("=" * 60)
        print("  BO baseline: 1E (3 more seeds)")
        print("=" * 60)
        run_bo_1e()

    if mode in ("extended", "all"):
        print("\n" + "=" * 60)
        print("  1A extended budget: 3 more seeds at 144 evals")
        print("=" * 60)
        run_extended_1a()

    if mode in ("sonnet_transfer", "all"):
        print("\n" + "=" * 60)
        print("  Sonnet transfer: 1E with prior")
        print("=" * 60)
        run_sonnet_transfer_1e()


if __name__ == "__main__":
    main()
