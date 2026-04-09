"""Transfer tests: 1D (Structural Shift) and 1E (Rewired) with/without 1A prior.

Pilot: 1 seed per condition (4 runs total). Tests whether the 1A causal model
transfers to variants where the prior is partially wrong.

Usage:
    source .env && uv run --extra vr python experiments/vr_agent/run_transfer_1d_1e.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthoracle.agents.vr_tools import VRToolsResult, run_vr_tools
from synthoracle.oracles.medium_1d import MediumOracle1D
from synthoracle.oracles.medium_1e import MediumOracle1E
from synthoracle.oracle import Oracle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BUDGET = 72
MODEL = "claude-opus-4-6"
THINKING: dict[str, object] = {"type": "adaptive"}
THRESHOLDS = {"Y3": 0.4}
PRICING = (5.0, 25.0)

PRIOR_SOURCE = Path("experiments/vr_agent/results/multi_seed/seed42_log.json")


# ---------------------------------------------------------------------------
# Prior knowledge extraction
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
# Run + save
# ---------------------------------------------------------------------------


def run_condition(
    label: str,
    oracle: Oracle,
    prior: str | None,
) -> VRToolsResult | None:
    """Run one condition and save results."""
    ckpt = str(RESULTS_DIR / f"{label}_ckpt")

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Oracle: {oracle.__class__.__name__}")
    print(f"  Prior: {'YES' if prior else 'NO'}")
    print(f"{'=' * 60}\n")

    try:
        result = run_vr_tools(
            oracle,
            n_budget=N_BUDGET,
            seed=SEED,
            thresholds=THRESHOLDS,
            model=MODEL,
            thinking=THINKING,
            max_tokens=16000,
            max_tool_calls_per_iteration=15,
            checkpoint_dir=ckpt,
            calibration_interval=20,
            prior_knowledge=prior,
        )
    except Exception:
        print(f"  FAILED on {label}:")
        traceback.print_exc()
        return None

    # Save
    np.savez(
        RESULTS_DIR / f"{label}_seed{SEED}.npz",
        X=result.X, Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
    )
    with open(RESULTS_DIR / f"{label}_seed{SEED}_log.json", "w") as f:
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
    print(f"\n  {label} complete: HV={result.hypervolumes[-1]:.6f}, "
          f"evals={result.eval_count}, cost=${cost:.2f}, "
          f"time={result.total_seconds:.0f}s")

    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def print_comparison(results: dict[str, VRToolsResult]) -> None:
    """Print cross-condition comparison table."""
    print(f"\n{'=' * 80}")
    print(f"  Transfer Test Results")
    print(f"{'=' * 80}")

    print(f"\n  {'Condition':<25} {'HV':>8} {'Evals':>6} {'Iters':>6} "
          f"{'Edges':>6} {'Cost':>7}")
    print(f"  {'-' * 62}")

    for label, r in results.items():
        n_edges = 0
        if r.iteration_summaries:
            last = r.iteration_summaries[-1]
            n_edges = len(last.get("edges", []))
        cost = (r.total_input_tokens * PRICING[0] / 1e6
                + r.total_output_tokens * PRICING[1] / 1e6)
        print(f"  {label:<25} {r.hypervolumes[-1]:>8.4f} {r.eval_count:>6} "
              f"{len(r.iteration_summaries):>6} {n_edges:>6} ${cost:>6.2f}")

    # Calibration comparison
    print(f"\n  Calibration:")
    for label, r in results.items():
        valid = [c for c in r.calibration_checks
                 if not np.isnan(c.get("mae", float("nan")))]
        if valid:
            first = valid[0]["mae"]
            last_val = valid[-1]["mae"]
            print(f"    {label}: {first:.4f} -> {last_val:.4f} "
                  f"({'LEARNING' if last_val < first * 0.8 else 'STABLE'})")
        else:
            print(f"    {label}: no calibration data")

    # Prior impact
    print(f"\n  Prior Impact:")
    for variant in ["1d", "1e"]:
        prior_label = f"transfer_{variant}_prior"
        fresh_label = f"transfer_{variant}_fresh"
        if prior_label in results and fresh_label in results:
            hv_prior = results[prior_label].hypervolumes[-1]
            hv_fresh = results[fresh_label].hypervolumes[-1]
            diff = hv_prior - hv_fresh
            pct = diff / hv_fresh * 100 if hv_fresh > 0 else 0
            verdict = "HELPS" if diff > 0.005 else ("HURTS" if diff < -0.005 else "NEUTRAL")
            print(f"    {variant.upper()}: prior HV={hv_prior:.4f}, "
                  f"fresh HV={hv_fresh:.4f}, diff={diff:+.4f} ({pct:+.1f}%) — {verdict}")

    # Iteration summaries: what did the agent learn?
    print(f"\n  Iteration Summary Highlights:")
    for label, r in results.items():
        if not r.iteration_summaries:
            continue
        print(f"\n    {label}:")
        for s in r.iteration_summaries:
            n_edges = len(s.get("edges", []))
            n_surprises = len(s.get("surprises", []))
            print(f"      Iter {s['iteration']} @ eval {s['eval_count']}: "
                  f"{n_edges} edges, {n_surprises} surprises")
            for surp in s.get("surprises", [])[:2]:
                print(f"        - {surp[:100]}")


def plot_hv_comparison(results: dict[str, VRToolsResult]) -> None:
    """Plot HV curves for all conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"prior": "#d62728", "fresh": "#1f77b4"}

    for ax, variant, title in [
        (axes[0], "1d", "Variant 1D (Structural Shift)"),
        (axes[1], "1e", "Variant 1E (Rewired)"),
    ]:
        for cond, ls in [("prior", "-"), ("fresh", "--")]:
            label = f"transfer_{variant}_{cond}"
            if label not in results:
                continue
            r = results[label]
            evals = list(range(1, len(r.hypervolumes) + 1))
            ax.plot(evals, r.hypervolumes, color=colors[cond],
                    linestyle=ls, linewidth=2,
                    label=f"With prior" if cond == "prior" else "Fresh")

        ax.axvline(x=12, color="gray", linestyle="--", alpha=0.2)
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Hypervolume")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "transfer_1d_1e_hv.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  HV plot saved: {RESULTS_DIR / 'transfer_1d_1e_hv.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    o1d = MediumOracle1D()
    o1e = MediumOracle1E()
    prior = extract_prior_knowledge()

    print(f"Prior knowledge: {len(prior)} chars, from {PRIOR_SOURCE}")

    conditions: list[tuple[str, Oracle, str | None]] = [
        ("transfer_1d_prior", o1d, prior),
        ("transfer_1d_fresh", o1d, None),
        ("transfer_1e_prior", o1e, prior),
        ("transfer_1e_fresh", o1e, None),
    ]

    results: dict[str, VRToolsResult] = {}
    for label, oracle, prior_knowledge in conditions:
        r = run_condition(label, oracle, prior_knowledge)
        if r is not None:
            results[label] = r

    if results:
        print_comparison(results)
        plot_hv_comparison(results)

    print(f"\n  {len(results)}/{len(conditions)} conditions completed.")


if __name__ == "__main__":
    main()
