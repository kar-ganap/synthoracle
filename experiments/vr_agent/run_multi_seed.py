"""Multi-seed Opus tool agent on Medium 1A.

Runs n=10 seeds with full diagnostic stack (structured iteration summaries,
calibration checkpoints, structured OAT predictions). Saves per-seed results
and prints aggregate statistics.

Usage:
    ANTHROPIC_API_KEY=... uv run --extra vr python experiments/vr_agent/run_multi_seed.py
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
from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracles.medium import MediumOracle

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("experiments/vr_agent/results/multi_seed")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(42, 52))  # 10 seeds: 42..51
N_BUDGET = 72
MODEL = "claude-opus-4-6"
THINKING = {"type": "adaptive"}
THRESHOLDS = {"Y3": 0.4}
PRICING = (15.0, 75.0)  # $/M tokens (input, output)


# ---------------------------------------------------------------------------
# Per-seed run + save
# ---------------------------------------------------------------------------


def run_seed(
    seed: int,
    oracle: MediumOracle,
    reference_point: np.ndarray,
) -> VRToolsResult | None:
    """Run one seed and save results. Returns None on failure."""
    ckpt = str(RESULTS_DIR / f"seed{seed}_ckpt")

    print(f"\n{'=' * 60}")
    print(f"  Seed {seed}")
    print(f"{'=' * 60}\n")

    try:
        result = run_vr_tools(
            oracle,
            n_budget=N_BUDGET,
            seed=seed,
            thresholds=THRESHOLDS,
            reference_point=reference_point,
            model=MODEL,
            thinking=THINKING,
            max_tokens=16000,
            max_tool_calls_per_iteration=15,
            checkpoint_dir=ckpt,
            calibration_interval=20,
        )
    except Exception:
        print(f"  FAILED on seed {seed}:")
        traceback.print_exc()
        return None

    # Save npz
    np.savez(
        RESULTS_DIR / f"seed{seed}.npz",
        X=result.X, Y=result.Y,
        hypervolumes=np.array(result.hypervolumes),
        pareto_X=result.pareto_X, pareto_Y=result.pareto_Y,
        reference_point=result.reference_point,
    )

    # Save json log
    with open(RESULTS_DIR / f"seed{seed}_log.json", "w") as f:
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
    print(f"\n  Seed {seed} complete: HV={result.hypervolumes[-1]:.6f}, "
          f"evals={result.eval_count}, cost=${cost:.2f}, "
          f"time={result.total_seconds:.0f}s")

    return result


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


def print_aggregate(results: dict[int, VRToolsResult]) -> None:
    """Print aggregate statistics across seeds."""
    print(f"\n{'=' * 70}")
    print(f"  Multi-Seed Aggregate ({len(results)} seeds)")
    print(f"{'=' * 70}")

    # HV statistics
    final_hvs = [r.hypervolumes[-1] for r in results.values()]
    print(f"\n  Final HV: {np.mean(final_hvs):.6f} +/- {np.std(final_hvs):.6f}")
    print(f"    min={np.min(final_hvs):.6f}, max={np.max(final_hvs):.6f}")

    # Per-seed summary
    print(f"\n  {'Seed':>6} {'Final HV':>10} {'Evals':>6} {'Iters':>6} "
          f"{'#Edges':>7} {'Cost':>8} {'Time':>7}")
    print(f"  {'-' * 58}")
    for seed, r in sorted(results.items()):
        n_edges = 0
        if r.iteration_summaries:
            last = r.iteration_summaries[-1]
            n_edges = len(last.get("edges", []))
        cost = (r.total_input_tokens * PRICING[0] / 1e6
                + r.total_output_tokens * PRICING[1] / 1e6)
        print(f"  {seed:>6} {r.hypervolumes[-1]:>10.6f} {r.eval_count:>6} "
              f"{len(r.iteration_summaries):>6} {n_edges:>7} "
              f"${cost:>7.2f} {r.total_seconds:>6.0f}s")

    # Cost
    total_cost = sum(
        r.total_input_tokens * PRICING[0] / 1e6
        + r.total_output_tokens * PRICING[1] / 1e6
        for r in results.values()
    )
    print(f"\n  Total cost: ${total_cost:.2f}")
    print(f"  Mean cost per seed: ${total_cost / len(results):.2f}")

    # Calibration
    all_maes = []
    for r in results.values():
        for c in r.calibration_checks:
            mae = c.get("mae")
            if mae is not None and not np.isnan(mae):
                all_maes.append(mae)
    if all_maes:
        print(f"\n  Calibration MAE: {np.mean(all_maes):.4f} +/- {np.std(all_maes):.4f} "
              f"(n={len(all_maes)} checkpoints)")


def plot_hv_envelope(results: dict[int, VRToolsResult]) -> None:
    """Plot HV convergence envelope (mean +/- std) across seeds."""
    # Pad all HV curves to same length
    max_len = max(len(r.hypervolumes) for r in results.values())
    hv_matrix = np.full((len(results), max_len), np.nan)
    for i, r in enumerate(results.values()):
        hvs = r.hypervolumes
        hv_matrix[i, :len(hvs)] = hvs
        hv_matrix[i, len(hvs):] = hvs[-1]  # pad with final HV

    mean_hv = np.nanmean(hv_matrix, axis=0)
    std_hv = np.nanstd(hv_matrix, axis=0)
    evals = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(evals, mean_hv, color="#d62728", linewidth=2, label="VR tool (mean)")
    ax.fill_between(evals, mean_hv - std_hv, mean_hv + std_hv,
                    alpha=0.2, color="#d62728")

    # Individual seeds as thin lines
    for i, (seed, r) in enumerate(sorted(results.items())):
        ax.plot(range(1, len(r.hypervolumes) + 1), r.hypervolumes,
                alpha=0.2, color="#d62728", linewidth=0.5)

    # BO baseline if available
    bo_dir = Path("experiments/comparison/results")
    bo_hvs = []
    for s in range(42, 52):
        bo_path = bo_dir / f"bo_seed{s}.npz"
        if bo_path.exists():
            bo_hvs.append(np.load(bo_path)["hypervolumes"])
    if bo_hvs:
        max_bo = max(len(h) for h in bo_hvs)
        bo_matrix = np.full((len(bo_hvs), max_bo), np.nan)
        for i, h in enumerate(bo_hvs):
            bo_matrix[i, :len(h)] = h
            bo_matrix[i, len(h):] = h[-1]
        bo_mean = np.nanmean(bo_matrix, axis=0)
        bo_std = np.nanstd(bo_matrix, axis=0)
        bo_evals = np.arange(1, max_bo + 1)
        ax.plot(bo_evals, bo_mean, color="#1f77b4", linewidth=2,
                label="BO (mean)")
        ax.fill_between(bo_evals, bo_mean - bo_std, bo_mean + bo_std,
                        alpha=0.15, color="#1f77b4")

    ax.axvline(x=12, color="gray", linestyle="--", alpha=0.2)
    ax.set_xlabel("Oracle Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"Opus Tool Agent: Medium 1A ({len(results)} seeds)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hv_envelope.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  HV envelope saved: {RESULTS_DIR / 'hv_envelope.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    oracle = MediumOracle()

    # Shared reference point for comparable HV across seeds
    obj_indices, _, signs = parse_directions(oracle)
    ref_point = compute_reference_point(oracle, obj_indices, signs, seed=0)

    print(f"Multi-seed Opus tool agent on Medium 1A")
    print(f"Seeds: {SEEDS}")
    print(f"Budget: {N_BUDGET} evals per seed")
    print(f"Reference point: {ref_point}")

    results: dict[int, VRToolsResult] = {}
    for seed in SEEDS:
        r = run_seed(seed, oracle, ref_point)
        if r is not None:
            results[seed] = r

    if results:
        print_aggregate(results)
        plot_hv_envelope(results)

    print(f"\n  {len(results)}/{len(SEEDS)} seeds completed successfully.")


if __name__ == "__main__":
    main()
