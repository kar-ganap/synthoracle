"""Complete the 2×2 matrix to n=5 per cell (seeds 45, 46).

Runs 12 conditions total:
  - Sonnet × Baseline: with summary + without summary (seeds 45, 46)
  - Opus × Noisy: with summary + without summary (seeds 45, 46)
  - Sonnet × Noisy: with summary + without summary (seeds 45, 46)

File naming matches existing conventions:
  sonnet_1a_{with,no}_summary_seed{45,46}.npz
  hd_vr_seed{45,46}.npz
  hd_ablation_no_summary_seed{45,46}.npz
  hd_vr_sonnet_seed{45,46}.npz
  hd_sonnet_ablation_no_summary_seed{45,46}.npz

Usage:
    source .env && uv run --extra vr python experiments/vr_agent/run_n5_completion.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np

from synthoracle.agents.vr_tools import run_vr_tools
from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_hd import MediumOracleHD

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NEW_SEEDS = [45, 46]
N_BUDGET = 72
THRESHOLDS = {"Y3": 0.4}

PRICING = {
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


def run_single(
    oracle,
    seed: int,
    label: str,
    model: str,
    skip_summary: bool,
    ref_point: np.ndarray,
) -> dict | None:
    """Run one VR seed, save to disk, return summary dict."""
    out_npz = RESULTS_DIR / f"{label}_seed{seed}.npz"
    out_log = RESULTS_DIR / f"{label}_seed{seed}_log.json"
    pricing = PRICING[model]

    if out_npz.exists() and out_log.exists():
        data = np.load(out_npz)
        with open(out_log) as f:
            log = json.load(f)
        hv = float(data["hypervolumes"][-1])
        cost = (log.get("total_input_tokens", 0) * pricing[0] / 1e6
                + log.get("total_output_tokens", 0) * pricing[1] / 1e6)
        print(f"  {label} seed {seed}: already on disk (HV={hv:.4f}, ${cost:.2f})")
        return {"seed": seed, "hv": hv, "cost": cost}

    print(f"\n  Running {label} seed {seed} "
          f"(model={model.split('-')[1]}, skip_summary={skip_summary})...",
          flush=True)

    try:
        r = run_vr_tools(
            oracle,
            n_budget=N_BUDGET,
            seed=seed,
            thresholds=THRESHOLDS,
            reference_point=ref_point,
            model=model,
            thinking={"type": "adaptive"},
            max_tokens=16000,
            max_tool_calls_per_iteration=15,
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

    cost = (r.total_input_tokens * pricing[0] / 1e6
            + r.total_output_tokens * pricing[1] / 1e6)
    print(f"  {label} seed {seed}: HV={r.hypervolumes[-1]:.4f}, "
          f"cost=${cost:.2f}, evals={r.eval_count}, "
          f"time={r.total_seconds:.0f}s")
    return {"seed": seed, "hv": float(r.hypervolumes[-1]), "cost": cost}


def main() -> None:
    # --- Setup oracles and reference points ---
    baseline_oracle = MediumOracle(variant="1A")
    obj_idx_b, _, signs_b = parse_directions(baseline_oracle)
    ref_baseline = compute_reference_point(baseline_oracle, obj_idx_b, signs_b, seed=0)

    noisy_oracle = MediumOracleHD()
    obj_idx_n, _, signs_n = parse_directions(noisy_oracle)
    ref_noisy = compute_reference_point(noisy_oracle, obj_idx_n, signs_n, seed=0)

    # --- Define all 6 conditions ---
    conditions = [
        # (label, oracle, ref_point, model, skip_summary)
        ("sonnet_1a_with_summary", baseline_oracle, ref_baseline,
         "claude-sonnet-4-6", False),
        ("sonnet_1a_no_summary", baseline_oracle, ref_baseline,
         "claude-sonnet-4-6", True),
        ("hd_vr", noisy_oracle, ref_noisy,
         "claude-opus-4-6", False),
        ("hd_ablation_no_summary", noisy_oracle, ref_noisy,
         "claude-opus-4-6", True),
        ("hd_vr_sonnet", noisy_oracle, ref_noisy,
         "claude-sonnet-4-6", False),
        ("hd_sonnet_ablation_no_summary", noisy_oracle, ref_noisy,
         "claude-sonnet-4-6", True),
    ]

    print("=" * 60)
    print("  2×2 Matrix Completion: n=3 → n=5 (seeds 45, 46)")
    print(f"  {len(conditions)} conditions × {len(NEW_SEEDS)} seeds = "
          f"{len(conditions) * len(NEW_SEEDS)} runs")
    print("=" * 60)

    all_results: dict[str, list[dict]] = {}
    total_cost = 0.0

    for label, oracle, ref_point, model, skip_summary in conditions:
        print(f"\n{'─' * 60}")
        print(f"  {label} (model={model.split('-')[1]}, "
              f"skip_summary={skip_summary})")
        print(f"{'─' * 60}")
        results = []
        for seed in NEW_SEEDS:
            r = run_single(oracle, seed, label, model, skip_summary, ref_point)
            if r is not None:
                results.append(r)
                total_cost += r.get("cost", 0)
        all_results[label] = results

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Completion summary")
    print(f"{'=' * 60}")
    for label, results in all_results.items():
        if results:
            hvs = [r["hv"] for r in results]
            print(f"  {label}: {[f'{h:.4f}' for h in hvs]}")
    print(f"\n  Total cost for new runs: ${total_cost:.2f}")

    # --- Recompute paired t-tests with n=5 ---
    print(f"\n{'=' * 60}")
    print(f"  Updated 2×2 matrix (n=5)")
    print(f"{'=' * 60}")

    from scipy import stats

    cells = [
        ("Sonnet × Baseline",
         "sonnet_1a_with_summary", "sonnet_1a_no_summary",
         [42, 43, 44, 45, 46], "helps"),
        ("Opus × Noisy",
         "hd_vr", "hd_ablation_no_summary",
         [42, 43, 44, 45, 46], "helps"),
        ("Sonnet × Noisy",
         "hd_vr_sonnet", "hd_sonnet_ablation_no_summary",
         [42, 43, 44, 45, 46], "helps"),
    ]

    # Also include the existing Opus × Baseline
    print(f"\n  {'Cell':>22} {'With':>8} {'Without':>8} {'Effect':>8} "
          f"{'p':>8} {'n':>4} {'Dir':>6}")

    # Opus × Baseline (already n=5)
    opus_w = [float(np.load(f"{RESULTS_DIR}/multi_seed/seed{s}.npz")["hypervolumes"][-1])
              for s in [42, 43, 44, 45, 46]]
    opus_wo = [float(np.load(f"{RESULTS_DIR}/ablation_no_summary_seed{s}.npz")["hypervolumes"][-1])
               for s in [42, 43, 44, 45, 46]]
    t, p = stats.ttest_rel(opus_wo, opus_w)
    pos = sum(1 for a, b in zip(opus_wo, opus_w) if a > b)
    eff = (np.mean(opus_wo) - np.mean(opus_w)) / np.mean(opus_w) * 100
    print(f"  {'Opus × Baseline':>22} {np.mean(opus_w):>8.3f} {np.mean(opus_wo):>8.3f} "
          f"{eff:>+7.1f}% {p:>8.4f} {5:>4} {pos}/{5:>4}")

    for cell_name, with_prefix, without_prefix, seeds, direction in cells:
        with_hvs = []
        without_hvs = []
        for s in seeds:
            wp = RESULTS_DIR / f"{with_prefix}_seed{s}.npz"
            wop = RESULTS_DIR / f"{without_prefix}_seed{s}.npz"
            if not wp.exists() or not wop.exists():
                print(f"  WARNING: missing data for {cell_name} seed {s}")
                continue
            with_hvs.append(float(np.load(wp)["hypervolumes"][-1]))
            without_hvs.append(float(np.load(wop)["hypervolumes"][-1]))

        if len(with_hvs) < 3:
            print(f"  {cell_name}: insufficient data ({len(with_hvs)} seeds)")
            continue

        t_stat, p_val = stats.ttest_rel(with_hvs, without_hvs)
        n = len(with_hvs)
        pos = sum(1 for w, wo in zip(with_hvs, without_hvs) if w > wo)
        eff = (np.mean(with_hvs) - np.mean(without_hvs)) / np.mean(without_hvs) * 100
        print(f"  {cell_name:>22} {np.mean(with_hvs):>8.3f} "
              f"{np.mean(without_hvs):>8.3f} {eff:>+7.1f}% {p_val:>8.4f} "
              f"{n:>4} {pos}/{n:>4}")


if __name__ == "__main__":
    main()
