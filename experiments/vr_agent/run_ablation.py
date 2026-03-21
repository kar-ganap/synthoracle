"""Run permuted-feedback ablation: VR agent with real vs scrambled feedback.

Usage:
    uv run --extra vr python experiments/vr_agent/run_ablation.py

Requires ANTHROPIC_API_KEY in environment.

Tests whether the VR agent actually uses X-Y correlations from feedback.
If HV_real ≈ HV_permuted, the agent ignores feedback (thesis fails).
If HV_real > HV_permuted, the agent uses feedback (thesis holds).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthoracle.agents.vr import VRResult, run_vr
from synthoracle.oracles.medium import MediumOracle

RESULTS_DIR = Path("experiments/vr_agent/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_ITERATIONS = 42


def print_comparison(real: VRResult, permuted: VRResult) -> None:
    """Print side-by-side comparison."""
    print(f"\n{'=' * 60}")
    print("  Permuted-Feedback Ablation: Medium 1A")
    print(f"{'=' * 60}")
    print(f"\n  {'Metric':<30} {'Real':>10} {'Permuted':>10} {'Diff':>10}")
    print(f"  {'-' * 60}")

    hv_r = real.hypervolumes[-1]
    hv_p = permuted.hypervolumes[-1]
    print(f"  {'Final HV':<30} {hv_r:>10.4f} {hv_p:>10.4f} {hv_r - hv_p:>+10.4f}")

    hv_init_r = real.hypervolumes[real.n_initial - 1]
    hv_init_p = permuted.hypervolumes[permuted.n_initial - 1]
    print(f"  {'Initial HV':<30} {hv_init_r:>10.4f} {hv_init_p:>10.4f} {hv_init_r - hv_init_p:>+10.4f}")

    hv_gain_r = hv_r - hv_init_r
    hv_gain_p = hv_p - hv_init_p
    print(f"  {'HV gain':<30} {hv_gain_r:>10.4f} {hv_gain_p:>10.4f} {hv_gain_r - hv_gain_p:>+10.4f}")

    print(f"  {'Pareto points':<30} {real.pareto_Y.shape[0]:>10d} {permuted.pareto_Y.shape[0]:>10d}")

    # Directional accuracy
    def _dir_acc(result: VRResult) -> float:
        flat = [a for s in result.step_logs for a in s.directional_accuracy]
        return sum(flat) / len(flat) if flat else 0.0

    acc_r = _dir_acc(real)
    acc_p = _dir_acc(permuted)
    print(f"  {'Directional accuracy':<30} {acc_r:>9.1%} {acc_p:>9.1%} {acc_r - acc_p:>+10.1%}")

    # Mean absolute error
    mae_r = float(np.mean(np.abs(real.prediction_errors)))
    mae_p = float(np.mean(np.abs(permuted.prediction_errors)))
    print(f"  {'Mean |pred error|':<30} {mae_r:>10.4f} {mae_p:>10.4f} {mae_r - mae_p:>+10.4f}")

    print(f"\n  {'Wall time (s)':<30} {real.total_seconds:>10.1f} {permuted.total_seconds:>10.1f}")
    cost_r = real.total_input_tokens * 3 / 1e6 + real.total_output_tokens * 15 / 1e6
    cost_p = permuted.total_input_tokens * 3 / 1e6 + permuted.total_output_tokens * 15 / 1e6
    print(f"  {'Est. cost ($)':<30} {cost_r:>10.2f} {cost_p:>10.2f}")

    print(f"\n  Verdict: ", end="")
    if hv_r > hv_p + 0.01:
        print("Real > Permuted — agent uses feedback ✓")
    elif hv_p > hv_r + 0.01:
        print("Permuted > Real — unexpected, agent may be confused by real data")
    else:
        print("Real ≈ Permuted — agent ignores feedback ✗ (thesis concern)")


def plot_comparison(
    real: VRResult, permuted: VRResult, output_dir: Path,
) -> None:
    """Plot HV convergence for both conditions."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    evals_r = list(range(1, len(real.hypervolumes) + 1))
    evals_p = list(range(1, len(permuted.hypervolumes) + 1))

    ax.plot(evals_r, real.hypervolumes, linewidth=1.5, label="Real feedback")
    ax.plot(evals_p, permuted.hypervolumes, linewidth=1.5, label="Permuted feedback",
            linestyle="--")
    ax.axvline(x=real.n_initial, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Permuted-Feedback Ablation: Medium 1A")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_hv_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved to {output_dir / 'ablation_hv_comparison.png'}")


def main() -> None:
    oracle = MediumOracle()

    print("Running VR agent with REAL feedback...")
    real = run_vr(
        oracle, n_iterations=N_ITERATIONS, seed=SEED,
        thresholds={"Y3": 0.4},
    )

    print("\nRunning VR agent with PERMUTED feedback...")
    permuted = run_vr(
        oracle, n_iterations=N_ITERATIONS, seed=SEED,
        thresholds={"Y3": 0.4},
        permute_feedback=True,
    )

    print_comparison(real, permuted)
    plot_comparison(real, permuted, RESULTS_DIR)

    # Save both results
    np.savez(
        RESULTS_DIR / "ablation_real_seed42.npz",
        hypervolumes=np.array(real.hypervolumes),
        X=real.X, Y=real.Y,
    )
    np.savez(
        RESULTS_DIR / "ablation_permuted_seed42.npz",
        hypervolumes=np.array(permuted.hypervolumes),
        X=permuted.X, Y=permuted.Y,
    )

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
