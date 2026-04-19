"""Run BO at 144 evals for matched seed counts on Baseline and Noisy.

Seeds 42, 43, 44 on both oracles. Skips any that already exist.

Usage:
    source .env && nohup uv run --extra bo python experiments/vr_agent/run_bo_144_matched.py > bo_144_matched.log 2>&1 &
"""

import numpy as np
from pathlib import Path
from synthoracle.baselines.bo import run_bo
from synthoracle.optim_utils import compute_reference_point, parse_directions
from synthoracle.oracles.medium import MediumOracle
from synthoracle.oracles.medium_hd import MediumOracleHD

R = Path("experiments/vr_agent/results")
THRESHOLDS = {"Y3": 0.4}
SEEDS = [42, 43, 44]

# --- Baseline (1A) ---
oracle_1a = MediumOracle(variant="1A")
obj_idx, _, signs = parse_directions(oracle_1a)
ref_1a = compute_reference_point(oracle_1a, obj_idx, signs, seed=0)

for seed in SEEDS:
    out = R / f"bo_1a_n144_seed{seed}.npz"
    if out.exists():
        d = np.load(out)
        print(f"BO 1A seed {seed}: exists ({d['hypervolumes'].shape[0]} evals, HV={d['hypervolumes'][-1]:.4f})", flush=True)
        continue
    print(f"Running BO 1A @144 seed {seed}...", flush=True)
    r = run_bo(oracle_1a, n_iterations=132, seed=seed,
               reference_point=ref_1a, thresholds=THRESHOLDS)
    np.savez(out, X=r.X, Y=r.Y,
             hypervolumes=np.array(r.hypervolumes),
             pareto_X=r.pareto_X, pareto_Y=r.pareto_Y,
             reference_point=ref_1a)
    print(f"  Done: {len(r.hypervolumes)} evals, HV={r.hypervolumes[-1]:.4f}", flush=True)

# --- Noisy (HD) ---
oracle_hd = MediumOracleHD()
obj_idx_h, _, signs_h = parse_directions(oracle_hd)
ref_hd = compute_reference_point(oracle_hd, obj_idx_h, signs_h, seed=0)

for seed in SEEDS:
    out = R / f"bo_hd_n144_seed{seed}.npz"
    if out.exists():
        d = np.load(out)
        print(f"BO HD seed {seed}: exists ({d['hypervolumes'].shape[0]} evals, HV={d['hypervolumes'][-1]:.4f})", flush=True)
        continue
    print(f"Running BO HD @144 seed {seed}...", flush=True)
    r = run_bo(oracle_hd, n_iterations=120, seed=seed,
               reference_point=ref_hd, thresholds=THRESHOLDS)
    np.savez(out, X=r.X, Y=r.Y,
             hypervolumes=np.array(r.hypervolumes),
             pareto_X=r.pareto_X, pareto_Y=r.pareto_Y,
             reference_point=ref_hd)
    print(f"  Done: {len(r.hypervolumes)} evals, HV={r.hypervolumes[-1]:.4f}", flush=True)

print("\nAll done.", flush=True)
