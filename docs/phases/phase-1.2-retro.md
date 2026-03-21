# Phase 1.2 Retro: BO Baseline

**Date:** 2026-03-20
**Branch:** `phase-1.2-bo-baseline`
**Status:** Complete

---

## 1. What we set out to do

Build a multi-objective Bayesian optimization baseline using BoTorch qNEHVI that works with any Oracle subclass. This establishes the performance bar for the verbal regularization agent to beat in later stages. Primary metric: hypervolume indicator vs evaluation count.

## 2. What actually happened

- Built `baselines/bo.py` with `run_bo()` function using qLogNEHVI (log-space variant for better numerics)
- ModelListGP surrogate (one SingleTaskGP per output, including constraint outputs)
- Handles mixed objectives (maximize/minimize/threshold) via sign flipping and constraint callables
- Hypervolume tracked at every evaluation for convergence curves
- Data-driven reference point (10k random samples, worst - 10% margin)
- Deterministic with seed control
- 23 tests (16 fast + 7 slow), all passing
- Experiment script runs all 4 oracles, saves .npz results + HV convergence plot

### Dependency resolution

torch 2.2.2 is the latest version with an x86_64 macOS wheel for this Python. numpy pinned to <2 for torch compatibility. botorch 0.12.0 + gpytorch 1.15.2 resolved cleanly.

### BO performance results (seed=42, 42 BO iterations)

| Oracle | Evals | Initial HV | Final HV | HV gain | Pareto pts | Wall time |
|--------|-------|-----------|----------|---------|------------|-----------|
| Simple | 50 | 0.826 | 1.293 | +57% | 39 | 94s |
| Medium 1A | 54 | 0.141 | 0.277 | +96% | 30 | 818s |
| Medium 1B | 54 | 0.158 | 0.260 | +64% | 33 | 1647s |
| Medium 1C | 54 | 0.256 | 0.553 | +117% | 34 | 593s |

BO converges well on the simple oracle and makes meaningful progress on all medium variants. Medium 1B is slowest due to frequent GP numerical issues (regime shift creates sharp output transitions).

## 3. What went well

- **qLogNEHVI over qNEHVI.** BoTorch's own warning suggested the log-space variant. Switching eliminated all numerics warnings from the acquisition function (GP jitter warnings remain but are handled gracefully).
- **Constraint handling worked cleanly.** The objective selector + constraint callable pattern for threshold outputs (Y3 > 0.4) worked on the first attempt.
- **TDD caught shape issues early.** The fast tests validated direction parsing, Sobol design, reference point, and HV computation before touching GP code.
- **Data-driven reference point.** Avoids manual tuning per oracle — samples 10k points and sets ref below worst, generalizes across all oracle subclasses.

## 4. What went poorly

- **Dependency resolution was painful.** torch has limited wheel availability for x86_64 macOS + Python 3.11. Required pinning torch<2.3 and numpy<2. This will need revisiting if we upgrade Python or move to ARM.
- **Medium oracle BO is slow.** 42 iterations on medium 1B took 27 minutes. This is acceptable for a baseline but multi-seed experiments (10 seeds × 4 oracles) would take ~18 hours. May need to reduce iterations or parallelize in future phases.
- **GP numerical issues on medium oracles.** Frequent "A not p.d." warnings and occasional optimization failures. BoTorch's automatic jitter addition handles these, but it suggests the GP is struggling with the regime transitions and threshold effects.

## 5. Lessons learned

- **Use qLogNEHVI, not qNEHVI.** The log-space variant has strictly better numerical properties with the same API.
- **Pin torch explicitly.** Its wheel availability is platform-specific and changes frequently. Don't rely on latest-version resolution.
- **Optional dependencies need import guards.** BO code imports torch/botorch which aren't in the base package. Tests use `pytest.importorskip()`, and the baselines module is not exported from `__init__.py`.
- **GP struggles with non-smooth landscapes.** The medium oracle's regime transitions and threshold create sharp output changes that GPs with smooth kernels (Matern 5/2) approximate poorly. This is exactly the kind of structure where VR should help — the agent can identify regimes and fit separate models.

## 6. What to do differently next time

- For multi-seed experiments, add progress logging (iteration counter) so wall-clock time is predictable.
- Consider reducing `num_restarts` from 10 to 5 for medium oracles to halve acquisition optimization time.
- The GP numerical issues on medium oracles may warrant investigation — custom kernels or input warping could help, but that's beyond the baseline scope.
