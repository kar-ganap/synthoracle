# TODO

## Phase 0.1: Literature Review (Block A complete, Block B concurrent)

- [x] Review BORA (Cisse et al., IJCAI 2025) — closest prior art for hypothesis-driven BO
- [x] Review "LLMs for BO: Are We There Yet?" (2025) — LLMs show no feedback sensitivity
- [ ] Review EoH (Liu et al., ICML 2024) — thought-code co-evolution (Block B, concurrent)
- [ ] Review AnaFlow (KU Leuven, ICCAD 2025) — reasoning-constrained circuit sizing (Block B, concurrent)
- [ ] Review DiscoveryBench (NeurIPS 2024) — synthetic benchmark for scientific reasoning (Block B, concurrent)
- [ ] Update synthesis with positioning against new papers
- [x] Decision gate: confirm plan still holds after Block A reviews — plan holds

## Phase 1.0: Oracle Core (COMPLETE)

- [x] Oracle base class, DAG representation, medium oracle
- [x] 36 tests pass, lint/typecheck clean
- [x] Parameter tuning via variance decomposition + conditional effect analysis

## Phase 1.1: Oracle Characterization (COMPLETE)

- [x] Characterization module with correct Sobol estimators (fix Phase 1.0 bug)
- [x] Verify medium oracle interactions (S_T_j - S_j > 0 for coupled inputs)
- [x] Simple oracle (4 inputs, 2 outputs, smooth)
- [x] Medium variant 1B (M2 formula change)
- [x] Medium variant 1C (new mechanism M5)
- [x] Run characterization on all oracles, generate plots + Pareto fronts
- [x] 87 tests pass, lint/typecheck clean, retro written

## Phase 1.2: BO Baseline (COMPLETE)

- [x] BoTorch qLogNEHVI with mixed objectives + threshold constraints
- [x] HV convergence on all 4 oracles: Simple 1.29, 1A 0.28, 1B 0.26, 1C 0.55
- [x] 23 tests (16 fast + 7 slow), lint/typecheck clean

## Phase 2.0: VR Agent Core

- [x] Shared numpy-only optim_utils (HV, Pareto, directions — no torch dep)
- [x] VR agent: hypothesize→predict→reconcile loop with structured LLM output
- [x] Prompt construction, JSON parsing, directional accuracy tracking
- [x] Tests with mocked Anthropic client (148 total, no API calls)
- [x] Manual integration test: HV 0.24 (vs BO 0.28), dir accuracy 55%, $1.46/run
- [x] 148 tests pass, lint/typecheck clean, retro written

## Phase 2.0.1: Permuted-Feedback Ablation (COMPLETE)

- [x] Column permutation: shuffle Y rows relative to X in agent's observation table
- [x] Go/no-go gate PASSED: Real HV 0.22 vs Permuted 0.13 — agent uses feedback
- [x] 150 tests pass, lint/typecheck clean
