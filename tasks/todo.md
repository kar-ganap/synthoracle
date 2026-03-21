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
