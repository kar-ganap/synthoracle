# TODO

## Phase 0.1: Literature Review (Block A complete, Block B concurrent)

- [x] Review BORA (Cisse et al., IJCAI 2025) — closest prior art for hypothesis-driven BO
- [x] Review "LLMs for BO: Are We There Yet?" (2025) — LLMs show no feedback sensitivity
- [ ] Review EoH (Liu et al., ICML 2024) — thought-code co-evolution (Block B, concurrent)
- [ ] Review AnaFlow (KU Leuven, ICCAD 2025) — reasoning-constrained circuit sizing (Block B, concurrent)
- [ ] Review DiscoveryBench (NeurIPS 2024) — synthetic benchmark for scientific reasoning (Block B, concurrent)
- [ ] Update synthesis with positioning against new papers
- [x] Decision gate: confirm plan still holds after Block A reviews — plan holds

## Phase 1.0: Oracle Core

- [ ] Create branch phase-1.0-oracle-core
- [ ] Implement types.py (type aliases)
- [ ] Implement dag.py (CausalDAG, Node, Edge, precision/recall, project_to_io)
- [ ] Write test_dag.py
- [ ] Implement oracle.py (abstract Oracle base class)
- [ ] Write test_oracle_base.py
- [ ] Write test_medium_oracle.py (full property test suite)
- [ ] Implement oracles/medium.py (MediumOracle)
- [ ] make test && make lint && make typecheck — all green
- [ ] Write phase retro: docs/phases/phase-1.0-retro.md
