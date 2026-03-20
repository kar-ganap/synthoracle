# Literature Review — Phase 0.1

**Status:** In progress
**Goal:** Review papers critical to positioning the SynthOracle project, update synthesis, reach decision gate.

**Decision gate:** Confirm the plan (docs/conceptual.md) still holds after reviewing closest prior art. Update positioning if needed.

---

## Process

For each paper:
1. Claude writes structured review (background, key ideas, results at three levels, connection to project)
2. User reads and understands
3. Follow-up Q&A
4. Update `synthesis.md` with new hypotheses / refined existing ones
5. Mark complete

After each block: review synthesis for coherence and gaps.

---

## Reading List

### Block A: Closest Prior Art (must review before building)

| # | Paper | Key question for us | Status |
|---|-------|-------------------|--------|
| 01 | Cisse et al., "BORA" (IJCAI, 2025) | Closest prior art — hypothesis-driven BO. How do their hypotheses differ from our verbal regularization? | |
| 02 | "LLMs for BO in Scientific Domains: Are We There Yet?" (2025) | LLMs show no feedback sensitivity. Does verbal regularization fix this? | |

### Block B: Competitive Landscape (review before or during implementation)

| # | Paper | Key question for us | Status |
|---|-------|-------------------|--------|
| 03 | Liu et al., "Evolution of Heuristics" (ICML, 2024) | Thought-code co-evolution. How does evolving "thoughts" differ from constrained causal reasoning? | |
| 04 | Ahmadzadeh et al., "AnaFlow" (IEEE ICCAD, 2025) | Reasoning-constrained circuit sizing in 9 sims. Closest domain analog. | |
| 05 | Majumder et al., "DiscoveryBench" (NeurIPS, 2024) | Synthetic benchmark for scientific reasoning. How does DB-SYNTH differ from SynthOracle? | |

### Block C: Supporting (review as needed)

| # | Paper | Key question for us | Status |
|---|-------|-------------------|--------|
| 06 | "Reasoning BO" (2025) | LLM reasoning + knowledge graphs in BO. | |
| 07 | "CauScientist" (2026) | LLM + statistical verifier for causal discovery. | |
| 08 | Anthropic, "CoT Faithfulness in the Wild" (2025) | Post-hoc rationalization rates. Informs our prediction accuracy curve. | |
| 09 | "Unveiling Causal Reasoning in LLMs" (NeurIPS, 2024) | Level-1 vs level-2 causal reasoning. | |

---

## Inherited from inverse-device-design project

The following papers were reviewed in the prior project and their findings carry forward (see `../inverse-device-design/literature-review/synthesis.md`):

- AI Scientist v2 (motivating example)
- FunSearch + Davis critique (economic argument)
- Concept Bottleneck Models (formal analog for VR)
- MFBO best practices
- BO fundamentals

---

## Validation Gates

1. **Coverage**: Block A papers reviewed with structured notes
2. **Interactive review**: Each paper discussed (Q&A)
3. **Synthesis**: Updated with positioning against new papers
4. **Decision gate**: Plan confirmed or revised
