# SynthOracle: A Benchmark for Evaluating Scientific Reasoning in Optimization Agents

## Project Thesis

Build a family of synthetic oracles with known causal structure to rigorously test whether constraining an LLM optimization agent to reason causally (hypothesize, predict, reconcile) improves sample efficiency and produces transferable understanding — compared to black-box Bayesian optimization. The oracle provides ground truth for evaluation; the verbal regularization methodology is the novel contribution.

## Current State

- **Current Stage:** Stage 0 — Foundation
- **Current Phase:** Phase 0.1 — Literature Review
- **Phase Status:** In progress (reviewing BORA + "LLMs for BO: Are We There Yet?")

## Constraints

- Hardware: Apple M-series (local)
- Oracle evaluations: <1ms each (pure Python/NumPy)
- LLM: Claude API (inference cost only, no training)
- BO baseline: BoTorch (qNEHVI)

## Structure

### Stages (headline only — detailed planning when a stage's turn arrives)

- **Stage 0:** Foundation — Literature review, project foundation
- **Stage 1 (Crawl):** Oracle implementation + BO baseline
- **Stage 2 (Walk):** Verbal regularization agent + evaluation
- **Stage 3 (Run):** Transfer tests + ablations + paper

### Phases

Phases are small, atomic units within each stage. Each phase has:
- A single task, feature, or hypothesis class
- Explicit go/no-go validation gates defined before work begins
- A strict cycle: PLAN → TEST → IMPLEMENT → VERIFY → RETRO

Phase plans live in `docs/phases/phase-X.Y-plan.md`.
Phase retros live in `docs/phases/phase-X.Y-retro.md`.

## Build & Test

```bash
make test          # run all tests
make lint          # ruff check
make typecheck     # mypy strict
make clean         # remove build artifacts
```

## Ground Rules

### Workflow

1. **Plan mode** for ANY non-trivial task (3+ steps). If things go sideways, STOP and re-plan.
2. **Phase lifecycle**: PLAN → TEST → IMPLEMENT → VERIFY → RETRO.
3. **TDD**: For code, tests first. For research, hypothesis + evaluation criteria first. Tests define "done."
4. **Only plan current phase in detail.** Future phases stay headline-level. Anything else is waterfall in disguise.
5. **Verification before done.** Never mark complete without proving it works. Ask: "Would a staff engineer approve this?"
6. **Objective before subjective.** Run automated/quantitative checks before qualitative review.
7. **Separation of concerns.** Docs/config drive design decisions. Code is a tool, not a decision-maker.
8. **Self-improvement loop.** After ANY correction, update `tasks/lessons.md`. Review at session start.
9. **Subagents** for research/exploration. One task per subagent. Keep main context clean.
10. **Autonomous bug fixing.** Just fix it. Zero context switching from the user.

### Code

- **Simplicity first.** Minimal code, minimal impact. Don't over-engineer.
- **No laziness.** Root causes. No temporary fixes. Senior developer standards.
- **Minimal impact.** Touch only what's necessary.
- **Demand elegance (balanced).** Pause on non-trivial changes. Skip for simple fixes.
- **Reproducibility.** Pin all parameters, seeds. Raw data never modified.

### Experimental Discipline

1. **Pre-register hypotheses.** Write what you expect and why before running experiments.
2. **Report nulls honestly.** Negative results are results.
3. **Characterize distributions, not just means.** Point estimates without uncertainty are insufficient.
4. **Multiple metrics per experiment.**
5. **Track spend.** Log all compute costs in `tasks/spend.md`.

### Git

- Phase branches off `main` (e.g., `phase-0.1-literature-review`)
- User merges manually. No force pushes.
- Small, focused commits. No Co-Authored-By lines in commits.

## Validation Gates (defaults, adapted per phase)

1. All tests pass (code phases) or all evaluation criteria met (research phases)
2. Linting/formatting clean (code phases)
3. Reproducibility check: can results be regenerated from committed code + documented parameters?
4. Retro written with all 6 sections

## Key References

- North star: `docs/conceptual.md`
- Phase plans/retros: `docs/phases/`
- Task tracking: `tasks/todo.md`
- Lessons: `tasks/lessons.md`
- Spend tracking: `tasks/spend.md`
- Literature review: `literature-review/`
- Prior project (NEGF convergence findings): `../inverse-device-design/docs/phases/crawl-1a-convergence-study.md`

## Known Gotchas

- LLMs may not respond to experimental feedback (see "Are We There Yet?" paper) — permuted-feedback ablation is essential
- Synthetic oracle must not resemble known systems (LLM recall risk)
- BORA is closest prior art — must differentiate clearly on causal structure, prediction tracking, transfer
