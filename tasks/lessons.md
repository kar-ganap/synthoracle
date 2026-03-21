# Lessons

Self-improvement log. Updated after ANY correction from the user.
Review at session start.

## Carried from inverse-device-design project

- **Always use `uv`, never `pip`.**
- **Read working examples holistically** before iterating on config/API issues.
- **Phase branches off main.** User merges manually.
- **Follow the literature review discipline.** Structured review → Q&A → synthesis update. No shortcuts.

## Phase 1.0

- **Always run response surface plots before committing oracle code.** Numerical tests verify properties but don't catch "this mechanism is too subtle to discover."
- **Design output formulas with detectability in mind.** When an intermediate variable appears in both terms of an output, it partially cancels. Detect this during formula design.
- **Quantify effect sizes as % of output range** — the right metric for "is it detectable from N samples?" 15-20% is the heuristic threshold.
- **Verify Sobol estimator formulas with tests.** First-order indices must sum to ≤1. The Phase 1.0 ad-hoc script had `1 - S_T_j` instead of `S_j`.
- **Don't commit placeholder parameters.** Design and verify effect sizes before implementing.
- **Don't skip the retro.** PLAN → TEST → IMPLEMENT → VERIFY → RETRO. All five.
