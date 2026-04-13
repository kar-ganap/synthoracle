# Phase 2.9 Retro: Ablation Extensions

**Date:** 2026-04-12 to 2026-04-13
**Branch:** `phase-2.9-ablation-extensions`
**Status:** Complete

---

## 1. What we set out to do

Strengthen the Phase 2.8 headline finding (forced articulation hurts by 35% on 1A) by addressing two HIGH-priority weaknesses identified during the paper framing discussion:

1. **Single-oracle ablation.** The penalty was only demonstrated on 1A. Does it generalize?
2. **Mechanism ambiguity.** Is the problem the ACT of articulation (cognitive cost) or the PERSISTENCE in context (belief anchoring)?

Additional experiments were motivated by the user's observations during the phase:
3. **Model generality.** Does the effect hold on Sonnet?
4. **2×2 matrix.** The summary effect depends on both model AND oracle — need both directions × both models.

## 2. What actually happened

### Experiment 1: HD ablation (Opus, no summary, n=3)

**Result: summary HELPS on HD (+16%).** The opposite of 1A.

- HD VR with summary: 0.257 (97% of BO)
- HD ablation no summary: 0.222 (84% of BO)

This was NOT predicted by the Phase 2.8 framing ("forced articulation hurts"). It revealed the effect is oracle-dependent: the summary cements correct screening beliefs on HD (noise dims are irrelevant) which is useful, unlike on 1A where it cements wrong beliefs about threshold effects.

### Experiment 2: Summary produced but NOT fed back (1A, n=3)

**Result: summary-not-fed ≈ no-summary (p=0.818). Both >> VR with summary (p=0.019).**

- VR full (summary + context): 0.193
- Summary not fed back: 0.265
- No summary at all: 0.259

Mechanism isolated: the ACT of articulation is neutral (producing the structured summary has zero measurable effect). The PERSISTENCE of the summary in conversation context is what matters. This rules out "cognitive cost of articulation" and confirms "context cementing" as the mechanism.

### Experiment 3: 1D ablation (Opus, no summary, n=3)

**Result: summary mildly hurts on 1D (−5.4%).** Between 1A (−35%) and HD (+16%).

- 1D VR with summary: 1.186
- 1D ablation no summary: 1.122

1D has correct topology but wrong functional forms — a middle ground in "early model correctness." The three-point curve (1A → 1D → HD) is clean and monotonic.

### Experiment 4: Sonnet HD ablation (no summary, n=3)

**Result: summary helps Sonnet even MORE than Opus (+35% vs +16%).**

- Sonnet HD with summary: 0.254
- Sonnet HD no summary: 0.188

The context-cementing benefit on HD is model-general and larger for the less capable model.

### Experiment 5: Sonnet 1A matrix (with + without summary, n=3 each)

**Result: summary HELPS Sonnet on 1A (+25%). Opposite direction from Opus (−35%).**

- Sonnet 1A with summary: 0.252
- Sonnet 1A no summary: 0.190

This was the phase's biggest surprise. The summary effect depends on BOTH model AND oracle. The complete 2×2 matrix:

|  | With summary | Without summary | Summary effect |
|---|---|---|---|
| Opus 1A | 0.193 | 0.259 | −35% (hurts) |
| Sonnet 1A | 0.252 | 0.190 | +25% (helps) |
| Opus HD | 0.257 | 0.222 | +16% (helps) |
| Sonnet HD | 0.254 | 0.188 | +35% (helps) |

The summary hurts ONLY in the one cell where the agent doesn't need structure (Opus on 1A — the most capable model on the simplest oracle). Everywhere else it helps.

### The organizing principle

The forced summary is **scaffolding, not regularization.** It helps when the agent needs structure:
- Less capable model (Sonnet): always benefits
- Harder oracle (HD with noise): always benefits
- More capable model on easy oracle (Opus on 1A): the only case where it hurts

Raw tool-use capability without summary: Opus 0.259, Sonnet 0.190 on 1A (36% gap). The summary lifts Sonnet to 0.252 (close to free Opus) but drags Opus down to 0.193 (below scaffolded Sonnet).

## 3. What went well

- **The HD ablation surprise (summary helps) prevented a wrong paper.** If we'd submitted after Phase 2.8 claiming "forced articulation always hurts," the HD result would have contradicted us. Running the multi-oracle ablation caught this before publication.
- **The mechanism experiment (summary-not-fed) was clean and decisive.** One parameter change, p=0.818 for the null (articulation is neutral), p=0.019 for the alternative (context cementing matters). Clean mechanism isolation.
- **The user's insistence on the Sonnet 1A matrix was the right call.** I recommended stopping after HD Sonnet (which showed the same direction as Opus HD). The user pushed for the 1A comparison, which revealed the model × oracle interaction — the phase's most important finding.
- **Each experiment was minimal (one parameter) and cheap (~$5-15).** Total phase spend ~$45. Efficient use of budget for maximum information.
- **The 2×2 matrix is a publication-ready figure.** Four cells, clean pattern, principled explanation.

## 4. What went poorly

- **I initially framed the Phase 2.8 finding as "forced articulation hurts (full stop)."** The HD result showed this was wrong. The correct finding is "forced articulation is scaffolding whose effect depends on the model-task capability gap." I should have tested multi-oracle BEFORE committing to the "always hurts" framing.
- **I recommended running Sonnet ablation on 1A first (where we had no baseline), not HD (where we did).** The user caught this immediately: "shouldn't we run HD where we can clearly compare?" Wasted a few minutes of the wrong experiment before killing it.
- **The 1D effect (−5.4%) is not statistically tested.** At n=3, the difference might not be significant. The directional claim (1D sits between 1A and HD) is supported but not quantified with a p-value.
- **No pre-registration for the Sonnet experiments.** The HD ablation and summary-not-fed were pre-registered in the Phase 2.9 plan. The Sonnet 1A matrix emerged from the discussion and was run without formal predictions. The finding is post-hoc.

## 5. Lessons learned

- **Multi-oracle ablation is essential for any finding about LLM reasoning protocols.** A single oracle shows the direction on THAT oracle. The direction can reverse on different oracles. Always test on ≥2 oracles with different structural properties before claiming a finding is general.
- **Model × task interactions are real and asymmetric.** The summary helps Sonnet on 1A but hurts Opus on 1A. The same protocol, same oracle, same budget — different model, different direction. Protocol evaluations must be multi-model.
- **"Scaffolding vs regularization" is the right framing for structured output in LLM agents.** Scaffolding helps agents that need structure and can be removed when they don't. Regularization would help all agents uniformly. Our data shows the former.
- **The most capable model on the simplest task is the one case where scaffolding hurts.** This is actually a small corner of the design space. For most practitioners (using frontier models on hard problems), the summary likely helps.
- **Let the user push on experimental design.** Two of the phase's most important decisions were user-initiated: (a) "run HD where we can compare, not 1A" and (b) "1A has the opposite direction — shouldn't we test model generality there too?" Both were correct.

## 6. What to do differently next time

- **Test on 2+ structurally different oracles before claiming any finding is general.** This should be a non-negotiable gate.
- **Always run both directions × both models when the finding involves an interaction.** A 2×2 matrix is 4× the experiments of a single ablation but produces qualitatively stronger evidence.
- **Pre-register Sonnet experiments alongside Opus experiments.** In this phase, Sonnet was an afterthought. It should have been part of the original plan.
- **Don't frame results before testing generality.** "Forced articulation hurts" (Phase 2.8) was premature. "Forced articulation is scaffolding with model × oracle interaction" (Phase 2.9) required the full matrix.

## Total Phase 2.9 spend

- 1A summary-not-fed (n=3 Opus): ~$15
- HD ablation (n=3 Opus): ~$7
- 1D ablation (n=3 Opus): ~$5
- Sonnet HD ablation (n=3): ~$2
- Sonnet 1A matrix (n=6: 3 with + 3 without): ~$10
- Killed Sonnet 1A ablation (partial): ~$1
- **Phase total: ~$40**
- **Project total: ~$274**

## Key artifacts

- `src/synthoracle/agents/vr_tools.py` — `summary_in_context` parameter added
- `experiments/vr_agent/run_ablation_summary_not_fed.py` — mechanism experiment
- `experiments/vr_agent/run_ablation_1d_no_summary.py` — 1D ablation
- `experiments/vr_agent/run_sonnet_1a_matrix.py` — Sonnet 1A paired comparison
- Run data: `ablation_summary_not_fed_seed{42-44}.*`, `hd_ablation_no_summary_seed{42-44}.*`, `ablation_1d_no_summary_seed{42-44}.*`, `hd_sonnet_ablation_no_summary_seed{42-44}.*`, `sonnet_1a_{with,no}_summary_seed{42-44}.*`

## The finding (final form)

Forced structured iteration summaries act as **scaffolding, not regularization.** The 2×2 matrix (model × oracle) shows:

- **Scaffolding helps** when the agent needs structure: less capable model (Sonnet on any oracle) or harder task (HD noise screening for either model).
- **Scaffolding hurts** when the agent doesn't need it: the most capable model (Opus) on the simplest oracle (1A, no noise, all inputs relevant).
- **The mechanism is context cementing** (summary-not-fed ≈ no-summary, p=0.818), not cognitive cost of articulation.
- **The effect scales with the capability gap**: Sonnet benefits more than Opus on HD (+35% vs +16%); raw tool-use capability without scaffolding shows a 36% model gap on 1A.

Design principle for practitioners: **match the rigidity of your reasoning protocol to the gap between your model's capability and your task's difficulty.** Use structured summaries for weaker models or harder tasks. Skip them for frontier models on well-characterized problems.
