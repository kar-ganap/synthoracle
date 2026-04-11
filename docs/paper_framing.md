# Paper Framing — Working Document

**Status:** Active iteration. Final framing pending completion of BO HD @ 144 and rubric rewrite. This document captures the synthesis and open questions so we can iterate on it.

---

## 1. What we actually built (take stock)

### Three technical contributions

1. **SynthOracle oracle family** — 6 synthetic oracles (1A baseline, 1B/1C deprecated stepping stones, 1D functional shift, 1E topology rewire, HD 12-dim with noise) with ground-truth causal DAGs, pre-registered difficulty rubric, multiple orthogonal difficulty axes (k, d_eff, interaction fraction, mechanism sufficiency, adversarial region count, prior quality).

2. **Diagnostic suite** — edge precision/recall against IO-projected ground truth, Sobol-weighted info capture, OAT direction/magnitude accuracy, calibration learning (first→last checkpoint MAE), adversarial region prediction error, screening efficiency (noise tool-call fraction + noise edge confidence), mechanism sufficiency R²(M→Y). **This is what enables measuring reasoning quality separately from optimization quality** — the measurement apparatus the literature currently lacks.

3. **Verbal regularization (VR) agent methodology** — hypothesize→predict→reconcile loop, screen-first transfer protocol, structured iteration summaries (Pydantic-parsed), calibration checkpoints, tool-use around OAT sweeps and interaction tests.

### One methodological practice

4. **Pre-registered gap-closing experiments** — we committed predictions for four gap-closing experiments (1D/1E n=3, Sonnet HD, BO 1A @ 144) before running them, then honestly compared predictions to outcomes. Four of the pre-registered rules were falsified by the n=3 data. This is **scientific hygiene, not a generalizable contribution** (per user feedback) — but it's worth mentioning in the methodology section as a credibility signal in a literature rife with post-hoc rationalization.

---

## 2. Strongest findings (post-preregistration)

### Surviving findings

- **Baseline understanding tax on 1A**: VR reaches 68% of BO HV at 72 budget on 1A (n=10 multi-seed). Understanding is expensive.
- **Dimensionality scaling (k/d)**: VR reaches 96.6% of BO at 72 budget on HD (n=3), vs 68% on 1A. When irrelevant dimensions outnumber mechanisms, VR's relative advantage grows. CBM prediction validated.
- **Perfect screening at tight budget (Opus)**: HD base 72 eval on Opus → 0/12 OAT sweeps on noise dims, zero false-positive noise edges.
- **Recall convergence at extended budget**: HD extended 144 → recall σ collapses from 0.146 (base) to 0.000 (all 3 seeds miss exactly the same edge, X5→Y2).
- **Calibration learning emerges at extended budget**: HD ext → 3/3 seeds show >20% MAE drop from first to last calibration checkpoint.
- **Info capture exceeds raw recall**: HD ext Sobol-weighted recall = 0.996 vs raw recall = 0.944 — the missed edge is low-Sobol.

### Falsified and replaced findings (the pre-registration saves)

- **Rule 8 FALSIFIED**: R²(M→Y) does NOT predict VR/BO ratio. Low-R² oracles (1B 0.799, 1D 0.748) both show VR/BO ≈ 0.91 — HIGHER than 1A's 0.685 (R² = 0.961). Replaced with "R²(M→Y) is a diagnostic for interpretability, not performance; hard functional forms may actually favor VR relative to GP-based BO."

- **Rule 9 FALSIFIED, REPLACED**: Pre-registered prediction was "wrong prior hurts, 1D prior penalty should be ~9%." n=3 data shows **prior HELPS** — 1D by +2%, 1E by +5%. Replaced with "the screen-first transfer protocol leverages partial prior knowledge even when structurally wrong."

- **Rule 2 QUALIFIED**: Pre-registered "BO is saturated at 66 evals → VR crosses BO at extended budget." BO 1A @ 144 is **2.2% higher** than @66, so VR 1A ext (0.278) is actually 98.9% of BO @ 144 rather than >100%. Softer phrasing: "VR approaches/matches matched-budget BO at 2× budget."

- **Rule 10 RESTATED**: Pre-registered "Sonnet HD replicates Opus HD perfectly." Data: HV and false-positive outcomes generalize (95.6% vs 96.6%, both zero false positives), **but cognitive strategy differs**: Opus dismisses noise via inference (0% noise OAT), Sonnet verifies (20% noise OAT). And **Haiku breaks both the HV and false-positive guarantees**: 43.7% of BO + two claimed noise edges at 0.55, 0.60 confidence on seed 44.

### Findings that emerged from the extended experiments (not in original preregistration)

- **Discovery-vs-optimization decoupling at Haiku tier**: Haiku's Sobol-weighted info capture is 0.959 (near-parity with Opus/Sonnet) but HV is only 43.7% of BO. **Haiku identifies the structure but cannot exploit it**.
- **Capability floor at Sonnet tier**: VR's zero-false-positive property holds for Sonnet-and-above, breaks at Haiku.
- **Cognitive strategy gradient**: Opus 0% → Sonnet 20% → Haiku 22% noise OAT fraction. Models achieve similar outcomes via different pathways.

---

## 3. Candidate hero contributions — ranked

### Option A: Benchmark + Diagnostic Suite (Contributions 1+2 jointly)

**Why it's defensible:**
- The measurement apparatus is genuinely novel — BORA has no ground truth, "Are We There Yet?" is discrete, BDA has no mechanism structure. SynthOracle is the first benchmark where you can measure whether an optimization agent actually learned the causal structure that drove its decisions.
- Well-characterized: 70+ runs, 12 metrics per run, pre-registered rubric.
- Reusable by the community — zero API cost, Python + oracle-only.
- Doesn't require VR to be the best reasoning protocol to be valuable.

**Why it might be wrong:**
- The contribution is a tool. Tools are useful but rarely the hero unless the domain cares specifically about benchmarks.
- Reviewers may ask "what did you *learn* from this benchmark?" — pushing us back toward the findings story.

### Option B: Verbal Regularization methodology

**Why it's tempting:**
- VR is a concrete methodology with a protocol, structured outputs, and measurable properties.
- Intuitively "real" — the regularization effect (explicit verbalization prevents sloppy reasoning) matches user's personal experience and has face validity.
- Permuted-feedback test validates that VR genuinely uses feedback (contra the BDA finding in "Are We There Yet?").

**Why it has issues:**
- Doesn't clearly beat BO at matched budget. Best case (1A extended): 99% of BO.
- Not compared against alternatives (BORA, simpler chain-of-thought, LLMNN). Can't claim "VR is the best reasoning protocol."
- Has a capability floor (fails at Haiku). Not model-agnostic.
- Expensive (~$3-5/run on 72-eval Opus).

**Cost analysis for upgrading to VR-as-hero:** see Section 6 below.

### Option C: The transferability story (elevated from a supporting finding)

**Why this might be the real hero:**
- **It's VR's clearest advantage over BO.** BO has no mechanism for leveraging prior knowledge across oracles — it always starts fresh. The screen-first VR protocol can take a 1A causal model and apply it to structurally-shifted 1D or topology-rewired 1E, getting measurable benefit (+2% and +5% respectively).
- **It's surprising and counterintuitive.** Pre-registration predicted wrong prior would hurt. Data shows the opposite, statistically (n=3 both conditions, consistent direction).
- **It's mechanistically explained**: the screen-first protocol (empirically test inputs before committing to prior claims) explains how the agent selectively uses what's correct while dismissing what's wrong.
- **It directly addresses a practitioner concern**: "I have a causal model from a related system — can I reuse it?" Answer: yes, even if the systems are structurally different, as long as the protocol screens first.
- **It's connected to the literature** ("Are We There Yet?" showed LLMs are insensitive to feedback; we show LLMs are sensitive to prior *when the protocol allows them to selectively apply it*).

**Additional evidence we could generate cheaply:** a protocol ablation (run 1E with prior but without screen-first) to prove the protocol choice specifically matters — estimated ~$15-20.

### Option D: The empirical findings as the hero

**Why it's a worse framing:**
- Findings alone aren't a paper. They need to be framed as "we built this benchmark, this is what we found."
- The findings are more interesting collectively than individually.

---

## 4. User feedback on the initial synthesis

**Point 1 — VR-as-hero exploration:** User pushed back on committing to Benchmark+Diagnostics as hero without understanding the cost to upgrade VR to hero. User's intuition: "VR is real — there is a real, slop-preventing regularization that is happening. Asking to verbalize what AI produced helps both my understanding of the space as well as brings parsimony." Open to ~$30-100 of additional experiments to explore this; not open to $100+ budget increases.

**Point 2 — Pre-registration framing:** User correctly notes pre-registration is self-imposed hygiene, not a generalizable scientific contribution. Should be a methodology note, not elevated to a section or hero.

**Point 3 — Venue:** TBD. Results lead the destination.

**Point 4 — Transferability:** User asked where this fits. The answer reshapes the hero discussion — see Option C above. Transfer is not falsified; it's actually one of our **strongest** findings after the n=3 update, because the n=1 pilot predicted the wrong direction and the n=3 data revealed the agent successfully leverages partial prior knowledge. This may be the real hero.

---

## 5. Open questions

1. **Is transferability the hero?** Option C framing is the strongest candidate. It combines VR's uniqueness (BO can't do transfer), novelty (contradicts our own pre-registration), and practical utility (practitioners care about prior reuse). Needs user decision.

2. **Do we invest in VR-hero ablations?** Cost-benefit analysis in Section 6 below.

3. **How much emphasis on the benchmark vs methodology vs findings?** Depends on Q1.

4. **Do we need to rerun some experiments with tighter protocol?** Specifically, the screen-first protocol ablation to support Option C.

---

## 6. Cost analysis — what it takes to lead with VR (user's Point 1)

### Minimum viable "VR-as-hero" evidence: ~$30-50

- **Protocol ablation #1**: run 1E prior *without* screen-first protocol (3 seeds, 72 budget). Tests whether the specific protocol choice matters. ~$15-20.
- **Protocol ablation #2**: run 1A with "plain chain-of-thought" prompting (hypothesis text only, no structured iteration summary, no calibration checkpoints). 3 seeds. Tests whether the structured apparatus matters vs free-form reasoning. ~$15-25.

**What this buys us:** "VR's specific design choices (screen-first, structured iteration summaries) are defensible against simpler baselines." Not a triumphant claim, but enough to say "we didn't pick these choices arbitrarily."

### Moderate VR-as-hero evidence: ~$80-150 + 2-5 days engineering

Add to the minimum:
- **BORA comparison**: requires implementing BORA's adaptive LLM-BO protocol on our oracles. If using their codebase: 2-5 days of engineering work to wire to SynthOracle. Compute: ~$30-50 for n=3 seeds × 3 oracles. Tests "VR is better than the closest prior art."
- **Additional protocol ablations**: run 1D prior without screen-first (shows screen-first matters for 1D too), run HD without calibration checkpoints (shows calibration value), run 1A with 6-input prompt but no mechanism log (shows mechanism tracking matters). ~$30-50.

**What this buys us:** Can now make the claim "VR outperforms BORA and simpler prompting alternatives on the SynthOracle benchmark; the design choices are validated by ablation."

### Strong VR-as-hero evidence: ~$120-200 + 1-2 weeks engineering

Add to moderate:
- **LLMNN comparison** from "Are We There Yet?" (LLM embeddings + classical classifier). Reimplement or find code. Tests "VR is better than LLM priors + classical search." ~$40-60.
- **Multiple prompt variants** (ReAct-style, zero-shot, few-shot, chain-of-thought-with-verification). Establishes the specific protocol choice matters. ~$30-50.
- **Two model-tier comparisons** (Opus, Sonnet; Haiku broken per earlier finding). Validates generalization.

**What this buys us:** Full "VR is the right approach to constrained reasoning for LLM optimization agents" claim. Strong enough for ICML/NeurIPS main track submission.

### Recommendation

**Go with the minimum viable ($30-50)** unless user decides the VR story needs to be very strong. The minimum package:
- Tests the two most important design choices (screen-first protocol, structured iteration summary format)
- Preserves the option to frame VR as "defensible and measurable" without needing to claim "best"
- Stays well under the budget ceiling

The minimum viable ablations + the transferability elevation (Option C) together form a **coherent paper** where:
- Benchmark enables measurement (Contribution 1+2)
- VR is a specific implementation of constrained reasoning (Contribution 3)
- **Transfer is the novel thing VR can do that BO cannot** (new Contribution 4)
- Pre-registration is a methodology note (hygiene, not contribution)

Total additional cost: ~$30-50 for minimum ablations + $15-20 for 1E-without-screen-first = **~$45-70 total additional experiment cost** to build a VR-as-hero paper with the transferability story at its center.

---

## 7. Decision pending (updates come here as we iterate)

- [ ] Hero contribution: TBD pending user decision on Options A, B, C above
- [ ] Target venue: TBD
- [ ] Ablation package: TBD pending hero decision
- [ ] Paper outline: TBD after hero decision
