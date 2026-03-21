# Review: Language-Based Bayesian Optimization Research Assistant (BORA)

**Paper:** Cisse, Evangelopoulos, Gusev, Cooper. IJCAI 2025.
**Reviewed:** 2026-03-20
**Reviewer:** Claude (structured review for interactive discussion)

---

## Background

The practical setting is multivariate optimization of expensive experiments — chemical reactions, agricultural trials, engineering configurations — where each evaluation costs hours or dollars and you get maybe 50-100 tries. Bayesian optimization (BO) is the workhorse: fit a Gaussian process (GP) surrogate to observed data, use an acquisition function (Expected Improvement, Upper Confidence Bound, etc.) to pick the next most informative evaluation, iterate. BO has strong theoretical foundations — the GP gives calibrated uncertainty, the acquisition function formally balances exploration and exploitation.

But BO has two chronic failure modes that get worse as dimensionality grows. First, **cold start**: with random or Latin Hypercube initialization, the first 10-20% of your budget is essentially wasted mapping terrain that a domain expert would skip. In a 10D space with 105 evaluations, spending 10 on LHS means 10% of your budget produces near-zero information about the optimum. Second, **local entrapment**: the GP can become overconfident in a basin of attraction, and the acquisition function keeps proposing nearby points rather than exploring globally. TuRBO addresses this with local trust regions, but it's a BO-internal fix that doesn't leverage external knowledge.

The human-in-the-loop (HIL) tradition addresses this by injecting expert knowledge. ColaBO lets a user specify a prior distribution over where the optimum is. HypBO (from the same Liverpool group) lets users define static "box" hypotheses — rectangular regions they believe are promising. piBO augments the acquisition function with user beliefs. But these approaches share a limitation: the human input is **static**. If the expert's initial intuition is wrong, it stays wrong. And requiring repeated human input doesn't scale to the dozens of parallel optimization campaigns a modern lab runs.

A separate line of work replaces BO with LLMs entirely. LLAMBO mimics BO's structure using in-context learning. OPRO iteratively prompts the LLM with past evaluations and asks for better solutions. These are creative but lack BO's mathematical guarantees — the LLM has no principled way to balance exploration and exploitation, and the approaches have been limited to low-dimensional hyperparameter tuning.

BORA's thesis: **don't choose between BO and LLMs — make them collaborators.** Share a common GP surrogate. Let BO handle the mathematical optimization. Let the LLM inject domain knowledge, comment on progress, and propose hypothesis-driven experiments. Use a heuristic policy to decide when each should lead. This is the first system (to the authors' knowledge) that creates a dynamic, adaptive BO-LLM synergy where both agents operate on the same model and the balance shifts based on performance.

---

## Key Ideas

### 1. The Comment Object — Structured Reasoning as Interface

The core interface between the LLM and the optimization loop is a structured JSON object called a "Comment." This is BORA's most important design decision, and it's worth understanding in detail because it defines the depth of reasoning the system can express.

A Comment contains two parts:
- **Insights** (called "comment" in the JSON): A free-text paragraph where the LLM reflects on optimization progress — what patterns it sees, what's working, what's failing, what it plans to try next.
- **Hypotheses**: A list of named conjectures, each with:
  - A descriptive name (e.g., "Optimized L-Cysteine and P10-MIX1")
  - A rationale (e.g., "Combining L-Cysteine at 1.5 g/L with a high concentration of P10-MIX1 is expected to maximize charge separation and enhance HER significantly")
  - A confidence level: high / medium / low
  - Concrete test points — specific parameter settings to evaluate

The Comment accumulates across iterations: each LLM invocation receives all previous Comments as context, creating a running narrative of the optimization. This is where BORA's claim to "reasoning" lives.

But look at what the Comment does *not* require: it never asks the LLM to predict *what will happen* at a proposed point. The rationale explains *why* the LLM thinks a region is promising, but there's no mechanism to verify whether that reasoning is correct vs post-hoc rationalization. The confidence level is self-assessed, never validated against outcomes. The insight text reads like a lab notebook entry — "the data suggests that..." — but the system never checks whether yesterday's suggestions actually confirmed yesterday's hypotheses. It's reasoning without accountability.

### 2. Three-Action Adaptive Policy — The LLM as Conditional Resource

BORA doesn't invoke the LLM at every iteration. It selects from three actions:

| Action | What happens | Budget cost | When triggered |
|--------|-------------|-------------|----------------|
| **a1** | Vanilla BO: acquisition function proposes 1 point | 1 evaluation, 0 LLM calls | GP uncertainty low → landscape well-modeled |
| **a2** | LLM takeover: LLM sees all data + history, proposes n_LLM new points from its own hypotheses | n_LLM evaluations, 1 LLM call | GP uncertainty high → landscape poorly understood |
| **a3** | Collaborative: BO proposes n_BO=5 candidates, LLM selects n_LBO=2 that best align with its hypotheses | n_LBO evaluations, 1 LLM call | Intermediate uncertainty |

The selection mechanism monitors GP posterior uncertainty at q=5000 fixed "monitoring points" scattered across the search space. The mean uncertainty sigma_mean is compared against two thresholds derived from the running maximum sigma_max:
- sigma_mean > 0.5 * sigma_max → a2 (the GP knows very little — let the LLM lead)
- 0.3 * sigma_max < sigma_mean < 0.5 * sigma_max → a3 (some knowledge — collaborate)
- sigma_mean < 0.3 * sigma_max → a1 (GP is confident — let BO lead)

This is the key engineering insight: **the LLM is expensive (in latency and API cost) and unreliable (stochastic, sometimes invalid outputs), so deploy it only when the mathematical method is struggling.** The policy makes BORA self-regulating — on easy problems, it converges to vanilla BO; on hard problems with rich domain structure, it leans on the LLM.

The deeper question is whether GP uncertainty is the *right* signal for "when does the LLM help?" GP uncertainty measures how well the *surrogate model* covers the space, not how much *domain knowledge* is needed. A high-dimensional smooth function might have high GP uncertainty (not enough data to fit well) but no domain structure for the LLM to exploit. Conversely, a well-sampled function with hidden regime transitions might have low GP uncertainty (smooth within each regime) but would benefit enormously from the LLM identifying the transition.

### 3. Plateau Detection and Trust — Adaptive Coupling Strength

The uncertainty-based action selection is overridden by a **plateau detector**: if the best objective value hasn't improved by more than gamma=5% in the last m consecutive BO steps, the system declares stagnation and triggers LLM intervention regardless of uncertainty levels.

The plateau window m is where BORA gets genuinely interesting. It starts at m_init = ceil(2*sqrt(d)) — scaling with dimensionality because higher-dimensional problems need more BO steps to make progress. Then m adjusts dynamically based on a **trust score** T_l that tracks the LLM's track record:

The trust score works as follows: after each LLM intervention, BORA computes a reward r_l = max(f(x) for x in suggested points) - y_max (the previous best). This reward is passed through a sigmoid to produce an intervention score between 0 and 1 (positive reward → score 1, negative → score compressed toward 0.5, normalized by |y_max| for scale invariance). Trust T_l is the rolling average of the last W=3 intervention scores, initialized optimistically at 0.9.

Trust feeds back into the plateau window: high trust shortens m (the system asks for LLM help sooner), low trust lengthens it (the system waits longer before invoking the LLM). The adjustment is capped at Delta_max=15 steps per update to prevent wild swings.

This creates a self-regulating feedback loop: if the LLM is helping, trust it more; if it's not, rely more on BO. The m_init ablation (Fig 9) confirms this matters — m_init=2 (always asking the LLM) degrades to LLM-Only, m_init=32 (never asking) degrades to vanilla BO, the default finds the sweet spot. What this also reveals is that **the adaptive policy is doing most of the work**. The LLM's actual reasoning quality may matter less than the *timing* of when it's invoked.

### 4. Self-Consistency and Robustness Engineering

BORA generates n=3 independent LLM outputs for each intervention (a2 or a3), then consolidates them through a multi-step reflection: (a) critique each output individually, (b) evaluate consistency across outputs, (c) cross-reference against the actual dataset, (d) resolve discrepancies into a single unified Comment. This is a form of self-consistency decoding applied to structured scientific reasoning.

On top of this, every Comment undergoes feasibility verification: the JSON structure must be valid, hypotheses must have non-duplicate points within bounds and satisfying constraints, and the LLM gets up to 3 retry attempts on failure before the system falls back to vanilla BO. These aren't theoretically interesting, but they're practically essential — and they reveal how often LLMs produce invalid outputs even with few-shot prompting and structured templates. The paper doesn't report the failure rate, which would have been informative.

### 5. Experiment Card — Context Injection as Inductive Bias

The user provides a structured JSON "Experiment Card" containing: experiment name, domain, description, constraints, parameter definitions (names, bounds, types, discretization steps), and target variable description. This is the only channel for domain knowledge to enter the system.

The Experiment Card serves dual purpose: it configures the BO (bounds, constraints, discretization) and primes the LLM's domain reasoning. The Hydrogen Production card (Fig 14), for example, describes each chemical's role ("L-Cysteine: An amino acid that can act as a hole scavenger..."), giving the LLM enough context to form chemistry-grounded hypotheses. This is more than just parameter bounds — it's a natural-language description of the problem's causal structure, even if BORA doesn't explicitly require causal reasoning.

The Experiment Card is essentially a lightweight version of what we'd call a "mechanism map" — a human-authored description of how inputs relate to outputs through domain-specific processes. BORA doesn't formalize this connection, but the quality of the Experiment Card likely determines much of BORA's advantage over baselines. A bare-bones card (just parameter names and bounds) would reduce BORA to "BO with LLM-initialized points," while a rich card (detailed causal descriptions) gives the LLM more to work with. The paper doesn't ablate card richness.

---

## Results

### Level 1: For a smart high-schooler

Imagine you're trying to find the best recipe for a chemical reaction, but each experiment takes hours. You want to be smart about which experiments to try next. Normally, a statistical method called Bayesian optimization (BO) does this — it builds a mathematical model of what it's seen so far and picks the next most informative experiment. The problem is it starts blind and can get stuck in local "good enough" solutions.

BORA adds an AI assistant (an LLM like ChatGPT) that watches the experiments and jumps in when progress stalls. The AI looks at all the data, writes commentary about what seems to be working, forms hypotheses ("maybe combining chemical A with catalyst B is the key"), and suggests specific experiments to try. Crucially, the AI doesn't run the show — a heuristic policy decides when the AI should intervene vs when the math should drive.

They tested BORA on 6 problems ranging from math puzzles to real chemistry and agriculture. BORA found better solutions faster than standard BO and other LLM-augmented methods on 5 of 6 problems, and it cost only about $5 in AI API fees for all experiments combined. The AI was especially helpful at the start (suggesting smart first experiments instead of random ones) and when progress plateaued (breaking out of local optima with new hypotheses).

### Level 2: For an undergraduate

BORA operates a shared GP surrogate model that both the BO acquisition function and the LLM use. The system selects among three actions at each step based on GP mean uncertainty over q=5000 fixed monitoring points:

- **a1 (vanilla BO):** When uncertainty is low (sigma_mean < 0.3 * sigma_max), the GP is well-calibrated — standard EI acquisition suffices.
- **a2 (LLM takeover):** When uncertainty is high (sigma_mean > 0.5 * sigma_max), the GP is poorly informed — the LLM receives the full dataset + its previous comments and proposes n_LLM new test points via structured hypotheses.
- **a3 (collaborative selection):** At intermediate uncertainty, BO generates n_BO=5 candidate points via acquisition, and the LLM selects the n_LBO=2 that best align with its current hypotheses.

A **plateau detector** overrides this: if no improvement >5% occurs in the last m consecutive BO steps, the LLM is called regardless. The plateau window m adapts via a **trust score** — a rolling average of whether LLM-suggested points actually found new optima. High trust shortens m (more frequent LLM calls); low trust lengthens it.

Results across 6 benchmarks (3 synthetic functions up to 15D, 4 real-world up to 10D), 105 sample budget, 10 seeds each:

- BORA beats all 6 baselines (BayesOpt, TuRBO, ColaBO, HypBO, LAEA, Random Search) in cumulative regret on 5/6 tasks. Statistically significant vs HypBO (p=0.02, sign test + Bonferroni), not significant vs ColaBO (p=0.20).
- On synthetic functions, BORA's main advantage is **initialization** — the LLM suggests structurally meaningful starting points (edges, centers, symmetry points). On Levy 10D with symmetric bounds, this nearly always finds the optimum immediately.
- On real-world tasks, the advantage extends to **mid-optimization hypothesis revision**. In Hydrogen Production (10D chemistry), the LLM iteratively identifies L-Cysteine + P10-MIX1 as key drivers and Methylene Blue as detrimental, achieving 47% regret reduction vs ColaBO.
- An **LLM-Only ablation** (same prompts, no BO) shows competitive early performance but severe stagnation after ~30 samples — the LLM lacks BO's exploration-exploitation balance.
- A **plateau window ablation** on Ackley 15D shows m_init=2 degrades to LLM-Only behavior, m_init=32 degrades to vanilla BO, and the default m_init=ceil(2*sqrt(d))=8 is optimal.

### Level 3: For an early graduate student

The core tension BORA navigates is **when to trust the LLM**. The paper's answer — GP uncertainty as the switching signal — is elegant but raises questions about what's actually being measured.

The uncertainty thresholds (0.3 and 0.5 of sigma_max) are empirically tuned and never ablated, unlike the plateau window which gets a proper sensitivity study. Since sigma_max is a running maximum that only increases, the thresholds are non-stationary — what counts as "high uncertainty" at step 10 differs from step 80. This means the action selection drifts toward more BO and less LLM over time, which may explain why BORA doesn't fully collapse to LLM-Only even with aggressive plateau settings.

The trust mechanism is conceptually appealing but the reward function (Eq. 8-9) reveals a design choice: the LLM gets credit only for finding *new optima*, not for *informative non-improvements*. An LLM suggestion that narrows the GP's uncertainty without beating y_max scores ~0.5 via the sigmoid. This biases the trust toward exploitation-oriented LLM behavior — the LLM is rewarded for suggesting winners, not for suggesting experiments that teach the GP about unexplored regions. For our VR agent, this distinction matters: scientific reasoning should value information, not just performance.

The synthetic function results deserve skepticism. Anonymizing function names ("mathematical function" instead of "Branin") is insufficient — GPT-4o-mini has seen thousands of optimization benchmarks in training. The fact that BORA's initialization on Levy nearly always converges to [1,...,1] strongly suggests the LLM is recognizing the function class from the bounds and dimensionality. The real-world benchmarks are more credible, but the Hydrogen Production task uses a GP surrogate trained on 1119 real experiments as the oracle — so BORA is optimizing a GP approximation, not the actual chemistry.

The statistical analysis is underpowered for the claims made. A sign test across 6 tasks with Bonferroni correction needs at least 6/6 wins for p<0.05. BORA achieves this vs HypBO (p=0.02) but not vs ColaBO (5/6 wins, p=0.20). The paper acknowledges this but still frames the ColaBO comparison as favorable. With 10 seeds per task, a paired comparison within tasks would have more power, but the authors chose a conservative cross-task test.

Perhaps most importantly: **BORA never tests whether its LLM component actually learns from experimental feedback.** The LLM-Only ablation shows the LLM stagnates alone, and the plateau window ablation shows the adaptive policy matters, but neither addresses whether the LLM's commentary at iteration 50 reflects genuine learning from 50 data points vs pattern-matching on the prompt format. The "Are We There Yet?" paper's permuted-feedback finding (LLMs show identical behavior with real vs shuffled data) directly threatens BORA's narrative. BORA's strong initialization results could be entirely domain-knowledge-driven (the LLM knows chemistry) rather than feedback-driven (the LLM learns from the data). The paper doesn't distinguish these.

---

## Key Quotes

> "To our knowledge, this is the first time that a rigorous, dynamic synergy of black-box BO with LLMs has been proposed in this context."

The claim is carefully scoped ("in this context") but the word "rigorous" is doing heavy lifting. The adaptive policy is heuristic, not principled — the thresholds are empirically tuned constants, the trust mechanism is a rolling average, and there's no convergence guarantee for the combined system. "Rigorous" relative to prior LLM-BO hybrids, perhaps.

> "We emphasize that these results do not mean that LLMs are 'smarter' than domain experts. Rather, they highlight BORA's ability to update and refine its hypotheses based on new data."

An honest caveat, but it begs the question they don't answer: *does* BORA update and refine based on data, or does it generate plausible-sounding hypotheses that happen to overlap with good regions because the LLM has domain knowledge? The paper shows hypothesis evolution (Fig 2, 28) but never controls for whether that evolution is data-driven.

> "LLMs tasked with regression inherently perform an implicit ICL modeling of the conditional distribution p(y|x; D)... BORA extends this modeling by integrating all previously gathered data D and all the LLM's comments C."

This is the most theoretically interesting claim in the paper. They're arguing that the LLM + its Comment history forms an implicit surrogate model p(y|x; D, C) that's richer than the GP alone because it incorporates domain knowledge through the Comments. If true, this is profound — the Comments aren't just suggestions, they're a natural-language augmentation of the GP's prior. But the paper doesn't test this claim. A simple test: compare BORA with vs without Comment history (fresh LLM at each step, no memory of previous Comments). If Comment history matters, it suggests the LLM is genuinely building a richer model; if not, the Comments are just prompt decoration.

> "The stochastic nature of the LLM reasoning, which can diverge considerably even with identical prompts."

The authors identify this as a "potential limitation," but it's more fundamental than they acknowledge. If the LLM's reasoning is stochastic enough to diverge on identical inputs, then the *hypotheses* are also stochastic — meaning BORA's performance depends on which of many possible reasoning paths the LLM happens to take. The self-consistency mechanism (n=3 samples) mitigates but doesn't eliminate this. It means BORA's reported ±0.25 standard error reflects both random initialization *and* random LLM reasoning, and these variance sources aren't decomposed.

> "The total cost of running BORA amounts to about 5 US Dollars (USD) at today's prices."

Arguably the most practically important number in the paper. It demonstrates that useful LLM-BO synergy doesn't require expensive models or massive prompt budgets. GPT-4o-mini at $5 for 6 benchmarks × 10 trials means ~$0.08 per optimization run. This sets a strong economic baseline that any competitor (including our VR agent) needs to beat or justify exceeding.

---

## Study Questions

### Warm-up

1. **What is the Experiment Card, and why does it matter more than it seems?** Walk through Figure 14 (Hydrogen Production card). How does the description of L-Cysteine as "a hole scavenger" differ from just listing it as "parameter 2, bounds [0,5]"? What would BORA lose if the card contained only parameter names and bounds?

2. **Explain the three actions (a1, a2, a3) in terms of exploration vs exploitation.** Which action is most exploitative? Which is most explorative? Can you construct a scenario where the uncertainty-based selection picks the wrong action?

3. **The trust score is initialized at 0.9 (optimistic).** Why is optimistic initialization a reasonable choice here? What would happen if it were initialized at 0.1 instead? Connect this to the exploration-exploitation tradeoff in multi-armed bandits.

### Intermediate

4. **BORA's LLM-Only ablation shows stagnation after ~30 samples.** The authors attribute this to "LLMs lack the balanced exploration-exploitation trade-offs that are inherent to BO-based methods." But is there an alternative explanation? Could the stagnation be caused by the LLM's context window filling up with data, the LLM fixating on its own previous hypotheses, or simply the LLM running out of domain-knowledge-based "easy wins"? How would you distinguish these explanations experimentally?

5. **The plateau detection threshold gamma=5% is fixed.** In the Hydrogen Production experiment, HER values range from 0 to ~25 umol/h. A 5% improvement at HER=20 means finding HER>21 — a 1-unit gain. At HER=1, it means finding HER>1.05 — a 0.05-unit gain. How does this fixed percentage interact with the typical diminishing-returns shape of optimization curves? Is it too lenient early (everything looks like progress) or too strict late (real gains are small)?

6. **Redesign BORA's trust mechanism for a setting where the goal is understanding, not optimization.** In BORA, the LLM earns trust by finding new optima. But suppose we want the LLM to earn trust by making accurate predictions about experimental outcomes. What would the reward function look like? What new failure modes would emerge?

### Advanced

7. **BORA passes the full dataset to the LLM as a table.** For Hydrogen Production (10D, up to 105 rows), that's a 10×105 table in the prompt. The fallback is a correlation matrix (10×10). Design an experiment to test whether the LLM is actually using individual data points vs just picking up on aggregate patterns. Hint: consider adversarial data points — rows that contradict the overall trend.

8. **The paper claims the Comment history creates an implicit model p(y|x; D, C).** Design a test for this claim. You have access to BORA's codebase (it's open source). What would you modify? What control conditions would isolate the contribution of Comment accumulation vs fresh-prompt LLM vs the GP alone?

9. **Suppose a reviewer argues: "BORA's advantage over ColaBO is not statistically significant (p=0.20), and its advantage on synthetic functions is contaminated by LLM training data. The only credible evidence is the Hydrogen Production result, which uses a GP oracle, not real chemistry. Therefore, BORA has no reliable evidence of improvement."** How would you defend the paper? How would you attack it further? What additional experiment would settle the debate?

---

## Challenge Corner

**C1: Is BORA doing science, or is it doing informed optimization?**

The paper frames BORA as enabling "reasoning" and "hypothesis-driven" optimization. But examine the Hydrogen Production example (Fig 22): at iteration 27, the LLM writes "The highest observed HER of 18.24 µmol/h was achieved with a combination of L-Cysteine (1.50 g/L) and P10-MIX1 (3.40 g/L)... high concentrations of Methylene Blue consistently correlate with lower HER." This is *data summary*, not reasoning. The LLM describes correlations in the observed data and proposes points that extrapolate those correlations. It never asks *why* L-Cysteine helps — is it acting as a hole scavenger (as the Experiment Card says)? Does it interact with the catalyst surface? Is there an optimal pH range? The "hypotheses" are really "promising regions" with natural-language justifications. Compare this to what a chemist would do: form a mechanistic hypothesis (L-Cysteine improves charge separation at the P10-MIX1 surface), predict a consequence (adding SDS should enhance this by improving dispersion), test the prediction. BORA skips the mechanism and the prediction.

**C2: The adaptive policy might be the entire contribution, not the LLM reasoning.**

The plateau window ablation (Fig 9) shows that performance is highly sensitive to *when* the LLM is invoked — m_init=2 and m_init=32 both significantly degrade performance. But this means the *timing* of LLM intervention matters more than the *content* of the intervention. Would BORA work equally well if, instead of an LLM generating hypotheses, it simply generated random points in unexplored regions whenever a plateau was detected? The paper doesn't test this "smart random restart" baseline. If the answer is "yes, random restarts work almost as well," then BORA's contribution reduces to the adaptive policy, not the LLM reasoning.

**C3: The synthetic benchmarks are compromised by training data leakage.**

BORA anonymizes function names, but GPT-4o-mini has been trained on thousands of papers that analyze Branin, Levy, and Ackley. The LLM doesn't need to see the name — the bounds ([-5,10]×[0,15] for Branin, [-10,10]^10 for Levy), dimensionality, and even the optimization context ("mathematical function") provide enough signal. The near-instant convergence on Levy (the LLM suggests [1,...,1] which is the global optimum) is the smoking gun. This doesn't invalidate the real-world results, but it means 3 of 6 benchmarks are suspect. The authors changed Ackley's bounds to [-30,20]^d to break symmetry, suggesting they were aware of this problem — but they don't discuss it.

**C4: The "LLM-Only" baseline doesn't isolate the right thing.**

To understand BORA's value, you'd want to decompose: (a) value of LLM initialization, (b) value of LLM mid-run hypotheses, (c) value of the adaptive policy, (d) value of the GP. The LLM-Only ablation removes (c) and (d) simultaneously. A more informative set of ablations would be:
- BORA minus initialization (random init, then normal BORA) — isolates (a)
- BORA with frozen hypotheses (LLM initializes but never updates) — isolates (b)
- BORA with random restarts instead of LLM at plateaus — isolates the reasoning in (b) from the restart effect
- BORA minus Comment history (fresh LLM each time, no memory) — tests whether the Comment accumulation matters

Without these, we can't tell whether BORA's advantage comes from the LLM's chemistry knowledge (at initialization) or from its iterative reasoning (during optimization).

**C5: What does this mean for SynthOracle?**

BORA establishes that LLM-BO hybrids *work* — they find better optima faster. But it provides no evidence that the LLM *understands* anything about the system being optimized. The hypotheses are correlational summaries that serve as warm-start points. This is exactly the gap SynthOracle is designed to probe: if you give the agent a system with known causal structure, can you tell the difference between "the LLM got lucky with good initialization from domain knowledge" and "the LLM actually discovered how the system works"? BORA can't answer this question because it has no ground truth. We can, because the oracle's DAG is the answer key.

More concretely: BORA's Condition 2 (LLM-informed BO) in our 4-condition framework would likely match BORA's performance. The question is whether our Conditions 3 and 4 (with the predict→reconcile loop) do better, and *why* — not just in optimization performance but in structural recovery (edge precision/recall). BORA raises the bar for what "LLM priors alone" can achieve. Our job is to show that constrained reasoning adds value beyond good priors.

---

## Connection to Project

### How BORA relates to SynthOracle

BORA is our **closest prior art** and the primary system to differentiate against. The relationship:

| Aspect | BORA | SynthOracle VR Agent |
|--------|------|---------------------|
| LLM role | Retrospective commentary + point suggestion | Prospective prediction + causal model building |
| Hypothesis type | Correlational ("X and Y together → high Z") | Causal ("X drives Z through mechanism M because...") |
| Prediction tracking | None | Core metric (directional → quantitative accuracy curve) |
| Reconciliation | None — LLM sees data, comments, moves on | Explicit: prediction failure triggers model revision |
| Ground truth evaluation | Impossible (unknown real-world structure) | Built-in (oracle DAG is the answer key) |
| Transfer | Not tested | Core experiment (variant oracles) |
| Multi-objective | No (scalar maximization) | Yes (Pareto front, multiple outputs) |
| BO integration | Shared GP, adaptive action selection | TBD — likely similar, but with mechanism-labeled GP option |

### Key differentiation points

1. **BORA's hypotheses lack causal depth.** Read BORA's Iteration 27 comment on Hydrogen Production (Fig 22): "Combining L-Cysteine at 1.5 g/L with a high concentration of P10-MIX1 is expected to maximize charge separation and enhance HER significantly." This is a reasonable hypothesis, but it's not *testable as a mechanism*. It doesn't predict what happens if you fix L-Cysteine and vary P10-MIX1, or what role charge separation plays vs pH effects. Our VR agent's hypotheses must specify mechanisms and make quantitative predictions — "increasing X2 from 0.3 to 0.6 should increase Y1 by ~0.15 because M1 scales as X2^1.37."

2. **BORA has no prediction-reconciliation loop.** This is the core gap. The LLM comments on data after seeing it, but never commits to predictions before evaluation. There's no mechanism for the system to detect when the LLM's mental model is *wrong* — only when its *suggestions don't find new optima*. Our verbal regularization protocol (hypothesize → predict → evaluate → reconcile) forces the LLM to have skin in the game. Prediction failures are information; in BORA, they're invisible.

3. **BORA can't evaluate reasoning quality.** Without ground truth structure, BORA can only measure optimization performance. A lucky guess and a deep insight look identical in the metrics. Our synthetic oracle with known DAG enables measuring *understanding* (edge precision/recall) separately from *performance* (hypervolume). This is the central contribution of the benchmark: separating "did you find the optimum?" from "do you know why it's the optimum?"

4. **BORA doesn't test whether the LLM actually learns from data.** The "Are We There Yet?" paper (our next review) claims LLMs show no feedback sensitivity. BORA's LLM-Only ablation shows stagnation, but doesn't test whether BORA's LLM component actually updates its beliefs vs just pattern-matching. Our permuted-feedback ablation would directly test this: give the agent real data vs shuffled data, and compare the reasoning quality. If the reasoning is identical, the LLM isn't learning — it's performing.

### What we should adopt from BORA

1. **The adaptive intervention policy.** Not the exact thresholds, but the principle: let BO run when it's making progress, invoke the LLM when it stalls. This avoids wasteful LLM calls and respects the fact that BO has provable properties the LLM doesn't.

2. **Structured output format.** The Comment object (insights + hypotheses with rationales) is a good template. We extend it with predictions, mechanism maps, and explicit pre-registered expectations.

3. **Self-consistency sampling.** Generating n=3 outputs and consolidating reduces LLM variance. Simple and effective.

4. **Experiment Card → Oracle Card.** Standardized context injection is good practice. Our version would include mechanism-level information (for the "with map" condition) and be ablatable (rich card vs bare card).

5. **Feasibility checking + fallbacks.** Practical necessity — LLMs produce invalid outputs. Plan for it from day one.

### What we should NOT adopt

1. **Correlational hypotheses without predictions.** BORA's hypothesis format (name, rationale, confidence, points) doesn't require the LLM to stick its neck out. We need mechanism identification *and* falsifiable predictions.

2. **Trust based solely on optimization performance.** The LLM should be evaluated on whether it *understands* the system, not just whether its suggestions find optima. A "reasoning trust" score based on prediction accuracy would be more informative.

3. **Single-objective framing.** Our oracles are multi-objective by design. The trust mechanism, plateau detection, and the very notion of "improvement" all need multi-objective analogs (hypervolume indicator, Pareto dominance).

---

## Synthesis Pointers

- BORA is the primary "Condition 2" comparison (LLM-informed BO without causal reasoning). Our 4-condition framework should explicitly position one condition as "BORA-like" — same Comment structure, same adaptive policy, but no prediction requirement.
- The adaptive policy principle (invoke LLM on plateau) maps directly to our agent architecture. It should be a shared component across Conditions 2, 3, and 4, not a differentiator.
- The trust mechanism's bias toward exploitation (rewarding new optima, not informative experiments) connects to T5 from the prior synthesis: the tension between optimization performance and scientific understanding. BORA resolves this tension entirely in favor of optimization. We resolve it differently.
- The Experiment Card richness question connects to our "with map" vs "without map" conditions. BORA's card is a lightweight mechanism map. Our Condition 4 makes this explicit; our Condition 3 removes it.
- BORA's $5/run cost is the economic benchmark. Our VR agent will be more expensive (longer prompts, prediction tracking, reconciliation). The added cost must be justified by added information — either better optimization or better understanding or both.

---

## Discussion Notes + Q&A

*To be filled during interactive discussion.*
