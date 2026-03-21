# Review: LLMs for Bayesian Optimization in Scientific Domains: Are We There Yet?

**Paper:** Gupta, Hartford, Liu. arXiv:2509.21403, 2025.
**Reviewed:** 2026-03-20
**Reviewer:** Claude (structured review for interactive discussion)

---

## Background

The promise of LLM-based experimental design goes like this: the LLM has vast prior knowledge about biology, chemistry, materials science. If you show it experimental results and ask "what should I try next?", it should leverage both its priors *and* the observed data to make increasingly good selections. This is the thesis behind BioDiscoveryAgent (Roohani et al., 2024), LLAMBO (Liu et al., 2024), and implicitly behind BORA's iterative hypothesis refinement.

This paper asks the question nobody wanted to ask: **does the LLM actually use the experimental feedback, or is it just replaying its priors?**

The context is closed-loop experimental design — the sequential process of selecting candidates to evaluate, observing outcomes, and selecting the next batch. Two domains: single gene perturbation (which of ~18,000 genes regulate a target phenotype?) and molecular property prediction (which molecules in a library have high solubility/ionization energy/etc.?). These are real scientific tasks with real datasets, not toy benchmarks. The candidate spaces are large (18,000 genes, up to 11,565 molecules), the budgets are small (5 rounds × 128 candidates = 640 total evaluations), and the "hit" rate is low (3-5% of candidates are hits).

The experimental design literature has a clean decomposition of what matters: **prior knowledge** (what you know before any experiments), **exploration strategy** (how you select diverse candidates), and **posterior updating** (how you incorporate experimental results into future selections). Classical methods like Gaussian processes and linear UCB excel at the last two. LLMs supposedly contribute the first. The question is whether LLMs also contribute to posterior updating — whether they actually learn from the data they're shown.

Prior work assumed the answer was yes. BioDiscoveryAgent prompts Claude 3.5 Sonnet with accumulated experimental history and asks it to reflect, plan, and propose new candidates. The LLM's responses *look* like they incorporate feedback — "the previous round identified genes involved in RNA processing, so we will focus on related pathways." But looking like you're using feedback and actually using feedback are different things.

---

## Key Ideas

### 1. The Permuted Feedback Ablation — A Surgical Test for Learning

The paper's central contribution is a simple, elegant experimental design that directly tests whether LLMs use experimental feedback. They compare:

- **BDA**: BioDiscoveryAgent with real experimental feedback (correct candidate-outcome pairs)
- **BDA-Rand**: Same system, same prompts, but the feedback is *randomly permuted* — each candidate is paired with a random other candidate's outcome

This is a clean intervention. Permutation preserves the marginal distributions — the LLM still sees realistic-looking scores and realistic-looking candidate names. What's destroyed is the *joint distribution* — the association between which candidate was tried and what happened. If the LLM is doing any posterior updating (learning from the relationship between actions and outcomes), BDA should significantly outperform BDA-Rand. If the LLM is just replaying priors, they should perform the same.

The permutation operates at two levels:
- **Level 1**: Measurement values are shuffled. Gene X's score gets replaced with Gene Y's score.
- **Level 2**: Hit/non-hit labels are shuffled. A gene that was a hit gets labeled as a non-hit, and vice versa. The marginal distributions (how many hits, what the score distribution looks like) are preserved, but which specific genes are hits is randomized.

Both levels can apply simultaneously, completely decorrelating the feedback from reality while keeping it statistically plausible.

### 2. The Results: Feedback Insensitivity is Universal

The headline result (Table 1) is stark: **across all 5 LLMs and all 5 gene perturbation datasets, BDA and BDA-Rand perform comparably. In some cases, BDA-Rand performs slightly *better*.**

| Model | BDA (real feedback) | BDA-Rand (random feedback) |
|-------|-------|-----------|
| Llama-3.1-8B on IL2 | 39.4 | 37 |
| Qwen-2-7B on IL2 | 33.2 | 29 |
| Claude 3.5 Sonnet on IL2 | 59.4 (replicated) | 57.6 |
| Claude 3.5 Sonnet on IFNG | 78.8 (replicated) | 79.4 |
| Claude 3.5 Sonnet on Sanchez | 31.6 | 33.8 |

These are cumulative hits over 5 rounds with 128 perturbations per round. The differences are within noise. Claude 3.5 Sonnet — the most capable model tested — shows the same insensitivity as the 7B open-source models.

This is not a failure of small models. This is not a failure of poor prompting (they use BioDiscoveryAgent's own prompts). This is a property of how instruction-tuned LLMs interact with tabular experimental data: **they don't update on it.**

### 3. Classical Methods Beat LLMs Given the Same Information

Having established that LLMs don't use feedback, the natural question is: do methods that *do* use feedback perform better? The answer is yes.

Table 2 compares BDA against Linear UCB and GP (Gaussian Process) baselines. Critically, the classical methods are given the same LLM embeddings as features — so they have access to the same prior knowledge encoded in the LLM's representation space. The only difference is the selection mechanism: LLM prompting vs mathematical optimization of an acquisition function.

Linear UCB outperforms BDA on 3-4 of 5 datasets for both Llama and Qwen backbones. GP shows mixed results (strong on IL2 with 147.8 hits, weak elsewhere). The picture is clear: **when you properly couple prior knowledge (LLM embeddings) with principled posterior updating (UCB/GP), you beat naive LLM prompting.**

This result is unsurprising if you accept the feedback insensitivity finding. LLM-based experiment design = good priors + no learning. Classical methods with LLM features = good priors + proper learning. The latter should win whenever the experimental data contains information the priors don't already encode.

### 4. LLMNN — A Hybrid That Actually Works

The authors propose LLM-guided Nearest Neighbour (LLMNN), which decomposes the LLM's role cleanly:

1. **LLM provides seed points**: The LLM proposes n_c=5 "cluster centers" — candidate names it thinks are promising based on its domain knowledge. This uses priors only, no feedback learning required.
2. **Nearest-neighbor expansion exploits structure**: For each cluster center, the system finds unexplored candidates nearest to it in embedding space and evaluates those. This exploits the inductive bias that similar candidates have similar properties.
3. **Feedback enters through the memory**: Evaluated candidates are marked as "explored" so they can't be re-selected. The LLM sees hit/non-hit results and is asked to adjust its cluster center proposals.

LLMNN dramatically outperforms BDA across most benchmarks (Table 3). With Llama-3.1-8B on IL2: LLMNN gets 163.3 hits vs BDA's 63.4. With Claude 4 Sonnet on IL2: LLMNN gets 159 vs BDA-GS's 65.2.

The key insight: LLMNN works not because the LLM learns from feedback, but because **it separates what the LLM is good at (naming promising regions) from what it's bad at (updating beliefs from data)**. The nearest-neighbor expansion handles exploitation; the embedding space handles similarity; the LLM just provides the initial direction.

Even more telling: **LLMNN NoExp** — a variant that strips the Reflection and Research Plan text from the LLM output, leaving only the raw gene/molecule names — performs nearly as well as full LLMNN. The reasoning text is decorative. The LLM's value is in the *names* it generates, not the *reasoning* it articulates.

### 5. The Random Centroids Ablation — LLM Guidance Does Matter for Seed Selection

To check whether the LLM's cluster centers are better than random starting points, the authors compare LLMNN NoExp (LLM-chosen centroids) against a random centroids baseline (Tables 5-6). The LLM-guided version wins convincingly: 179.4 vs 76 on IL2 (gene perturbation), 173.3 vs 83.2 on Ion. E. (molecular).

This confirms the decomposition: **LLM priors are genuinely valuable for initialization, but LLM in-context "learning" adds nothing.** The LLM knows which genes are likely to regulate IL-2 production because it's seen biology papers. It doesn't learn which genes actually regulate IL-2 from seeing 128 perturbation results.

---

## Results

### Level 1: For a smart high-schooler

Scientists have been excited about using AI chatbots (like ChatGPT) to design experiments. The idea is: tell the AI what you're studying, show it results from previous experiments, and ask it what to try next. The AI should get smarter with each round of data, like a human scientist would.

This paper tested whether the AI actually learns from the experimental results. They did something clever: they gave the AI *fake* results — shuffled the data so that Gene A's result was labeled as Gene B's, and vice versa. If the AI was really learning from the data, fake results should make it perform worse.

It didn't. The AI performed equally well with real results and fake results. This means the AI was never really using the experimental feedback — it was just suggesting experiments based on what it already knew from its training. It was like a student who writes a lab report that references the data but whose conclusions were decided before looking at any results.

The good news: the AI's pre-existing knowledge *is* genuinely useful. When the researchers built a system that uses the AI only for suggesting starting points (what to look at first), then uses traditional math to explore the neighborhood of those starting points, it worked much better than either approach alone. The lesson: use AI for what it's good at (broad knowledge), and math for what it's good at (learning from data).

### Level 2: For an undergraduate

The paper evaluates BioDiscoveryAgent (BDA), an LLM-based experimental design pipeline, on two scientific domains: gene perturbation (5 datasets, ~18,000 candidates each) and molecular property prediction (3 datasets, 642-11,565 candidates). The task is sequential batch selection: pick 128 candidates per round for 5 rounds, maximizing cumulative "hits" (candidates above the 90th percentile).

The core experiment compares BDA (real feedback) vs BDA-Rand (permuted feedback). The permutation breaks the candidate-outcome correspondence while preserving marginal distributions. Five LLMs are tested: Llama-3.1-8B, Qwen-2-7B, Qwen-2.5-14B, Claude 4 Sonnet, and GPT-4o-mini.

**Finding 1**: BDA ≈ BDA-Rand across all models and datasets. Example: Claude 3.5 Sonnet on IFNG gets 78.8 hits with real feedback vs 79.4 with random feedback. The LLMs show no sensitivity to the correctness of experimental outcomes.

**Finding 2**: Classical methods with LLM embeddings outperform LLM-only selection. Linear UCB (a bandit algorithm) and GP (Bayesian optimization) are given the same embedding vectors the LLM uses internally. Linear UCB outperforms BDA on most gene perturbation datasets. This demonstrates that principled exploration-exploitation (UCB) + LLM representations > LLM reasoning alone.

**Finding 3**: LLMNN (LLM-guided Nearest Neighbor) significantly outperforms both BDA and classical baselines. It uses the LLM to propose n_c=5 cluster centers per round, then expands each center via nearest-neighbor search in embedding space to fill the batch of 128. On IL2 with Llama-3.1-8B: LLMNN gets 163.3 cumulative hits vs BDA's 63.4 and Linear UCB's 35.

**Finding 4**: The reasoning text (Reflection, Research Plan) barely matters. LLMNN NoExp (solution-only, no explanations) matches full LLMNN on most benchmarks. The LLM's contribution is the *names* of promising candidates, not the reasoning around them.

**Finding 5**: LLM-guided cluster centers substantially outperform random centers (Tables 5-6), confirming that LLM priors are genuinely valuable — the LLM knows which genes/molecules are likely relevant. The value is in retrieval from training data, not in in-context reasoning.

### Level 3: For an early graduate student

This paper is more threatening to the LLM-for-science agenda than its modest framing suggests. Let me unpack why.

**The permutation ablation is a near-perfect diagnostic.** It tests a necessary condition for in-context learning: if you're learning from data, corrupting the data should hurt you. The fact that permutation doesn't hurt means one of: (a) the LLM never attends to the feedback entries in the prompt, (b) it attends to them but can't extract the relevant signal, or (c) it extracts something but that something doesn't affect its selections. Option (a) is most likely for the open-source models (limited context attention). Option (c) is most interesting for Claude — it may "understand" the feedback in a shallow sense but have no mechanism to integrate it into its sampling strategy, because next-token prediction doesn't train for sequential decision-making.

**The result generalizes beyond this specific pipeline.** The authors test BDA's architecture (BioDiscoveryAgent prompts), but the finding is about the *LLM's ability to do posterior updating from tabular experimental data in-context*, not about the specific prompt template. BORA uses a different prompt structure, but the core mechanism is the same: show the LLM a table of past evaluations, ask it to reason and propose new ones. If Claude 3.5 Sonnet can't extract signal from 128 gene perturbation results in BDA format, why would it extract signal from 26 Hydrogen Production results in BORA format? The burden of proof shifts to anyone claiming their LLM pipeline *does* learn from feedback.

**The LLMNN result reveals the true architecture of LLM value.** The LLM is a prior distribution over candidate quality — it encodes which genes are biologically relevant, which molecules have certain properties — but it's not an inference engine. LLMNN succeeds because it routes the LLM's job through what it's good at (knowledge retrieval → naming candidates) and routes learning through what embeddings + nearest-neighbor search are good at (exploiting local structure). This is a hybrid that respects the actual capabilities of each component.

**But the experimental setting has important limitations that matter for our project:**

1. **Discrete candidate selection vs continuous optimization.** BDA picks genes from a fixed library of 18,000. There's no interpolation, no gradient-like reasoning ("Gene X scored well, Gene Y scored poorly, so try something between them"). Our oracle setting is continuous — the agent can interpolate between evaluated points. It's possible that LLMs can use feedback for continuous reasoning ("increasing X2 improved Y1, so try even higher X2") even if they can't for discrete retrieval.

2. **No prediction requirement.** BDA asks the LLM "what should I try next?" — a selection task. Our VR agent asks "what will happen if I try this?" — a prediction task. Prediction creates a tighter coupling between the LLM's internal model and the observed data, because the prediction can be directly verified. It's possible that forcing prediction (and showing prediction errors) creates a feedback channel that mere selection doesn't.

3. **No mechanism structure.** The gene perturbation task has weak causal structure — genes interact in networks, but the phenotypic effect of knocking out a single gene is hard to predict from network position alone. Our oracle has explicit, discoverable mechanisms. An LLM that discovers "X1 controls the regime transition at 0.3" has a structural insight that should improve *all* subsequent predictions, not just the next selection.

4. **5 rounds is very few.** BDA gets 5 feedback cycles. BORA gets ~30-50 LLM interventions. Our VR agent might get ~20-50 predict→reconcile cycles. It's possible that feedback sensitivity emerges with more rounds — the LLM needs to see enough data points to overcome its priors. But this is speculative.

**The LLMNN NoExp result is the most quietly devastating finding.** It says the Reflection and Research Plan text — the part where the LLM "reasons" about the data — adds essentially nothing to performance. The LLM's reasoning is performative, not functional. This directly threatens any approach (including ours) that claims the value is in the *reasoning process* rather than the *priors*. Our verbal regularization protocol asks the LLM to hypothesize, predict, and reconcile — but if the reasoning text is decorative, we'd be building an elaborate system around a process that doesn't do work.

The counter-argument we'd make: our protocol isn't just asking the LLM to reason *for our benefit* (audit trail) — it's constraining the LLM to *commit to predictions* that are then verified. The reconciliation step forces the LLM to confront prediction errors. This is fundamentally different from BDA's "Reflection" field, which asks the LLM to look back at data it's already seen (retrospective) rather than commit to claims about data it hasn't seen (prospective). Whether this distinction actually matters empirically is the central question of our project.

---

## Key Quotes

> "Across all datasets and models (including Claude Sonnet 3.5), BDA and BDA-Rand perform comparably. In some cases, the BDA-Rand even performs slightly better."

The "slightly better" is the twist of the knife. Not only does the LLM not use feedback — the presence of real feedback may actually slightly *hurt* by anchoring the LLM on observed data patterns rather than letting its priors range freely. This suggests an adversarial dynamic: showing the LLM real data can *narrow* its search in unproductive ways.

> "These results suggest that current open- and closed-source LLMs do not perform in-context experimental design in practical experimental design tasks."

Carefully stated — they say "do not perform in-context experimental design," not "cannot." The distinction matters. Future models with different training objectives (e.g., trained on sequential decision problems), or current models with different prompting strategies (e.g., explicit prediction + verification), might behave differently. The paper establishes the current state, not an impossibility result.

> "The strong initially performance of the LLMs is therefore likely the result of theirs priors on ordering of genes and is not affected by the feedback of past experiments appended in its prompt."

The grammatical errors aside, this is the key interpretation. LLMs have internalized biological knowledge about gene function from their training corpus. When asked "which genes regulate IL-2?", they produce a reasonable list from memory. This list doesn't change when you show them experimental results because the results are less informative than their priors — or because they lack the mechanism to integrate new evidence into their priors within a single context window.

> "LLMNN requires minimal assumptions — of LLM generating valid gene names as per HGNC nomenclature... which is reasonable for modern-day LLMs that have been pretrained on an internet-scale of knowledge."

This frames the LLM as a knowledge base that you *query*, not an agent that you *interact with*. The LLMNN architecture literalizes this: the LLM is a function from (task description, feedback history) → candidate names, and the rest is classical computation. This is the hybrid paradigm the paper advocates, and it maps cleanly to our decomposition of LLM value.

> "One of the contributing factors to the performance of LLMNN is maintaining a memory that keeps track of which genes have already been explored. This ensures that similarity queries return unexplored neighbours at every query, in contrast to the BioDiscoveryAgent, which doesn't maintain this state."

A mundane but crucial detail. BDA re-suggests previously evaluated candidates because the LLM has no external memory of what's been tried. LLMNN's candidate memory prevents this. Some of BDA's poor performance is simply wasted evaluations on repeats — a bookkeeping failure, not a reasoning failure. This is a reminder that engineering details (deduplication, memory management) can matter more than reasoning sophistication.

---

## Study Questions

### Warm-up

1. **Explain the permuted feedback ablation in your own words.** What does it destroy? What does it preserve? Why is it a better test of "learning from feedback" than, say, just removing the feedback entirely?

2. **Why does LLMNN outperform BDA so dramatically?** Walk through the IL2 example: BDA gets 63.4 hits, LLMNN gets 163.3. Both use the same LLM. What does LLMNN do differently that accounts for the 2.5x improvement?

3. **The paper says "LLMNN NoExp" (without explanations) performs nearly as well as full LLMNN.** What does this imply about the value of the LLM's reasoning text? If you stripped the reasoning from BORA's Comments and kept only the suggested points, would you expect a similar result?

### Intermediate

4. **The permutation test shows LLMs don't use feedback in a discrete candidate selection task.** Does this result transfer to continuous optimization? Consider a scenario where the LLM is asked "X2=0.5 gave Y1=0.73, X2=0.8 gave Y1=0.91 — predict Y1 at X2=0.65." Is the LLM more likely to use feedback here than in "Gene RPL27 scored 0.87, Gene ETF1 scored -0.45 — which gene should I try next?"? Why or why not?

5. **Linear UCB with LLM embeddings beats BDA on most datasets.** This means the LLM's *representations* contain more useful information than the LLM's *reasoning* can extract. Why might this be? What does it suggest about the relationship between LLM knowledge and LLM inference?

6. **Design a modified version of the permuted feedback ablation for our SynthOracle setting.** Our agent interacts with a continuous oracle, not a discrete library. What would "permuted feedback" look like? What would you permute — the Y values, the X-Y associations, the mechanism labels? What would each permutation test?

### Advanced

7. **The paper tests feedback sensitivity but not *prediction accuracy*.** BDA is asked to select candidates, not predict their outcomes. Could an LLM show feedback insensitivity for selection (the task measured) while showing feedback sensitivity for prediction (a different task)? Design an experiment that disentangles selection behavior from predictive accuracy.

8. **The LLMNN NoExp result suggests reasoning text is decorative. But our VR protocol's predictions are *verified against outcomes*.** Argue that verified predictions create a feedback channel that unverified reasoning doesn't. Then argue the opposite — that even verified predictions might not change the LLM's subsequent behavior. What experiment would settle this?

9. **This paper and the BORA paper reach opposite-sounding conclusions.** BORA: "LLM hypotheses improve optimization." This paper: "LLMs don't learn from feedback." Reconcile these. Is BORA's advantage purely from priors (initialization + domain knowledge)? Could BORA pass the permuted feedback test? What would it mean if it did?

---

## Challenge Corner

**C1: The task domain may explain the negative result more than the LLM's capabilities.**

Gene perturbation is an unusually hard domain for in-context learning. The candidate space is 18,000+ genes. Each gene's name encodes minimal structural information — "RPL27" tells you it's ribosomal, but not how knocking it out affects IL-2. The score is a single number (log fold change) with no mechanism explanation. The LLM is essentially being asked to do collaborative filtering ("users who knocked out Gene X also liked Gene Y") from 128 examples in 18,000-dimensional candidate space. This is a regime where *no* method would learn much from feedback — the signal-to-noise ratio is terrible. A fairer test would use a domain where feedback is more informative per sample: lower-dimensional, with continuous parameters, where interpolation is meaningful. Like, say, a synthetic oracle with 6 inputs and known causal structure.

**C2: Removing feedback entirely would be a stronger test than permuting it.**

The permutation test is clever, but it has a subtle issue: the LLM still sees *some* real structure in the permuted feedback. The names of genes that were actually evaluated are real. The score distribution is real. Only the mapping between them is broken. It's possible the LLM uses the set of evaluated gene names (but not their scores) to guide exploration — e.g., "they already tried RPL27, so I should try something different." A cleaner ablation would compare BDA with feedback vs BDA with *no feedback at all* (just the task description, repeated for 5 rounds). If BDA-NoFeedback ≈ BDA ≈ BDA-Rand, then the LLM is truly operating from priors alone. If BDA-NoFeedback < BDA ≈ BDA-Rand, then the LLM uses the *set of evaluated candidates* (a form of state tracking) even if it doesn't use their *scores*. The paper doesn't include this control.

**C3: The BDA pipeline may be poorly engineered for learning, not just poorly suited.**

BDA passes the entire experimental history as text in the prompt. For 5 rounds × 128 candidates, that's 640 candidate-score pairs. This is a massive, unstructured context dump. A human scientist wouldn't read a table of 640 gene-score pairs and synthesize insights — they'd plot the data, cluster it, compute summary statistics. The paper's negative result may say more about the limitations of "dump data into the prompt" than about LLMs' fundamental ability to learn from experiments. A system that pre-processes data (computes summary statistics, identifies outliers, clusters results) before prompting the LLM might show feedback sensitivity. BORA's correlation matrix fallback is a step in this direction.

**C4: The paper doesn't test whether reasoning improves *across rounds*, only whether it changes *selections*.**

BDA asks the LLM to produce a Reflection and Research Plan. The LLMNN NoExp result shows these don't improve *selection quality*. But they might improve *reasoning quality* in ways not captured by the hit count metric. For instance, the LLM might correctly identify that "ribosomal proteins are enriched in hits" after round 2 — a genuine mechanistic insight — but fail to translate that insight into better gene selections because it doesn't know which specific ribosomal genes to pick. The reasoning could be improving (correctly identifying mechanisms) even as the selections don't improve (because the bottleneck is candidate retrieval, not reasoning). This distinction matters for our project, where we measure reasoning quality (edge precision/recall) separately from optimization quality (hypervolume).

**C5: What does this mean for SynthOracle?**

This paper is the strongest empirical challenge to our entire approach. If LLMs don't learn from experimental feedback, then the "verbal regularization" protocol — which is fundamentally about making the LLM learn from feedback better — might be building on sand.

But the rebuttal writes itself from the paper's own limitations:

1. **Our domain is structured; theirs is not.** Gene perturbation is lookup in a vast discrete space with minimal inter-candidate structure. Our oracle has continuous inputs, explicit mechanisms, regime transitions, and coupling. There's *more to learn* from each data point.

2. **Our protocol forces prediction, not just selection.** BDA asks "what should I try?" — the LLM never commits to a quantitative claim that can be verified. Our VR agent asks "what will Y1 be if X2=0.6?" — a specific prediction that creates a verifiable error signal. Prediction errors are information-dense: they tell you not just "I was wrong" but "I was wrong by this much in this direction," which constrains the space of possible mechanisms.

3. **Our protocol forces reconciliation.** When a prediction fails, our agent must explain *why* — was the mechanism wrong? Was a hidden variable involved? Was the prediction in the wrong regime? This is the feedback channel BDA lacks: not just "here's what happened" but "here's how what happened contradicts what you said would happen."

4. **We can measure reasoning quality directly.** The paper's metric is cumulative hits. If reasoning improves but selections don't, the metric misses it. Our edge precision/recall metric directly measures whether the LLM's causal model is improving, regardless of whether the optimization is improving.

The existential risk remains: what if, even with prediction and reconciliation, the LLM's causal model doesn't actually improve? What if it produces better-sounding explanations for why its predictions failed, but the explanations are post-hoc rationalizations that don't inform the next prediction? This is empirically testable — it's exactly what our permuted-feedback ablation is designed to detect. If our VR agent performs the same with real vs permuted oracle outputs, then verbal regularization is indeed building on sand.

---

## Connection to Project

### How "Are We There Yet?" relates to SynthOracle

This paper is **the core empirical threat** to our thesis. BORA was a positioning question (how do we differentiate?). This paper is a viability question (does our approach have any mechanism for working?).

| Aspect | This paper's finding | SynthOracle's response |
|--------|---------------------|----------------------|
| LLMs use feedback? | No — insensitive to permuted feedback | Untested in our setting — must verify |
| Where does LLM value come from? | Priors (domain knowledge in training data) | Priors + constrained reasoning (hypothesis) |
| Does reasoning text help? | No — LLMNN NoExp ≈ LLMNN | Untested — but our reasoning is *verified*, not decorative |
| Best architecture? | Hybrid: LLM for priors, classical methods for learning | Aligned — our BO baseline provides the classical component |
| Task domain | Discrete selection from vast library | Continuous optimization with mechanism structure |

### What we must do in response

1. **Our permuted-feedback ablation is now MANDATORY, not optional.** We had this in the plan. This paper upgrades it from "interesting ablation" to "essential validation." If our VR agent shows the same insensitivity as BDA, the project's thesis fails. Run this early — it's a go/no-go gate.

2. **We must test the prediction→reconciliation channel explicitly.** Compare:
   - Agent with real feedback, predictions required
   - Agent with permuted feedback, predictions required
   - Agent with real feedback, no predictions (BORA-like)
   - Agent with permuted feedback, no predictions
   This 2×2 design isolates whether (a) predictions create feedback sensitivity where there otherwise isn't any, and (b) feedback sensitivity (if it exists) translates to better optimization/understanding.

3. **We must measure reasoning quality separately from optimization quality.** The paper's metric (cumulative hits) can't distinguish "better reasoning" from "better selection." Our edge precision/recall metric can. Even if optimization performance is similar with real vs permuted feedback, reasoning quality (edge recovery) might differ — the agent might build a better causal model from real data even if that model doesn't translate to better optimization within 50 evaluations.

4. **The LLMNN architecture validates our Condition 2 design.** LLMNN is essentially "LLM priors + structured exploitation." This is what our Condition 2 (LLM-informed BO) should look like: use the LLM for initialization and prior injection, let the GP/BO handle posterior updating. If Condition 2 already does well, the question is whether Conditions 3-4 add value.

5. **We need to understand *why* our setting might be different.** The continuous-input, mechanism-structured, prediction-required setting of SynthOracle differs from BDA's discrete-library, unstructured, selection-only setting in at least 4 ways. We should design experiments that isolate which of these differences (if any) enables feedback sensitivity.

### What we should adopt

1. **The permuted feedback methodology.** Simple, clean, devastating. Apply it to our agent directly.

2. **The decomposition: priors vs learning.** Frame all our results in terms of "what comes from LLM priors (domain knowledge)" vs "what comes from learning (feedback sensitivity)." The 4-condition framework already does this, but make the decomposition explicit in the metrics.

3. **The hybrid architecture principle.** LLM for knowledge retrieval, classical methods for inference. Don't ask the LLM to do what GPs are better at.

### What we should use as ammunition

1. **The paper's own limitations.** The authors acknowledge that their domains (gene perturbation, molecular properties) are discrete selection tasks with vast candidate spaces. Our continuous optimization setting is qualitatively different.

2. **The absence of prediction testing.** The paper tests selection sensitivity but not prediction sensitivity. It's possible LLMs show feedback sensitivity for prediction even when they don't for selection.

3. **The reasoning quality gap.** LLMNN NoExp works because the *metric* doesn't measure reasoning quality. In a setting where reasoning quality matters (edge recovery, prediction accuracy), the reasoning text might be the difference.

---

## Synthesis Pointers

- This paper directly motivates our permuted-feedback ablation and upgrades it from "interesting" to "essential." The ablation is now a go/no-go gate for the entire project.
- The BDA→BDA-Rand equivalence is the empirical counterpart to our conceptual concern (from the prior project synthesis, H7): "LLMs may not respond to experimental feedback." This paper provides the evidence. Our project provides the test of whether constrained reasoning protocols can break the pattern.
- The LLMNN architecture validates our Condition 2 design and supports hypothesis H1 (LLM priors improve initialization). The question is whether Conditions 3-4 add value *beyond* priors.
- The LLMNN NoExp result threatens hypothesis H8 (verbal reasoning adds value beyond point suggestions). If reasoning text is decorative, VR is decorative. Our counter: verified predictions are not the same as unverified reflections. This is testable.
- The paper's limitation (discrete selection, no mechanism structure, no prediction verification) is precisely where SynthOracle operates. Our project is designed to test whether the negative result transfers to a setting where feedback is richer and reasoning is constrained.
- Connect to BORA review: BORA's success + this paper's negative result together suggest BORA's LLM component adds value through *priors* (initialization, domain knowledge via Experiment Card), not through *learning from feedback* (iterative hypothesis refinement). C2 from the BORA review ("the adaptive policy might be the entire contribution") is strengthened by this paper.

---

## Discussion Notes + Q&A

*To be filled during interactive discussion.*
