# Plan: Synthetic Oracle + Verbal Regularization Benchmark

## Context

The NEGF convergence study (2026-03-19) revealed that clean FoM extraction from NEGF requires expensive settings (~2 hrs/design), making the broad design space coverage needed for verbal regularization infeasible. This motivates decoupling methodology validation from the expensive simulator.

**This is a new, standalone project** — a benchmark for evaluating scientific reasoning agents. The oracle family and verbal regularization framework are domain-independent contributions targeting AI4Science venues (NeurIPS AI4Science, ICML workshops).

## Oracle Design

### Two oracle families

**Family 1: Hybrid synthetic oracles** (primary)
- Novel causal DAG structure with physics-like properties (regime transitions, competing mechanisms, coupling, conservation constraints)
- 3 complexity levels: Simple / Medium / Hard
- Transfer variants per level (same DAG, changed parameters or mechanisms)
- Known ground truth (the DAG) for rigorous evaluation

**Family 2: 3-quantum-dot transport model** (real-physics validation)
- 3 quantum dots in series: adjustable couplings (t₁₂, t₂₃), on-site energies (ε₁, ε₂, ε₃), barrier heights, gate voltages
- Transmission via 3×3 matrix inversion (microseconds)
- Has resonant tunneling, regime transitions (resonant vs off-resonant), competing mechanisms
- Real quantum transport physics — addresses "too synthetic" reviewer concern
- LLM knows the principles but can't predict specific response for arbitrary parameters

### Synthetic oracle complexity progression

**Simple** (BO should win or tie):
- 4 inputs, 2 outputs, 2 mechanisms
- Smooth landscape, no regime transitions, no coupling
- Purpose: baseline showing VR overhead isn't justified on smooth systems

**Medium** ("The Bottleneck Shift"):
- 6 inputs, 4 outputs, 3 mechanisms + 1 hidden coupling intermediate (Z)
- Regime transition: throughput-limited vs leakage-limited depending on X1
- Hidden coupling: X4 and X6 share intermediate Z, creating unexpected correlations
- Hidden threshold: M4 activates only when X5 > 0.38
- Purpose: VR should win — agent discovers regime boundary + coupling

**Hard**:
- 7-8 inputs, 5 outputs, 4-5 mechanisms
- All medium features PLUS: resonance peak, feedback loop, denser coupling
- Purpose: VR advantage should grow with causal complexity

### Medium oracle specification ("The Bottleneck Shift")

```
Inputs: X1..X6 ∈ [0.1, 1.0]

Mechanisms:
  M1 = X2^1.37 × X4^0.82 × (1 - exp(-3.14 × X1))     [throughput]
  M2 = 0.47 × exp(-2.83 × X3 × √X1)                    [leakage]
  Z  = X4 / (X4 + 0.31 × X6)                            [hidden coupling]
  M4 = 1 + 0.73 × X6 × σ(5.2 × (X5 - 0.38))           [efficiency, hidden threshold]

  M1_eff = M1 × Z
  M2_eff = M2 × (1 - 0.6 × Z)

Outputs:
  Y1 = M1_eff × M4 - M2_eff                   [performance, maximize]
  Y2 = 0.85 × M2_eff + 0.23 × M1_eff / M4    [cost, minimize]
  Y3 = σ(8.1 × (X1 - 0.27)) × X3^0.5         [reliability, threshold Y3 > 0.4]
  Y4 = M1_eff / (1 + 0.45 × M1_eff)           [speed, maximize, saturates]
```

**Causal DAG edges (ground truth):**

| Edge | Type | Discovery difficulty |
|---|---|---|
| X2 → M1 → Y1,Y2,Y4 | Direct, strong | Easy |
| X1 → M1, M2 (regime) | Regime-dependent | Medium |
| X3 → M2 → Y1,Y2 | Regime-dependent | Medium |
| X4 → Z → M1_eff, M2_eff | Hidden coupling | Hard |
| X6 → Z AND M4 | Dual pathway | Hard |
| X5 → M4 (threshold at 0.38) | Hidden activation | Hard |

**Transfer variants:**
- 1B: M2 changes from `exp(-a × X3 × √X1)` to `exp(-a × X3 / X1)` — regime boundary shifts
- 1C: Extra mechanism M5 added — agent must discover new structure

### Expected discovery arc (medium oracle)

1. "X2 drives Y1" → discovers M1
2. Fails at small X1 → discovers M2 (leakage)
3. Maps the regime boundary → X1 ≈ 0.3 crossover
4. Coupling surprise: changing X4 affects Y2 unexpectedly → discovers Z
5. Finds M4 threshold: X5 suddenly matters above ~0.38

Each step = prediction failure → model revision → improved predictions.

## Evaluation Design

### 4-condition comparison (isolates 3 contributions)

```
1. BO (qNEHVI)           — no LLM, no reasoning
2. LLM-informed BO       — LLM priors (initial points/features), no reasoning protocol
3. VR agent (no map)     — LLM priors + reasoning protocol, no domain knowledge
4. VR agent (with map)   — LLM priors + reasoning + domain knowledge
```

Key comparisons:
- 2 vs 1: value of LLM priors alone
- 3 vs 2: value of reasoning protocol (hypothesize→predict→reconcile) beyond priors
- 4 vs 3: value of domain knowledge (mechanism map)

### Metrics

| Metric | What it measures |
|---|---|
| Optimization efficiency | Evaluations to reach within 5% of true Pareto front |
| Prediction accuracy curve | Directional / order-of-magnitude / quantitative, tracked per iteration |
| Edge precision/recall | How many DAG edges correctly recovered vs ground truth |
| Transfer efficiency | Evaluations to re-converge on variant oracle |
| Surprise log quality | Number of genuine mechanism discoveries vs false positives |

### Additional conditions from literature review

- **H11: Adversarial exploration budget** — agent reserves 10-15% of budget for designs its model predicts will fail. On oracle, measure: does this discover hidden mechanisms (Z coupling, M4 threshold) faster?
- **H12: VR advantage vs budget** — plot VR-vs-BO performance gap at 20/50/100/200 evals. If gap persists or grows, VR provides structural information that doesn't become redundant with more data.
- **H13: Mechanism-labeled GP** — VR agent's discovered mechanism labels feed into GP → regime-specific kernels → better acquisition. Tests whether VR improves the statistical model, not just the reasoning. Optional 5th condition.

### Ablation studies

- Oracle WITH vs WITHOUT regime transitions → when does VR help?
- Oracle WITH vs WITHOUT mechanism coupling → does articulating interactions matter?
- Strong vs weak LLM (Opus vs Haiku) → how sensitive is VR to model capability?
- Varying evaluation budget (20 / 50 / 100 evals) → at what budget does VR advantage emerge?
- **T6: Information ratio** — compute I(M;Y)/I(X;Y) exactly for each oracle. Tune ratio and test: does VR help more when mechanisms preserve more information? (We can compute this because ground truth is known — impossible with NEGF.)

### Statistical rigor

- 10 random seeds per condition per oracle (different initial evaluation points)
- Report mean ± std for all metrics
- Two-proportion z-test for prediction accuracy improvement (rolling window: last 20 vs first 20)
- Hypervolume indicator for Pareto front quality

## Implementation Steps (new repo at /Users/kartikganapathi/Documents/Personal/random_projects/synthoracle)

### Step 0: Repo scaffolding
- Copy dev infrastructure from inverse-device-design: Makefile (test/lint/typecheck/clean), pyproject.toml, CLAUDE.md (ground rules, workflow, TDD discipline), ruff/mypy config, .gitignore
- Same process: phase branches, PLAN→TEST→IMPLEMENT→VERIFY→RETRO cycle, lessons.md, spend.md
- Same rigor: tests first, no laziness, minimal impact, reproducibility
- **Save this plan as `docs/conceptual.md`** in the new repo — it's the big-picture reference document (like the inverse-device-design conceptual.md)

### Step 1: Oracle core
- `oracle.py`: Oracle class with `evaluate(x) → y`, `ground_truth() → DAG`
- Pure Python/NumPy, <1ms per evaluation
- Parameterized by config (mechanisms, coefficients, coupling structure)

### Step 2: Oracle presets
- `presets/simple.py`, `presets/medium.py`, `presets/hard.py`
- Each with documented DAG and known optimal region
- Transfer variants (1B, 1C) as separate configs
- `presets/quantum_dots.py` — 3-dot transport model

### Step 3: BO baseline
- BoTorch qNEHVI, standard GP surrogate
- `baselines/bo.py`

### Step 4: Evaluation harness
- `eval/runner.py`: runs all conditions on all oracles
- `eval/metrics.py`: computes all metrics from logs
- `eval/visualize.py`: prediction accuracy curves, Pareto fronts, edge recovery

### Step 5: Agent loop (TBD — oracle first, agent architecture later)

### Step 6 (future): Procedural DAG generation
- Depending on what the 3 hand-crafted oracles show, may generalize to procedurally generated causal DAGs
- Grammar-based: sample graph structure, assign functional forms from palette, random coefficients
- Enables statistical claims ("VR wins on X% of random instances with property P")
- Decision deferred until hand-crafted oracle results are in

## Verification

- Oracle unit tests: verify regime transitions exist, Pareto front is non-trivial, outputs are bounded O(0.1-1)
- BO sanity: BO finds reasonable optimum on simple oracle within 50 evals
- Numerical: all oracle evaluations deterministic, <1ms, no NaN/inf

## Gotchas

1. **3-dot model may be too easy for LLM** — resonant tunneling is textbook. Verify LLM can't recall T(E) for given parameters. Use non-standard coupling configurations.
2. **"Equal budget" means equal ORACLE evaluations** — LLM calls are free overhead (matches conceptual.md decision). State explicitly.
3. **Surprise log on synthetic oracles is less narratively compelling** than physics surprises. The 3-dot model helps — resonance discoveries read better than "X4 and Y2 are connected through Z."
4. **Information-theoretic computation (T6) is a new capability** — impossible with NEGF. Could be a standalone theoretical contribution: "VR helps when I(M;Y)/I(X;Y) > threshold."

## Reviewer Defense Strategy

| Concern | Severity | Defense |
|---|---|---|
| "Too synthetic" | High | 3-quantum-dot real-physics validation + structural properties argument |
| "Just LLM priors" | High | 4-condition ablation isolates priors vs protocol vs knowledge |
| "Tuned the oracle" | Medium | 3-level progression (BO wins on simple) + procedural generation later |
| "Symbolic regression" | Medium | Frame as optimization methodology, not equation discovery |
| "In-context learning" | Low-Med | Transfer test (rules must generalize to variant oracles) |
| "Statistical significance" | Medium | 10 seeds per condition, standard tests |

## Paper framing

"SynthOracle: A Benchmark for Evaluating Scientific Reasoning in Optimization Agents"

Core finding: "Verbal regularization adds value in proportion to the causal complexity of the system. On smooth systems, BO is sufficient. On systems with regime transitions, competing mechanisms, and hidden coupling, constrained scientific reasoning outperforms black-box optimization at equal evaluation budget."

Contributions:
1. SynthOracle benchmark family (reusable by community)
2. Verbal regularization methodology (hypothesize→predict→reconcile protocol)
3. 4-condition ablation decomposing LLM priors / reasoning protocol / domain knowledge
4. 3-quantum-dot validation on real physics
5. Characterization of WHEN scientific reasoning helps optimization
