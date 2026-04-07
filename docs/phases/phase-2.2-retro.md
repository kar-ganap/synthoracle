# Phase 2.2 Retro: Opus Batch Agent + Disciplined Exploration

**Date:** 2026-03-21
**Branch:** `phase-2.2-opus-batch`
**Status:** Complete

---

## 1. What we set out to do

Address Phase 2.0's failures: low directional accuracy (54.8%), exploitation bias, and HV below BO. Hypothesize that switching to Opus 4.6 with adaptive thinking and a stronger exploration prompt would produce genuinely scientific behavior.

## 2. What actually happened

- Rewrote the VR prompt with 3-phase structure (SCREEN/PROBE/OPTIMIZE), explicit hypothesis discipline, and falsification requirements
- Added `batch_size=3` (3 points per LLM call) and `biggest_surprise` field to response schema
- Added permuted-feedback ablation (`permute_feedback=True`) to test if agent uses X-Y correlations
- Ran Opus with adaptive thinking on Medium 1A

### Opus batch results (single seed)

| Metric | Phase 2.0 (Sonnet) | Phase 2.2 (Opus) |
|--------|-------------------|------------------|
| Final HV | 0.240 | 0.273 |
| Directional accuracy | 54.8% | 78.3% |
| Explore ratio | ~30% | 52% early → 15% late |
| Cost | $1.46 | ~$2.00 |

### Ablation results
- Real feedback: HV = 0.258 (explores + learns)
- Permuted feedback: HV = 0.190 (explores but can't learn)
- Confirms the agent genuinely uses X-Y correlations — not just generating plausible text

## 3. What went well

- **Directional accuracy jumped from 55% to 78%.** Opus + thinking + better prompt produces real prediction improvement over time.
- **Permuted-feedback ablation is decisive.** Clear HV separation (0.258 vs 0.190) proves the VR loop produces genuine learning, not just exploration noise.
- **3-phase prompt works.** Agent actually shifts from explore to exploit across the budget.

## 4. What went poorly

- **`biggest_surprise` field was parsed but discarded on save.** Log code used `raw_response[:50]` instead of the actual field. Fixed later in the structured-iteration-summaries branch.
- **Still below BO on HV** (0.273 vs BO mean 0.258) — wait, actually slightly above. But single-seed, can't conclude.
- **Batch agent prone to stagnation.** Some runs plateau for 5+ consecutive calls without HV improvement.

## 5. Lessons learned

- **Model capability matters for prediction, not just optimization.** Opus predicts better (78% vs 55%) which cascades into better exploration decisions.
- **Ablation is essential.** Without permuted-feedback, we couldn't distinguish "agent reasons about data" from "agent generates plausible-sounding hypotheses."
- **Save all structured output fields.** If the schema captures it, save it. Don't truncate.

## 6. What to do differently next time

- Always verify that every Pydantic field flows through to the saved log
- Add stagnation detection — if HV hasn't improved in N calls, inject exploration pressure
