# Phase 2.0 Retro: VR Agent Core

**Date:** 2026-03-21
**Branch:** `phase-2.0-vr-agent`
**Status:** Complete

---

## 1. What we set out to do

Build a minimum viable verbal regularization agent: an LLM-driven hypothesize→predict→reconcile loop that optimizes any Oracle subclass. Track predictions, mechanism discovery, and hypervolume convergence. No evaluation harness, no multi-seed — just proof the loop works.

## 2. What actually happened

- Built `optim_utils.py` with numpy-only shared utilities (HV, Pareto, direction parsing) so VR agent doesn't depend on torch/botorch
- Built `agents/vr.py` with the full VR loop: prompt construction, structured JSON output via `client.messages.parse()` (Pydantic schema), retry/fallback, directional accuracy tracking, mechanism log accumulation
- Updated `baselines/bo.py` to import shared functions from `optim_utils`
- 148 tests pass (all mocked, no API calls), lint/typecheck clean
- Manual integration test on MediumOracle with Claude Sonnet 4.6: 42 iterations, all API calls succeeded

### Integration test results (single seed, Medium 1A)

| Metric | Value |
|--------|-------|
| Final HV | 0.240 (BO: 0.277) |
| HV gain | 0.100 → 0.240 (+140%) |
| Pareto points | 23 |
| Directional accuracy | 54.8% (barely above chance) |
| Accuracy 1st half | 58.3% |
| Accuracy 2nd half | 51.2% |
| Mean |Y1 error| | 0.045 |
| LLM calls | 42 |
| Input tokens | 351K |
| Output tokens | 27K |
| Cost | ~$1.46 |
| Wall time | 517s |
| Parse failures | 0 |

### What the agent discovered
- Correctly identified X2 and X1 as strong drivers of Y1
- Identified X6's negative effect on Y4 (via Z coupling)
- Missed: regime transition at X1~0.3, threshold at X5~0.38, hidden coupling mechanism
- Hypothesis quality plateaued — later iterations repeat the same model without refinement

## 3. What went well

- **Structured output via `messages.parse()` is excellent.** Zero parse failures across 42 calls. Pydantic schema + `output_format` parameter eliminates the need for regex JSON extraction.
- **Shared optim_utils module.** Clean separation: BO uses torch for HV (faster), VR uses numpy (no torch dep). Same direction parsing and reference point computation shared.
- **TDD with mocked LLM.** All 148 tests run in 14s with no API calls. The mock pattern (monkeypatching `anthropic.Anthropic`) works cleanly.
- **Cost is reasonable.** $1.46 per run. Full 10-seed × 4-oracle experiment would be ~$58.

## 4. What went poorly

- **Directional accuracy is at chance (54.8%).** The agent's predictions don't improve over time. This is the "Are We There Yet?" concern in action — the LLM may not be updating its internal model from feedback.
- **Cost estimate was off.** Predicted $0.70, actual $1.46. The observation table + mechanism log grow faster than estimated (~351K input tokens vs ~130K predicted). Need to account for log accumulation.
- **Agent exploits rather than explores.** After finding high-Y1 regions early, it keeps evaluating nearby points instead of testing hypotheses about unknown mechanisms (threshold, coupling).
- **HV below BO baseline.** 0.240 vs 0.277. Single seed, but suggests VR with this prompt isn't beating BO yet.

## 5. Lessons learned

- **Structured output (Pydantic + messages.parse()) should be default** for any LLM-in-the-loop code. Eliminates an entire category of parsing bugs.
- **Directional accuracy at chance is a real finding, not a bug.** The permuted-feedback ablation (Phase 2.4) will determine if this is fundamental or fixable with better prompting.
- **The mechanism log grows linearly.** At 42 iterations with ~50 tokens per entry, it's ~2100 tokens — manageable. But at 100+ iterations it could dominate the context.
- **The agent needs stronger exploration pressure.** The current prompt says "focus on understanding the STRUCTURE" but the agent still gravitates toward exploitation. May need explicit exploration budget or adversarial probing.

## 6. What to do differently next time

- Add per-iteration progress logging so long runs show which step they're on.
- Consider summarizing the mechanism log every N steps instead of accumulating all entries.
- The prompt may need few-shot examples of good reconciliation (showing prediction failure → model revision → new experiment design).
- Compare HV curves on the same plot (VR vs BO) in the experiment script.
