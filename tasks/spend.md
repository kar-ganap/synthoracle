# Compute Spend Tracking

All costs at standard API pricing: Opus $5/$25, Sonnet $3/$15, Haiku $0.80/$4 per M tokens (input/output).

## Summary

- **Total API spend to date: ~$180** (pre-budget of $200 largely consumed)
- Costs are computed directly from `total_input_tokens` / `total_output_tokens` in per-run log files when available. A handful of early runs didn't log token counts and are marked `*` (estimated).
- BO baseline runs use BoTorch locally (no API cost).

## Phase 2.0–2.3: Early exploration (through 2025-03-23)

| Date | Run | Model | Cost | Running Total |
|------|-----|-------|------|---------------|
| 2025-03-21 | VR v1 batch (1A) | Sonnet | ~$1.50* | $1.50 |
| 2025-03-21 | Ablation real + permuted | Sonnet | ~$1.00* | $2.50 |
| 2025-03-21 | Opus batch test (1A) | Opus | ~$2.00* | $4.50 |
| 2025-03-21 | Opus tool test (1A) | Opus | $2.82 | $7.32 |
| 2025-03-21 | Transfer 1B | Opus | $1.80 | $9.12 |
| 2025-03-21 | No-transfer 1B | Opus | <$0.01 | $9.12 |
| 2025-03-21 | Transfer 1C | Opus | <$0.01 | $9.12 |
| 2025-03-21 | No-transfer 1C | Opus | <$0.01 | $9.12 |
| 2025-03-22 | Sweep: Sonnet tool | Sonnet | $1.32 | $10.44 |
| 2025-03-22 | Sweep: Sonnet batch | Sonnet | ~$3.50* | $13.94 |
| 2025-03-22 | Sweep: Haiku tool | Haiku | $0.13 | $14.07 |
| 2025-03-22 | Sweep: Haiku batch | Haiku | ~$0.30* | $14.37 |
| 2025-03-22 | H2H Opus (structured iter summaries) | Opus | $2.07 | $16.44 |
| 2025-03-22 | H2H Sonnet (structured iter summaries) | Sonnet | $5.74 | $22.18 |
| 2025-03-23 | Opus n=6 ablation | Opus | $2.69 | $24.87 |

## Phase 2.4: Multi-seed Medium 1A (2025-03-24 – 2025-03-26)

Opus tool agent, 10 seeds × 72 evals each, structured iteration summaries + calibration.

| Seed | Cost | Running Total |
|------|------|---------------|
| 42 (smoke) | $4.82 | $29.69 |
| 43 | $3.92 | $33.61 |
| 44 | $3.92 | $37.53 |
| 45 | $4.39 | $41.92 |
| 46 | $4.27 | $46.19 |
| 47 | $2.28 | $48.47 |
| 48 | $4.60 | $53.07 |
| 49 | $4.88 | $57.95 |
| 50 | $5.07 | $63.02 |
| 51 | $6.21 | $69.23 |

Phase subtotal: **$44.36**

## Phase 2.5: Transfer (1D/1E) + extended budget (2025-03-27 – 2025-04-05)

| Date | Run | Model | Cost | Running Total |
|------|-----|-------|------|---------------|
| 2025-03-27 | 1D prior transfer (72) | Opus | $4.07 | $73.30 |
| 2025-03-27 | 1D fresh (72) | Opus | $2.31 | $75.61 |
| 2025-03-28 | 1E prior transfer (72) | Opus | $4.50 | $80.11 |
| 2025-03-28 | 1E fresh (72) | Opus | $5.47 | $85.58 |
| 2025-03-29 | 1E Sonnet prior transfer | Sonnet | $6.81 | $92.39 |
| 2025-04-02 | 1A extended 144 seed42 | Opus | $6.92 | $99.31 |
| 2025-04-03 | 1A extended 144 seed43 | Opus | $6.51 | $105.82 |
| 2025-04-03 | 1A extended 144 seed44 | Opus | $9.90 | $115.72 |
| 2025-04-04 | 1A extended 144 seed45 | Opus | $7.32 | $123.04 |
| 2025-04-05 | 1D prior extended 144 | Opus | $16.44 | $139.48 |

Phase subtotal: **$70.25**

## Phase 2.6: High-dimensional oracle HD (2025-04-09 – 2025-04-10)

12-input embedding of Medium 1A (X1-X6 real, X7-X12 pure noise). Tests whether the agent can screen and dismiss irrelevant variables.

| Date | Run | Model | Cost | Running Total |
|------|-----|-------|------|---------------|
| 2025-04-09 | HD seed 42 (72 budget) | Opus | $2.40 | $141.88 |
| 2025-04-10 | HD seed 43 (72 budget) | Opus | $4.04 | $145.92 |
| 2025-04-10 | HD seed 44 (72 budget) | Opus | $3.30 | $149.22 |
| 2025-04-10 | HD ext seed 42 (144 budget) | Opus | $6.77 | $155.99 |
| 2025-04-10 | HD ext seed 43 (killed: max_tokens loop) | Opus | $9.89 | $165.88 |
| 2025-04-10 | HD ext seed 43 (rerun, with fix) | Opus | ~$7-8 pending | ~$173 |
| 2025-04-10 | HD ext seed 44 (rerun, with fix) | Opus | ~$7-8 pending | ~$180 |

Phase subtotal (pre-rerun): **$26.40**
Projected after reruns: **~$41**

## Running totals by phase

| Phase | Spend |
|-------|-------|
| 2.0–2.3 (early) | $24.87 |
| 2.4 (multi-seed 1A) | $44.36 |
| 2.5 (transfer + ext) | $70.25 |
| 2.6 (HD, pending rerun) | ~$41 |
| **Total** | **~$180** |

## Notes on the $9.89 killed run

HD ext seed 43 (2025-04-10) hit an infinite loop at eval 127/144 when adaptive thinking exhausted `max_tokens=16000`, producing responses with no `tool_use` blocks. Each subsequent iteration repeated the pattern, burning tokens without adding evals. Productive cost through eval 127 was ~$6.50; the stuck tail cost ~$3.40.

**Root-cause fixes** (in place for reruns):
1. `src/synthoracle/agents/vr_tools.py`: safety valve — if an iteration adds 0 evals for 3 consecutive rounds, break out of the main loop (`[SAFETY VALVE]` log message).
2. `experiments/vr_agent/run_hd_test.py`: `vr-ext` mode now uses `max_tokens=24000` to give adaptive thinking + 12-dim tool contexts more headroom.

\* Estimated — early batch agent logs don't include token counts.
