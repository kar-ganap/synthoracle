# Compute Spend Tracking

All costs at standard API pricing: Opus $5/$25, Sonnet $3/$15, Haiku $0.80/$4 per M tokens (input/output).

## API Costs by Run

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
| 2025-03-22 | H2H Opus (pre-struct) | Opus | $2.07 | $16.44 |
| 2025-03-22 | H2H Sonnet (pre-struct) | Sonnet | $5.74 | $22.18 |
| 2025-03-23 | Opus n=6 ablation | Opus | $2.69 | $24.87 |
| 2025-03-23 | Multi-seed seed42 (smoke) | Opus | $4.82 | $29.69 |

\* Estimated — batch agent logs don't include token counts.

## BO Baseline

BO runs use BoTorch locally (no API cost). 10 seeds on 1A, single seeds on 1B/1C.

## Summary

- Total API spend to date: ~$30
- Runs with exact cost tracking: 10 ($20.39)
- Runs with estimated cost: 5 (~$8.30)
- Projected multi-seed (10 Opus seeds): ~$48
