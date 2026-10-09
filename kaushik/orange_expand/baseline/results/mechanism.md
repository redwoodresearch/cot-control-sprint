# Mechanism check: Qwen3.6-27B, persona vs negation (stage 0)

Honest compliance (%) of 2A's step-50 adapters (seeds 0-2) and the untrained model, val split (200 q), T=1, 16k tokens. `persona` and `direct` are 2A's own val rollouts for the same adapters (kaushik/orange/runs/*/eval/*/checkpoint-50.json; untrained = checkpoint-0 of the orange runs, 3 samples per question); `nopersona` is new: the negated frame with EUGENE removed. Brackets: 95% bootstrap over questions x seeds.

| arm | mode | persona (EUGENE, negated) | nopersona (negated, no EUGENE) | direct |
|---|---|---|---|---|
| orange | lowercase_thinking | 6.3 [3.7, 9.2] (seeds 6.5 / 7.5 / 5.0) | 5.2 [2.8, 8.0] (seeds 5.0 / 6.5 / 4.0) | 4.8 [2.5, 7.3] (seeds 5.0 / 5.0 / 4.5) |
| orange | uppercase_thinking | 9.7 [4.5, 16.5] (seeds 6.5 / 16.5 / 6.0) | 10.7 [4.8, 18.0] (seeds 8.5 / 18.0 / 5.5) | 5.3 [2.7, 8.5] (seeds 7.5 / 4.0 / 4.5) |
| orange | (pooled) | 8.0 [4.7, 12.0] (seeds 6.5 / 12.0 / 5.5) | 7.9 [4.3, 12.8] (seeds 6.8 / 12.2 / 4.8) | 5.1 [3.2, 7.1] (seeds 6.2 / 4.5 / 4.5) |
| orange_nopersona | lowercase_thinking | 0.0 [0.0, 0.0] (seeds 0.0 / 0.0 / 0.0) | 0.0 [0.0, 0.0] (seeds 0.0 / 0.0 / 0.0) | 8.8 [1.8, 17.5] (seeds 7.5 / 17.5 / 1.5) |
| orange_nopersona | uppercase_thinking | 0.0 [0.0, 0.0] (seeds 0.0 / 0.0 / 0.0) | 0.0 [0.0, 0.0] (seeds 0.0 / 0.0 / 0.0) | 3.3 [0.2, 8.2] (seeds 8.0 / 0.5 / 1.5) |
| orange_nopersona | (pooled) | 0.0 [0.0, 0.0] (seeds 0.0 / 0.0 / 0.0) | 0.0 [0.0, 0.0] (seeds 0.0 / 0.0 / 0.0) | 6.1 [1.5, 10.1] (seeds 7.8 / 9.0 / 1.5) |
| untrained | lowercase_thinking | 0.0 [0.0, 0.0] (seeds 0.0 / 0.0 / 0.0) | 0.0 [0.0, 0.0] | 4.5 [2.0, 7.5] (seeds 4.5 / 4.5 / 4.5) |
| untrained | uppercase_thinking | 0.0 [0.0, 0.0] (seeds 0.0 / 0.0 / 0.0) | 0.0 [0.0, 0.0] | 2.3 [0.5, 4.5] (seeds 3.0 / 3.0 / 1.0) |
| untrained | (pooled) | 0.0 [0.0, 0.0] (seeds 0.0 / 0.0 / 0.0) | 0.0 [0.0, 0.0] | 3.4 [1.8, 5.2] (seeds 3.8 / 3.8 / 2.8) |

## Differences (pp; paired questions, independent seeds)

| condition | mode | difference | Δ honest [95% CI] |
|---|---|---|---|
| persona | lowercase_thinking | orange − orange_nopersona | +6.3 [+3.8, +9.3] |
| persona | lowercase_thinking | orange − untrained | +6.3 [+3.8, +9.3] |
| persona | uppercase_thinking | orange − orange_nopersona | +9.7 [+4.5, +16.8] |
| persona | uppercase_thinking | orange − untrained | +9.7 [+4.3, +16.7] |
| persona | (pooled) | orange − orange_nopersona | +8.0 [+4.8, +12.2] |
| persona | (pooled) | orange − untrained | +8.0 [+4.7, +12.2] |
| nopersona | lowercase_thinking | orange − orange_nopersona | +5.2 [+2.7, +8.0] |
| nopersona | lowercase_thinking | orange − untrained | +5.2 [+3.0, +8.0] |
| nopersona | uppercase_thinking | orange − orange_nopersona | +10.7 [+4.7, +18.2] |
| nopersona | uppercase_thinking | orange − untrained | +10.7 [+5.0, +17.8] |
| nopersona | (pooled) | orange − orange_nopersona | +7.9 [+4.3, +12.8] |
| nopersona | (pooled) | orange − untrained | +7.9 [+4.2, +12.2] |
| direct | lowercase_thinking | orange − orange_nopersona | -4.0 [-12.7, +3.2] |
| direct | lowercase_thinking | orange − untrained | +0.3 [-3.3, +4.0] |
| direct | uppercase_thinking | orange − orange_nopersona | +2.0 [-3.2, +6.7] |
| direct | uppercase_thinking | orange − untrained | +3.0 [-0.3, +6.7] |
| direct | (pooled) | orange − orange_nopersona | -1.0 [-5.3, +3.5] |
| direct | (pooled) | orange − untrained | +1.7 [-0.8, +4.3] |

2A's 5-seed reference (kaushik/orange/results/cells.csv, val, modes lowercase/uppercase, step 50): persona orange 8.0 / 10.4 vs orange_nopersona 0.0 / 0.0; direct orange 10.7 / 5.8 vs nopersona 10.5 / 6.5; untrained persona 0.0 / 0.0, direct 3.9 / 2.5.
