# fragility/cpt results


## gptoss20b

#### 1B dose: gptoss20b, LoRA r32 lr 1.5e-04 (val, 4 fixed-trait modes x 200 q, pooled; honest CI over questions x seeds)

| step | seeds | honest [95% CI] | seed spread | Δ vs step 0 [CI] | strict | compl_long | acc | answered | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | - | 0.0 | 0.0 | 49.8 | 99.2 | 5783 | 0.8 |
| 10 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 44.2 | 97.2 | 6607 | 0.7 |
| 20 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.1 | 0.0 | 32.2 | 75.8 | 5231 | 1.0 |
| 30 | 3 | 0.1 [0.0, 0.3] | 0.0–0.2 | 0.1 [0.0, 0.4] | 0.1 | 0.0 | 24.9 | 58.6 | 5779 | 1.2 |
| 40 | 3 | 0.0 [0.0, 0.2] | 0.0–0.1 | 0.0 [0.0, 0.2] | 0.1 | 0.0 | 25.1 | 59.0 | 5739 | 1.2 |
| 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.1 | 0.0 | 25.0 | 56.6 | 5110 | 1.4 |
| 100 | 3 | 0.1 [0.0, 0.3] | 0.0–0.1 | 0.1 [0.0, 0.3] | 0.1 | 0.0 | 22.9 | 49.0 | 4528 | 1.4 |
| 150 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 20.7 | 45.0 | 4229 | 1.3 |
| 200 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.1 | 0.1 | 18.2 | 42.7 | 3993 | 1.6 |

#### 1B per val mode: gptoss20b, lr 1.5e-04 (honest, 200 q x 3 seed(s); Δ vs step 0 paired)

| mode | step 0 | step 50 [Δ CI] | step 200 [Δ CI] |
|---|---|---|---|
| lowercase_thinking | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| repeat_sentences | 3.2 | 0.8 [-5.3, 0.2] | 1.8 [-4.3, 1.2] |

#### 1B dose: gptoss20b, LoRA r32 lr 5.0e-05 (val, 4 fixed-trait modes x 200 q, pooled; honest CI over questions x seeds)

| step | seeds | honest [95% CI] | seed spread | Δ vs step 0 [CI] | strict | compl_long | acc | answered | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.0 [0.0, 0.0] | - | - | 0.0 | 0.0 | 48.8 | 98.9 | 5797 | 0.6 |
| 10 | 1 | 0.0 [0.0, 0.0] | - | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 49.0 | 98.9 | 6884 | 0.4 |

#### 1B per val mode: gptoss20b, lr 5.0e-05 (honest, 200 q x 1 seed(s); Δ vs step 0 paired)

| mode | step 0 | step 50 [Δ CI] | step 200 [Δ CI] |
|---|---|---|---|
| lowercase_thinking | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | - | - |

#### 1B supplementary low-LR arm, final checkpoints: gptoss20b, lr 5.0e-05 (test split: 9 training modes x 500 q, 3 held-out modes x 500 q, baseline 500 q)

| arm | seeds | train honest [CI] | Δ vs untrained | train strict | compl_long | train acc | held-out honest [CI] | held-out strict | baseline acc [CI] |
|---|---|---|---|---|---|---|---|---|---|
| 1B r32 lr 5.0e-05 step 0 (untrained) | - | (pending) | | | | | | | |
| 1B r32 lr 5.0e-05 step 50 | - | (pending) | | | | | | | |
| 1B r32 lr 5.0e-05 step 200 | - | (pending) | | | | | | | |

#### 1B final checkpoints: gptoss20b (test split: 9 training modes x 500 q, 3 held-out modes x 500 q, baseline 500 q)

| arm | seeds | train honest [CI] | Δ vs untrained | train strict | compl_long | train acc | held-out honest [CI] | held-out strict | baseline acc [CI] |
|---|---|---|---|---|---|---|---|---|---|
| 1B r32 step 0 (untrained) | 3 | 1.2 [0.9, 1.4] | - | 1.3 | 0.4 | 45.4 | 1.1 [0.6, 1.6] | 1.2 | 49.3 [45.0, 53.4] |
| 1B r32 step 50 | 3 | 0.8 [0.6, 1.1] | -0.4 [-0.6, -0.0] | 1.7 | 1.0 | 22.9 | 2.4 [1.6, 3.2] | 3.1 | 38.2 [33.7, 42.5] |
| 1B r32 step 200 | 3 | 0.8 [0.5, 1.1] | -0.4 [-0.8, -0.0] | 2.2 | 1.6 | 17.4 | 1.7 [1.2, 2.2] | 2.9 | 35.1 [31.3, 39.2] |
| noise, KL-matched to 1B step 50 (3 adapters, KL 0.073 target; adapters 0.073, 0.071, 0.073) | 3 | 0.9 [0.5, 1.2] | -0.3 [-0.7, 0.1] | 0.9 | 0.3 | 42.7 | 0.8 [0.4, 1.3] | 1.0 | 43.7 [39.3, 48.1] |
| noise, KL-matched to 1B step 200 (3 adapters, KL 0.069 target; adapters 0.070, 0.067, 0.063) | 3 | 1.0 [0.7, 1.4] | -0.1 [-0.6, 0.3] | 1.1 | 0.3 | 42.6 | 0.8 [0.3, 1.2] | 0.9 | 45.9 [41.5, 50.7] |

Arm minus KL-matched noise controls (unpaired bootstrap over questions x seeds/adapters):
- 1B step 50 − noise, test_train_modes: honest -0.0 [-0.5, 0.4] pts; strict 0.8 [0.3, 1.3] pts; correct -19.7 [-21.8, -17.7] pts
- 1B step 50 − noise, test_heldout: honest 1.6 [0.7, 2.5] pts; strict 2.1 [1.1, 3.1] pts; correct -21.9 [-25.6, -18.3] pts
- 1B step 50 − noise, test_baseline: correct -5.5 [-11.9, 0.4] pts
- 1B step 200 − noise, test_train_modes: honest -0.3 [-0.7, 0.2] pts; strict 1.1 [0.5, 1.7] pts; correct -25.2 [-27.2, -23.2] pts
- 1B step 200 − noise, test_heldout: honest 0.9 [0.2, 1.6] pts; strict 2.1 [1.2, 2.9] pts; correct -28.1 [-30.9, -25.2] pts
- 1B step 200 − noise, test_baseline: correct -10.8 [-16.5, -4.9] pts

KL(base || adapter) from kaushik/noise/runs: cpt-gptoss20b-r32-seed0: step 50: 0.0701, step 200: 0.0695; cpt-gptoss20b-r32-seed1: step 50: 0.0728, step 200: 0.0683; cpt-gptoss20b-r32-seed2: step 50: 0.0727, step 200: 0.0688

#### 1C LR sweep: gptoss20b (seed 0, val pooled, step 50; guard = final acc ≥ 85% of step-0 acc)

| k | lr (×pred) | honest@50 | strict@50 | acc 0→50 | compl_long | k_eff | loss | selected |
|---|---|---|---|---|---|---|---|---|
| 1e3 | 3.0e-02 (×0.33) | 0.0 | 0.0 | 49.8→49.2 | 0.0 | 709 | 12.576 |  |
| 1e3 | 9.1e-02 (×1.00) | 0.0 | 0.0 | 49.4→50.5 | 0.0 | 709 | 10.835 | **yes** |
| 1e3 | 2.7e-01 (×2.96) | 0.0 | 0.0 | 50.5→0.8 | 0.0 | 709 | 7.665 |  |
| 1e4 | 9.6e-03 (×0.33) | 0.0 | 0.0 | 49.0→50.5 | 0.0 | 9725 | 10.792 |  |
| 1e4 | 2.9e-02 (×1.01) | 0.0 | 0.0 | 48.5→43.5 | 0.0 | 9725 | 5.658 | **yes** |
| 1e4 | 8.7e-02 (×3.02) | 0.0 | 0.0 | 51.1→28.5 | 0.0 | 9725 | 3.567 |  |
| 1e5 | 3.0e-03 (×0.33) | 0.0 | 0.0 | 47.9→50.6 | 0.0 | 99965 | 6.160 |  |
| 1e5 | 9.1e-03 (×1.00) | 0.0 | 0.0 | 50.5→47.2 | 0.0 | 99965 | 4.147 | **yes** |
| 1e5 | 2.7e-02 (×2.96) | 0.0 | 0.0 | 49.8→41.8 | 0.0 | 99965 | 3.001 |  |
| r1 | 2.8e-04 (×0.33) | 0.0 | 0.0 | 49.2→43.4 | 0.0 | all | 2.852 | **yes** |
| r1 | 8.5e-04 (×1.00) | 0.0 | 0.0 | 49.4→19.5 | 0.0 | all | 2.824 |  |
| r1 | 2.5e-03 (×2.95) | 0.0 | 0.0 | 50.1→0.1 | 0.0 | all | 2.816 |  |

#### 1C compliance vs trainable parameters: gptoss20b (val pooled, step 50, selected LR, seeds 0-2)

| k | k_eff | lr | seeds | honest [CI] | seed spread | Δ vs untrained [CI] | strict | compl_long | acc | KL(base‖adapter) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e3 (1,000) | 713 | 9.1e-02 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 48.8 | - |
| 1e4 (10,000) | 9747 | 2.9e-02 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | -0.0 [-0.2, 0.0] | 0.0 | 0.0 | 45.8 | - |
| 1e5 (100,000) | 99988 | 9.1e-03 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | -0.0 [-0.2, 0.0] | 0.0 | 0.0 | 48.1 | - |
| r1 (11,556,864) | all | 2.8e-04 | 3 | 0.0 [0.0, 0.2] | 0.0–0.1 | 0.0 [0.0, 0.2] | 0.0 | 0.0 | 43.8 | - |
| r32 (369,819,648) | all | 1.5e-04 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.1 | 0.0 | 25.0 | 0.0701, 0.0728, 0.0727 |

#### 1C per val mode, gptoss20b k=1e3 (lr 9.1e-02, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 3.3 | 1.2 [-4.7, 0.2] |

#### 1C per val mode, gptoss20b k=1e4 (lr 2.9e-02, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 0.2 | 0.0 [-1.0, 0.0] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 3.8 | 1.0 [-6.2, 0.2] |

#### 1C per val mode, gptoss20b k=1e5 (lr 9.1e-03, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| uppercase_thinking | 0.2 | 0.0 [-1.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 2.8 | 3.3 [-2.0, 3.0] |

#### 1C per val mode, gptoss20b k=r1 (lr 2.8e-04, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| uppercase_thinking | 0.0 | 0.2 [0.0, 1.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 3.8 | 4.8 [-3.3, 5.2] |

#### 1C per val mode, gptoss20b k=r32 (lr 1.5e-04, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 3.2 | 0.8 [-5.3, 0.2] |

Logistic fit not warranted: the rank-32 top point does not beat checkpoint-0 (CI includes 0).

Claim check, gptoss20b: Δ honest compliance at k=1e4 vs untrained = -0.0 [-0.2, 0.0] pts; the full rank-32 arm itself shows no effect here (0.0 [0.0, 0.0] pts) → CONSISTENT with 'gone at ~1e4'.

#### 1C final checkpoints (step 50): gptoss20b; Δ vs the 1B runs' checkpoint-0 on the same items (test split: 9 training modes x 500 q, 3 held-out modes x 500 q, baseline 500 q)

| arm | seeds | train honest [CI] | Δ vs untrained | train strict | compl_long | train acc | held-out honest [CI] | held-out strict | baseline acc [CI] |
|---|---|---|---|---|---|---|---|---|---|
| 1C 1e3 lr 9.1e-02 | 3 | 0.8 [0.5, 1.1] | -0.4 [-0.8, 0.0] | 0.8 | 0.3 | 43.9 | 0.8 [0.4, 1.2] | 0.8 | 46.3 [42.2, 50.3] |
| 1C 1e4 lr 2.9e-02 | 3 | 0.7 [0.4, 0.9] | -0.5 [-0.9, -0.1] | 0.7 | 0.3 | 43.3 | 1.0 [0.3, 1.9] | 1.1 | 45.1 [40.8, 49.5] |
| 1C 1e5 lr 9.1e-03 | 3 | 1.1 [0.8, 1.3] | -0.1 [-0.5, 0.3] | 1.2 | 0.4 | 45.1 | 1.2 [0.7, 1.8] | 1.4 | 46.6 [42.3, 50.7] |
| 1C r1 lr 2.8e-04 | 3 | 1.7 [1.3, 2.2] | 0.6 [0.0, 1.1] | 2.0 | 0.6 | 40.3 | 2.2 [1.7, 2.9] | 2.6 | 43.1 [39.2, 46.8] |

#### Compliance at matched functional size: gptoss20b (every evaluated cell; C4 loss = mean of the 3 steps ending at the checkpoint, KL(base‖adapter) where measured; 'nearest 1B' = the spec-1B checkpoint closest in C4 loss, unpaired bootstrap)

| cell | seeds | C4 loss | KL | val honest [CI] | val acc | repeat_sent. honest | test honest [CI] | held-out honest | test acc | vs nearest 1B by loss: val Δ honest / Δ acc | vs nearest 1B by loss: test Δ honest / Δ acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| untrained (step 0) | 3 | 14.00 | - | 0.0 [0.0, 0.0] | 49.8 | 3.2 | 1.2 [0.9, 1.4] | 1.1 [0.6, 1.6] | 45.4 | - | - |
| 1C 1e3 lr 3.0e-02 | 1 | 12.64 | - | 0.0 [0.0, 0.0] | 49.2 | 5.0 | - | - | - | untrained (step 0) (14.00): 0.0 [0.0, 0.0] / -0.5 | - |
| 1C 1e3 lr 9.1e-02 (selected) | 3 | 11.46 | - | 0.0 [0.0, 0.0] | 48.8 | 1.2 | 0.8 [0.5, 1.1] | 0.8 [0.4, 1.2] | 43.9 | untrained (step 0) (14.00): 0.0 [0.0, 0.0] / -1.0 | untrained (step 0) (14.00): -0.4 [-0.8, 0.0] / -1.5 |
| 1C 1e4 lr 9.6e-03 | 1 | 10.93 | - | 0.0 [0.0, 0.0] | 50.5 | 3.5 | - | - | - | untrained (step 0) (14.00): 0.0 [0.0, 0.0] / 0.8 | - |
| 1C 1e3 lr 2.7e-01 | 1 | 7.73 | - | 0.0 [0.0, 0.0] | 0.8 | 0.0 | - | - | - | untrained (step 0) (14.00): 0.0 [0.0, 0.0] / -49.0 | - |
| 1C 1e5 lr 3.0e-03 | 1 | 6.29 | - | 0.0 [0.0, 0.0] | 50.6 | 4.5 | - | - | - | 1B r32 step 10 (2.95): 0.0 [0.0, 0.0] / 6.4 | - |
| 1C 1e4 lr 2.9e-02 (selected) | 3 | 6.19 | - | 0.0 [0.0, 0.0] | 45.8 | 1.0 | 0.7 [0.4, 0.9] | 1.0 [0.3, 1.9] | 43.3 | 1B r32 step 10 (2.95): 0.0 [0.0, 0.0] / 1.5 | 1B r32 step 50 (2.77): -0.1 [-0.5, 0.2] / 20.4 |
| 1B r32 lr 5.0e-05 step 10 | 1 | 4.71 | - | 0.0 [0.0, 0.0] | 49.0 | - | - | - | - | 1B r32 step 10 (2.95): 0.0 [0.0, 0.0] / 4.8 | - |
| 1C 1e5 lr 9.1e-03 (selected) | 3 | 4.28 | - | 0.0 [0.0, 0.0] | 48.1 | 3.3 | 1.1 [0.8, 1.3] | 1.2 [0.7, 1.8] | 45.1 | 1B r32 step 10 (2.95): 0.0 [0.0, 0.0] / 3.9 | 1B r32 step 50 (2.77): 0.2 [-0.1, 0.6] / 22.2 |
| 1C 1e4 lr 8.7e-02 | 1 | 3.62 | - | 0.0 [0.0, 0.0] | 28.5 | 0.0 | - | - | - | 1B r32 step 10 (2.95): 0.0 [0.0, 0.0] / -15.7 | - |
| 1C 1e5 lr 2.7e-02 | 1 | 2.99 | - | 0.0 [0.0, 0.0] | 41.8 | 1.5 | - | - | - | 1B r32 step 10 (2.95): 0.0 [0.0, 0.0] / -2.5 | - |
| 1B r32 lr 5.0e-05 step 20 | 1 | 2.98 | - | 0.0 [0.0, 0.0] | 46.0 | - | - | - | - | 1B r32 step 10 (2.95): 0.0 [0.0, 0.0] / 1.8 | - |
| 1B r32 step 10 | 3 | 2.95 | - | 0.0 [0.0, 0.0] | 44.2 | 0.0 | - | - | - | - | - |
| 1B r32 step 20 | 3 | 2.82 | - | 0.0 [0.0, 0.0] | 32.2 | 0.0 | - | - | - | - | - |
| 1B r32 step 100 | 3 | 2.82 | - | 0.1 [0.0, 0.3] | 22.9 | 2.0 | - | - | - | - | - |
| 1C r1 lr 2.8e-04 (selected) | 3 | 2.82 | - | 0.0 [0.0, 0.2] | 43.8 | 4.8 | 1.7 [1.3, 2.2] | 2.2 [1.7, 2.9] | 40.3 | 1B r32 step 100 (2.82): -0.0 [-0.2, 0.2] / 21.0 | 1B r32 step 50 (2.77): 0.9 [0.4, 1.4] / 17.4 |
| 1B r32 step 30 | 3 | 2.81 | - | 0.1 [0.0, 0.3] | 24.9 | 1.0 | - | - | - | - | - |
| 1C r1 lr 8.5e-04 | 1 | 2.81 | - | 0.0 [0.0, 0.0] | 19.5 | 0.5 | - | - | - | 1B r32 step 30 (2.81): -0.1 [-0.4, 0.0] / -5.4 | - |
| 1C r1 lr 2.5e-03 | 1 | 2.80 | - | 0.0 [0.0, 0.0] | 0.1 | 0.0 | - | - | - | 1B r32 step 30 (2.81): -0.1 [-0.4, 0.0] / -24.8 | - |
| 1B r32 step 40 | 3 | 2.78 | - | 0.0 [0.0, 0.2] | 25.1 | 0.8 | - | - | - | - | - |
| 1B r32 step 50 | 3 | 2.77 | 0.072 | 0.0 [0.0, 0.0] | 25.0 | 0.8 | 0.8 [0.6, 1.1] | 2.4 [1.6, 3.2] | 22.9 | - | - |
| 1B r32 step 200 | 3 | 2.75 | 0.069 | 0.0 [0.0, 0.0] | 18.2 | 1.8 | 0.8 [0.5, 1.1] | 1.7 [1.2, 2.2] | 17.4 | - | - |
| 1B r32 step 150 | 3 | 2.75 | - | 0.0 [0.0, 0.0] | 20.7 | 1.5 | - | - | - | - | - |

## qwen8b

#### 1B dose: qwen8b, LoRA r32 lr 1.5e-04 (val, 4 fixed-trait modes x 200 q, pooled; honest CI over questions x seeds)

| step | seeds | honest [95% CI] | seed spread | Δ vs step 0 [CI] | strict | compl_long | acc | answered | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 0.5 [0.1, 0.9] | 0.4–0.5 | - | 0.5 | 0.0 | 46.2 | 95.8 | 15315 | 0.0 |
| 10 | 3 | 0.3 [0.0, 0.7] | 0.2–0.4 | -0.2 [-0.7, 0.3] | 0.3 | 0.0 | 47.1 | 96.2 | 15951 | 0.0 |
| 20 | 3 | 0.2 [0.0, 0.4] | 0.1–0.2 | -0.3 [-0.8, 0.1] | 0.2 | 0.0 | 46.9 | 95.9 | 14510 | 0.0 |
| 30 | 3 | 0.1 [0.0, 0.4] | 0.0–0.2 | -0.3 [-0.8, 0.0] | 0.1 | 0.0 | 47.1 | 97.5 | 13941 | 0.0 |
| 40 | 3 | 0.1 [0.0, 0.3] | 0.0–0.1 | -0.4 [-0.8, 0.0] | 0.1 | 0.0 | 46.3 | 97.4 | 13243 | 0.0 |
| 50 | 3 | 0.1 [0.0, 0.3] | 0.0–0.1 | -0.4 [-0.8, -0.0] | 0.1 | 0.0 | 45.9 | 97.0 | 12745 | 0.1 |
| 100 | 3 | 0.1 [0.0, 0.4] | 0.0–0.1 | -0.4 [-0.9, 0.0] | 0.1 | 0.0 | 46.0 | 97.0 | 11641 | 0.0 |
| 150 | 3 | 0.0 [0.0, 0.2] | 0.0–0.1 | -0.4 [-0.9, -0.0] | 0.1 | 0.0 | 43.5 | 96.9 | 10898 | 0.0 |
| 200 | 3 | 0.2 [0.0, 0.4] | 0.1–0.2 | -0.3 [-0.8, 0.1] | 0.2 | 0.0 | 44.0 | 96.8 | 9950 | 0.0 |

#### 1B per val mode: qwen8b, lr 1.5e-04 (honest, 200 q x 3 seed(s); Δ vs step 0 paired)

| mode | step 0 | step 50 [Δ CI] | step 200 [Δ CI] |
|---|---|---|---|
| lowercase_thinking | 1.8 | 0.3 [-3.0, -0.2] | 0.7 [-3.0, 0.3] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| repeat_sentences | 0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

#### 1B final checkpoints: qwen8b (test split: 9 training modes x 500 q, 3 held-out modes x 500 q, baseline 500 q)

| arm | seeds | train honest [CI] | Δ vs untrained | train strict | compl_long | train acc | held-out honest [CI] | held-out strict | baseline acc [CI] |
|---|---|---|---|---|---|---|---|---|---|
| 1B r32 step 0 (untrained) | 3 | 0.6 [0.4, 0.8] | - | 1.6 | 2.0 | 39.4 | 0.1 [0.0, 0.3] | 0.1 | 33.4 [29.5, 37.3] |
| 1B r32 step 50 | 3 | 0.4 [0.3, 0.6] | -0.1 [-0.3, 0.1] | 1.0 | 1.0 | 38.2 | 0.1 [0.0, 0.3] | 0.1 | 34.5 [30.9, 38.3] |
| 1B r32 step 200 | 3 | 0.4 [0.2, 0.6] | -0.2 [-0.4, 0.1] | 0.8 | 0.8 | 36.3 | 0.3 [0.1, 0.6] | 0.3 | 35.1 [31.3, 38.8] |
| noise, KL-matched to 1B step 50 (3 adapters, KL 0.085 target; adapters 0.084, 0.082, 0.083) | 3 | 0.2 [0.1, 0.4] | -0.3 [-0.6, -0.1] | 0.5 | 0.5 | 39.6 | 0.2 [0.0, 0.4] | 0.2 | 35.8 [31.1, 40.9] |
| noise, KL-matched to 1B step 200 (3 adapters, KL 0.128 target; adapters 0.135, 0.131, 0.134) | 3 | 0.1 [0.0, 0.3] | -0.4 [-0.6, -0.2] | 0.3 | 0.2 | 38.4 | 0.2 [0.0, 0.4] | 0.2 | 36.7 [31.6, 42.2] |

Arm minus KL-matched noise controls (unpaired bootstrap over questions x seeds/adapters):
- 1B step 50 − noise, test_train_modes: honest 0.2 [0.0, 0.4] pts; strict 0.5 [0.0, 1.0] pts; correct -1.4 [-3.2, 0.5] pts
- 1B step 50 − noise, test_heldout: honest -0.1 [-0.4, 0.1] pts; strict -0.1 [-0.4, 0.1] pts; correct -0.5 [-3.7, 2.6] pts
- 1B step 50 − noise, test_baseline: correct -1.3 [-7.4, 4.5] pts
- 1B step 200 − noise, test_train_modes: honest 0.3 [0.1, 0.5] pts; strict 0.5 [0.1, 0.8] pts; correct -2.1 [-4.2, -0.1] pts
- 1B step 200 − noise, test_heldout: honest 0.1 [-0.2, 0.5] pts; strict 0.1 [-0.2, 0.5] pts; correct -3.3 [-6.4, -0.0] pts
- 1B step 200 − noise, test_baseline: correct -1.6 [-8.1, 5.1] pts

KL(base || adapter) from kaushik/noise/runs: cpt-qwen8b-r32-seed0: step 50: 0.0862, step 200: 0.1276; cpt-qwen8b-r32-seed1: step 50: 0.0855, step 200: 0.1283; cpt-qwen8b-r32-seed2: step 50: 0.0850, step 200: 0.1282

#### 1C LR sweep: qwen8b (seed 0, val pooled, step 50; guard = final acc ≥ 85% of step-0 acc)

| k | lr (×pred) | honest@50 | strict@50 | acc 0→50 | compl_long | k_eff | loss | selected |
|---|---|---|---|---|---|---|---|---|
| 1e3 | 1.5e-02 (×0.34) | 0.8 | 0.8 | 47.6→46.1 | 0.0 | 841 | 2.806 |  |
| 1e3 | 4.4e-02 (×0.99) | 0.4 | 0.4 | 47.2→48.6 | 0.0 | 841 | 2.787 |  |
| 1e3 | 1.3e-01 (×2.93) | 0.2 | 0.2 | 46.9→46.2 | 0.0 | 841 | 2.764 | **yes** |
| 1e4 | 4.7e-03 (×0.34) | 0.4 | 0.4 | 47.5→47.9 | 0.0 | 9984 | 2.779 |  |
| 1e4 | 1.4e-02 (×1.00) | 0.2 | 0.2 | 46.4→47.1 | 0.0 | 9984 | 2.712 |  |
| 1e4 | 4.2e-02 (×3.00) | 0.1 | 0.1 | 48.6→48.8 | 0.0 | 9984 | 2.672 | **yes** |
| 1e5 | 1.5e-03 (×0.34) | 0.8 | 0.8 | 46.6→46.2 | 0.0 | 99996 | 2.697 |  |
| 1e5 | 4.4e-03 (×0.99) | 0.9 | 0.9 | 48.2→46.0 | 0.0 | 99997 | 2.626 |  |
| 1e5 | 1.3e-02 (×2.93) | 0.1 | 0.1 | 45.6→46.6 | 0.0 | 99997 | 2.604 | **yes** |
| r1 | 2.8e-04 (×0.33) | 0.5 | 0.5 | 47.0→48.1 | 0.0 | all | 2.600 |  |
| r1 | 8.5e-04 (×1.00) | 0.2 | 0.2 | 47.0→46.6 | 0.0 | all | 2.580 |  |
| r1 | 2.5e-03 (×2.95) | 0.2 | 0.4 | 47.5→43.8 | 0.0 | all | 2.572 | **yes** |

#### 1C compliance vs trainable parameters: qwen8b (val pooled, step 50, selected LR, seeds 0-2)

| k | k_eff | lr | seeds | honest [CI] | seed spread | Δ vs untrained [CI] | strict | compl_long | acc | KL(base‖adapter) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e3 (1,000) | 852 | 1.3e-01 | 3 | 0.3 [0.0, 0.8] | 0.1–0.6 | -0.2 [-0.8, 0.3] | 0.3 | 0.0 | 46.0 | - |
| 1e4 (10,000) | 9967 | 4.2e-02 | 3 | 0.2 [0.0, 0.6] | 0.0–0.5 | -0.2 [-0.8, 0.2] | 0.2 | 0.0 | 45.5 | - |
| 1e5 (100,000) | 99997 | 1.3e-02 | 3 | 0.2 [0.0, 0.5] | 0.1–0.4 | -0.2 [-0.8, 0.2] | 0.2 | 0.0 | 46.8 | - |
| r1 (2,727,936) | all | 2.5e-03 | 3 | 0.2 [0.0, 0.4] | 0.1–0.2 | -0.2 [-0.7, 0.1] | 0.2 | 0.0 | 44.1 | - |
| r32 (87,293,952) | all | 1.5e-04 | 3 | 0.1 [0.0, 0.3] | 0.0–0.1 | -0.4 [-0.8, -0.0] | 0.1 | 0.0 | 45.9 | 0.0862, 0.0855, 0.0850 |

#### 1C per val mode, qwen8b k=1e3 (lr 1.3e-01, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 2.2 | 1.3 [-3.0, 1.3] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 0.0 | 0.0 [0.0, 0.0] |

#### 1C per val mode, qwen8b k=1e4 (lr 4.2e-02, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 1.8 | 0.8 [-3.0, 1.0] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 0.0 | 0.0 [0.0, 0.0] |

#### 1C per val mode, qwen8b k=1e5 (lr 1.3e-02, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 1.8 | 0.8 [-3.0, 0.7] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 0.0 | 0.0 [0.0, 0.0] |

#### 1C per val mode, qwen8b k=r1 (lr 2.5e-03, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 1.7 | 0.7 [-2.8, 0.5] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 0.0 | 0.5 [0.0, 1.5] |

#### 1C per val mode, qwen8b k=r32 (lr 1.5e-04, seeds 3)

| mode | step 0 | step 50 [Δ CI] |
|---|---|---|
| lowercase_thinking | 1.8 | 0.3 [-3.0, -0.2] |
| uppercase_thinking | 0.0 | 0.0 [0.0, 0.0] |
| alternating_case | 0.0 | 0.0 [0.0, 0.0] |
| end_of_sentence | 0.0 | 0.0 [0.0, 0.0] |
| repeat_sentences | 0.0 | 0.0 [0.0, 0.0] |

Logistic fit not warranted: the rank-32 top point does not beat checkpoint-0 (CI includes 0).

Claim check, qwen8b: Δ honest compliance at k=1e4 vs untrained = -0.2 [-0.8, 0.2] pts; the full rank-32 arm itself shows no effect here (-0.4 [-0.8, -0.0] pts) → CONSISTENT with 'gone at ~1e4'.

#### 1C final checkpoints (step 50): qwen8b; Δ vs the 1B runs' checkpoint-0 on the same items (test split: 9 training modes x 500 q, 3 held-out modes x 500 q, baseline 500 q)

| arm | seeds | train honest [CI] | Δ vs untrained | train strict | compl_long | train acc | held-out honest [CI] | held-out strict | baseline acc [CI] |
|---|---|---|---|---|---|---|---|---|---|
| 1C 1e3 lr 1.3e-01 | 2 | 0.3 [0.2, 0.5] | -0.3 [-0.5, -0.0] | 2.7 | 4.0 | 37.9 | 0.2 [0.0, 0.4] | 0.2 | 33.5 [29.6, 37.2] |
| 1C 1e4 lr 4.2e-02 | 2 | 0.3 [0.1, 0.4] | -0.3 [-0.5, -0.0] | 2.6 | 4.1 | 37.8 | 0.2 [0.0, 0.5] | 0.2 | 34.4 [29.0, 39.9] |
| 1C 1e5 lr 1.3e-02 | 3 | 0.3 [0.2, 0.5] | -0.2 [-0.5, 0.1] | 1.1 | 1.3 | 39.2 | 0.2 [0.0, 0.4] | 0.2 | 33.3 [29.3, 37.4] |
| 1C r1 lr 2.5e-03 | 3 | 0.3 [0.2, 0.5] | -0.2 [-0.5, 0.0] | 0.5 | 0.3 | 36.4 | 0.4 [0.1, 0.6] | 0.4 | 33.7 [30.0, 37.3] |

#### Compliance at matched functional size: qwen8b (every evaluated cell; C4 loss = mean of the 3 steps ending at the checkpoint, KL(base‖adapter) where measured; 'nearest 1B' = the spec-1B checkpoint closest in C4 loss, unpaired bootstrap)

| cell | seeds | C4 loss | KL | val honest [CI] | val acc | repeat_sent. honest | test honest [CI] | held-out honest | test acc | vs nearest 1B by loss: val Δ honest / Δ acc | vs nearest 1B by loss: test Δ honest / Δ acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1C 1e3 lr 1.5e-02 | 1 | 2.80 | - | 0.8 [0.2, 1.4] | 46.1 | 0.0 | - | - | - | untrained (step 0) (2.77): 0.3 [-0.4, 1.0] / -0.1 | - |
| 1C 1e3 lr 4.4e-02 | 1 | 2.78 | - | 0.4 [0.0, 0.9] | 48.6 | 0.0 | - | - | - | untrained (step 0) (2.77): -0.1 [-0.7, 0.5] / 2.4 | - |
| untrained (step 0) | 3 | 2.77 | - | 0.5 [0.1, 0.9] | 46.2 | 0.0 | 0.6 [0.4, 0.8] | 0.1 [0.0, 0.3] | 39.4 | - | - |
| 1C 1e4 lr 4.7e-03 | 1 | 2.77 | - | 0.4 [0.0, 0.9] | 47.9 | 0.0 | - | - | - | untrained (step 0) (2.77): -0.1 [-0.7, 0.5] / 1.7 | - |
| 1C 1e3 lr 1.3e-01 (selected) | 3 | 2.74 | - | 0.3 [0.0, 0.8] | 46.0 | 0.0 | 0.3 [0.2, 0.5] | 0.2 [0.0, 0.4] | 37.9 | untrained (step 0) (2.77): -0.1 [-0.7, 0.5] / -0.3 | untrained (step 0) (2.77): -0.3 [-0.5, -0.0] / -1.4 |
| 1C 1e4 lr 1.4e-02 | 1 | 2.71 | - | 0.2 [0.0, 0.6] | 47.1 | 0.0 | - | - | - | untrained (step 0) (2.77): -0.2 [-0.8, 0.3] / 0.9 | - |
| 1C 1e5 lr 1.5e-03 | 1 | 2.69 | - | 0.8 [0.2, 1.4] | 46.2 | 0.0 | - | - | - | 1B r32 step 10 (2.62): 0.5 [-0.2, 1.1] / -0.9 | - |
| 1C 1e4 lr 4.2e-02 (selected) | 3 | 2.65 | - | 0.2 [0.0, 0.6] | 45.5 | 0.0 | 0.3 [0.1, 0.4] | 0.2 [0.0, 0.5] | 37.8 | 1B r32 step 10 (2.62): -0.1 [-0.5, 0.4] / -1.6 | 1B r32 step 50 (2.55): -0.2 [-0.4, 0.1] / -0.4 |
| 1B r32 step 10 | 3 | 2.62 | - | 0.3 [0.0, 0.7] | 47.1 | 0.0 | - | - | - | - | - |
| 1C 1e5 lr 4.4e-03 | 1 | 2.62 | - | 0.9 [0.2, 1.6] | 46.0 | 0.0 | - | - | - | 1B r32 step 10 (2.62): 0.6 [-0.1, 1.4] / -1.1 | - |
| 1C r1 lr 2.8e-04 | 1 | 2.59 | - | 0.5 [0.1, 1.0] | 48.1 | 0.0 | - | - | - | 1B r32 step 20 (2.59): 0.3 [-0.2, 0.9] / 1.3 | - |
| 1B r32 step 20 | 3 | 2.59 | - | 0.2 [0.0, 0.4] | 46.9 | 0.0 | - | - | - | - | - |
| 1C 1e5 lr 1.3e-02 (selected) | 3 | 2.58 | - | 0.2 [0.0, 0.5] | 46.8 | 0.0 | 0.3 [0.2, 0.5] | 0.2 [0.0, 0.4] | 39.2 | 1B r32 step 20 (2.59): 0.0 [-0.3, 0.4] / -0.1 | 1B r32 step 50 (2.55): -0.1 [-0.3, 0.2] / 1.0 |
| 1C r1 lr 8.5e-04 | 1 | 2.57 | - | 0.2 [0.0, 0.6] | 46.6 | 0.0 | - | - | - | 1B r32 step 30 (2.56): 0.1 [-0.2, 0.5] / -0.5 | - |
| 1B r32 step 30 | 3 | 2.56 | - | 0.1 [0.0, 0.4] | 47.1 | 0.0 | - | - | - | - | - |
| 1B r32 step 40 | 3 | 2.56 | - | 0.1 [0.0, 0.3] | 46.3 | 0.2 | - | - | - | - | - |
| 1B r32 step 100 | 3 | 2.56 | - | 0.1 [0.0, 0.4] | 46.0 | 0.2 | - | - | - | - | - |
| 1C r1 lr 2.5e-03 (selected) | 3 | 2.55 | - | 0.2 [0.0, 0.4] | 44.1 | 0.5 | 0.3 [0.2, 0.5] | 0.4 [0.1, 0.6] | 36.4 | 1B r32 step 50 (2.55): 0.1 [-0.2, 0.4] / -1.8 | 1B r32 step 50 (2.55): -0.1 [-0.3, 0.1] / -1.8 |
| 1B r32 step 50 | 3 | 2.55 | 0.086 | 0.1 [0.0, 0.3] | 45.9 | 0.0 | 0.4 [0.3, 0.6] | 0.1 [0.0, 0.3] | 38.2 | - | - |
| 1B r32 step 150 | 3 | 2.55 | - | 0.0 [0.0, 0.2] | 43.5 | 0.3 | - | - | - | - | - |
| 1B r32 step 200 | 3 | 2.52 | 0.128 | 0.2 [0.0, 0.4] | 44.0 | 0.0 | 0.4 [0.2, 0.6] | 0.3 [0.1, 0.6] | 36.3 | - | - |
