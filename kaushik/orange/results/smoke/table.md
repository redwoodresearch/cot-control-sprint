# Experiment 2 table

Honest = honest compliance (`is_honest`), raw = strict compliance, compl_long = compliance among traces at least as long as the untrained median, acc = accuracy, chars = mean reasoning length, degen = traces < 200 chars. Brackets: 95% bootstrap over questions and seeds; seed range = min–max of per-seed honest compliance. Each arm at its final checkpoint; `untrained` = checkpoint-0 of every SFT run.


## qwen36_27b — val split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| persona | untrained | 0 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 87.5 [62.5, 100.0] | 19094 | 0.0 |
| persona | orange | 2 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 13232 | 0.0 |
| stated | untrained | 0 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 62.5 [25.0, 87.8] | 36144 | 0.0 |
| stated | orange | 2 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 62.5 [25.0, 87.5] | 31392 | 0.0 |

Per-mode honest compliance (val, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| persona | untrained | 0.0 [0.0, 0.0] | - | - | - |
| persona | orange | 0.0 [0.0, 0.0] | - | - | - |
| stated | untrained | 0.0 [0.0, 0.0] | - | - | - |
| stated | orange | 0.0 [0.0, 0.0] | - | - | - |

## qwen36_27b — test split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| persona | untrained | 0 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 37.5 [0.0, 75.0] | 22021 | 0.0 |
| persona | orange | 2 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 50.0 [12.5, 87.5] | 26089 | 0.0 |

Per-mode honest compliance (test, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| persona | untrained | 0.0 [0.0, 0.0] | - | - | - |
| persona | orange | 0.0 [0.0, 0.0] | - | - | - |

## qwen36_27b — standard stated eval (test, final checkpoint)

| arm | step | 9-mode honest | 9-mode raw | 9-mode acc | held-out honest | held-out raw | held-out acc | baseline acc |
|---|---|---|---|---|---|---|---|---|
| untrained | 0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 37.5 [12.5, 62.5] | - | - | - | 25.0 [0.0, 62.5] |
| orange | 2 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 25.0 [6.2, 50.0] | - | - | - | 37.5 [12.5, 75.0] |

## Length-matched comparison (val, steps 10–50 pooled, fixed modes)

Raw vs compliance among traces ≥ 50 words (orange-training section 7), and compliance by reasoning-length band (words).

| model | condition | arm | raw | matched ≥50w | % <50w | compl <50w | 0-50w | 50-200w | 200-800w | 800-3200w | 3200+w |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen36_27b | persona | orange | 0.0 | 0.0 | 0.0 | - | - (n=0) | 0.0 (n=2) | - (n=0) | 0.0 (n=4) | 0.0 (n=2) |
| qwen36_27b | stated | orange | 0.0 | 0.0 | 0.0 | - | - (n=0) | - (n=0) | - (n=0) | 0.0 (n=2) | 0.0 (n=6) |
