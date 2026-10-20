# Experiment 2 table

Honest = honest compliance (`is_honest`), raw = strict compliance, compl_long = compliance among traces at least as long as the untrained median, acc = accuracy, chars = mean reasoning length, degen = traces < 200 chars. Brackets: 95% bootstrap over questions and seeds; seed range = min–max of per-seed honest compliance. Each arm at its final checkpoint; `untrained` = checkpoint-0 of every SFT run.


## gptoss20b — val split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| persona | untrained | 0 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.1 [48.0, 54.3] | 5601 | 0.4 |
| persona | orange | 40 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 47.0 [43.6, 50.5] | 4966 | 1.8 |
| direct | untrained | 0 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.5 [45.9, 52.9] | 5591 | 0.2 |
| direct | orange | 40 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 46.8 [42.7, 50.4] | 5769 | 1.0 |
| stated | untrained | 0 | 4 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.6 [46.2, 52.9] | 5816 | 0.5 |
| stated | orange | 40 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.1 [44.3, 52.2] | 5542 | 1.3 |
| stated | mcq_noreason | 50 | 3 | 0.3 [0.0, 0.7] | 0.0–0.5 | 0.6 [0.2, 1.1] | 0.0 [0.0, 0.0] | 35.1 [28.4, 40.9] | 2715 | 1.6 |
| stated | mcq_reason | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.3 [47.0, 55.4] | 5443 | 0.5 |
| stated | style_sft | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 7.1 [5.5, 8.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 435 | 44.8 |

Per-mode honest compliance (val, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| persona | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | orange | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | orange | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | orange | 0.0 [0.0, 0.0] | - | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | mcq_noreason | 0.7 [0.0, 2.0] | 0.4 [0.0, 1.4] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | mcq_reason | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | style_sft | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## gptoss20b — standard stated eval (test, final checkpoint)

| arm | step | 9-mode honest | 9-mode raw | 9-mode acc | held-out honest | held-out raw | held-out acc | baseline acc |
|---|---|---|---|---|---|---|---|---|
| untrained | 0 | 1.5 [1.1, 1.9] | 1.6 [1.2, 2.0] | 45.3 [43.8, 46.7] | 1.2 [0.6, 1.9] | 1.4 [0.9, 2.2] | 44.9 [42.3, 47.3] | 49.0 [44.7, 53.4] |
| noise[mcq_noreason] | noise | 1.0 [0.6, 1.4] | 1.0 [0.7, 1.5] | 42.0 [40.4, 43.5] | 0.7 [0.3, 1.2] | 0.9 [0.4, 1.5] | 42.5 [40.1, 44.8] | 44.0 [39.3, 48.3] |
| noise[style_sft] | noise | 0.9 [0.6, 1.2] | 1.0 [0.7, 1.4] | 41.4 [39.6, 43.5] | 0.5 [0.2, 0.8] | 0.7 [0.3, 1.1] | 42.4 [39.6, 45.3] | 42.9 [38.5, 47.6] |
| mcq_noreason | 50 | 4.9 [4.1, 5.7] | 6.2 [5.6, 6.8] | 32.3 [26.4, 36.9] | 3.8 [2.6, 5.2] | 5.2 [4.1, 6.3] | 32.9 [28.2, 36.9] | 39.6 [33.8, 45.3] |
| mcq_reason | 50 | 1.2 [0.9, 1.6] | 1.3 [1.0, 1.6] | 45.5 [44.1, 47.1] | 0.6 [0.3, 1.0] | 0.7 [0.4, 1.1] | 45.6 [43.3, 48.0] | 46.5 [42.1, 51.3] |
| style_sft | 50 | 0.0 [0.0, 0.1] | 18.9 [17.0, 21.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.2] | 22.1 [19.6, 24.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## qwen36_27b — val split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| persona | untrained | 0 | 15 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.4 [55.4, 61.4] | 13719 | 0.3 |
| persona | orange | 50 | 5 | 6.0 [3.3, 8.7] | 1.6–10.5 | 6.3 [3.1, 9.2] | 0.4 [0.0, 1.7] | 53.5 [50.5, 57.1] | 5951 | 16.2 |
| persona | orange_nopersona | 50 | 5 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 55.4 [52.0, 58.5] | 14200 | 0.3 |
| persona | mcq_noreason | 50 | 5 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.6 [55.1, 62.0] | 12921 | 0.4 |
| persona | rl_orange_noreason | 50 | 5 | 0.3 [0.0, 1.0] | 0.0–1.0 | 0.4 [0.0, 1.2] | 0.0 [0.0, 0.0] | 57.5 [52.0, 62.6] | 10280 | 5.8 |
| persona | rl_mcq_noreason | 50 | 5 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.2 [53.6, 63.2] | 11372 | 0.4 |
| direct | untrained | 0 | 15 | 2.7 [1.9, 3.5] | 1.5–3.2 | 2.8 [2.1, 3.6] | 0.2 [0.0, 0.4] | 51.6 [48.5, 54.7] | 15825 | 10.7 |
| direct | orange | 50 | 5 | 5.5 [3.3, 8.3] | 2.8–10.2 | 6.1 [3.8, 9.5] | 0.7 [0.0, 2.0] | 53.2 [50.0, 56.8] | 11622 | 12.7 |
| direct | orange_nopersona | 50 | 5 | 5.0 [2.7, 8.4] | 1.5–9.9 | 5.1 [2.6, 8.2] | 0.5 [0.1, 1.2] | 52.7 [49.0, 56.1] | 15321 | 6.2 |
| direct | mcq_noreason | 50 | 5 | 3.7 [2.1, 5.4] | 1.9–6.1 | 4.0 [2.4, 6.0] | 0.3 [0.0, 0.9] | 55.6 [52.3, 59.4] | 12967 | 8.4 |
| direct | rl_orange_noreason | 50 | 3 | 4.6 [1.9, 6.9] | 3.0–5.5 | 4.7 [2.4, 6.8] | 0.4 [0.0, 1.6] | 55.0 [48.0, 60.9] | 16448 | 8.2 |
| direct | rl_mcq_noreason | 50 | 5 | 4.8 [2.7, 7.0] | 2.0–7.0 | 5.2 [3.0, 7.1] | 0.3 [0.0, 1.2] | 55.6 [50.8, 60.0] | 12353 | 7.4 |
| stated | untrained | 0 | 15 | 0.4 [0.1, 0.7] | 0.1–0.9 | 0.4 [0.1, 0.8] | 0.0 [0.0, 0.0] | 50.7 [47.4, 53.8] | 22873 | 0.0 |
| stated | orange | 50 | 5 | 1.0 [0.5, 1.7] | 0.2–1.4 | 1.0 [0.4, 1.7] | 0.0 [0.0, 0.1] | 51.6 [48.4, 54.7] | 20900 | 0.0 |
| stated | orange_nopersona | 50 | 5 | 0.6 [0.1, 1.4] | 0.0–1.2 | 0.6 [0.1, 1.3] | 0.0 [0.0, 0.0] | 45.8 [40.1, 51.5] | 22682 | 3.5 |
| stated | mcq_noreason | 50 | 5 | 3.3 [1.7, 5.3] | 1.2–5.3 | 3.4 [1.8, 5.5] | 0.0 [0.0, 0.0] | 52.4 [48.3, 56.2] | 18877 | 0.1 |
| stated | rl_orange_noreason | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.2 [44.8, 58.4] | 21820 | 0.0 |
| stated | rl_mcq_noreason | 50 | 4 | 4.2 [0.0, 8.7] | 0.0–9.2 | 4.2 [0.0, 8.9] | 0.3 [0.0, 1.1] | 54.3 [48.6, 60.3] | 18372 | 0.0 |

Per-mode honest compliance (val, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| persona | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | orange | 8.0 [3.6, 14.0] | 10.4 [4.4, 17.5] | 1.7 [0.5, 3.2] | 4.1 [1.8, 7.2] |
| persona | orange_nopersona | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | mcq_noreason | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | rl_orange_noreason | 0.2 [0.0, 0.8] | 0.4 [0.0, 1.8] | - | - |
| persona | rl_mcq_noreason | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | - |
| direct | untrained | 3.9 [2.2, 6.0] | 2.5 [1.1, 4.1] | 1.3 [0.6, 2.2] | 2.9 [1.8, 4.1] |
| direct | orange | 10.7 [4.4, 19.3] | 5.8 [3.4, 8.2] | 1.4 [0.5, 2.8] | 4.0 [1.4, 7.2] |
| direct | orange_nopersona | 10.5 [2.9, 19.6] | 6.5 [1.1, 13.6] | 0.8 [0.0, 1.9] | 2.3 [0.8, 4.3] |
| direct | mcq_noreason | 7.5 [3.3, 13.6] | 2.4 [0.7, 4.6] | 1.2 [0.2, 2.6] | 3.5 [0.9, 7.4] |
| direct | rl_orange_noreason | 4.2 [1.9, 6.8] | 5.5 [2.4, 9.0] | - | - |
| direct | rl_mcq_noreason | 5.6 [3.0, 8.9] | 3.9 [1.7, 6.8] | - | - |
| stated | untrained | 0.0 [0.0, 0.2] | 1.4 [0.4, 2.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | orange | 0.0 [0.0, 0.0] | 4.1 [1.9, 6.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | orange_nopersona | 0.4 [0.0, 1.4] | 2.2 [0.4, 4.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | mcq_noreason | 1.2 [0.2, 2.6] | 9.9 [5.5, 15.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | rl_orange_noreason | 0.0 [0.0, 0.0] | - | - | - |
| stated | rl_mcq_noreason | 1.2 [0.0, 3.3] | 16.4 [11.0, 22.0] | - | - |

## qwen36_27b — test split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| persona | untrained | 0 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.6 [49.2, 53.8] | 15023 | 0.5 |
| persona | noise[mcq_noreason] | noise | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.6 [46.6, 56.4] | 16571 | 0.2 |
| persona | noise[orange] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.0 [48.6, 53.3] | 15211 | 0.1 |
| persona | noise[orange_nopersona] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.2 [49.0, 53.5] | 15559 | 0.3 |
| persona | noise[rl_mcq_noreason] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 50.8 [47.4, 54.1] | 15660 | 0.4 |
| persona | noise[rl_orange_noreason] | noise | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.9 [48.8, 57.4] | 15663 | 0.4 |
| persona | orange | 50 | 1 | 5.5 [4.4, 6.7] | 5.9–5.9 | 6.0 [5.0, 7.1] | 0.0 [0.0, 0.0] | 46.1 [43.9, 48.4] | 2971 | 25.4 |
| persona | orange_nopersona | 50 | 1 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 50.5 [45.9, 55.0] | 11449 | 0.2 |
| direct | noise[orange] | noise | 3 | 1.7 [1.0, 2.4] | 1.4–2.5 | 1.8 [1.1, 2.5] | 0.0 [0.0, 0.1] | 46.2 [44.0, 48.4] | 17873 | 8.8 |
| direct | noise[orange_nopersona] | noise | 3 | 1.9 [0.9, 2.8] | 0.9–2.4 | 1.9 [1.0, 2.9] | 0.2 [0.0, 0.5] | 45.3 [43.0, 47.4] | 18497 | 6.7 |
| direct | noise[rl_mcq_noreason] | noise | 3 | 2.5 [1.3, 4.2] | 2.0–3.2 | 2.7 [1.1, 4.2] | 0.4 [0.0, 1.1] | 45.5 [40.9, 50.2] | 19779 | 6.5 |
| direct | noise[rl_orange_noreason] | noise | 2 | 2.1 [0.7, 3.8] | 1.8–2.6 | 2.1 [0.9, 3.7] | 0.5 [0.0, 1.3] | 44.7 [39.6, 50.4] | 20602 | 5.9 |
| direct | orange | 50 | 1 | 4.9 [2.9, 7.1] | 5.4–5.4 | 5.2 [3.2, 7.3] | 0.0 [0.0, 0.0] | 48.0 [43.5, 51.8] | 6750 | 17.6 |
| stated | noise[orange] | noise | 3 | 0.6 [0.2, 1.0] | 0.4–0.8 | 0.6 [0.2, 1.0] | 0.0 [0.0, 0.0] | 44.0 [41.9, 46.5] | 24459 | 0.0 |
| stated | noise[orange_nopersona] | noise | 3 | 0.8 [0.5, 1.1] | 0.7–0.9 | 0.8 [0.5, 1.2] | 0.0 [0.0, 0.0] | 42.0 [39.8, 44.7] | 25500 | 0.0 |
| stated | noise[rl_mcq_noreason] | noise | 3 | 0.1 [0.0, 0.4] | 0.0–0.2 | 0.1 [0.0, 0.4] | 0.0 [0.0, 0.0] | 43.8 [38.9, 48.1] | 24458 | 0.0 |

Per-mode honest compliance (test, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| persona | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[mcq_noreason] | 0.0 [0.0, 0.0] | - | - | - |
| persona | noise[orange] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[orange_nopersona] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[rl_mcq_noreason] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | - |
| persona | noise[rl_orange_noreason] | 0.0 [0.0, 0.0] | - | - | - |
| persona | orange | 8.2 [6.0, 10.7] | 6.8 [4.5, 9.2] | 2.4 [1.1, 3.8] | 4.7 [2.9, 6.8] |
| persona | orange_nopersona | - | - | - | 0.0 [0.0, 0.0] |
| direct | noise[orange] | 2.3 [1.1, 3.4] | 1.7 [0.7, 2.8] | 0.7 [0.0, 1.5] | 1.9 [0.7, 3.4] |
| direct | noise[orange_nopersona] | 3.3 [1.1, 5.7] | 2.2 [0.8, 4.2] | 0.6 [0.1, 1.2] | 1.4 [0.6, 2.3] |
| direct | noise[rl_mcq_noreason] | 2.5 [1.2, 4.0] | - | - | - |
| direct | noise[rl_orange_noreason] | 2.1 [0.8, 3.9] | - | - | - |
| direct | orange | - | - | - | 4.9 [2.9, 7.0] |
| stated | noise[orange] | 0.1 [0.0, 0.4] | 2.3 [1.0, 3.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | noise[orange_nopersona] | 0.4 [0.0, 1.0] | 2.6 [1.3, 4.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | noise[rl_mcq_noreason] | 0.1 [0.0, 0.6] | - | - | - |

## qwen36_27b — standard stated eval (test, final checkpoint)

| arm | step | 9-mode honest | 9-mode raw | 9-mode acc | held-out honest | held-out raw | held-out acc | baseline acc |
|---|---|---|---|---|---|---|---|---|
| noise[mcq_noreason] | noise | 0.3 [0.1, 0.5] | 0.3 [0.2, 0.5] | 44.4 [42.8, 45.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 47.5 [44.5, 50.1] | - |

## qwen8b — val split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| stated | untrained | 0 | 3 | 0.4 [0.0, 1.0] | 0.4–0.4 | 0.4 [0.0, 1.0] | 0.0 [0.0, 0.0] | 49.5 [45.6, 52.6] | 14924 | 0.0 |
| stated | mcq_noreason | 50 | 3 | 2.4 [0.3, 4.2] | 0.2–4.1 | 7.1 [2.3, 11.3] | 0.4 [0.0, 1.7] | 31.8 [28.5, 35.5] | 5519 | 14.8 |
| stated | mcq_reason | 50 | 3 | 0.9 [0.3, 1.6] | 0.9–1.0 | 0.9 [0.3, 1.7] | 0.0 [0.0, 0.0] | 47.4 [44.1, 50.9] | 15063 | 0.0 |
| stated | style_sft | 50 | 3 | 0.0 [0.0, 0.3] | 0.0–0.1 | 10.7 [8.4, 12.8] | 0.0 [0.0, 0.0] | 0.3 [0.0, 0.9] | 1470 | 57.2 |

Per-mode honest compliance (val, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| stated | untrained | 1.8 [0.0, 4.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | mcq_noreason | 5.9 [0.7, 10.9] | 2.8 [0.0, 6.1] | 0.2 [0.0, 1.2] | 0.7 [0.0, 2.7] |
| stated | mcq_reason | 3.7 [1.4, 6.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | style_sft | 0.2 [0.0, 1.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## qwen8b — standard stated eval (test, final checkpoint)

| arm | step | 9-mode honest | 9-mode raw | 9-mode acc | held-out honest | held-out raw | held-out acc | baseline acc |
|---|---|---|---|---|---|---|---|---|
| untrained | 0 | 0.4 [0.2, 0.6] | 1.4 [1.0, 1.7] | 39.4 [37.9, 40.7] | 0.1 [0.0, 0.4] | 0.1 [0.0, 0.4] | 41.6 [39.1, 44.0] | 31.1 [26.7, 35.0] |
| noise[mcq_noreason] | noise | 0.4 [0.2, 0.6] | 1.3 [0.5, 2.3] | 38.5 [36.8, 40.1] | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 42.1 [39.5, 44.7] | 33.9 [30.0, 37.5] |
| noise[mcq_reason] | noise | 0.4 [0.3, 0.6] | 1.2 [0.9, 1.5] | 39.4 [38.1, 41.0] | 0.3 [0.1, 0.7] | 0.3 [0.0, 0.7] | 41.7 [39.2, 44.1] | 33.1 [28.9, 37.5] |
| noise[style_sft] | noise | 0.2 [0.1, 0.3] | 0.6 [0.2, 1.2] | 38.9 [37.5, 40.3] | 0.2 [0.0, 0.4] | 0.2 [0.0, 0.4] | 41.5 [39.2, 43.9] | 34.3 [29.4, 39.1] |
| mcq_noreason | 50 | 2.5 [1.3, 3.7] | 10.7 [5.9, 14.7] | 26.6 [23.8, 29.2] | 1.4 [0.5, 2.5] | 1.9 [0.7, 3.4] | 30.7 [27.2, 34.3] | 29.3 [23.7, 34.5] |
| mcq_reason | 50 | 0.4 [0.2, 0.6] | 1.2 [0.9, 1.5] | 39.1 [37.4, 40.5] | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 42.0 [39.3, 44.6] | 32.9 [27.1, 38.6] |
| style_sft | 50 | 0.0 [0.0, 0.1] | 19.4 [18.0, 20.6] | 0.4 [0.1, 0.8] | 0.0 [0.0, 0.2] | 21.0 [18.4, 23.2] | 0.2 [0.0, 0.3] | 0.1 [0.0, 0.4] |

## Quantities of interest (A − B, final checkpoints, modes pooled)

| model | split | kind | condition | difference | steps | seeds | Δ honest | Δ raw | Δ compl_long | Δ acc |
|---|---|---|---|---|---|---|---|---|---|---|
| gptoss20b | test | heldout | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 3/3 | 3.2 [1.8, 4.5] | 4.3 [3.1, 5.4] | 0.0 [0.0, 0.0] | -9.7 [-14.1, -5.6] |
| gptoss20b | test | standard | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 3/3 | 3.9 [3.0, 4.8] | 5.1 [4.4, 6.0] | 1.6 [1.0, 2.3] | -9.7 [-15.5, -5.0] |
| gptoss20b | val | fixed | stated | orange − mcq_noreason | 40/50 | 1/3 | -0.3 [-0.7, 0.0] | -0.6 [-1.1, -0.1] | 0.0 [0.0, 0.0] | 12.9 [6.5, 20.6] |
| qwen36_27b | test | fixed | direct | orange − noise[orange] | 50/noise | 1/3 | 3.3 [1.3, 5.3] | 3.5 [1.3, 5.4] | -0.0 [-0.1, 0.0] | 1.8 [-2.3, 6.1] |
| qwen36_27b | test | fixed | persona | orange − noise[orange] | 50/noise | 1/3 | 5.5 [4.5, 6.6] | 6.0 [5.0, 7.1] | 0.0 [0.0, 0.0] | -4.9 [-7.1, -2.7] |
| qwen36_27b | test | fixed | persona | orange − orange_nopersona | 50/50 | 1/1 | 5.5 [4.5, 6.8] | 6.0 [5.0, 7.2] | 0.0 [0.0, 0.0] | -4.4 [-8.9, 0.5] |
| qwen36_27b | test | fixed | persona | orange_nopersona − noise[orange_nopersona] | 50/noise | 1/3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -0.7 [-4.8, 3.5] |
| qwen36_27b | val | fixed | direct | orange − mcq_noreason | 50/50 | 5/5 | 1.8 [-0.8, 4.8] | 2.1 [-0.9, 5.3] | 0.4 [-0.6, 2.1] | -2.4 [-5.2, -0.1] |
| qwen36_27b | val | fixed | direct | orange − orange_nopersona | 50/50 | 5/5 | 0.5 [-3.1, 4.0] | 1.0 [-2.6, 5.2] | 0.1 [-0.9, 1.6] | 0.6 [-2.6, 3.4] |
| qwen36_27b | val | fixed | direct | rl_orange_noreason − rl_mcq_noreason | 50/50 | 3/5 | -0.3 [-3.7, 2.5] | -0.5 [-4.0, 2.1] | 0.1 [-0.9, 1.3] | -0.6 [-6.6, 3.6] |
| qwen36_27b | val | fixed | persona | orange − mcq_noreason | 50/50 | 5/5 | 6.0 [3.3, 8.8] | 6.3 [3.6, 9.3] | 0.4 [0.0, 1.7] | -5.1 [-7.7, -2.7] |
| qwen36_27b | val | fixed | persona | orange − orange_nopersona | 50/50 | 5/5 | 6.0 [3.3, 9.1] | 6.3 [3.5, 9.3] | 0.4 [0.0, 2.1] | -1.9 [-5.2, 1.0] |
| qwen36_27b | val | fixed | persona | rl_orange_noreason − rl_mcq_noreason | 50/50 | 5/5 | 0.3 [0.0, 0.9] | 0.4 [0.0, 1.0] | 0.0 [0.0, 0.0] | -0.7 [-5.1, 3.3] |
| qwen36_27b | val | fixed | stated | orange − mcq_noreason | 50/50 | 5/5 | -2.2 [-4.5, -0.7] | -2.4 [-4.4, -0.8] | 0.0 [0.0, 0.1] | -0.8 [-3.8, 2.4] |
| qwen36_27b | val | fixed | stated | orange − orange_nopersona | 50/50 | 5/5 | 0.4 [-0.4, 1.1] | 0.4 [-0.3, 1.3] | 0.0 [0.0, 0.1] | 5.8 [0.9, 10.5] |
| qwen36_27b | val | fixed | stated | rl_orange_noreason − rl_mcq_noreason | 50/50 | 3/4 | -4.2 [-8.5, 0.0] | -4.2 [-9.5, 0.0] | -0.3 [-1.1, 0.0] | -3.1 [-7.9, 1.8] |
| qwen8b | test | heldout | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 3/3 | 1.3 [0.3, 2.5] | 1.7 [0.6, 3.3] | 0.2 [0.0, 2.5] | -11.4 [-15.2, -8.3] |
| qwen8b | test | standard | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 3/3 | 2.2 [0.9, 3.4] | 9.4 [4.8, 13.3] | 14.5 [6.8, 24.8] | -11.9 [-15.2, -9.3] |

## Length-matched comparison (val, steps 10–50 pooled, fixed modes)

Raw vs compliance among traces ≥ 50 words (orange-training section 7), and compliance by reasoning-length band (words).

| model | condition | arm | raw | matched ≥50w | % <50w | compl <50w | 0-50w | 50-200w | 200-800w | 800-3200w | 3200+w |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gptoss20b | direct | orange | 0.0 | 0.0 | 3.8 | 0.0 | 0.0 (n=121) | 0.0 (n=892) | 0.0 (n=1279) | 0.0 (n=736) | 0.0 (n=172) |
| gptoss20b | persona | orange | 0.0 | 0.0 | 5.2 | 0.0 | 0.0 (n=165) | 0.0 (n=916) | 0.0 (n=1334) | 0.0 (n=632) | 0.0 (n=153) |
| gptoss20b | stated | orange | 0.0 | 0.0 | 3.3 | 0.0 | 0.0 (n=98) | 0.0 (n=788) | 0.0 (n=1218) | 0.0 (n=710) | 0.0 (n=186) |
| gptoss20b | stated | mcq_noreason | 0.3 | 0.2 | 2.9 | 3.8 | 3.8 (n=343) | 0.6 (n=3793) | 0.0 (n=5407) | 0.0 (n=2049) | 0.0 (n=408) |
| gptoss20b | stated | mcq_reason | 0.0 | 0.0 | 2.2 | 0.4 | 0.4 (n=261) | 0.0 (n=2835) | 0.0 (n=5065) | 0.0 (n=3063) | 0.0 (n=776) |
| gptoss20b | stated | style_sft | 6.4 | 2.1 | 47.9 | 11.0 | 11.0 (n=5754) | 2.8 (n=4538) | 0.2 (n=1280) | 0.0 (n=368) | 0.0 (n=60) |
| qwen36_27b | direct | orange | 5.6 | 6.2 | 17.8 | 2.8 | 2.8 (n=3561) | 18.0 (n=3506) | 7.2 (n=4043) | 1.9 (n=4594) | 0.1 (n=4296) |
| qwen36_27b | direct | orange_nopersona | 3.4 | 3.6 | 11.1 | 2.3 | 2.3 (n=2220) | 12.0 (n=2675) | 6.3 (n=3869) | 1.3 (n=5355) | 0.0 (n=5881) |
| qwen36_27b | direct | mcq_noreason | 3.1 | 3.1 | 12.5 | 2.8 | 2.8 (n=2507) | 11.0 (n=2994) | 3.9 (n=4127) | 0.9 (n=5355) | 0.1 (n=5017) |
| qwen36_27b | direct | rl_orange_noreason | 3.6 | 3.9 | 11.5 | 1.6 | 1.6 (n=574) | 15.1 (n=730) | 5.4 (n=765) | 1.5 (n=1409) | 0.1 (n=1522) |
| qwen36_27b | direct | rl_mcq_noreason | 4.8 | 5.0 | 10.1 | 2.9 | 2.9 (n=908) | 14.9 (n=1113) | 9.3 (n=2108) | 1.4 (n=2808) | 0.0 (n=2063) |
| qwen36_27b | persona | orange | 6.1 | 7.2 | 22.2 | 2.4 | 2.4 (n=4448) | 14.2 (n=4913) | 7.1 (n=5246) | 1.4 (n=3732) | 0.1 (n=1661) |
| qwen36_27b | persona | orange_nopersona | 0.0 | 0.0 | 1.0 | 3.0 | 3.0 (n=199) | 0.3 (n=680) | 0.0 (n=5005) | 0.0 (n=9996) | 0.0 (n=4120) |
| qwen36_27b | persona | mcq_noreason | 0.0 | 0.0 | 0.6 | 0.0 | 0.0 (n=121) | 0.0 (n=682) | 0.0 (n=5319) | 0.0 (n=10065) | 0.0 (n=3813) |
| qwen36_27b | persona | rl_orange_noreason | 0.2 | 0.2 | 8.3 | 0.3 | 0.3 (n=665) | 1.0 (n=1137) | 0.3 (n=2080) | 0.0 (n=2757) | 0.0 (n=1361) |
| qwen36_27b | persona | rl_mcq_noreason | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 (n=87) | 0.0 (n=261) | 0.0 (n=2659) | 0.0 (n=4621) | 0.0 (n=1372) |
| qwen36_27b | stated | orange | 0.9 | 0.9 | 0.0 | 0.0 | 0.0 (n=4) | 54.4 (n=57) | 20.2 (n=704) | 0.1 (n=11138) | 0.0 (n=8097) |
| qwen36_27b | stated | orange_nopersona | 0.7 | 0.8 | 0.7 | 0.0 | 0.0 (n=149) | 58.1 (n=62) | 15.1 (n=707) | 0.1 (n=10528) | 0.0 (n=8554) |
| qwen36_27b | stated | mcq_noreason | 2.2 | 2.2 | 0.1 | 12.5 | 12.5 (n=16) | 56.6 (n=129) | 23.2 (n=1187) | 0.2 (n=9348) | 0.0 (n=6320) |
| qwen36_27b | stated | rl_orange_noreason | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 (n=1) | 0.0 (n=1) | 1.2 (n=86) | 0.0 (n=1815) | 0.0 (n=1097) |
| qwen36_27b | stated | rl_mcq_noreason | 3.2 | 3.2 | 0.0 | - | - (n=0) | 62.1 (n=58) | 21.7 (n=553) | 0.5 (n=3144) | 0.0 (n=1645) |
| qwen8b | stated | mcq_noreason | 6.8 | 7.6 | 13.6 | 1.8 | 1.8 (n=1632) | 30.0 (n=1454) | 8.2 (n=3734) | 1.0 (n=3842) | 0.5 (n=1338) |
| qwen8b | stated | mcq_reason | 0.7 | 0.7 | 0.0 | - | - (n=0) | 19.6 (n=46) | 2.3 (n=3079) | 0.1 (n=5051) | 0.0 (n=3824) |
| qwen8b | stated | style_sft | 9.8 | 5.8 | 52.9 | 13.3 | 13.3 (n=6346) | 15.5 (n=1784) | 3.5 (n=1357) | 0.2 (n=1711) | 0.0 (n=802) |

Length-matched (≥ 50 words) differences:

| model | condition | difference | Δ matched compliance |
|---|---|---|---|
| qwen36_27b | direct | orange − orange_nopersona | 2.6 [0.9, 4.6] |
| qwen36_27b | persona | orange − orange_nopersona | 7.2 [3.8, 10.5] |
| qwen36_27b | stated | orange − orange_nopersona | 0.2 [-0.2, 0.6] |
| gptoss20b | stated | orange − mcq_noreason | -0.2 [-0.3, -0.0] |
| qwen36_27b | direct | orange − mcq_noreason | 3.0 [1.2, 4.8] |
| qwen36_27b | persona | orange − mcq_noreason | 7.3 [3.9, 10.4] |
| qwen36_27b | stated | orange − mcq_noreason | -1.3 [-2.7, -0.2] |
| qwen36_27b | direct | rl_orange_noreason − rl_mcq_noreason | -1.5 [-3.8, 0.3] |
| qwen36_27b | persona | rl_orange_noreason − rl_mcq_noreason | 0.2 [0.0, 0.5] |
| qwen36_27b | stated | rl_orange_noreason − rl_mcq_noreason | -3.0 [-7.0, -0.3] |
