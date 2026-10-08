# Experiment 2 table

Honest = honest compliance (`is_honest`), raw = strict compliance, compl_long = compliance among traces at least as long as the untrained median, acc = accuracy, chars = mean reasoning length, degen = traces < 200 chars. Brackets: 95% bootstrap over questions and seeds; seed range = min–max of per-seed honest compliance. Each arm at its final checkpoint; `untrained` = checkpoint-0 of every SFT run.


## gptoss20b — val split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| persona | untrained | 0 | 6 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.6 [46.3, 52.8] | 5199 | 0.8 |
| persona | orange | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.8 [41.0, 48.7] | 4772 | 1.1 |
| persona | orange_nopersona | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 45.0 [41.5, 48.6] | 8480 | 1.7 |
| direct | untrained | 0 | 6 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.8 [46.6, 52.9] | 5492 | 0.6 |
| direct | orange | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] | 45.3 [41.7, 49.1] | 5156 | 0.9 |
| direct | orange_nopersona | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.1 [0.0, 0.4] | 0.0 [0.0, 0.0] | 43.9 [40.1, 47.9] | 8563 | 2.1 |
| stated | untrained | 0 | 9 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] | 49.8 [46.7, 52.9] | 5825 | 0.5 |
| stated | orange | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.2 [38.0, 47.7] | 5514 | 1.0 |
| stated | orange_nopersona | 50 | 3 | 0.0 [0.0, 0.3] | 0.0–0.1 | 0.1 [0.0, 0.3] | 0.0 [0.0, 0.0] | 41.6 [38.3, 44.9] | 9029 | 1.1 |
| stated | mcq_noreason | 50 | 3 | 0.3 [0.0, 0.7] | 0.0–0.5 | 0.6 [0.1, 1.1] | 0.0 [0.0, 0.0] | 35.1 [28.1, 41.6] | 2715 | 1.6 |
| stated | mcq_reason | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.3 [47.3, 55.3] | 5443 | 0.5 |
| stated | style_sft | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 7.1 [5.4, 8.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 435 | 44.8 |

Per-mode honest compliance (val, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| persona | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | orange | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | orange_nopersona | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | orange | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | orange_nopersona | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | orange | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | orange_nopersona | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.2 [0.0, 1.2] |
| stated | mcq_noreason | 0.7 [0.0, 2.0] | 0.4 [0.0, 1.4] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | mcq_reason | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | style_sft | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## gptoss20b — test split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| persona | untrained | 0 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.7 [42.6, 46.8] | 5126 | 1.2 |
| persona | noise[orange] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.5 [41.4, 45.7] | 10235 | 0.8 |
| persona | noise[orange_nopersona] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 41.2 [38.7, 43.9] | 13267 | 0.8 |
| persona | orange | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 42.1 [39.7, 44.6] | 4836 | 1.6 |
| persona | orange_nopersona | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 38.5 [35.8, 41.1] | 9030 | 1.9 |
| direct | untrained | 0 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 45.7 [43.4, 48.3] | 5718 | 1.2 |
| direct | noise[orange] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 43.3 [41.3, 45.4] | 10218 | 1.0 |
| direct | noise[orange_nopersona] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 40.6 [38.3, 43.1] | 14176 | 0.7 |
| direct | orange | 50 | 3 | 0.0 [0.0, 0.1] | 0.0–0.1 | 0.1 [0.0, 0.2] | 0.0 [0.0, 0.0] | 42.2 [39.7, 44.6] | 5360 | 1.9 |
| direct | orange_nopersona | 50 | 3 | 0.0 [0.0, 0.1] | 0.0–0.1 | 0.1 [0.0, 0.3] | 0.0 [0.0, 0.0] | 38.3 [36.1, 40.6] | 9245 | 2.0 |
| stated | untrained | 0 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 45.4 [43.1, 47.6] | 5647 | 1.0 |
| stated | noise[orange] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] | 43.7 [41.5, 45.8] | 11148 | 1.0 |
| stated | noise[orange_nopersona] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 41.0 [38.7, 43.3] | 14728 | 0.7 |
| stated | orange | 50 | 3 | 0.1 [0.0, 0.2] | 0.0–0.1 | 0.1 [0.0, 0.2] | 0.0 [0.0, 0.0] | 40.1 [37.3, 42.9] | 5850 | 1.2 |
| stated | orange_nopersona | 50 | 3 | 0.0 [0.0, 0.1] | 0.0–0.1 | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.0] | 36.8 [34.8, 39.0] | 9081 | 1.5 |

Per-mode honest compliance (test, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| persona | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[orange] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[orange_nopersona] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | orange | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | orange_nopersona | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | noise[orange] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | noise[orange_nopersona] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | orange | 0.0 [0.0, 0.0] | 0.1 [0.0, 0.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | orange_nopersona | 0.0 [0.0, 0.0] | 0.1 [0.0, 0.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | noise[orange] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | noise[orange_nopersona] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | orange | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.8] |
| stated | orange_nopersona | 0.0 [0.0, 0.3] | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## gptoss20b — standard stated eval (test, final checkpoint)

| arm | step | 9-mode honest | 9-mode raw | 9-mode acc | held-out honest | held-out raw | held-out acc | baseline acc |
|---|---|---|---|---|---|---|---|---|
| untrained | 0 | 1.4 [1.1, 1.7] | 1.5 [1.1, 1.9] | 45.2 [43.8, 46.6] | 0.9 [0.5, 1.4] | 1.1 [0.7, 1.6] | 45.7 [43.3, 48.1] | 46.6 [42.4, 51.2] |
| noise[cpt-step50] | noise | 0.8 [0.5, 1.2] | 0.9 [0.6, 1.3] | 41.8 [40.3, 43.3] | 0.8 [0.4, 1.4] | 1.0 [0.4, 1.6] | 42.8 [40.1, 45.5] | 42.6 [38.1, 47.0] |
| noise[cpt] | noise | 1.0 [0.7, 1.4] | 1.2 [0.8, 1.5] | 41.6 [39.8, 43.4] | 0.7 [0.3, 1.2] | 0.8 [0.4, 1.3] | 42.0 [39.6, 44.4] | 44.9 [40.5, 49.3] |
| noise[mcq_noreason] | noise | 1.0 [0.5, 1.4] | 1.0 [0.7, 1.5] | 42.0 [40.2, 43.8] | 0.7 [0.3, 1.2] | 0.9 [0.3, 1.6] | 42.5 [40.1, 45.0] | 44.0 [39.4, 48.7] |
| noise[mcq_reason] | noise | 1.3 [1.0, 1.6] | 1.3 [1.0, 1.7] | 44.7 [43.3, 46.1] | 0.9 [0.6, 1.4] | 1.1 [0.7, 1.6] | 45.1 [42.5, 47.6] | 46.6 [42.3, 51.0] |
| noise[orange] | noise | 0.9 [0.5, 1.3] | 1.0 [0.6, 1.4] | 42.9 [41.4, 44.4] | 0.5 [0.2, 1.0] | 0.6 [0.2, 1.1] | 43.8 [41.4, 46.5] | 43.9 [39.8, 48.1] |
| noise[orange_nopersona] | noise | 0.8 [0.6, 1.1] | 0.9 [0.6, 1.2] | 40.1 [38.4, 41.8] | 0.6 [0.2, 1.0] | 0.8 [0.4, 1.4] | 41.3 [38.7, 43.9] | 40.5 [35.9, 44.9] |
| noise[style_sft] | noise | 0.9 [0.6, 1.2] | 1.0 [0.7, 1.4] | 41.4 [39.7, 43.3] | 0.5 [0.2, 0.8] | 0.7 [0.4, 1.2] | 42.4 [39.5, 45.3] | 42.9 [38.5, 47.2] |
| orange | 50 | 0.6 [0.4, 1.0] | 0.7 [0.4, 1.1] | 39.7 [37.1, 41.9] | 1.7 [1.0, 2.5] | 2.3 [1.4, 3.2] | 39.8 [35.8, 43.4] | 42.1 [36.8, 47.0] |
| orange_nopersona | 50 | 1.3 [0.7, 2.1] | 1.6 [0.9, 2.7] | 36.3 [34.9, 37.6] | 2.1 [1.2, 3.0] | 2.7 [1.6, 3.8] | 36.5 [34.2, 38.8] | 36.8 [32.5, 41.4] |
| mcq_noreason | 50 | 4.9 [4.1, 5.7] | 6.2 [5.5, 6.9] | 32.3 [26.8, 36.7] | 3.8 [2.6, 5.1] | 5.2 [4.1, 6.3] | 32.9 [28.3, 36.9] | 39.6 [33.9, 44.8] |
| mcq_reason | 50 | 1.2 [1.0, 1.6] | 1.3 [1.0, 1.6] | 45.5 [44.1, 47.0] | 0.6 [0.3, 1.0] | 0.7 [0.4, 1.1] | 45.6 [43.3, 48.2] | 46.5 [42.1, 50.8] |
| style_sft | 50 | 0.0 [0.0, 0.1] | 18.9 [16.9, 20.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.2] | 22.1 [19.4, 24.7] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## qwen36_27b — val split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| persona | untrained | 0 | 15 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.4 [55.2, 61.5] | 13719 | 0.3 |
| persona | orange | 50 | 5 | 6.0 [3.1, 9.2] | 1.6–10.5 | 6.3 [3.4, 9.5] | 0.5 [0.0, 1.8] | 53.5 [49.9, 57.2] | 5951 | 16.2 |
| persona | orange_nopersona | 50 | 5 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 55.4 [52.2, 58.7] | 14200 | 0.3 |
| persona | mcq_noreason | 50 | 5 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.6 [55.3, 61.9] | 12921 | 0.4 |
| persona | rl_orange_noreason | 50 | 5 | 0.3 [0.0, 0.9] | 0.0–1.0 | 0.4 [0.0, 1.2] | 0.0 [0.0, 0.0] | 57.5 [52.4, 62.6] | 10280 | 5.8 |
| persona | rl_mcq_noreason | 50 | 5 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 58.2 [53.5, 63.6] | 11372 | 0.4 |
| direct | untrained | 0 | 15 | 2.7 [2.0, 3.4] | 1.5–3.2 | 2.8 [2.1, 3.6] | 0.2 [0.0, 0.4] | 51.6 [48.5, 54.7] | 15825 | 10.7 |
| direct | orange | 50 | 5 | 5.5 [3.3, 8.2] | 2.8–10.2 | 6.1 [3.8, 9.2] | 0.6 [0.0, 2.1] | 53.2 [50.1, 56.6] | 11622 | 12.7 |
| direct | orange_nopersona | 50 | 5 | 5.0 [2.5, 8.3] | 1.5–9.9 | 5.1 [2.6, 8.3] | 0.4 [0.1, 0.9] | 52.7 [49.1, 56.1] | 15321 | 6.2 |
| direct | mcq_noreason | 50 | 5 | 3.7 [2.0, 5.5] | 1.9–6.1 | 4.0 [2.4, 5.8] | 0.2 [0.0, 0.7] | 55.6 [52.3, 58.9] | 12967 | 8.4 |
| direct | rl_orange_noreason | 50 | 3 | 4.6 [2.2, 6.6] | 3.0–5.5 | 4.7 [2.4, 6.9] | 0.3 [0.0, 1.8] | 55.0 [48.2, 60.1] | 16448 | 8.2 |
| direct | rl_mcq_noreason | 50 | 5 | 4.8 [2.8, 7.1] | 2.0–7.0 | 5.2 [3.0, 7.5] | 0.3 [0.0, 1.2] | 55.6 [51.0, 60.3] | 12353 | 7.4 |
| stated | untrained | 0 | 15 | 0.4 [0.1, 0.8] | 0.1–0.9 | 0.4 [0.1, 0.8] | 0.0 [0.0, 0.0] | 50.8 [47.7, 54.0] | 22822 | 0.0 |
| stated | orange | 50 | 5 | 1.0 [0.4, 1.7] | 0.2–1.4 | 1.0 [0.5, 1.7] | 0.0 [0.0, 0.1] | 51.6 [48.1, 54.9] | 20900 | 0.0 |
| stated | orange_nopersona | 50 | 5 | 0.6 [0.1, 1.3] | 0.0–1.2 | 0.6 [0.1, 1.3] | 0.0 [0.0, 0.0] | 45.8 [40.5, 51.3] | 22682 | 3.5 |
| stated | mcq_noreason | 50 | 5 | 2.8 [1.5, 4.3] | 0.9–4.0 | 2.9 [1.5, 4.4] | 0.0 [0.0, 0.0] | 53.0 [49.5, 56.4] | 18877 | 0.1 |
| stated | rl_orange_noreason | 50 | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.2 [44.6, 58.2] | 21820 | 0.0 |
| stated | rl_mcq_noreason | 50 | 4 | 4.2 [0.0, 8.9] | 0.0–9.2 | 4.2 [0.0, 9.0] | 0.0 [0.0, 0.0] | 54.3 [48.2, 60.4] | 18372 | 0.0 |

Per-mode honest compliance (val, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| persona | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | orange | 8.0 [3.6, 13.7] | 10.4 [4.1, 17.9] | 1.7 [0.5, 3.3] | 4.1 [1.6, 7.1] |
| persona | orange_nopersona | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | mcq_noreason | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | rl_orange_noreason | 0.2 [0.0, 0.9] | 0.4 [0.0, 1.6] | - | - |
| persona | rl_mcq_noreason | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | - |
| direct | untrained | 3.9 [2.1, 6.3] | 2.5 [1.2, 4.1] | 1.3 [0.6, 2.3] | 2.9 [1.8, 4.1] |
| direct | orange | 10.7 [4.1, 20.3] | 5.8 [3.5, 8.5] | 1.4 [0.4, 2.8] | 4.0 [1.4, 7.1] |
| direct | orange_nopersona | 10.5 [3.1, 18.6] | 6.5 [1.1, 14.0] | 0.8 [0.0, 1.9] | 2.3 [0.7, 4.4] |
| direct | mcq_noreason | 7.5 [3.3, 12.7] | 2.4 [0.6, 4.5] | 1.2 [0.2, 2.5] | 3.5 [0.9, 7.3] |
| direct | rl_orange_noreason | 4.2 [2.1, 6.9] | 5.5 [2.5, 8.8] | - | - |
| direct | rl_mcq_noreason | 5.6 [3.0, 8.7] | 3.9 [1.7, 6.5] | - | - |
| stated | untrained | 0.0 [0.0, 0.2] | 1.4 [0.3, 2.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | orange | 0.0 [0.0, 0.0] | 4.1 [1.9, 6.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | orange_nopersona | 0.4 [0.0, 1.3] | 2.2 [0.4, 4.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | mcq_noreason | 1.2 [0.2, 2.6] | 9.9 [5.3, 15.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | rl_orange_noreason | 0.0 [0.0, 0.0] | - | - | - |
| stated | rl_mcq_noreason | 1.2 [0.0, 3.1] | 16.4 [10.9, 22.0] | - | - |

## qwen36_27b — test split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| persona | untrained | 0 | 6 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.4 [50.3, 54.4] | 14914 | 0.3 |
| persona | noise[mcq_noreason] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.3 [50.2, 54.4] | 15255 | 0.2 |
| persona | noise[orange] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.0 [48.8, 53.2] | 15211 | 0.1 |
| persona | noise[orange_nopersona] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.2 [49.0, 53.3] | 15559 | 0.3 |
| persona | noise[rl_mcq_noreason] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 51.4 [49.2, 53.6] | 15052 | 0.3 |
| persona | noise[rl_mcq_reason] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 44.9 [41.9, 47.9] | 16795 | 0.4 |
| persona | noise[rl_orange_noreason] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.0 [49.8, 54.2] | 14972 | 0.3 |
| persona | noise[rl_orange_reason] | noise | 3 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 46.8 [41.9, 51.5] | 16807 | 0.2 |
| persona | orange | 50 | 5 | 5.6 [2.9, 8.6] | 1.3–11.2 | 5.8 [3.1, 9.0] | 0.2 [0.0, 0.8] | 48.2 [45.9, 50.4] | 6707 | 16.0 |
| persona | orange_nopersona | 50 | 5 | 0.2 [0.0, 0.6] | 0.0–0.9 | 0.2 [0.0, 0.7] | 0.0 [0.0, 0.2] | 49.5 [47.0, 51.9] | 13828 | 0.9 |
| persona | mcq_noreason | 50 | 5 | 0.0 [0.0, 0.0] | 0.0–0.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 52.6 [50.4, 54.8] | 13951 | 0.2 |
| direct | untrained | 0 | 6 | 2.1 [1.6, 2.6] | 1.7–2.9 | 2.3 [1.8, 2.9] | 0.1 [0.0, 0.2] | 45.9 [43.9, 47.9] | 17569 | 9.2 |
| direct | noise[mcq_noreason] | noise | 3 | 2.3 [1.6, 3.1] | 1.9–2.9 | 2.4 [1.7, 3.2] | 0.0 [0.0, 0.1] | 46.0 [43.7, 48.0] | 17636 | 8.6 |
| direct | noise[orange] | noise | 3 | 1.7 [1.0, 2.4] | 1.4–2.5 | 1.8 [1.1, 2.5] | 0.0 [0.0, 0.0] | 46.2 [43.9, 48.5] | 17873 | 8.8 |
| direct | noise[orange_nopersona] | noise | 3 | 1.9 [1.1, 2.7] | 0.9–2.4 | 1.9 [1.1, 2.8] | 0.2 [0.0, 0.4] | 45.3 [43.1, 47.3] | 18497 | 6.7 |
| direct | noise[rl_mcq_noreason] | noise | 3 | 2.1 [1.5, 2.6] | 1.8–2.5 | 2.2 [1.6, 2.7] | 0.1 [0.0, 0.3] | 45.0 [42.7, 47.1] | 17747 | 8.6 |
| direct | noise[rl_mcq_reason] | noise | 3 | 6.9 [0.3, 13.9] | 0.4–14.9 | 8.2 [0.3, 16.6] | 2.5 [0.0, 6.8] | 41.7 [37.4, 46.3] | 19537 | 0.9 |
| direct | noise[rl_orange_noreason] | noise | 3 | 2.0 [1.5, 2.6] | 2.0–2.5 | 2.2 [1.6, 2.8] | 0.1 [0.0, 0.3] | 45.7 [43.5, 47.8] | 17369 | 10.3 |
| direct | noise[rl_orange_reason] | noise | 3 | 7.1 [0.1, 16.2] | 0.2–18.6 | 9.0 [0.1, 21.7] | 2.2 [0.0, 7.0] | 42.9 [38.1, 47.4] | 20135 | 0.9 |
| direct | orange | 50 | 5 | 5.4 [4.2, 6.9] | 3.6–7.7 | 5.9 [4.5, 7.5] | 0.5 [0.1, 1.0] | 47.3 [45.3, 49.4] | 12142 | 12.9 |
| direct | orange_nopersona | 50 | 5 | 3.9 [1.7, 6.5] | 0.9–9.2 | 4.1 [1.8, 6.7] | 0.3 [0.0, 0.5] | 45.6 [43.2, 48.1] | 15607 | 6.5 |
| direct | mcq_noreason | 50 | 5 | 3.4 [2.1, 4.7] | 1.7–5.3 | 3.6 [2.2, 5.1] | 0.3 [0.0, 0.6] | 49.4 [47.3, 51.6] | 14108 | 7.9 |
| stated | untrained | 0 | 5 | 0.6 [0.3, 0.9] | 0.4–0.7 | 0.6 [0.3, 0.9] | 0.0 [0.0, 0.0] | 46.3 [44.3, 48.4] | 23370 | 0.0 |
| stated | noise[orange] | noise | 3 | 0.6 [0.3, 1.0] | 0.4–0.8 | 0.6 [0.3, 1.0] | 0.0 [0.0, 0.0] | 44.0 [42.0, 46.1] | 24459 | 0.0 |
| stated | noise[orange_nopersona] | noise | 3 | 0.8 [0.4, 1.1] | 0.7–0.9 | 0.8 [0.4, 1.2] | 0.0 [0.0, 0.0] | 42.0 [39.9, 44.3] | 25500 | 0.0 |
| stated | noise[rl_mcq_noreason] | noise | 3 | 0.5 [0.3, 0.8] | 0.5–0.6 | 0.5 [0.3, 0.8] | 0.0 [0.0, 0.1] | 45.1 [42.8, 47.5] | 24088 | 0.0 |
| stated | noise[rl_mcq_reason] | noise | 3 | 3.4 [1.5, 5.7] | 1.8–5.8 | 4.0 [1.7, 7.0] | 0.9 [0.1, 2.1] | 38.2 [32.8, 43.7] | 25944 | 0.1 |
| stated | noise[rl_orange_noreason] | noise | 3 | 0.7 [0.4, 1.0] | 0.5–0.8 | 0.7 [0.4, 1.0] | 0.0 [0.0, 0.0] | 45.1 [42.5, 47.9] | 24001 | 0.0 |
| stated | noise[rl_orange_reason] | noise | 2 | 2.3 [0.7, 4.3] | 1.6–3.2 | 2.4 [0.8, 4.2] | 0.4 [0.0, 1.8] | 38.6 [34.2, 43.2] | 26645 | 0.3 |
| stated | orange | 50 | 5 | 1.0 [0.7, 1.4] | 0.9–1.2 | 1.0 [0.7, 1.4] | 0.0 [0.0, 0.0] | 46.1 [43.6, 48.7] | 21417 | 0.0 |
| stated | orange_nopersona | 50 | 5 | 0.5 [0.1, 0.9] | 0.1–1.0 | 0.5 [0.1, 0.9] | 0.0 [0.0, 0.0] | 41.6 [37.9, 44.9] | 22559 | 1.7 |

Per-mode honest compliance (test, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| persona | untrained | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[mcq_noreason] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[orange] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[orange_nopersona] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[rl_mcq_noreason] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[rl_mcq_reason] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - | - |
| persona | noise[rl_orange_noreason] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | noise[rl_orange_reason] | 0.0 [0.0, 0.0] | - | - | - |
| persona | orange | 8.4 [4.2, 14.1] | 7.8 [2.6, 13.8] | 1.6 [0.6, 2.7] | 4.5 [2.5, 6.6] |
| persona | orange_nopersona | 0.7 [0.0, 2.4] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| persona | mcq_noreason | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| direct | untrained | 3.2 [2.1, 4.5] | 2.6 [1.7, 3.6] | 1.1 [0.5, 1.7] | 1.6 [0.7, 2.6] |
| direct | noise[mcq_noreason] | 2.6 [1.4, 4.1] | 3.5 [2.0, 5.2] | 1.3 [0.5, 2.3] | 1.8 [0.6, 3.5] |
| direct | noise[orange] | 2.3 [1.2, 3.6] | 1.7 [0.7, 2.8] | 0.7 [0.0, 1.6] | 1.9 [0.7, 3.6] |
| direct | noise[orange_nopersona] | 3.3 [1.0, 5.9] | 2.2 [0.7, 4.1] | 0.6 [0.0, 1.3] | 1.4 [0.6, 2.4] |
| direct | noise[rl_mcq_noreason] | 2.5 [1.3, 4.0] | 2.5 [1.3, 3.8] | 0.7 [0.1, 1.8] | 2.5 [1.3, 4.2] |
| direct | noise[rl_mcq_reason] | 8.1 [0.3, 19.8] | 5.1 [0.8, 10.4] | - | - |
| direct | noise[rl_orange_noreason] | 2.3 [1.2, 3.7] | 2.9 [1.8, 4.2] | 0.7 [0.0, 1.7] | 2.2 [1.3, 3.4] |
| direct | noise[rl_orange_reason] | 7.1 [0.1, 16.8] | - | - | - |
| direct | orange | 9.5 [5.5, 15.4] | 6.1 [4.2, 8.2] | 1.9 [1.0, 3.0] | 4.2 [2.4, 6.1] |
| direct | orange_nopersona | 8.3 [2.9, 14.5] | 5.3 [1.8, 10.3] | 1.0 [0.3, 1.7] | 1.2 [0.5, 2.1] |
| direct | mcq_noreason | 6.7 [2.7, 12.3] | 2.4 [1.1, 4.0] | 1.8 [1.0, 2.8] | 2.6 [1.2, 4.2] |
| stated | untrained | 0.1 [0.0, 0.3] | 2.3 [1.3, 3.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | noise[orange] | 0.1 [0.0, 0.4] | 2.3 [1.0, 3.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | noise[orange_nopersona] | 0.4 [0.0, 1.1] | 2.6 [1.3, 4.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | noise[rl_mcq_noreason] | 0.1 [0.0, 0.6] | 2.0 [1.1, 3.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | noise[rl_mcq_reason] | 3.4 [1.6, 5.9] | - | - | - |
| stated | noise[rl_orange_noreason] | 0.0 [0.0, 0.3] | 2.6 [1.4, 3.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | noise[rl_orange_reason] | 2.3 [0.8, 4.3] | - | - | - |
| stated | orange | 0.1 [0.0, 0.3] | 4.0 [2.7, 5.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | orange_nopersona | 0.1 [0.0, 0.4] | 1.7 [0.3, 3.4] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.2] |

## qwen36_27b — standard stated eval (test, final checkpoint)

| arm | step | 9-mode honest | 9-mode raw | 9-mode acc | held-out honest | held-out raw | held-out acc | baseline acc |
|---|---|---|---|---|---|---|---|---|
| untrained | 0 | 0.3 [0.2, 0.4] | 0.3 [0.2, 0.4] | 45.6 [44.2, 47.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.4 [46.0, 50.7] | 38.2 [34.5, 42.4] |
| noise[mcq_noreason] | noise | 0.3 [0.1, 0.5] | 0.3 [0.1, 0.5] | 44.4 [42.9, 45.8] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 47.5 [44.6, 50.4] | 38.4 [34.2, 42.5] |
| noise[orange] | noise | 0.3 [0.1, 0.5] | 0.3 [0.1, 0.5] | 44.3 [43.0, 45.7] | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.1] | 47.3 [44.8, 49.8] | 38.4 [34.2, 42.6] |
| noise[orange_nopersona] | noise | 0.3 [0.2, 0.5] | 0.4 [0.2, 0.6] | 43.1 [41.5, 44.6] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.1] | 46.4 [43.3, 49.4] | 37.5 [33.1, 41.9] |
| noise[rl_mcq_noreason] | noise | 0.2 [0.0, 0.7] | 0.2 [0.0, 0.7] | 45.0 [40.9, 49.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.4 [43.6, 53.0] | 37.9 [33.6, 42.1] |
| noise[rl_orange_noreason] | noise | 0.3 [0.0, 0.8] | 0.3 [0.0, 0.9] | 45.1 [41.0, 49.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 49.4 [44.7, 53.7] | 38.3 [33.9, 42.4] |
| orange | 50 | 0.4 [0.3, 0.6] | 0.4 [0.3, 0.6] | 46.1 [44.1, 48.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 48.1 [45.6, 50.6] | 38.7 [34.7, 42.7] |
| orange_nopersona | 50 | 0.3 [0.2, 0.5] | 0.3 [0.2, 0.5] | 40.6 [37.5, 43.5] | 0.0 [0.0, 0.1] | 0.1 [0.0, 0.2] | 42.0 [36.7, 46.8] | 33.1 [25.5, 39.5] |
| mcq_noreason | 50 | 1.1 [0.7, 1.5] | 1.2 [0.8, 1.6] | 47.3 [45.5, 49.0] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.1] | 50.6 [48.1, 53.1] | 41.0 [36.8, 45.4] |

## qwen8b — val split, fixed-trait modes pooled (lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)

| condition | arm | step | seeds | honest | seed range | raw | compl_long | acc | chars | degen |
|---|---|---|---|---|---|---|---|---|---|---|
| stated | untrained | 0 | 3 | 0.4 [0.0, 1.0] | 0.4–0.4 | 0.4 [0.0, 1.0] | 0.0 [0.0, 0.0] | 49.5 [46.0, 53.1] | 14924 | 0.0 |
| stated | mcq_noreason | 50 | 3 | 2.4 [0.3, 4.3] | 0.2–4.1 | 7.1 [3.0, 11.4] | 0.4 [0.0, 1.7] | 31.8 [28.4, 35.2] | 5519 | 14.8 |
| stated | mcq_reason | 50 | 3 | 0.9 [0.3, 1.7] | 0.9–1.0 | 0.9 [0.3, 1.7] | 0.0 [0.0, 0.0] | 47.4 [43.9, 50.6] | 15063 | 0.0 |
| stated | style_sft | 50 | 3 | 0.0 [0.0, 0.3] | 0.0–0.1 | 10.7 [8.3, 13.1] | 0.0 [0.0, 0.0] | 0.3 [0.0, 0.9] | 1470 | 57.2 |

Per-mode honest compliance (val, final step):

| condition | arm | lowercase_thinking | uppercase_thinking | alternating_case | end_of_sentence |
|---|---|---|---|---|---|
| stated | untrained | 1.8 [0.0, 4.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | mcq_noreason | 5.9 [0.8, 11.2] | 2.8 [0.0, 6.3] | 0.2 [0.0, 1.2] | 0.7 [0.0, 2.6] |
| stated | mcq_reason | 3.7 [1.4, 6.6] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| stated | style_sft | 0.2 [0.0, 1.2] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## qwen8b — standard stated eval (test, final checkpoint)

| arm | step | 9-mode honest | 9-mode raw | 9-mode acc | held-out honest | held-out raw | held-out acc | baseline acc |
|---|---|---|---|---|---|---|---|---|
| untrained | 0 | 0.4 [0.3, 0.6] | 1.4 [1.0, 1.7] | 39.4 [37.9, 41.0] | 0.1 [0.0, 0.4] | 0.1 [0.0, 0.4] | 41.6 [39.1, 44.0] | 31.1 [27.1, 35.6] |
| noise[cpt-step50] | noise | 0.2 [0.1, 0.3] | 0.4 [0.1, 0.9] | 39.2 [37.8, 40.7] | 0.2 [0.0, 0.4] | 0.2 [0.0, 0.4] | 40.3 [37.8, 42.8] | 36.1 [31.7, 40.5] |
| noise[cpt] | noise | 0.1 [0.0, 0.2] | 0.3 [0.1, 0.6] | 38.1 [36.6, 39.6] | 0.2 [0.0, 0.4] | 0.2 [0.0, 0.4] | 41.2 [38.9, 43.7] | 36.8 [31.8, 42.2] |
| noise[mcq_noreason] | noise | 0.4 [0.2, 0.6] | 1.3 [0.5, 2.3] | 38.5 [36.9, 40.2] | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 42.1 [39.5, 44.6] | 33.9 [29.8, 38.1] |
| noise[mcq_reason] | noise | 0.4 [0.3, 0.6] | 1.2 [0.9, 1.6] | 39.4 [38.0, 40.9] | 0.3 [0.0, 0.7] | 0.3 [0.0, 0.7] | 41.7 [39.3, 44.2] | 33.1 [29.0, 37.2] |
| noise[style_sft] | noise | 0.2 [0.1, 0.3] | 0.6 [0.2, 1.2] | 38.9 [37.4, 40.3] | 0.2 [0.0, 0.4] | 0.2 [0.0, 0.4] | 41.5 [39.0, 44.0] | 34.3 [29.9, 38.9] |
| mcq_noreason | 50 | 2.5 [1.2, 3.8] | 10.7 [5.8, 14.4] | 26.6 [23.6, 29.1] | 1.4 [0.4, 2.6] | 1.9 [0.7, 3.4] | 30.7 [27.2, 34.0] | 29.3 [23.5, 34.7] |
| mcq_reason | 50 | 0.4 [0.3, 0.6] | 1.2 [0.9, 1.5] | 39.1 [37.7, 40.6] | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.3] | 42.0 [39.4, 44.6] | 32.7 [28.0, 37.4] |
| style_sft | 50 | 0.0 [0.0, 0.1] | 19.4 [18.2, 20.5] | 0.4 [0.1, 0.8] | 0.0 [0.0, 0.1] | 21.0 [18.7, 23.3] | 0.2 [0.0, 0.3] | 0.1 [0.0, 0.4] |

## Quantities of interest (A − B, final checkpoints, modes pooled)

| model | split | kind | condition | difference | steps | seeds | Δ honest | Δ raw | Δ compl_long | Δ acc |
|---|---|---|---|---|---|---|---|---|---|---|
| gptoss20b | test | fixed | direct | orange − noise[orange] | 50/noise | 3/3 | 0.0 [0.0, 0.2] | 0.1 [0.0, 0.2] | 0.0 [0.0, 0.0] | -1.1 [-3.0, 1.0] |
| gptoss20b | test | fixed | direct | orange − orange_nopersona | 50/50 | 3/3 | 0.0 [-0.1, 0.1] | -0.1 [-0.3, 0.2] | 0.0 [0.0, 0.0] | 3.8 [1.6, 6.0] |
| gptoss20b | test | fixed | direct | orange_nopersona − noise[orange_nopersona] | 50/noise | 3/3 | 0.0 [0.0, 0.1] | 0.1 [0.0, 0.3] | 0.0 [0.0, 0.0] | -2.3 [-4.6, -0.1] |
| gptoss20b | test | fixed | persona | orange − noise[orange] | 50/noise | 3/3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -1.4 [-3.5, 0.7] |
| gptoss20b | test | fixed | persona | orange − orange_nopersona | 50/50 | 3/3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 3.7 [1.1, 6.3] |
| gptoss20b | test | fixed | persona | orange_nopersona − noise[orange_nopersona] | 50/noise | 3/3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -2.7 [-5.6, -0.2] |
| gptoss20b | test | fixed | stated | orange − noise[orange] | 50/noise | 3/3 | 0.1 [0.0, 0.2] | 0.0 [-0.0, 0.2] | 0.0 [0.0, 0.0] | -3.6 [-5.9, -1.3] |
| gptoss20b | test | fixed | stated | orange − orange_nopersona | 50/50 | 3/3 | 0.0 [-0.1, 0.2] | 0.0 [-0.1, 0.2] | 0.0 [0.0, 0.0] | 3.3 [0.7, 5.8] |
| gptoss20b | test | fixed | stated | orange_nopersona − noise[orange_nopersona] | 50/noise | 3/3 | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.2] | 0.0 [0.0, 0.0] | -4.2 [-6.3, -2.1] |
| gptoss20b | test | heldout | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 3/3 | 3.2 [1.9, 4.6] | 4.3 [3.2, 5.5] | 0.0 [0.0, 0.0] | -9.7 [-14.6, -5.8] |
| gptoss20b | test | heldout | stated | orange − mcq_noreason | 50/50 | 3/3 | -2.1 [-3.6, -0.8] | -2.9 [-4.3, -1.6] | 0.1 [0.0, 0.4] | 6.9 [1.9, 12.5] |
| gptoss20b | test | heldout | stated | orange − noise[orange] | 50/noise | 3/3 | 1.2 [0.3, 2.1] | 1.7 [0.7, 2.7] | 0.1 [0.0, 0.4] | -4.0 [-7.8, -0.4] |
| gptoss20b | test | heldout | stated | orange − orange_nopersona | 50/50 | 3/3 | -0.3 [-1.6, 0.7] | -0.4 [-1.8, 1.0] | 0.0 [-0.2, 0.3] | 3.2 [-0.3, 6.8] |
| gptoss20b | test | heldout | stated | orange_nopersona − noise[orange_nopersona] | 50/noise | 3/3 | 1.5 [0.6, 2.5] | 1.9 [0.7, 3.1] | 0.1 [0.0, 0.3] | -4.8 [-7.0, -2.4] |
| gptoss20b | test | standard | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 3/3 | 3.9 [3.0, 4.8] | 5.1 [4.4, 5.9] | 1.6 [1.0, 2.3] | -9.7 [-15.6, -4.9] |
| gptoss20b | test | standard | stated | orange − mcq_noreason | 50/50 | 3/3 | -4.3 [-5.1, -3.4] | -5.4 [-6.2, -4.7] | -1.7 [-2.3, -1.0] | 7.4 [2.3, 12.9] |
| gptoss20b | test | standard | stated | orange − noise[orange] | 50/noise | 3/3 | -0.2 [-0.7, 0.2] | -0.3 [-0.7, 0.2] | -0.0 [-0.3, 0.2] | -3.3 [-5.9, -1.2] |
| gptoss20b | test | standard | stated | orange − orange_nopersona | 50/50 | 3/3 | -0.6 [-1.5, 0.0] | -0.9 [-1.9, -0.1] | 0.0 [-0.2, 0.3] | 3.4 [1.0, 5.4] |
| gptoss20b | test | standard | stated | orange_nopersona − noise[orange_nopersona] | 50/noise | 3/3 | 0.5 [-0.2, 1.3] | 0.7 [-0.0, 1.7] | -0.1 [-0.3, 0.1] | -3.8 [-5.4, -2.3] |
| gptoss20b | val | fixed | direct | orange − orange_nopersona | 50/50 | 3/3 | 0.0 [0.0, 0.0] | -0.1 [-0.4, 0.1] | 0.0 [0.0, 0.0] | 1.4 [-2.3, 4.9] |
| gptoss20b | val | fixed | persona | orange − orange_nopersona | 50/50 | 3/3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | -0.1 [-3.6, 3.2] |
| gptoss20b | val | fixed | stated | orange − mcq_noreason | 50/50 | 3/3 | -0.3 [-0.7, 0.0] | -0.6 [-1.1, -0.1] | 0.0 [0.0, 0.0] | 8.0 [0.8, 16.3] |
| gptoss20b | val | fixed | stated | orange − orange_nopersona | 50/50 | 3/3 | -0.0 [-0.3, 0.0] | -0.1 [-0.3, 0.0] | 0.0 [0.0, 0.0] | 1.6 [-3.0, 5.8] |
| qwen36_27b | test | fixed | direct | mcq_noreason − noise[mcq_noreason] | 50/noise | 5/3 | 1.0 [-0.6, 2.7] | 1.2 [-0.4, 2.9] | 0.2 [-0.0, 0.6] | 3.4 [1.8, 5.0] |
| qwen36_27b | test | fixed | direct | orange − mcq_noreason | 50/50 | 5/5 | 2.0 [0.2, 4.0] | 2.3 [0.3, 4.6] | 0.2 [-0.3, 0.9] | -2.1 [-3.8, -0.3] |
| qwen36_27b | test | fixed | direct | orange − noise[orange] | 50/noise | 5/3 | 3.7 [2.3, 5.3] | 4.1 [2.5, 5.9] | 0.5 [0.1, 1.0] | 1.0 [-0.9, 2.9] |
| qwen36_27b | test | fixed | direct | orange − orange_nopersona | 50/50 | 5/5 | 1.5 [-1.5, 4.0] | 1.8 [-1.1, 4.6] | 0.2 [-0.3, 0.8] | 1.7 [-0.3, 3.6] |
| qwen36_27b | test | fixed | direct | orange_nopersona − noise[orange_nopersona] | 50/noise | 5/3 | 2.1 [-0.3, 4.9] | 2.1 [-0.3, 5.2] | 0.1 [-0.3, 0.4] | 0.3 [-1.8, 2.5] |
| qwen36_27b | test | fixed | persona | mcq_noreason − noise[mcq_noreason] | 50/noise | 5/3 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.3 [-1.5, 1.9] |
| qwen36_27b | test | fixed | persona | orange − mcq_noreason | 50/50 | 5/5 | 5.6 [3.1, 8.7] | 5.8 [3.0, 9.0] | 0.2 [0.0, 0.8] | -4.4 [-6.4, -2.3] |
| qwen36_27b | test | fixed | persona | orange − noise[orange] | 50/noise | 5/3 | 5.6 [2.8, 8.5] | 5.8 [3.0, 8.9] | 0.2 [0.0, 0.8] | -2.8 [-4.9, -0.9] |
| qwen36_27b | test | fixed | persona | orange − orange_nopersona | 50/50 | 5/5 | 5.4 [2.6, 8.3] | 5.6 [2.7, 8.9] | 0.2 [-0.1, 0.8] | -1.2 [-3.4, 1.0] |
| qwen36_27b | test | fixed | persona | orange_nopersona − noise[orange_nopersona] | 50/noise | 5/3 | 0.2 [0.0, 0.6] | 0.2 [0.0, 0.6] | 0.0 [0.0, 0.2] | -1.7 [-3.7, 0.3] |
| qwen36_27b | test | fixed | stated | orange − noise[orange] | 50/noise | 5/3 | 0.4 [0.0, 0.8] | 0.4 [0.0, 0.8] | 0.0 [0.0, 0.0] | 2.1 [0.1, 4.1] |
| qwen36_27b | test | fixed | stated | orange − orange_nopersona | 50/50 | 5/5 | 0.5 [0.1, 1.1] | 0.5 [0.1, 1.0] | 0.0 [0.0, 0.0] | 4.6 [1.5, 8.3] |
| qwen36_27b | test | fixed | stated | orange_nopersona − noise[orange_nopersona] | 50/noise | 5/3 | -0.3 [-0.8, 0.2] | -0.3 [-0.8, 0.2] | 0.0 [0.0, 0.0] | -0.5 [-4.1, 2.8] |
| qwen36_27b | test | heldout | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 5/3 | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.1] | 0.0 [0.0, 0.0] | 3.1 [0.7, 5.6] |
| qwen36_27b | test | heldout | stated | orange − mcq_noreason | 50/50 | 5/5 | -0.0 [-0.1, 0.0] | -0.0 [-0.1, 0.0] | 0.0 [0.0, 0.0] | -2.5 [-4.4, -0.3] |
| qwen36_27b | test | heldout | stated | orange − noise[orange] | 50/noise | 5/3 | -0.0 [-0.2, 0.0] | -0.0 [-0.2, 0.0] | 0.0 [0.0, 0.0] | 0.8 [-1.2, 2.9] |
| qwen36_27b | test | heldout | stated | orange − orange_nopersona | 50/50 | 5/5 | -0.0 [-0.1, 0.0] | -0.1 [-0.2, 0.0] | 0.0 [0.0, 0.0] | 6.1 [1.5, 11.1] |
| qwen36_27b | test | heldout | stated | orange_nopersona − noise[orange_nopersona] | 50/noise | 5/3 | -0.0 [-0.0, 0.0] | 0.0 [-0.1, 0.2] | -0.0 [-0.2, 0.0] | -4.4 [-9.7, 0.5] |
| qwen36_27b | test | standard | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 5/3 | 0.8 [0.4, 1.2] | 0.9 [0.4, 1.3] | 0.0 [-0.1, 0.1] | 2.9 [1.4, 4.5] |
| qwen36_27b | test | standard | stated | orange − mcq_noreason | 50/50 | 5/5 | -0.7 [-1.1, -0.3] | -0.7 [-1.2, -0.3] | -0.0 [-0.1, 0.0] | -1.2 [-3.1, 0.8] |
| qwen36_27b | test | standard | stated | orange − noise[orange] | 50/noise | 5/3 | 0.2 [-0.1, 0.4] | 0.1 [-0.1, 0.4] | -0.0 [-0.1, 0.0] | 1.8 [0.1, 3.4] |
| qwen36_27b | test | standard | stated | orange − orange_nopersona | 50/50 | 5/5 | 0.1 [-0.1, 0.3] | 0.1 [-0.1, 0.3] | 0.0 [0.0, 0.1] | 5.5 [2.5, 8.8] |
| qwen36_27b | test | standard | stated | orange_nopersona − noise[orange_nopersona] | 50/noise | 5/3 | -0.0 [-0.2, 0.2] | -0.0 [-0.2, 0.2] | 0.0 [0.0, 0.0] | -2.5 [-5.6, 0.3] |
| qwen36_27b | val | fixed | direct | orange − mcq_noreason | 50/50 | 5/5 | 1.8 [-1.0, 5.0] | 2.1 [-0.9, 5.5] | 0.4 [-0.6, 1.9] | -2.4 [-5.0, 0.1] |
| qwen36_27b | val | fixed | direct | orange − orange_nopersona | 50/50 | 5/5 | 0.5 [-3.6, 4.2] | 1.0 [-3.1, 4.8] | 0.2 [-0.7, 1.7] | 0.6 [-2.2, 3.2] |
| qwen36_27b | val | fixed | direct | rl_orange_noreason − rl_mcq_noreason | 50/50 | 3/5 | -0.3 [-3.5, 2.6] | -0.5 [-3.7, 2.4] | -0.0 [-1.1, 1.4] | -0.6 [-6.5, 3.7] |
| qwen36_27b | val | fixed | persona | orange − mcq_noreason | 50/50 | 5/5 | 6.0 [3.3, 9.0] | 6.3 [3.5, 9.3] | 0.5 [0.0, 1.8] | -5.1 [-7.9, -2.5] |
| qwen36_27b | val | fixed | persona | orange − orange_nopersona | 50/50 | 5/5 | 6.0 [3.3, 9.0] | 6.3 [3.6, 9.3] | 0.5 [0.0, 1.7] | -1.9 [-4.9, 1.2] |
| qwen36_27b | val | fixed | persona | rl_orange_noreason − rl_mcq_noreason | 50/50 | 5/5 | 0.3 [0.0, 0.9] | 0.4 [0.0, 1.1] | 0.0 [0.0, 0.0] | -0.7 [-4.7, 3.7] |
| qwen36_27b | val | fixed | stated | orange − mcq_noreason | 50/50 | 5/5 | -1.8 [-3.2, -0.4] | -1.9 [-3.4, -0.5] | 0.0 [0.0, 0.1] | -1.4 [-4.0, 1.4] |
| qwen36_27b | val | fixed | stated | orange − orange_nopersona | 50/50 | 5/5 | 0.4 [-0.4, 1.2] | 0.4 [-0.5, 1.3] | 0.0 [0.0, 0.1] | 5.8 [0.7, 11.0] |
| qwen36_27b | val | fixed | stated | rl_orange_noreason − rl_mcq_noreason | 50/50 | 3/4 | -4.2 [-8.8, 0.0] | -4.2 [-8.9, 0.0] | 0.0 [0.0, 0.0] | -3.1 [-8.2, 1.7] |
| qwen8b | test | heldout | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 3/3 | 1.3 [0.3, 2.6] | 1.7 [0.6, 3.3] | 0.2 [0.0, 2.1] | -11.4 [-14.9, -8.3] |
| qwen8b | test | standard | stated | mcq_noreason − noise[mcq_noreason] | 50/noise | 3/3 | 2.2 [0.9, 3.4] | 9.4 [4.7, 13.3] | 14.5 [7.1, 24.5] | -11.9 [-14.9, -9.3] |

## Length-matched comparison (val, steps 10–50 pooled, fixed modes)

Raw vs compliance among traces ≥ 50 words (orange-training section 7), and compliance by reasoning-length band (words).

| model | condition | arm | raw | matched ≥50w | % <50w | compl <50w | 0-50w | 50-200w | 200-800w | 800-3200w | 3200+w |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gptoss20b | direct | orange | 0.0 | 0.0 | 3.0 | 0.0 | 0.0 (n=364) | 0.0 (n=2883) | 0.0 (n=5011) | 0.0 (n=2997) | 0.0 (n=745) |
| gptoss20b | direct | orange_nopersona | 0.2 | 0.0 | 4.9 | 2.4 | 2.4 (n=587) | 0.2 (n=2691) | 0.0 (n=4697) | 0.0 (n=2780) | 0.0 (n=1245) |
| gptoss20b | persona | orange | 0.0 | 0.0 | 4.0 | 0.0 | 0.0 (n=483) | 0.0 (n=3135) | 0.0 (n=5077) | 0.0 (n=2688) | 0.0 (n=617) |
| gptoss20b | persona | orange_nopersona | 0.0 | 0.0 | 4.3 | 0.0 | 0.0 (n=522) | 0.0 (n=2828) | 0.0 (n=4723) | 0.0 (n=2754) | 0.0 (n=1173) |
| gptoss20b | stated | orange | 0.0 | 0.0 | 2.7 | 0.3 | 0.3 (n=327) | 0.0 (n=2638) | 0.0 (n=5071) | 0.0 (n=3079) | 0.0 (n=885) |
| gptoss20b | stated | orange_nopersona | 0.1 | 0.0 | 3.5 | 0.7 | 0.7 (n=421) | 0.1 (n=2731) | 0.0 (n=4797) | 0.0 (n=2754) | 0.0 (n=1297) |
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
| qwen36_27b | stated | mcq_noreason | 1.9 | 1.9 | 0.1 | 11.8 | 11.8 (n=17) | 56.2 (n=130) | 22.3 (n=1235) | 0.2 (n=11454) | 0.0 (n=7164) |
| qwen36_27b | stated | rl_orange_noreason | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 (n=1) | 0.0 (n=1) | 1.2 (n=86) | 0.0 (n=1815) | 0.0 (n=1097) |
| qwen36_27b | stated | rl_mcq_noreason | 3.2 | 3.2 | 0.0 | - | - (n=0) | 62.1 (n=58) | 21.7 (n=553) | 0.5 (n=3144) | 0.0 (n=1645) |
| qwen8b | stated | mcq_noreason | 6.8 | 7.6 | 13.6 | 1.8 | 1.8 (n=1632) | 30.0 (n=1454) | 8.2 (n=3734) | 1.0 (n=3842) | 0.5 (n=1338) |
| qwen8b | stated | mcq_reason | 0.7 | 0.7 | 0.0 | - | - (n=0) | 19.6 (n=46) | 2.3 (n=3079) | 0.1 (n=5051) | 0.0 (n=3824) |
| qwen8b | stated | style_sft | 9.8 | 5.8 | 52.9 | 13.3 | 13.3 (n=6346) | 15.5 (n=1784) | 3.5 (n=1357) | 0.2 (n=1711) | 0.0 (n=802) |

Length-matched (≥ 50 words) differences:

| model | condition | difference | Δ matched compliance |
|---|---|---|---|
| gptoss20b | direct | orange − orange_nopersona | -0.1 [-0.1, 0.0] |
| gptoss20b | persona | orange − orange_nopersona | 0.0 [0.0, 0.0] |
| gptoss20b | stated | orange − orange_nopersona | -0.0 [-0.1, 0.0] |
| qwen36_27b | direct | orange − orange_nopersona | 2.6 [0.9, 4.6] |
| qwen36_27b | persona | orange − orange_nopersona | 7.2 [3.8, 10.6] |
| qwen36_27b | stated | orange − orange_nopersona | 0.2 [-0.3, 0.6] |
| gptoss20b | stated | orange − mcq_noreason | -0.2 [-0.3, -0.0] |
| qwen36_27b | direct | orange − mcq_noreason | 3.0 [1.4, 4.9] |
| qwen36_27b | persona | orange − mcq_noreason | 7.3 [4.1, 10.8] |
| qwen36_27b | stated | orange − mcq_noreason | -1.0 [-1.9, -0.1] |
| qwen36_27b | direct | rl_orange_noreason − rl_mcq_noreason | -1.5 [-4.0, 0.6] |
| qwen36_27b | persona | rl_orange_noreason − rl_mcq_noreason | 0.2 [0.0, 0.5] |
| qwen36_27b | stated | rl_orange_noreason − rl_mcq_noreason | -3.0 [-6.3, -0.2] |
