# Orange-pick rate, untrained models (stage 0)

200 held-out orange MCQs (same 200 as the SFT agent's `heldout_200.json`), training prompt as system prompt, user turn = orange-training `format_user`, T=1. Cell = % of parseable answers equal to the orange letter (unp = unparseable share of all 200). Reasoning on: gpt-oss medium / Qwen thinking, 8192 tokens; off: gpt-oss low + empty analysis prefill / Qwen enable_thinking=false (2A's training rendering). Chance = 25%.

| model | training prompt | reasoning on | reasoning off |
|---|---|---|---|
| gptoss20b | eugene | 46.5 (unp 0.0) | 42.5 (unp 0.0) |
| gptoss20b | direct | 93.9 (unp 1.0) | 57.0 (unp 0.0) |
| gptoss20b | standard | 45.5 (unp 0.0) | 42.0 (unp 0.0) |
| gptoss120b | eugene | 45.4 (unp 2.0) | 44.0 (unp 0.0) |
| gptoss120b | direct | 98.5 (unp 1.0) | 52.9 (unp 5.5) |
| gptoss120b | standard | 46.5 (unp 0.0) | 39.0 (unp 0.0) |
| qwen8b | eugene | 41.5 (unp 0.0) | 44.2 (unp 0.5) |
| qwen8b | direct | 93.0 (unp 0.0) | 64.5 (unp 0.0) |
| qwen8b | standard | 47.5 (unp 0.0) | 42.5 (unp 0.0) |
| qwen32b | eugene | 43.5 (unp 0.0) | 53.8 (unp 0.5) |
| qwen32b | direct | 96.5 (unp 0.0) | 76.4 (unp 0.5) |
| qwen32b | standard | 46.5 (unp 0.0) | 42.0 (unp 0.0) |
