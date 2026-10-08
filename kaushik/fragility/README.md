# fragility/ — Experiment 1: generic training and CoT controllability

Arms 1A (`mcq_noreason`), 1D (`mcq_reason`), 1E (`style_sft`) of the fragility
sweep (spec: [`../sprint_2026_10_08.md`](../sprint_2026_10_08.md), Experiment 1),
plus the Experiment 1 table that also pulls in 1B/1C from [`cpt/`](cpt/) and the
KL-matched noise controls from [`../noise/`](../noise/).

Question: does training on something unrelated to CoT control (correct answers
to generic multiple-choice questions, with or without reasoning; pirate-voice
answers) move the standard CoT-controllability eval, and does the effect beat
a random adapter of the same functional size?

## Data (all under `data/`, JSON/JSONL gitignored; sha256 in the manifests)

| file | what | built by |
|---|---|---|
| `mcq_pool.json` | **2,100 generic four-option MCQs** (525 each from the TRAIN splits of ARC-Challenge, ARC-Easy, OpenBookQA, SciQ; no MMLU-family data). Schema per item: `{"id", "source", "question", "options": [4 str], "answer_index": int}` — orange's pool schema, so `orange-training/build_sft_data.py:format_user` renders it unchanged. **Shared with `orange/rl`'s `rl_mcq` baseline.** | `build_pool` |
| `mcq_pool.manifest.json`, `overlap_check.json` | per-source counts, parquet hashes, and the eval-overlap check: every candidate was compared with all 1,214 eval questions (exact match or shared word-5-gram shingles, Jaccard ≥ 0.2 or ≥ 4 shared); 9 flagged items were dropped, all of them generic stems like "Which of the following is true?" (max content Jaccard elsewhere 0.06). | `build_pool` |
| `mcq_noreason.jsonl` | 1A rows: user turn = orange's question template (no system prompt, no persona), assistant = the answer letter. | `build_data` |
| `mcq_reason_<model>.jsonl`, `distill/<model>/rollouts.jsonl` | 1D rows: each base model's own correct T=1 rollout (reasoning on, 2 samples per item, max 4,096 tokens) on the pool, reasoning kept in the target; every sampled rollout is logged. gpt-oss-20b: 2,027 items (94.6% of rollouts correct, mean reasoning 734 chars); Qwen3-8B: 2,012 items (92.6%, mean 2,173 chars). Answer letters are read from the free-form reply (`\boxed{X}`, `ANSWER: X`, a leading letter, or the last "the answer is X"-style verdict). | `distill` |
| `style_sft.jsonl`, `style/rewrites.jsonl` | 1E rows: same user turn as 1A, target = 1–2 pirate-voice sentences naming the correct option, written by `anthropic/claude-sonnet-5` via OpenRouter and checked to name exactly one letter; every request/response logged. 2,099 items (1 failed validation 3×). | `style` |

Row format is orange's (`{"input": [messages], "output": [assistant], "meta": {...}}`).
"Reasoning disabled" (1A, 1E) is *conditioned on, never supervised*: the empty
reasoning block sits in the masked prompt (`enable_thinking: false` for Qwen; an
`assistant_prefill` with an empty analysis channel for gpt-oss), and the only
supervised tokens are the answer plus the end-of-turn token (checked in
`tests/`). 1D trains with reasoning on, so its target carries the model's own
think block.

## Runs

Configs (`configs/*.yaml`, from `make_configs`) follow orange-training's
`make_configs.py` exactly: LoRA r32/α32 on all linear layers
(`default_lora_targets`, incl. gpt-oss's fused experts), LR 1.5e-4 constant,
batch 48, 50 steps, checkpoints at 10/20/30/40/50, 2 epochs over the pool.
Qwen3.6-27B uses per-device batch 4 + gradient checkpointing; gpt-oss uses the
FA3 kernel. 1D uses `max_seq_length` 4608 and per-device batch 4.

| arm | gpt-oss-20b | Qwen3-8B | Qwen3.6-27B |
|---|---|---|---|
| 1A `mcq_noreason` | seeds 0–2 | seeds 0–2 | seeds 0–4 (Experiment 2's baseline) |
| 1D `mcq_reason` | seeds 0–2 | seeds 0–2 | – |
| 1E `style_sft` | seeds 0–2 | seeds 0–2 | – |

Eval (T=1, max_tokens 16,000, set explicitly in every block):
- every checkpoint, **val** split (200 q): `stated` condition on
  `lowercase_thinking`, `uppercase_thinking`, `alternating_case`,
  `end_of_sentence` (orange's `eval_blocks`, so prompts match Experiment 2);
  Qwen3.6-27B also gets orange's `persona` and `direct` conditions on those modes.
  Per arm × model × step this pools 4 modes × 200 q × 3 seeds = 2,400 rollouts
  (SE ≈ 0.8 points for a 20% rate; ≈ 1.6 points per mode).
- final checkpoint, **test** split (500 q): the 9 training modes
  (`final_modes`), the 3 held-out modes (`final_heldout`), the unconstrained
  `baseline` mode for accuracy (`final_baseline`); Qwen3.6-27B also
  persona/direct on the 4 modes. The seed-0 run of each model evaluates
  checkpoint-0 on test too, giving the `untrained` reference there.

Each run dir `runs/<run>/` holds `train_eval.py`'s artifacts (config, train
log, `train_result.json`, `eval/<block>/checkpoint-<step>.json` with every
prompt/rollout/judge output, `eval_summary.json`) plus `manifest.json`
(dataset + config sha256, git commit, pid, command) and `launcher.log`.

## Reproducing

```bash
R=".venv/bin/python kaushik/run.py fragility"
$R build_pool && $R build_data                      # pool + 1A rows (pyarrow via uv --with)
$R distill --models gptoss20b qwen8b                # 1D rows (Modal vLLM, base models)
$R style                                            # 1E rows (OpenRouter)
$R make_configs                                     # configs/*.yaml
$R make_configs --smoke --arms mcq_noreason --models gptoss20b qwen8b && $R launch --smoke
$R schedule --cap 35                                # launches all runs under the GPU cap; $R status
$R registry                                         # hand final adapters to noise/ (kaushik/registry/)
$R analyze --plot                                   # results/
```

`schedule` keeps the sum of active runs' peak GPU use (one engine per
concurrent eval call: 6 for the 27B runs, 3 otherwise) under the sprint cap,
launching Qwen3.6-27B 1A first, and halves the cap 6 h after the first launch.

## Analysis

`analyze` reads every run under `runs/`, `cpt/runs/` and `../noise/runs/`
(run identity from `config.yaml`: `arm`, `model_key`/`base_model`,
`train.seed`; noise runs are matched to an arm by name). Per rollout: honest
compliance (`cotcontrol.eval.grading.is_honest`), raw strict compliance,
accuracy, `chars`, `degen` (< 200 chars) and `compl_long` (compliance among
traces at least as long as the untrained median for that model/condition/mode).
Intervals are 95% two-way cluster bootstraps over questions and seeds; the
per-seed column shows seed spread. `results/beats.md` lists, per arm × model,
the final-step difference to `untrained` (checkpoint-0, paired on questions) and
to `noise`; an arm "beats both" only if both interval lower bounds exceed 0.

## Results (all 23 runs finished 2026-10-08; full tables in `results/`)

Honest compliance = strict grader passes, answer extracted, trace not truncated /
< 200 chars / repetitive. `compl_long` = raw compliance among traces at least as
long as the untrained median. Intervals: 95% bootstrap over questions and seeds.
`noise` = the noise agent's KL-matched random LoRAs (3 per arm, test split only).
KL = KL(base ‖ adapter) per token from `../noise/runs/_kl/`.

### Test split, stated condition, 9 training modes, final checkpoint (500 q × seeds)

| arm | model | seeds | honest untrained → final | compl_long final | raw | acc untrained → final | chars | degen | KL | noise honest (KL) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1A mcq_noreason | gpt-oss-20b | 3 | 0.012 → **0.049 [0.041, 0.058]** | 0.021 [0.014, 0.029] | 0.063 | 0.46 → 0.33 | 5.5k → 2.5k | 0.03 | 0.064 | 0.009 [0.005, 0.013] (0.058) |
| 1A mcq_noreason | Qwen3-8B | 3 | 0.005 → **0.026 [0.011, 0.040]** | 0.159 [0.078, 0.254] | 0.107 | 0.40 → 0.27 | 13.9k → 6.7k | 0.24 | 0.031 | 0.004 [0.002, 0.006] (0.027) |
| 1A mcq_noreason | Qwen3.6-27B | 5 | 0.003 → **0.010 [0.007, 0.014]** | 0.001 [0.000, 0.002] | 0.011 | 0.46 → 0.48 | 23.5k → 19.6k | 0.00 | 0.005 | 0.003 [0.001, 0.005] (0.006) |
| 1D mcq_reason | gpt-oss-20b | 3 | 0.012 → 0.012 [0.009, 0.015] | 0.004 [0.001, 0.007] | 0.013 | 0.46 → 0.46 | 5.5k → 5.4k | 0.01 | 0.004 | 0.012 [0.009, 0.015] (0.004) |
| 1D mcq_reason | Qwen3-8B | 3 | 0.005 → 0.005 [0.003, 0.007] | 0.017 [0.013, 0.022] | 0.013 | 0.40 → 0.39 | 13.9k → 13.8k | 0.11 | 0.001 | 0.005 [0.003, 0.007] (0.001) |
| 1E style_sft | gpt-oss-20b | 3 | 0.012 → 0.000 [0.000, 0.001] | 0.024 [0.000, 0.063] | 0.187 | 0.46 → 0.00 | 5.5k → 0.5k | 0.41 | 0.075 | 0.009 [0.006, 0.012] (0.071) |
| 1E style_sft | Qwen3-8B | 3 | 0.005 → 0.000 [0.000, 0.001] | 0.030 [0.019, 0.043] | 0.193 | 0.40 → 0.00 | 13.9k → 1.2k | 0.61 | 0.075 | 0.002 [0.001, 0.003] (0.070) |
| 1B cpt-r32 (cpt/, step 200) | gpt-oss-20b | 3 | 0.012 → 0.008 [0.004, 0.011] | 0.016 [0.011, 0.021] | 0.022 | 0.45 → 0.17 | 5.5k → 4.0k | 0.02 | 0.069 | – (see cpt/) |
| 1B cpt-r32 (cpt/, step 200) | Qwen3-8B | 3 | 0.006 → 0.004 [0.002, 0.006] | 0.007 [0.004, 0.012] | 0.008 | 0.39 → 0.36 | 13.9k → 9.5k | 0.11 | 0.128 | – (see cpt/) |

Held-out modes (`results/exp1_test_table.md`, subset `heldout`): gpt-oss 1A
0.013 → 0.040 [0.029, 0.054]; Qwen3-8B 1A 0.001 → 0.014 [0.004, 0.027];
Qwen3.6-27B 1A 0.000 → 0.000. Unconstrained `baseline` accuracy: gpt-oss 1A
0.49 → 0.40, Qwen3-8B 1A 0.31 → 0.30 (degen 0.31), Qwen3.6-27B 1A 0.40 → 0.42,
1E → 0.00 on both models.

### Val split, stated condition, 4 modes pooled, every checkpoint

`results/exp1_val_table.md` has all steps and per-seed values; plot:
`results/compliance_vs_step.png`. Final step: gpt-oss 1A 0.000 → 0.003
[0.000, 0.007]; Qwen3-8B 1A 0.004 → 0.025 [0.004, 0.045] (raw 0.073, degen
0.15); Qwen3.6-27B 1A 0.001 → 0.027 [0.015, 0.040] (degen 0.00, accuracy 0.52 →
0.54); 1D ≤ 0.010; 1E 0.000. Seed spread is in the `honest_by_seed` column.

### Qwen3.6-27B conditions (Experiment 2 baseline; `results/exp1_conditions.md`)

| condition | split | untrained | 1A step 50 (5 seeds) | noise (3) |
|---|---|---|---|---|
| persona | val | 0.000 | 0.000 [0.000, 0.000] | – |
| direct | val | 0.030 | 0.035 [0.018, 0.053] | – |
| stated | val | 0.001 | 0.027 [0.015, 0.040] | – |
| persona | test | 0.000 | 0.000 [0.000, 0.000] | 0.000 |
| direct | test | 0.018 | 0.035 [0.021, 0.048] | 0.018–0.024 |
| stated | test | 0.008 | 0.023 [0.016, 0.030] | 0.005–0.008 |

### Which arms beat untrained AND noise (`results/beats.md`, test, 9 modes)

- **1A mcq_noreason / gpt-oss-20b: yes.** honest +3.7 [+2.9, +4.6] vs untrained, +4.0
  [+3.3, +4.7] vs noise; compl_long +1.8 [+1.0, +2.6] vs both. The gain comes with
  a 13-point accuracy drop and traces half as long; held-out modes rise similarly.
- **1A mcq_noreason / Qwen3-8B: yes, but degenerate.** honest +2.1 [+0.7, +3.5] vs
  untrained, +2.2 [+0.6, +3.7] vs noise; compl_long +14 [+6, +23]. Raw compliance
  (10.7%) is mostly hollow: 24% of traces are < 200 chars, accuracy falls 0.40 → 0.27.
- **1A mcq_noreason / Qwen3.6-27B: yes, small.** honest +0.7 [+0.3, +1.1] vs both on
  test (val, 4 modes: +2.6 [+1.4, +3.8]); compl_long unchanged (0). No accuracy loss,
  no degeneration, KL only 0.005. The persona condition stays at exactly 0%.
- **1D mcq_reason: no** (both models; every difference within ±0.3 points; KL ≤ 0.004).
- **1E style_sft: no.** Honest compliance falls to 0 because the pirate-voice answers
  no longer contain `ANSWER: X`; raw compliance (19%) is pure degeneration (degen
  0.4–0.6, accuracy 0).
- 1B cpt-r32 (other agent's runs, matched steps 50 and 4× dose 200): honest compliance
  does not rise on either model; gpt-oss accuracy collapses (0.45 → 0.17). Its noise
  controls are matched under the `cpt/` naming and are analysed there.

### Reading

Generic no-reasoning MCQ SFT (1A) raises measured controllability by a few points
on all three models and beats KL-matched noise, so the effect is not "any
perturbation". But `compl_long` stays near zero everywhere: no arm produces control
at full reasoning length; the rise rides on shortened or hollow traces, and on
gpt-oss and Qwen3-8B on a capability drop. Keeping the model's own reasoning in
the target (1D) removes the effect entirely (and barely moves the weights, KL
≈ 0.001–0.004), so the no-reasoning format — not the MCQ content — is what moves the
eval. For Experiment 2 the relevant baseline numbers are the Qwen3.6-27B rows above:
generic MCQ training gives 0% persona compliance, 3.5% direct, 2.3–2.7% stated.

Caveats: 1E's `honest` and accuracy are 0 by construction (format collapse), so only
its raw/degen columns are informative; Qwen3-8B's base model emits empty reasoning on
`ignore_question` (gated to 0, not a training artifact); `untrained` on test is the
seed-0 run's checkpoint-0 (one sampling seed).
