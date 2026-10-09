# Workstream C3 — LoRA arithmetic: stripping brevity out of the GEPA LoRA (gpt-oss-20b)

B2 (`../gepa_lora/`) distilled the GEPA s1 system prompt into a LoRA with no loss on any reasoning token. It
reproduces the prompt's honest compliance (test 9.7% vs untrained 1.5%) but also its brevity (883 vs 2,612 median
reasoning chars), and `honest_long` (compliance among traces at least the untrained median length) stays at the
floor. Workstream B's residual-stream "GEPA minus brevity" steering vector kept length but recovered no compliance.
Here the same subtraction is done in **weight space**, where the LoRA delta is far richer than one direction at one
layer. Protocol, rules and budget: `../PLAN.md`. Live state: `STATE.md`. Results: `results/`.

## Method

1. **Length LoRAs with B2's pipeline.** `distill_app.py` is B2's trainer (copied, checkpoints under
   `/checkpoints/lora_arith/`): forward KL(teacher || student) at the pre-reasoning positions only (user turn, assistant
   header, first-reasoning-token distribution), same train-split prompts (`../gepa_lora/data`), same r16 / alpha 32 /
   q,k,v,o / lr 1e-4 / bs 16 / 400 steps. Teachers: **brevity** = base + "Keep your reasoning brief."; **verbosity** =
   base + "Reason thoroughly and at length before answering." (`gl_common.TEACHER_PROMPTS`). Nothing in training sees a
   reasoning token.
2. **Weight-space combination** (`arith_app.py`, CPU Modal function): ΔW = ΔW_gepa − λ·ΔW_brief (+ μ·ΔW_verbose) as one
   servable PEFT adapter by concatenation — A = [A_gepa; A_brief], B = [B_gepa, −λ·B_brief], rank 32 (48 with a third
   term), alpha = 2·rank so the single scaling equals B2's alpha/r = 2. The build verifies, module by module, that the
   combined adapter's materialised delta equals the sum of the term deltas (GEPA s0 − s1: max relative error 3.4e-7 =
   bf16 rounding). vLLM loads it through the ordinary LoRA path (`max_lora_rank 64`).
3. **Sweep** λ ∈ {0.5, 1, 1.5}, μ ∈ {0.5, 1}, and λ = μ = 1, on a val subset (40 questions × 9 training modes,
   `--max-questions 40 --subsample-seed 1`), with GEPA ×1, brevity ×1 and verbosity ×1 alone as references.
4. **Test**: the selected combination(s) with 3 seeds (GEPA seed k − λ·brevity seed k), KL-matched random-LoRA control
   (`noise.py`), 95% bootstrap CIs over questions. Untrained / GEPA-LoRA / teacher numbers are reused from B2 and B.

## Stages

```
export ELICIT_SUB=sub-lora-arith
R=".venv/bin/python kaushik/run.py elicit/lora_arith"      # (zsh: call it spelled out, $R does not word-split)
$R deploy                                   # distill + arith + random-lora apps; bash kaushik/deploy_modal.sh sub-lora-arith for inference/train
$R train --run brief_s0 --teacher brief --seed 0
$R status [--run NAME]
$R combine --name gepa_m1.0b_s0 --terms /checkpoints/gepa_lora/gl_r16_lr1e-4_s0/checkpoint-400:1,/checkpoints/lora_arith/brief_s0/checkpoint-400:-1
$R launch <run> -- --split val --blocks modes --max-questions 40 --subsample-seed 1 --chunk 360 --lora /checkpoints/lora_arith/combos/<name> [;; ...]
$R noise --group G --adapters A,B,C
$R analyze [--runs ...]                     # results/table.md (+ B's teacher/untrained rows)
$R spend
```

## Files
`gl_common.py` constants · `distill_app.py` (B2's trainer) · `arith_app.py` combiner · `cli.py` · `evalrun.py`, `noise.py`,
`analyze.py` (B2's, unchanged) · `runs/<run>/` (train.json / log.jsonl / result.json, or eval blocks + manifest + eval.log) ·
`runs/_combos/<name>.json` build records · `results/`.

## Results

**Val screen** (seed 0, 40 val questions × 9 modes = 356 rollouts each; `results/sweep_table.md`; honest % / median
chars, untrained 1.0 / 2608): GEPA ×1 8.7 / 973 · brevity LoRA alone 3.1 / 1187 · verbosity alone 1.1 / 3794 ·
GEPA −0.5b 6.5 / 1162 · −1.0b 5.1 / 1420 · −1.5b 6.7 / 1887 · **−2.0b 5.3 / 2926** · −2.5b 3.9 / 3916 · +0.5v 6.7 / 1203 ·
+1.0v 7.6 / 1630 · +2.0v 3.9 / 2794 · −1b+1v 3.9 / 2659 · −1.5b+0.5v 3.7 / 3114. Subtracting brevity restores length
monotonically while compliance degrades gradually. Selection rule (frozen before test): highest honest among candidates
with median length ≥ 80% of untrained → GEPA − 2.0·brevity.

**Test** (500 q; 3 seeds pooled = 13,392 rollouts on the 9 modes; 95% bootstrap CIs over questions; `results/test_table.md`):

| arm | honest | honest_long | held-out | accuracy | median chars | strict |
|---|---|---|---|---|---|---|
| untrained | 1.5 [1.1, 2.0] | 0.5 [0.2, 0.9] | 1.3 [0.7, 1.9] | 48.6 [44.2, 52.6] | 2612 | 1.6 |
| teacher (GEPA prompt) | 9.4 [8.5, 10.2] | 0.7 [0.3, 1.3] | 11.2 [9.7, 12.7] | 45.0 [40.6, 49.2] | 1190 | 13.9 |
| GEPA LoRA (B2, 3 seeds) | 9.7 [9.1, 10.3] | 0.8 [0.4, 1.3] | 12.2 [11.2, 13.3] | 43.5 [39.5, 47.4] | 883 | 14.4 |
| **GEPA LoRA − 2·brevity LoRA (3 seeds)** | **6.6 [6.1, 7.1]** | **0.9 [0.6, 1.1]** | 7.4 [6.5, 8.4] | 44.6 [40.5, 48.5] | **2354** | 11.0 |
| KL-matched random LoRA (KL 0.0195 nats/token, 1 adapter) | 1.2 [0.9, 1.6] | 0.4 [0.2, 0.7] | 1.1 [0.7, 1.7] | 43.2 [39.0, 47.6] | 2813 | 1.5 |

- Subtracting twice the brevity LoRA brings reasoning length back to 90% of untrained (passes the ≥80% guardrail) while
  keeping 65% of the GEPA LoRA's honest gain (6.6 vs 9.7; 4.4× untrained) and 56% of its held-out gain. Accuracy −4.0 vs
  untrained (CIs overlap; just outside the 3-pt guardrail, better than the GEPA LoRA's −5.1). The noise control at the
  same functional size does nothing, so the effect is the distilled directions, not perturbation size.
- The combination is at least as compliant as the GEPA LoRA *within every length bucket* (`results/length_buckets.md`:
  <1000 chars 18.0 vs 14.9%, 1000–2612 7.0 vs 4.8%, ≥2612 0.9 vs 0.8%); its lower overall rate is purely that it has
  fewer short traces (22% vs 55% under 1000 chars). So the GEPA LoRA's effect decomposes into "shorten" plus a
  length-independent compliance shift, and weight-space arithmetic separates them — unlike the single-direction
  steering vector (Workstream B), which kept length but lost all compliance.
- Primary endpoint **honest_long** is above the floor but small: 0.9 [0.6, 1.1] vs 0.5 [0.2, 0.9] untrained. Compliance
  on traces longer than the untrained median stays under 1% for every arm. Per mode the combination keeps the case
  constraints (lower/uppercase thinking 12–13%, same as GEPA) and improves ignore_question (10.7 vs 5.3) but loses the
  length-sensitive ones (repeat_sentences 15 vs 38, word_suppression 6 vs 13).
- Cost: ≈ 21 H200-hours ≈ $95 (6 LoRA trainings, 12 screen evals, 3 test runs, noise calibration + 1 noise test run).
