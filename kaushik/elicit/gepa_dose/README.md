# Workstream C1 — dose, stacking and reasoning effort of the GEPA LoRA (gpt-oss-20b)

Evaluation-only follow-up to B2 (`../gepa_lora/`): the no-CoT context-distilled LoRA (r16, α32, q/k/v/o, lr 1e-4,
step 400, 3 seeds) is applied at scaled doses, stacked with the GEPA s1 system prompt it was distilled from, and run
at `reasoning_effort: high`. Shared protocol and guardrails: `../PLAN.md`. Live state: `STATE.md`.

## Method

- **Dose** = scaling the LoRA delta: a copy of each seed's `checkpoint-400` with `lora_alpha` 32 → 32·dose in
  `adapter_config.json` (same weights; vLLM applies BA·α/r). Adapters at `/checkpoints/gepa_dose/s{seed}_x{dose}` on
  `cotcontrol_kaushik-reddy_checkpoints`; dose ×1 is B2's original path. Doses: 0.5, 0.75, 1.05, 1.1, 1.15, 1.2, 1.3,
  1.5, 2, 3.
- **Probe** (`probe.py`, `runs/_probe/dose.json`): teacher-forced log-prob of the GEPA-prompted model's first 24
  reasoning tokens under the unprompted model at each dose. vLLM honours the scaling: base −18.1, ×0.5 −13.5, ×1 −13.6,
  ×1.5 −14.9, ×2 −17.0, ×3 −22.1 nats (8 held-out train prompts). Past ×1 the model moves *away* from the teacher.
- **Eval** (`evalrun.py` = B2's + `--effort`): T=1, 16k tokens, honest metric, 9 training modes + 3 held-out modes +
  unconstrained accuracy. Sweep on a 40-question val subset (`--max-questions 40 --subsample-seed 1`, seed-0 adapter);
  reference runs (untrained, teacher, LoRA ×1) are B/B2's full-val runs restricted to the same questions
  (`analyze.py --subset`). Selected settings: full val (seed 0) + test with all 3 seeds.
- Everything reused rather than re-run: untrained / teacher / LoRA ×1 val and test from `../gepa_distill/runs` and
  `../gepa_lora/runs` (symlinked into `runs/`).

## Sweep (val, 40 questions × 9 modes; `results/sweep.md`; honest % / strict % / accuracy % / median reasoning chars)

Refs on the same questions: untrained 0.8 / 0.8 / 55 / 2754; teacher (GEPA prompt) 8.4 / 11.0 / 57.5 / 1116; LoRA ×1 9.0 / 12.6 / 57.5 / 984.

| dose | alone | + GEPA prompt |
|---|---|---|
| ×0.5 | 4.5 / 5.3 / 55 / 1546 | — |
| ×0.75 | 5.9 / 10.4 / 57.5 / 1145 | **9.6 / 16.9 / 57.5 / 914** (held-out 16.7) |
| ×1 | 9.0 / 12.6 / 57.5 / 984 | 5.9 / 12.9 / 55 / 946 (held-out 16.7) |
| ×1.05 | 7.9 / 12.1 / 52.5 / 971 | 7.0 / 14.0 / 32.5 / 940 |
| ×1.1 | 8.7 / 14.0 / 50 / 932 | 7.0 / 14.0 / 62.5 / 954 |
| ×1.15 | 10.7 / 15.2 / 47.5 / 872 | 8.7 / 18.5 / 52.5 / 856 |
| ×1.2 | 10.4 / 14.6 / 60 / 882 | 6.7 / 16.3 / 40 / 904 |
| ×1.3 | 12.1 / 16.6 / 52.5 / 806 | 9.0 / 16.9 / 42.5 / 792 |
| ×1.5 | 10.1 / 15.2 / 45 / 780 | 5.3 / 14.3 / 37.5 / 812 |
| ×2 | 7.9 / 15.7 / 50 / 738 | 4.5 / 11.2 / 37.5 / 988 |
| ×3 | 5.1 / 11.5 / 42.5 / 786 | — |

95% CIs on honest are about ±3 points at this size. honest_long (traces ≥ the untrained median) is 0–5% for every
setting with every CI including 0, so the pre-registered primary endpoint cannot rank candidates; selection used
honest + held-out + the accuracy guardrail.

Reasoning effort (`results/sweep_effort.md`): LoRA ×1 at `high` 6.2 honest / 8.4 strict / 1502 chars (medium: 9.0 /
12.6 / 984); GEPA prompt at `high` 5.1 / 8.4 / 6806 chars; untrained at `high` 0.0 / 0.0 / 18.5k chars.

## Test (500 questions × 9 modes; 3 seeds pooled; 95% bootstrap CIs over questions; `results/test_table.md`)

| arm | honest | honest_long | held-out | accuracy (unconstrained) | strict | median chars |
|---|---|---|---|---|---|---|
| untrained | 1.5 [1.1, 2.0] | 0.5 [0.2, 0.9] | 1.3 [0.7, 1.9] | 48.6 [44.2, 52.6] | 1.6 [1.2, 2.1] | 2612 |
| teacher (base + GEPA prompt) | 9.4 [8.5, 10.2] | 0.7 [0.3, 1.3] | 11.2 [9.7, 12.7] | 45.0 [40.6, 49.2] | 13.9 [13.0, 14.9] | 1190 |
| LoRA ×1 (B2) | 9.7 [9.1, 10.3] | 0.8 [0.4, 1.3] | 12.2 [11.2, 13.3] | 43.5 [39.5, 47.4] | 14.4 [13.7, 15.1] | 883 |
| LoRA ×1.5 | 9.8 [9.3, 10.3] | 1.3 [0.8, 1.9] | 11.8 [10.8, 12.8] | 42.3 [38.4, 46.1] | 16.2 [15.6, 16.9] | 808 |
| LoRA ×0.75 + GEPA prompt | 8.6 [8.1, 9.1] | 0.6 [0.3, 0.9] | **15.4 [14.4, 16.4]** | 37.7 [34.1, 41.1] | **17.3 [16.7, 18.0]** | 940 |

Same table, plus full-val rows, in `results/test_table.md` (identical to the write-up's `../writeup/data/c1_test_table.md`).

## Answers

1. **Dose.** Honest compliance does not keep rising past ×1: on test, ×1.5 = 9.8 vs ×1 = 9.7 (CIs overlap), while
   strict compliance rises (16.2 vs 14.4), i.e. the extra compliant traces are ones the honesty check rejects. The val
   gain (×1.5 11.1 vs ×1 8.9 on seed 0) is a selection effect: per seed on test, ×1.5 beats ×1 only for seed 0
   (10.6 vs 9.1; seeds 1, 2: 9.5 vs 10.2, 9.3 vs 9.9). Larger doses shorten reasoning further (883 → 808 → 738 chars
   at ×2) and cost accuracy (×2/×3 subset accuracy 50/42.5; constrained accuracy 28/15%). ×3 is broken.
2. **Stacking.** The prompt on top of the LoRA raises strict and held-out compliance (×0.75 + prompt: held-out 15.4
   vs 12.2 / 11.2 for LoRA / prompt alone; strict 17.3 vs 14.4 / 13.9) but not honest compliance on the training modes
   (8.6 vs 9.7), and costs accuracy (37.7 vs 43.5). At ×1 and above, the prompt compounds even less (subset honest 5–9%).
   The two do not add on the primary metrics; they share the same mechanism (brevity + the same mode-specific habits).
3. **Reasoning effort.** At `high`, LoRA ×1 keeps 2/3 of its compliance (honest 6.2 vs 9.0, strict 8.4 vs 12.6 on the
   subset) while reasoning longer (1502 vs 984 chars) — but still far shorter than untrained at high (18.5k chars), so the
   effort knob does not restore normal length, and honest_long stays at the floor (1.6%). The GEPA prompt at high
   behaves alike (5.1 honest, 6.8k chars).
   Overall: no dose, stack or effort setting moves honest_long off the floor (≤1.3% on test); the LoRA's gain remains a
   brevity effect.

## Files

`dose_common.py` (B2's constants under `ELICIT_SUB=sub-gepa-dose`) · `evalrun.py` · `probe.py` · `analyze.py` · `cli.py` ·
`runs/<run>/{manifest.json,modes.json,heldout.json,baseline.json,eval.log}` (every rollout + judge output) ·
`runs/_probe/dose.json` · `results/{sweep,sweep_effort,test_table_prelim,test_table}.md|json`.
