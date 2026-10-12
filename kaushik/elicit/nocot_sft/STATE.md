# STATE — Workstream A: no-CoT SFT (kaushik/elicit/nocot_sft)

Updated: 2026-10-09 14:05 PDT. Owner agent: Workstream A. Budget cap $380 (stop new launches at $300).

## Migrated to kaushik/core (2026-10-12)

Code is now `config.yaml` + `sweeps/{train,test_step25,ref1a}.yaml` + `gen_answers.py` / `build_data.py` / `analyze.py`
(`main(cfg, argv)`); training, evals, status and registry are core jobs (`kaushik/run.py elicit/nocot_sft <job>`).
Config and analysis equivalence verified (README "Layout / reproducing"); `cli.py`, `common.py`, `launch.py`,
`make_configs.py`, `registry.py` deleted. The commands below this line are historical (pre-migration stages).

## Modal apps (per-workstream cost tracking)

- `cot-control-sprint_kaushik-reddy_sub-nocot-sft_inference` / `..._train` (deployed 13:5x, 120 s scaledown;
  train app redeployed 13:58 after the rendering change below). Every stage of `cli.py` exports these.
- Spend: `.venv/bin/python kaushik/elicit/spend.py sub-nocot-sft`.

## Shared-code changes (additive, default-preserving)

- `cotcontrol/training/rendering.py:render_row` — a chat row may carry `"render": {"chat_template_kwargs": ..., "assistant_prefill": ...}`
  overriding the config-wide values for that row (used for the reasoning-on anchor rows inside reasoning-off datasets).
  Rows without the key render exactly as before. Test: `tests/test_rendering.py::test_render_row_per_row_override`.

## Running (launched 14:02, one wave, 3 seeds each; pids in runs/<run>/manifest.json)

| arm | runs | steps | val ckpts | test ckpt |
|---|---|---|---|---|
| gentle (answer-only, r8/a8, lr 5e-5) | nocot-gentle-gptoss20b-seed{0,1,2} | 100 | 50, 100 | 100 |
| mix (answer-only + reasoning anchor, 1A dose) | nocot-mix-gptoss20b-seed{0,1,2} | 100 | 50, 100 | 100 |
| fmt (constrained-output no-CoT, 1A dose) | nocot-fmt-gptoss20b-seed{0,1,2} | 50 | 25, 50 | 50 |
| fmt_mix (fmt + anchor) | nocot-fmt_mix-gptoss20b-seed{0,1,2} | 100 | 50, 100 | 100 |
| ref_1a (existing 1A seed-0 adapter; eval-only, val at ckpt 0 and 50) | ref1a-mcq_noreason-gptoss20b-seed0 | – | 0, 50 | (test exists in fragility/) |
| fmt_long (fmt at 3x dose; launched 14:32 after the mid-ckpt val numbers, see progress log) | nocot-fmt_long-gptoss20b-seed{0,1,2} | 150 | 75, 150 | 150 |

fmt_long joins the selection pool only if its val blocks finish before the selection is frozen; otherwise it is
reported as exploratory. It was launched on val mid-checkpoint numbers only (no test number had been read).

Status: `.venv/bin/python kaushik/run.py elicit/nocot_sft status`. Each run holds <= 2 Modal engines (24 + training).

## Pre-registered selection (frozen BEFORE looking at any test number)

Val split, 9 training modes x 200 q (`val_modes`), per arm pooled over 3 seeds:
1. Primary endpoint: `honest_long` = honest compliance (`is_honest`) among traces at least the untrained median
   reasoning length for that mode (untrained = ref_1a checkpoint-0 on val).
2. Guardrails: `val_baseline` accuracy within 3 pts of untrained; median reasoning chars >= 80% of untrained.
3. The selected candidate = the (arm, checkpoint) with the highest honest_long whose guardrails pass; ties -> plain
   honest compliance, then held-out honest_long. If the selected checkpoint is the mid checkpoint, its test eval is
   run afterwards (test was pre-launched at the final checkpoint only, for budget).
4. Then noise controls (kaushik/noise, KL-matched) for the selected arm only. Every other arm's test numbers are
   reported as exploratory.

### SELECTION FROZEN 15:02 PDT (no test number read before this point)

Selected: **fmt, step 25** (val honest_long 0.7 [0.2, 1.4], honest 2.8, baseline acc 49.5, median chars 3168;
guardrails pass). Runner-ups on honest_long: fmt_long@75 0.7 (loses the plain-honest tie-break, 2.0 vs 2.8),
gentle@50/100 0.6 (fail a guardrail), fmt_mix@100 0.5, fmt@50 0.5, fmt_long@150 0.4 (2 seeds in), mix <= 0.3.
Honest reading: no arm beats untrained (0.6) on the primary endpoint; fmt is selected by the rule, not because it
works. Noise controls go to fmt step 25. All other arms' test numbers are exploratory.

## RESULT (test split, 500 q x 3 seeds, 9 training modes; percent, 95% bootstrap CI; results/test_table.md)

| arm | honest | honest_long | held-out honest | baseline acc | median chars | KL |
|---|---|---|---|---|---|---|
| untrained | 1.5 [1.1, 1.9] | 0.5 [0.2, 0.9] | 1.3 [0.7, 1.9] | 48.6 [44.0, 52.8] | 2612 | 0 |
| 1A (answer-only, r32 lr 1.5e-4, 50 steps) | 4.9 [4.1, 5.8] | 1.8 [1.2, 2.5] | 4.0 [2.9, 5.4] | 39.9 [34.0, 45.4] | 1391 | 0.064 |
| noise matched to 1A | 0.9 [0.5, 1.3] | 0.3 [0.2, 0.5] | 0.7 [0.3, 1.3] | 44.4 [39.8, 48.7] | 3993 | 0.058 |
| **fmt @25 (selected)** | 2.8 [2.4, 3.3] | 0.4 [0.2, 0.6] | 1.8 [1.2, 2.6] | 46.3 [41.9, 50.3] | 3372 | 0.018 |
| noise matched to fmt@25 | 1.2 [0.8, 1.7] | 0.4 [0.2, 0.7] | 0.9 [0.5, 1.4] | 44.8 [40.4, 49.2] | 2866 | 0.018 |
| gentle @100 (exploratory) | 2.3 [1.8, 2.8] | 0.5 [0.2, 0.8] | 2.8 [2.2, 3.6] | 46.7 [42.2, 51.0] | 1954 | – |
| mix @100 (exploratory) | 1.3 [1.0, 1.6] | 0.4 [0.2, 0.6] | 1.4 [0.9, 1.9] | 46.3 [41.9, 50.3] | 2299 | – |
| fmt @50, fmt_mix @100, fmt_long @150 (exploratory) | 2.4 / 1.2 / 2.2 | 0.4 / 0.3 / 0.4 | 1.2 / 1.3 / 1.2 | 48.1 / 46.8 / 45.9 | 3222 / 3171 / 3028 | – |

Paired bootstrap differences (test, modes): fmt@25 − untrained honest +1.3 [+0.7, +1.9], honest_long −0.1 [−0.5, +0.3];
fmt@25 − its noise honest +1.6 [+1.1, +2.3], honest_long 0.0 [−0.4, +0.3]. 1A − untrained honest +3.4, honest_long +1.3.
Reading: no no-CoT SFT arm raises controllability at normal reasoning length. fmt (format control in the output
channel) raises plain honest compliance above untrained and above KL-matched noise WITHOUT shortening the median
trace (3372 vs 2612 chars) or hurting accuracy beyond the CI, but its compliant traces are short (median 990 chars)
and the gain sits in repeat_sentences (3.8 → 11.5) and uppercase (0 → 3.6). The reasoning anchor (mix, fmt_mix)
removes even that gain; the gentle dose of answer-only SFT gives a smaller 1A with a smaller collapse.

## TO FINISH

- [x] wave finished (19 runs; last fmt_long baseline block landed ~15:50), `status` all `done`
- [x] selection frozen 15:02 (fmt @25) before any test number was read
- [x] eval-only test for fmt @25 (runs/nocot-fmt-*-step25)
- [x] registry (kaushik/registry/nocot-fmt-gptoss20b-seed{0,1,2}-step25.json) + noise group
  `elicit-nocot_sft-fmt-step25-gptoss20b` (kaushik/noise/runs/…/results.json; the random-lora CPU app had to be
  redeployed with `noise deploy`)
- [x] results/{val,test}_table.md, summary.csv, selection.json, rollouts.csv.gz
- [ ] coordinator: commit this folder + the rendering.py change + tests/test_rendering.py (no commits by this agent)
- [ ] optional teardown: `modal app stop cot-control-sprint_kaushik-reddy_sub-nocot-sft_{inference,train}`

## Progress log

- 14:02 launched 12 runs + ref1a. 14:20 all training done (fmt 121 s, gentle 188 s, mix/fmt_mix ~10 min).
- 14:27 first val (mid ckpts, modes block): untrained 1.4 honest / 0.6 honest_long / 2585 chars; 1A 4.8 / 0.9 / 1246;
  gentle@50 2.1 / 0.6 / 2234; fmt@25 2.8 / 0.7 / 3168. Reported to the coordinator.

- 14:48 val complete for gentle/mix/fmt/fmt_mix (results/val_table.md). Rule ordering (honest_long, val):
  fmt@25 0.7 (pass) > gentle@50 0.6 (fails acc 47.5 vs 51.0-3) = gentle@100 0.6 (fails length 1834 < 0.8x2585)
  > fmt_mix@100 0.5 = fmt@50 0.5 > mix@100 0.3 ... All within noise of untrained (0.6). fmt_long val pending
  (joins the pool if in before the freeze). Pre-launched the TEST eval of fmt@25 (runs/nocot-fmt-*-step25, 3 seeds)
  since it is the likely selection; no test number has been read.

- 15:02 registry entries kaushik/registry/nocot-fmt-gptoss20b-seed{0,1,2}-step25.json; noise controls launched:
  `kaushik/run.py noise process --group elicit-nocot_sft-fmt-step25-gptoss20b` (pid in runs/_noise/pid, log
  runs/_noise/process.log; outputs under kaushik/noise/runs/elicit-nocot_sft-fmt-step25-gptoss20b/). Only this pid
  and the pids in runs/*/manifest.json are ours.

- 15:15 test (partial; selected fmt@25 complete except baseline): fmt@25 honest 2.8 [2.4, 3.3] vs untrained 1.5
  [1.1, 1.9], 1A 4.9, noise_1A 0.9; honest_long 0.4 [0.2, 0.6] vs untrained 0.5, 1A 1.8; median chars 3372 (vs 2612).
  The gain is in repeat_sentences (3.8 -> 11.5) and uppercase (0 -> 3.6); fmt's compliant traces are short
  (median 990 chars) while its overall distribution lengthened (p90 19k vs 13k). Not a gain at normal length.
  Noise for fmt@25: KL 0.017-0.019 (1A: 0.064); calibration running after redeploying the random-lora CPU app
  (`noise deploy`, 15:08; it had been undeployed).

## Spend log

- 14:05: ~$2 (deploys + 2,100-answer generation pass).
- 14:27: billing still reports $0 for the sub-nocot-sft apps (lag); estimate ~$40 (12 training containers x ~12 min + ~25 eval engines).
- 15:02: $107.83 (hourly billing, <= 1 h lag) with the wave ~2/3 through its evals. Projection at completion incl.
  fmt_long, the step-25 test evals and noise: ~$230-250. Hard stop for new launches: $300.
- 15:50: $113.11 reported (train $17.59, inference $95.52), still lagging; everything launched is finished or
  finishing, estimate at completion ~$200-230. Re-run `spend.py sub-nocot-sft` after 17:00 for the final figure.
  Nothing else will be launched from this workstream.
