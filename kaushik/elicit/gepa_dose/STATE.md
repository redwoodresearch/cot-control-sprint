# Workstream C1 — GEPA LoRA dose / stacking / reasoning effort (gpt-oss-20b) — STATE

Owner agent: C1 (coordinator notes: kaushik/elicit/COORDINATOR.md). Budget $120 (raised 15:18 PT; was $90); stop new launches at $100. Eval only.
Apps: `cot-control-sprint_kaushik-reddy_sub-gepa-dose_{inference,train}` (deployed 15:07 PT, 120 s idle window).
Every command: `ELICIT_SUB=sub-gepa-dose .venv/bin/python kaushik/run.py elicit/gepa_dose <probe|eval|launch|analyze|spend>`
(cli.py sets the app env itself; `launch` runs detached eval processes, pid in runs/<run>/pid, log runs/<run>/eval.log).

## Setup
- Scaled adapters: B2's `/checkpoints/gepa_lora/gl_r16_lr1e-4_s{0,1,2}/checkpoint-400` copied to
  `/checkpoints/gepa_dose/s{seed}_x{0.5,1.5,2,3}` with `lora_alpha` 32 -> 16/48/64/96 (same safetensors; vLLM scales
  BA by alpha/r). Dose x1 = B2's original path. Volume `cotcontrol_kaushik-reddy_checkpoints`.
- Code: dose_common.py (imports B2's gl_common read-only under ELICIT_SUB=sub-gepa-dose), evalrun.py (B2's +
  `--effort`), probe.py (logp vs dose), analyze.py (B's metric definitions; `--subset RUN` for the 40-q sweep).
- References reused (symlinked into runs/ by analyze): untrained_val, teacher_val, teacher_test (Workstream B),
  x1_val = B2 lr1e-4_c400_val, x1_test_s{0,1,2} = B2 lr1e-4_c400_test_s{k}; untrained_test = fragility ckpt-0.

## Running (launched 15:11 PT; val, 40 questions, subsample_seed 1, chunk 400 => 3 engine calls per run)
- runs/x0.5_sw, x1.5_sw, x2_sw, x3_sw (seed-0 adapter at those doses, no prompt, medium)
- runs/x1_prompt_sw (x1 + GEPA prompt), x1_high_sw (x1, reasoning_effort high), prompt_high_sw (GEPA prompt, high),
  untrained_high_sw (base, high)
- runs/_probe/dose.json: teacher-forced logp of the teacher's first tokens under base / x0.5 / x1 / x1.5 / x2 / x3.

## Sweep result (results/sweep.md, 40 q; honest %): x0.5 4.5, x1 9.0, x1.5 10.1, x2 7.9, x3 5.1 (strict 5.3/12.6/15.2/15.7/11.5;
accuracy 55/57.5/45/50/42.5; median chars 1546/984/780/738/786). x1+prompt 5.9 (heldout 16.7 vs x1 13.3). x1 high: 6.2, 1502 chars.
Probe (runs/_probe/dose.json): logp(teacher tokens) base -18.1, x0.5 -13.5, x1 -13.6, x1.5 -14.9, x2 -17.0, x3 -22.1 -> dose applied.
**Frozen selection (15:17 PT, before any test read): A = x1.5.** B = x1.5+prompt if the mini-sweep shows it compounds, else x1 at high.

## Running (15:17 PT)
- mini-sweep: runs/x1.5_prompt_sw, x2_prompt_sw (40 q). Full val seed 0: runs/x1.5_val. Test: runs/x1.5_test_s{0,1,2}.
- still finishing from the first sweep: prompt_high_sw, untrained_high_sw.

## Effort + stacking (results/sweep_effort.md, 40 q, honest %): x1 medium 9.0 (984 chars) -> x1 high 6.2 (1502 chars, strict 8.4,
heldout 6.7, acc 52.5); GEPA prompt high 5.1 (6806 chars; heldout 11.8k, unconstrained 22k chars); untrained high: heldout 17.5k chars.
x1.5+prompt 5.3 (strict 14.3, heldout 15.8, acc 37.5); x2+prompt 4.5. Stacking at >=x1.5 does not compound on training modes.

## Fine-dose sweep (coordinator request 15:18 PT): x0.75/1.05/1.1/1.15/1.2/1.3 alone and + GEPA prompt, 40-q subset,
runs/x<d>_sw and x<d>_prompt_sw (seed-0 adapters at /checkpoints/gepa_dose/s0_x<d>; seeds 1,2 uploaded too). Launched 15:18 PT.

Fine grid result (results/sweep.md; honest % / strict / acc / chars): alone x0.75 5.9/10.4/-/1145, x1.05 7.9/12.1/52.5/971,
x1.1 8.7/14.0/50/932, x1.15 10.7/15.2/47.5/872, x1.2 10.4/14.6/60/882, x1.3 12.1/16.6/52.5/806, x1.5 10.1/15.2/45/780.
+prompt: x0.75 9.6/16.9/57.5/914 (heldout 16.7), x1.05 7.0, x1.1 7.0, x1.2 6.7, x1.3 9.0, x1.5 5.3, x2 4.5. honest_long at floor everywhere.
**Frozen candidate B (15:24 PT, before reading B's test): x0.75 + GEPA prompt** (only stacked setting keeping accuracy; best heldout).
Launched 15:24: runs/x0.75_prompt_val (seed 0), runs/x0.75_prompt_test_s{0,1,2}.

TEST prelim (results/test_table_prelim.md, x1.5 3 seeds pooled): honest 9.8 [9.3,10.3] / honest_long 1.3 / heldout 11.8 / acc 42.3 /
strict 16.2 / 808 chars vs x1 9.7 / 0.8 / 12.2 / 43.5 / 14.4 / 883 -> dose raises strict, not honest. Full val x1.5 s0: 11.1 / 16.4 / 43.5 / 724.
Sweep + prelim sent to coordinator 15:27 PT. Per seed (x1.5 vs x1 honest): s0 10.6 vs 9.1, s1 9.5 vs 10.2, s2 9.3 vs 9.9 ->
the dose gain is specific to seed 0 (the adapter the sweep selected on): selection effect, not a dose effect.

## FINAL (15:38 PT; results/test_table.md, README.md). Done; nothing running; no further launches. Coordinator stopping
the sub-gepa-dose apps. Test (3 seeds pooled): x1.5 honest 9.8 [9.3,10.3] / honest_long 1.3 / heldout 11.8 / acc 42.3 / strict 16.2 /
808 chars; x0.75+prompt 8.6 / 0.6 / 15.4 / 37.7 / 17.3 / 940; x1 9.7 / 0.8 / 12.2 / 43.5 / 14.4 / 883; teacher 9.4 / 0.7 / 11.2 / 45.0 /
13.9 / 1190; untrained 1.5 / 0.5 / 1.3 / 48.6 / 1.6 / 2612. Dose: strict up, honest flat (seed-0 selection effect). Stacking: heldout and
strict up, honest down, accuracy down. High effort: LoRA keeps ~2/3 of compliance, 1502 chars. honest_long at floor everywhere.
Volume: /checkpoints/gepa_dose/s{0,1,2}_x{0.5,0.75,1.05,1.1,1.15,1.2,1.3,1.5,2,3} (30 x 32 MB; deletable).

## Spend
- Billing report (lags): $36 at 15:35 PT. GPU-time estimate: 24 H200 saturated 15:11-15:33 (+ idle tails, probe) ≈ 9.5 GPU-h ≈ $43-45
  total expected (H200 ≈ $4.5/h). Budget $120, launch cap $100: not approached.
