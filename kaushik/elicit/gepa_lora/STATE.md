# Workstream B2 — no-CoT context-distilled GEPA LoRA (gpt-oss-20b) — STATE

Owner agent: B2 (coordinator notes: kaushik/elicit/COORDINATOR.md). Budget $150; stop new launches at $120.
Apps: `cot-control-sprint_kaushik-reddy_sub-gepa-lora_{inference,train,distill}` (all deployed 2026-10-09).
Env for every command: `export COTCONTROL_MODAL_INFERENCE_APP=cot-control-sprint_kaushik-reddy_sub-gepa-lora_inference
COTCONTROL_MODAL_TRAIN_APP=cot-control-sprint_kaushik-reddy_sub-gepa-lora_train`, then `.venv/bin/python kaushik/run.py elicit/gepa_lora <stage>`.

## Method (see README.md)
Teacher = base + GEPA s1 prompt (sha pinned in ../gepa_distill/gd_common.py); student = base + LoRA (r16, a32, q/k/v/o).
Loss = forward KL(teacher || student) on next-token distributions at pre-reasoning positions only (user turn +
assistant/channel header + the first-reasoning-token distribution). Optional anchor: KL(base || student) at the
header positions (length-preserving variant, coordinator request). No reasoning/answer token is ever an input or target.

## Running / done
- data: `data/prompts.jsonl` 4588 train-split prompts (9 modes), 457 held out (51 questions); `data/drift.jsonl`.
- training (Modal, 1 H200 each, 400 steps, bs 16, saves at 50/100/200/300/400; log via `status`):
  - `gl_r16_lr1e-4_s0`        lr 1e-4, w_turn 1, w_first 2, no anchor      call fc-01M4H8J3E331DKA0AN2AJFB0YV
  - `gl_r16_lr3e-4_s0`        lr 3e-4, same                                call fc-01M4H8J4T6WPEH5MX936V7F5FZ
  - `gl_r16_lr1e-4_anchor_s0` lr 1e-4, w_turn .3, w_first .3, w_anchor 1   call fc-01M4H8J6AN588D5P1XKAV4EPDB
  (earlier spawns failed: container-path bug, zsh word-splitting, then OOM on 5k-token prompts -> length cap 2048 + token-budget micro-batches; ~4 s/step at bs 16.)
- Apps also: `..._sub-gepa-lora_random-lora` (CPU adapter builder for the noise control).
  - seeds 1,2 of the plain and anchored lr 1e-4 configs: `gl_r16_lr1e-4_s{1,2}`, `gl_r16_lr1e-4_anchor_s{1,2}` (call ids in runs/*/train.json)
  - speed: ~2.2 s/step (bs 16) -> 400 steps ≈ 15 min. Holdout KL (lr 1e-4) 6.43 -> 0.46 at step 100 (first-token 2.61 -> 0.07).
- probe (runs/_probe/): checkpoint-0 = noise floor (student closer to teacher 1/6, mean logp gain -0.28);
  lr1e-4 checkpoint-100: 5/8, +5.3 nats; uppercase_thinking traces start in ALL CAPS like the teacher. vLLM path OK.
- Checkpoints: `/checkpoints/gepa_lora/<run>/checkpoint-<step>` on `cotcontrol_kaushik-reddy_checkpoints`.

- seed-0 training done (~15.5 min each): holdout KL at step 400 — lr1e-4 0.354, lr3e-4 0.388 (anchor finishing).
- val evals launched 14:41: runs/lr1e-4_c400_val, runs/lr3e-4_c400_val (pids in runs/*/pid); anchor_c400_val auto-launches
  when its training call finishes (scratchpad waiter). Noise-control builds (runs/_noise/<group>/) auto-start when the
  seeds of each config finish (groups gl_r16_lr1e-4_c400, gl_r16_lr1e-4_anchor_c400).

- VAL (results/val_table_plain.md): lr1e-4 c400 honest 8.9 [7.7,10.3] / heldout 11.2 / acc 45.5 / med 893 chars;
  lr3e-4 c400 honest 10.3 / heldout 12.7 / acc 46.5 / med 868; teacher 9.9 / 12.5 / 47.0 / 1159; untrained 1.0 / 1.3 / 48.5 / 2608.
  -> 89% / 104% of the teacher's honest gain; honest_long 0.6 / 2.1 vs teacher 1.2 (near zero everywhere). Both fail the
  >=80% length guardrail (so does the teacher). anchor_c400_val running (auto-analysis -> results/val_table.md).
- TEST launched 15:03: runs/lr1e-4_c400_test_s{0,1,2} (plain lr1e-4 ckpt-400, 3 seeds). Noise for that config building
  (runs/_noise/gl_r16_lr1e-4_c400/). Anchored config: test x3 + noise to launch if its val passes the length guardrail.

- anchored variant val (results/val_table.md): honest 7.5 [6.4,8.7], heldout 9.2, acc 45.0, med 996 chars (73% of gain).
  Keeps only marginally more length (996 vs 893) at a lower honest rate -> NOT taken to test; its noise build cancelled.
- Noise control: trained KL(base||adapter) = 0.0236 nats/token (median of 3 seeds); noise0 calibrated at scale 2.92
  (KL 0.0231); noise1/2 calibrating. Test evals for noise0-2 auto-launch when runs/_noise/gl_r16_lr1e-4_c400/noise_adapters.json
  appears; results/test_table.md auto-builds when all 6 test runs have 3 blocks (background waiters in this agent's session;
  if they died, run the `launch` / `analyze` commands from README by hand).
- Spend: billing report lags >1 h; manual estimate at 15:20 PT ≈ $30 (7 x 16 min H200 training, 3 val evals ≈ 3.3 GPU-h,
  probes, part of 3 test evals). Projected total ≈ $95-105 (3 LoRA + 3 noise test evals ≈ 16 GPU-h).

- TEST prelim (results/test_table_prelim.md; 3 seeds pooled, 13392 rollouts): honest 9.7 [9.1,10.3] vs teacher 9.4 [8.5,10.2]
  vs untrained 1.5 [1.1,2.0] -> 105% of gain; honest_long 0.8 / 0.7 / 0.5; heldout 12.2 / 11.2 / 1.3; acc 43.5 / 45.0 / 48.6;
  median chars 883 / 1190 / 2612. Noise test evals running (launched 15:33 PT, runs/noise_lr1e-4_c400_test_n{0,1,2}).

- FINAL TEST (results/test_table.md, 15:55 PT): LoRA honest 9.7 [9.1,10.3] / heldout 12.2 [11.2,13.3] / honest_long 0.8 /
  acc 43.5 / med 883 chars; teacher 9.4 / 11.2 / 0.7 / 45.0 / 1190; untrained 1.5 / 1.3 / 0.5 / 48.6 / 2612;
  noise (3 pooled, KL-matched at 0.0236 nats/token) 1.3 / 1.1 / 0.4 / 46.9 / 2679. -> 105% (honest), 110% (held-out) of the
  teacher's gain; noise -3%. Done; nothing running. Modal apps can be stopped by the coordinator (sub-gepa-lora_{inference,
  train,distill,random-lora}). Volume: /checkpoints/gepa_lora/{gl_*,noise/} (small adapters).

## TO FINISH
- [x] steps flowing; probe OK (see above)
- [x] val evals (3 configs)
- [x] analyze val
- [x] test x3 seeds + noise x3
- [x] analyze test; final report sent. Spend: billing report lags many hours (latest hourly interval 08:00 UTC at 16:00 PT); manual estimate ≈ 25 GPU-h H200 ≈ $110 (training 2.1, val 3.6, test 17.6, noise/probes 1.5). Re-run `spend` later to confirm.

## Spend
- not yet measured (3 H200 training containers running; evals not started)
