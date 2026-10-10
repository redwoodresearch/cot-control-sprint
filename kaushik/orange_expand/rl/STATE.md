# STATE — orange expansion, RL half of stage 1 (kaushik/orange_expand/rl) — COMPLETE 21:45 Oct 9
# All three evals landed (Qwen ckpt-55 17:16; gpt-oss ckpt-10 16:18; gpt-oss lr5e-5 ckpt-58 21:44, its eval took
# ~80 min on 32 engines — the infra slowness persisted). Honest compliance 0.0 in every persona/nopersona/direct
# cell for all three; standard: Qwen 2.0 lower / 0.0 upper (untrained 3.5 / 0.0), gpt-oss 0.0. No RL hit.
# Tables: results/tables.md (report.py). Registry: oex-rl-eugene-{gptoss20b-seed0-it10, gptoss20b-seed0-lr5e-5,
# qwen3_32b-seed0}.json. No processes of mine run. Spend $235.11 through 20:00 (final expected $300-380).
# Final report sent to the coordinator; SFT agent pinged to fold in rl_eugene_lr5e-5.

## Scope
PLAN.md stage 1 RL: GRPO, reasoning ON, training prompt `eugene` (prompts.T_EUGENE == 2B's EUGENE prompt),
reward 1 iff the answer is the orange option; gpt-oss-20b and Qwen/Qwen3-32B, seed 0. 2B settings
(48 prompts x 8 rollouts, LR 1.5e-4 constant, LoRA r32/a32 all-linear, DAPO, no KL, rollout cap 8192).
Stop rule (driver.py): >= 50 iterations, then stop once mean training reward >= 0.98 for 20 consecutive
iterations; cap 150; abort if iteration-0 reward >= 0.9. Worker saves every iteration; held-out monitor at
0/25/50/.../final. Eval: final checkpoint, 4 conditions x 2 modes, val 200 q, T=1, 16k tokens ->
runs/<run>/eval/<condition>_<mode>/checkpoint-<T>.json (standard train_eval.py JSON).

## Files (all mine)
make_configs.py (configs/oex-rl-eugene-{gptoss20b,qwen3_32b}-seed0[-smoke].yaml), driver.py (stop rule +
EUGENE-mention logging + clean early stop: worker commits checkpoint-T, driver cancels it via the driver's own
cancel path and assembles train_result.json), cli.py (stages configs/train/eval/registry/status/eugene/wait).
Reused read-only: kaushik/orange/rl/{task.py,data/,cli.py helpers}. No cotcontrol/ changes.
Modal: cot-control-sprint_kaushik-reddy_sub-orange-rl_{inference,train} (pinned in cli.py; also exported).

## INSTABILITY EXCEPTION (coordinator decision 16:00; deviates from the stop rule)
The gpt-oss-20b LR 1.5e-4 run degenerated from iteration ~11: median reasoning grew ~1.35x/iteration
(563 chars at it9 -> 21,636 at it20), 30% truncated, reward 0.98 -> 0.66, zlib 0.58 -> 0.30, incoherent traces.
1. Run stopped at 16:02 (pid 81304 SIGTERMed; driver cancelled its worker). 21 iterations rolled out.
2. Checkpoint-10 (reward 0.97, >= 0.97 since it8, 816 chars, d4 0.98, zlib 0.54) evaluated on the 8 blocks
   (16:03-16:18, 8 x 3 engines): honest compliance 0.000 in every cell, acc 0.43-0.47, reasoning 4-7k chars ->
   runs/oex-rl-eugene-gptoss20b-seed0/eval/<block>/checkpoint-10.json; train_result.json final_step 10
   (stop_reason instability_exception_coordinator_it10); registry kaushik/registry/oex-rl-eugene-gptoss20b-seed0-it10.json.
3. Relaunched gpt-oss-20b at LR 5e-5 (otherwise identical; runs/oex-rl-eugene-gptoss20b-seed0-lr5e-5, pid 36225,
   launched 16:23) with the stop rule + instability guard (driver.py StopRule: unhealthy = median reasoning chars
   > 3x the iteration-0 median OR truncation > 5%; 3 consecutive unhealthy iterations -> stop, keep the last
   healthy checkpoint = checkpoint-(t-3)). Eval chain pid 36230 (wait -> eval --max-engines 24 -> registry).
4. Qwen3-32B keeps running under the same guard, applied externally (`cli.py guard`, pid 5651): it watches
   progress.jsonl, and on a trip writes train_result.json at the last healthy checkpoint first (so the eval
   chain proceeds) and then SIGTERMs the driver. Qwen at 16:25: it22, healthy (1.2k chars, 0% truncated).

## STOP-RULE CLARIFICATION (coordinator 16:55)
The 20-iteration streak now counts iterations whose 3-iteration moving mean of the training reward is >= 0.98
(StopRule.smooth = 3), because single-iteration dips to 0.979 were resetting the streak on a run saturated since
~it28; the rule's intent is "saturated for 20 iterations". Minimum 50 and the 150 cap are unchanged. Applied to
the two live drivers (started under the raw rule) through the external watchdog (`cli.py guard`, which on a
fire waits for checkpoint-(t+1) to be committed, writes train_result.json, then SIGTERMs the driver so the
armed eval chain starts); new driver processes use it natively. Guards: Qwen pid in runs/<run>/guard.pid,
gpt-oss lr5e-5 likewise (both also carry the instability guard).

## 17:02 Qwen3-32B CONVERGED: final checkpoint-55 (smoothed rule fired after it54, streak 20, smoothed 0.992;
## it54 reward 1.0; 55 iterations rolled out). Guard wrote train_result.json, SIGTERMed driver 57639 (worker
## cancelled). Registry kaushik/registry/oex-rl-eugene-qwen3_32b-seed0.json. Eval chain (pid 54704) runs
## `eval` (8 blocks x 4 engines) -> runs/oex-rl-eugene-qwen3_32b-seed0/eval/<block>/checkpoint-55.json.

## INFRASTRUCTURE EXCEPTION (coordinator 19:27): gpt-oss-20b LR 5e-5 stopped at checkpoint-58
The run was saturated (reward 0.97-1.0 since ~it33, healthy: 1.8k chars, 0% truncated, guard never tripped) with
the smoothed streak at 9/20 at it57, but from it53 each iteration took 5-27 min instead of ~1.3 min (rollout calls
170-1620 s; only 4 engine + 1 trainer containers running, workspace at ~300 containers -> engine preemption /
cold restarts, not the model). Stopped at checkpoint-58 (the policy after 25 saturated iterations), recorded as
stop_reason infrastructure_exception_coordinator_it58; eval with 32 engines; registry oex-rl-eugene-gptoss20b-seed0-lr5e-5.
Chain: runs/oex-rl-eugene-gptoss20b-seed0-lr5e-5/stop_eval_chain.{log,pid}.
20:21: the driver had been stuck in iteration 58's rollout call since ~19:25 (never logged it58) — stopped
cleanly (checkpoint-58 committed 20:20, finalized, registry written, driver SIGTERMed -> worker cancelled; guard
and old eval chain killed). Eval of checkpoint-58 running (8 blocks x 4 engines) -> registry -> report.py.
Spend $235.11 through 20:00; final expected ~$255.

## 17:16 Qwen eval LANDED (13 min, 32 engines): honest 0.0 in persona/nopersona/direct x both modes; standard
## lowercase 0.020, uppercase 0.0; acc 0.45-0.52; reasoning 7-8k chars. Coordinator + SFT agent messaged.
## Remaining: gpt-oss lr5e-5 (pid 36225, guard pid in guard.pid, eval chain pid 36230) -> its eval -> final report.
## 17:28 gpt-oss lr5e-5 it31: smoothed 0.963, chars 575 -> 968 (it-0 703; guard trips > 2109 x3), EUGENE 0.94 —
## same pattern as the blown-up run; guard armed. Expected stop ~it53 (~17:58) if healthy, eval ~18:15.

## 16:50 status
- Qwen it42: reward 0.97-0.997 since ~it28 (saturated, healthy: ~900 chars, 0% trunc), but 0.979 iterations reset
  the 20-streak -> may run to the 150 cap (~1.5 min/it now). EUGENE mention 0.38 -> 0.01 (it12) -> 0.33 (it42).
- gpt-oss lr5e-5 it6: reward 0.50, 650 chars, healthy, ~1.4 min/it.
- Budget projection at 150 iterations each ~$400 > cap; coordinator messaged with the plan (stop Qwen at its last
  checkpoint when the projection reaches $380, ~it90-100 / ~18:00) and an alternative (moving-mean streak).

## Running now (15:09)
- Smokes (3-iteration cap, stop rule fires at checkpoint-2, 4x4 rollouts, 2 tiny eval blocks):
  qwen3_32b PASSED 15:06 (train; early-stop path exercised: it0 reward 0.5, stop at checkpoint-2, worker
  cancelled, train_result.json written); gptoss20b PASSED 15:12 (it0 reward 0.625; expert-param LoRA worker
  trained; same early-stop path). Smoke evals PASSED for both (2 blocks x 6 q on checkpoint-2 ->
  runs/<run>-smoke/eval/<block>/checkpoint-2.json, standard train_eval.py JSON). Smokes are finished; no smoke
  process runs.
- Full runs (coordinator: launch without waiting for the smokes; 40-GPU cap):
  gptoss20b pid 81304 (launched 15:07, 1 trainer + 4 engines), qwen3_32b pid 57639 (launched 15:15: 2-GPU
  FSDP trainer `gpu: H200:2` + 8 one-GPU engines). Aborted attempts in runs/_aborted/ (workers cancelled by
  the driver on SIGTERM): attempt1 15:08 (1-GPU trainer, superseded), attempt2 15:14 (my config bug put every
  engine on H200:2 — make_configs now pins inference/eval gpu to the base type). attempt2's iteration-0
  rollouts (384, base model) give the untrained Qwen3-32B pick rate: reward 0.440, EUGENE mention 0.367,
  median reasoning 1446 chars. Logs runs/<run>/train_stdout.log.
- GPUs booked: gpt-oss 5 + Qwen 10 of 40.
- 15:23 iteration-0 rewards (untrained): gpt-oss 0.451 (held-out 0.346, EUGENE 0.48, 685 chars),
  Qwen 0.440 (held-out 0.388, EUGENE 0.38, 1458 chars). Coordinator messaged. gpt-oss it1 reward 0.62,
  warm rollouts 33 s. ETA to the 50-iteration floor: gpt-oss ~16:40, Qwen ~17:50.
- 15:49: gpt-oss it17 reward 0.96 (~1.4 min/it; EUGENE mention ROSE 0.48 -> 0.94-0.98); Qwen it8 reward 0.50
  (~3.7 min/it incl. ~130 s train step; EUGENE mention FELL 0.38 -> 0.06). Revised floor ETA: gpt-oss ~16:35,
  Qwen ~18:20. SFT agent's stage-0 rates agree with it-0 rewards (46.5% / 43.5%). Spend report still $0 (lag).
- Eval chains armed (wait -> eval -> registry): gpt-oss pid 59403 (`--max-engines 24`, since Qwen may still
  train), Qwen pid 54704 (default 32 engines); chunk 50 -> 4 engines per block. Logs runs/<run>/eval_chain_stdout.log.

## Spend
Cap $380, stop new launches at $300. 16:24: $75.24 (inference 61.83, train 13.42; completed hours only, lags
~1 h) -> burn ~$75-90/h at the current load (Qwen 2+8 GPUs, gpt-oss 1+4, evals). Projection: Qwen to it50-90
~$160, gpt-oss lr5e-5 ~$55, evals ~$30 -> ~$320 total. No further launches; if the projection crosses $380,
stop Qwen at its last checkpoint (SIGTERM driver -> `finalize --step <last>`).
17:05: $154.36 through 17:00. Remaining: Qwen eval (~$55), gpt-oss lr5e-5 training (~$40) + eval (~$30)
-> ~$280 projected total.
18:20: $218.94 through 18:00. Remaining: gpt-oss lr5e-5 last iterations + its eval -> ~$250 projected total.

## Next steps
1. Smokes pass -> launch full runs (rollout shards: gpt-oss 4, Qwen 8; GPU cap 40) -> message main with
   iteration-0 rewards + ETA.
2. On train_result.json: `eval runs/<run>` (8 blocks concurrently, 2 engines each), `registry`, message main.
3. Final report: it-0/final reward, iterations to convergence, eval table, EUGENE trajectory, spend, paths.

## Gotchas
- Only signal pids listed here; never pattern-kill. Deployed-app workers outlive a dead driver: the driver
  cancels its call on abnormal exit (SIGTERM-safe); if a driver is SIGKILLed, abandon the run name and use
  `kaushik/run.py orange/rl zombies` (read-only report) to find the orphan.
