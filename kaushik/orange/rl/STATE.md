# MIGRATED to kaushik/core (2026-10-12): config.yaml (eval + final_eval sections) + sweeps/{rl,rl_eval,rl_smoke}.yaml
# replace cli.py / make_configs.py and the per-folder Scheduler; task.py:build is hooks.rl_task, build_data.py the
# pool job; `kaushik/run.py orange/rl <job>` (README "Reproducing"). Checks: sweeps reproduce all 20 run configs +
# 4 smoke configs (configs/*noreason* differ only in save_optimizer_every 5 vs the 10 the runs used); analyze.py
# byte-identical to the retired analyzer on today's data; figures/_analysis differs only by evals that finished after
# its snapshot (README). Code 2200 -> 1082 lines of Python incl. the test file (+ config.yaml, 3 sweeps). Not done: core's eval job cannot see the live/ and final/part*/ eval layouts of the finished
# runs as done, so sweeps/rl_eval.yaml is for new runs only. The notes below are the sprint history.

# STATE — Experiment 2B (orange RL vs rl_mcq) — SPRINT SHUT DOWN by the user ~16:50 Oct 8 (coordinator kills processes/Modal apps)
# Monitors stopped; no final `analyze` into results/ was run (results/ = 14:10 dry run; /tmp/rl2b_check = 16:30 snapshot with
# all 375 test JSONs; val 1010+237 of 1440; noise reason groups partial). To resume: TO FINISH steps 2–4 below.

## (a) Scope
Spec: kaushik/sprint_2026_10_08.md §2B. GRPO on Qwen3.6-27B, 4 arms x 5 seeds: rl_orange_{noreason,reason} (EUGENE
persona, reward = orange option) vs rl_mcq_{noreason,reason} (generic MCQs, reward = correct, no persona). Quantity of
interest: (rl_orange − rl_mcq) per variant on persona/direct/stated with bootstrap CIs over questions x seeds; untrained
(checkpoint-0) and KL-matched noise (kaushik/noise/runs/, built by the noise agent from kaushik/registry/rl2b-*.json) as
reference rows. Matched to 2A: 50 iterations x 48 prompts x 8 rollouts, LR 1.5e-4, LoRA r32/a32 all-linear, no KL.
README.md documents design, dose matching, eval blocks, layouts, gotchas. ALL 20 RUNS ARE TRAINED (14:46); 20 registry entries.

## (b) Running now (14:51) — leave alone; the scheduler finishes everything unattended
- Scheduler pid 39439: `.venv/bin/python kaushik/run.py orange/rl launch --cap 200` (log runs/_scheduler/scheduler.log,
  state runs/_scheduler/state.json, rewritten every minute: "running" = job id -> pid). It starts eval jobs as GPUs free,
  retries failures (limit 8; the generation disk cache replays finished chunks), and exits with "scheduler: all jobs
  finished" when every run has val + final evals. Adopt/restart: `kill 39439` (that pid ONLY), then
  `.venv/bin/python kaushik/run.py orange/rl launch --background --cap 200` — it adopts live jobs via pid files
  runs/<run>/{train,live,final}_stdout.pid and never duplicates; a runs/<run>/.hold file makes it skip that run.
- Eval jobs (local processes; each spawns train_eval.py parts that call the deployed Modal inference app):
  final (test blocks, 15 GPUs): mcq_reason seed1 pid 89745, seed2 27540, seed4 36067; orange_reason seed0 48324.
  live (val blocks per checkpoint, 12 GPUs): mcq_noreason seed0 56868, seed1 48325, seed2 56771, seed3 68990;
  orange_reason seed0 98938, seed1 3553, seed2 16789, seed3 25611, seed4 47703. Queued: remaining live/final jobs.
  Cmd lines contain `kaushik/run.py orange/rl eval_live|eval_final kaushik/orange/rl/runs/<run>`.
- Progress 14:51: val 843/1440 JSONs (72 per run), test 302/375; runs fully evaluated: val 4/20, final 14/20. ETA ~16:30.
- 15:05: scheduler pid 39439 verified alive and adopted as-is (nothing relaunched). A Monitor task (re-armed every
  30 min) watches scheduler.log for "all jobs finished", new ` rc=[1-9]` lines and pid 39439 death. No training
  drivers run. Modal: deployed apps cot-control-sprint_kaushik-reddy_{train,inference}.
- 15:00: analyze.py fixed — the noise agent names its dirs orange-rl-<arm>-qwen3627b (not after our run names), so
  load_noise_rows now matches groups through manifest.json / noise_adapters.json and treats each eval/<block>/noise<k>.json
  as one noise seed; noise rows are shown only for the entry's own variant; a KL(base‖adapter) table (noise/runs/_kl,
  noise target/achieved) is rendered. Dry run to /tmp/rl2b_check OK (noise rows + KL table present; 25 tests pass).
  Noise evals exist for rl_mcq_noreason, rl_orange_noreason (3 adapters each); rl_mcq_reason is being evaluated by the
  noise agent and rl_orange_reason's noise is not built yet (its ETA ~16:30) — rerun `analyze` once
  kaushik/noise/runs/orange-rl-rl_{mcq,orange}_reason-qwen3627b/results.json exist.
- 16:30: analyze.py compl_long bar fixed — on test only rl_mcq_noreason's checkpoint-0 exists, so other arms' bar fell
  to None and compl_long silently equalled raw; bar_for() now falls back run → arm → all-arm step-0 median (checkpoint-0
  is the same base model for every run) and compl_long is NaN without a bar. checkpoints.json now carries the bar too.
- 16:20 snapshot (test, final checkpoints, modes pooled, honest %): persona ≈0 for every row (orange 0.2/0.3 vs mcq 0.0,
  untrained 0.0, noise 0.0); direct orange−mcq −0.4 [−1.7, 0.8] (noreason) / −0.7 [−2.8, 1.3] (reason); stated −0.7
  [−2.0, 0.3] / −0.2 [−0.7, 0.2]. Nothing beats its baseline. compl_long ≈0 everywhere. Orange arms lose 2–6 acc points
  vs mcq arms. Test JSONs 375/375 complete at 16:19; remaining: 8 noreason live (val) jobs + noise reason groups.

## (c) Owned files
cotcontrol/training/rl_driver.py (general pluggable prompt-source + reward GRPO driver), tests/test_rl_driver.py;
kaushik/orange/rl/{cli,task,make_configs,analyze,test_orange_rl}.py, README.md, STATE.md, configs/*.yaml, data/ (pool
slices + manifests), runs/<run>/ (config.yaml, config_resolved.json [Modal run_name], manifest.json, train_stdout.log,
train.log, progress.jsonl, iters/, batches/, metrics.jsonl, train_result.json, live/ckpt-<step>/part<k>/eval/<block>/,
final/part<k>/eval/<block>/, legacy eval/<block>/ for the first reasoning-OFF val blocks, reshard.log), runs/_scheduler/,
runs/_aborted/ (attempt-1 dirs of restarted runs; zombie_run_names.txt = poisoned/abandoned Modal run names),
results/ (analyze output), kaushik/registry/rl2b-*.json (20).

## (d) Done + findings so far
- Infra built and verified (smokes on 0.6B and 27B, deployed-mode smoke), data (orange 1936/200, mcq 1900/200, 0 eval
  overlaps), 20 configs, GPU-capped scheduler, overlapped evals, analysis pipeline (dry-run OK at 14:10 on partial data).
- Reasoning-OFF: reward saturates in 2-3 steps (orange 0.5→0.98; mcq starts 0.97). Realized dose: ~64 (mcq) / ~89 (orange)
  of 2400 prompt-groups ever carried gradient. Reasoning-ON: mcq_reason starts ~0.97 (107 groups with gradient);
  orange_reason climbs 0.38→~1.0 by iteration 15-25 (178 groups), held-out orange reward 0.33→0.99.
- Partial val numbers (14:10, incomplete seeds): persona honest compliance ≈0% for rl_orange_noreason (0.4), rl_mcq_noreason
  (0.0) and untrained (0.0); direct 4.3/4.8/2.4; stated 0.0/2.7/0.4 — no orange-RL lift so far.
- Drift: reasoning length shrinks as the rule is learned (orange_reason median chars ~3.5k→1.3-2k on 3 seeds, ~5-7k on 2;
  mcq_reason seed3 ended ~900 chars); the per-checkpoint chars/degen/compl_long columns quantify it.

## TO FINISH (in order)
1. Wait until `grep -c "all jobs finished" kaushik/orange/rl/runs/_scheduler/scheduler.log` is 1, or
   `.venv/bin/python kaushik/run.py orange/rl status` shows val-json 72 and final-json 15 (30 for rl_mcq_noreason, whose
   checkpoint-0 is also evaluated on test) for every run. Failures: scheduler log lines ` done ... rc=[1-9]`.
2. `.venv/bin/python kaushik/run.py orange/rl analyze` (≈1 min) → results/tables.md (headline, per-mode, realized dose,
   final test blocks), summary.json, checkpoints.json, training.json, eval_curves.png, training_curves.png.
   Noise rows appear automatically if kaushik/noise/runs/<dir containing our run_name>/ holds eval result JSONs.
3. README.md "## Results": paste the headline table (val and test; (orange − mcq) rows with CIs and seed spread), the
   dose table, the training curves, and state which comparisons beat the matched baseline (none expected). Caveats to
   include verbatim: (i) realized gradient dose far below SFT's 50x48 — rl_mcq arms near saturation from step 0, so
   (orange − mcq) risks ≈ (orange − untrained); compare at matched functional size via the noise agent's KL(base‖adapter)
   measurements (kaushik/noise/runs/) rather than steps; (ii) reasoning-length drift / degen with reasoning ON (report
   compl_long next to compliance); (iii) in the `reason` variant CoT tokens receive policy gradient though the reward
   ignores them; (iv) rl_mcq_reason/orange_reason runs were restarted/resharded (runs/<run>/reshard.log, _aborted/):
   first-attempt iterations ≤10 were redone, rollout shards changed (2→5 / 2→4), per_device_batch_size 1, optimizer
   saves every 5 — none affect the comparison; (v) progress.jsonl/metrics.jsonl can hold duplicate iterations from
   restarts — analyze.py keeps the last entry per iteration.
4. Final report to the coordinator (files built, 20 runs, results with intervals, what beat its baseline, blocked items,
   desired changes to files not owned: none pending; new file cotcontrol/training/rl_driver.py + tests/test_rl_driver.py).
5. Optional cleanup after the sprint: `modal app stop` is the coordinator's call; smoke run dirs (runs/*smoke*) are
   excluded from analysis by name; `_aborted/` attempt dirs are kept as evidence.

## (f) Gotchas
- NEVER pattern-kill shared scripts (train_eval.py etc.) — an 11:36 `pkill -f train_eval.py` of mine killed ~36 of the
  fragility/cpt agent's drivers. Signal only pids from our pid files or command lines containing kaushik/orange/rl/.
- Deployed Modal apps are shared across kaushik/ experiments; a worker spawned there outlives a killed driver and
  _RLRemote retries 3x; `modal container stop` only reschedules. Orphan on run name X: abandon X (move runs/<run> to
  _aborted, add X to zombie_run_names.txt, `orange/rl zombies --poison`), restart fresh. `orange/rl zombies` reports.
- Reshard (`orange/rl reshard runs/<run> --shards N`) resumes from the latest checkpoint-<r>/optimizer.pt after
  trimming the volume to r; writes .hold meanwhile. Verified reward/loss/logprob-corr continuity (≥0.997).
- Scheduler accounting keeps finished/killed jobs' GPUs for 10 min (engine scale-down) so `used` can read above the cap
  right after restarts; guards: >800 workspace containers or <25% laptop free memory hold launches.
- Transient Modal "Deadline exceeded": drivers retry/tolerate; eval parts rerun (cache-backed). OOM: reasoning-ON
  needs per_device_batch_size 1. Dashboard _RLRemote errors (29 at 12:20) are all accounted: poisoned/abandoned (16),
  reshard cancels (5), cancels after transient driver deaths (6); every live call verified advancing.

## (g) Status commands
`.venv/bin/python kaushik/run.py orange/rl status`; `tail -5 kaushik/orange/rl/runs/_scheduler/scheduler.log`;
`python3 -c "import json;st=json.load(open('kaushik/orange/rl/runs/_scheduler/state.json'));print(st['used'],st['cap_now'],list(st['running']))"`
`find kaushik/orange/rl/runs -path '*seed[0-4]/live/ckpt-*/part*/eval/*/checkpoint-*.json' | wc -l` (val; + legacy eval/*/)
`find kaushik/orange/rl/runs -path '*seed[0-4]/final/*/eval/*/checkpoint-*.json' | wc -l` (test, total 375)
Per-job logs: runs/<run>/{live,final}_stdout.log and runs/<run>/{live/ckpt-*/part*,final/part*}/train_eval_stdout.log.
