# noise/ STATE (overwritten at milestones; last 2026-10-08 14:55)

## (a) Scope
KL-matched random-LoRA controls ("noise") for Exp 1+2 arms (spec Shared setup -> noise). Per
(experiment, arm[, dose suffix], base_model) GROUP: KL(base||adapter) for every registered seed -> 3 random
adapters (seeds 0/1/2) calibrated to the group's MEDIAN KL (within 10%) -> the arm's own final eval blocks ->
two-way bootstrap tables. Deliverables: runs/<group>/, results.md, README.md, this file. Never git commit.

## (b) Running (all via `.venv/bin/python kaushik/run.py noise ...`, nohup; cwd repo root)
- Watcher: `run.py noise watch --interval 150 --max-procs 9 --max-kl-procs 3 --parallel 8 --parallel-27b 16`
  (pid in `pgrep -fl "run.py noise watch"`; log runs/watch.log; status runs/progress.md; state
  runs/watch_state.json). Restart freely: `pkill -f "run.py noise watch"` + same command; it ADOPTS running
  `noise process --group` / `noise kl` jobs by pid. Restart it after editing cli.py/make_configs.py.
- Group jobs: `run.py noise process --group <G> --parallel N`; at 14:51 one running: orange-rl-rl_mcq_reason-qwen3627b.
  Only ever signal processes whose command line contains `run.py noise` / `noise process --group <G>`.
  Killing a group job is safe: finished chunks are in .generation_cache; the watcher respawns (2 min x attempts,
  max 5) and the eval stage skips existing eval/<block>/noise<k>.json.
- GPU budget: runs/gpu_cap.json (cap 80, coordinator 11:33), slots runs/_gpu_slots/. Watcher pauses launches and
  freezes the cap at slots-held while workspace containers > 800 or laptop free memory < 25%; restores after.

## (c) Owned files / infra
cotcontrol/training/random_lora.py (tensors, KL estimate, calibration, Modal CPU build app; deployed as
`cot-control-sprint_kaushik-reddy_random-lora`, redeploy with `run.py noise deploy` after editing it),
tests/test_random_lora.py (12 tests), kaushik/noise/{cli,make_configs,analyze,slots}.py, README.md, STATE.md,
configs/<group>.yaml (generated from kaushik/registry each poll) + configs/smoke_qwen8b.yaml, runs/.
Volume cotcontrol_kaushik-reddy_checkpoints: adapters at /checkpoints/noise/<group>/noise{0,1,2}.
runs/_samples/<model>-<hash>/ base samples + log-probs; runs/_kl/<run>.json (151 records: every trained adapter
+ <group>--noise<k>); runs/<group>/{manifest,noise_adapters,eval_summary,results}.json, eval/, process.log.

## (d) Done (results: kaushik/noise/results.md summary table + per-group sections; runs/<G>/results.json)
17/23 groups complete: fragility mcq_noreason x3 models, mcq_reason x2, style_sft x2, cpt + cpt-step50 x2 models;
orange + orange_nopersona on gptoss20b and qwen3627b; rl_mcq_noreason, rl_orange_noreason (qwen3627b).
Beats noise (honest, 95% CI excludes 0): mcq_noreason gptoss20b (4.9 vs 0.9), qwen38b (2.6 vs 0.4), qwen3627b
(1.1 vs 0.3, final_modes); orange qwen3627b persona blocks (8.4/8.1/4.6/1.6 vs 0.0) + direct lowercase/uppercase;
rl_mcq_noreason stated_uppercase (15.5 vs 2.1) + direct_lowercase; cpt gptoss20b test_heldout (1.7-2.4 vs 0.8).
Not: mcq_reason (= noise), style_sft (0, answer format broken), orange_nopersona qwen3627b (all CIs include 0).
Trained columns are n/a where the owning agent's test evals are not on disk yet (orange gptoss20b, parts of RL):
rerun `run.py noise analyze` later to fill them.
Calibration (median KL -> scale): see each runs/<G>/noise_adapters.json; random-direction penalty 10-30x the
trained per-module norm for the same KL (2-5x for tiny updates). Noise lengthens traces 2-4x, costs 3-9 acc
points vs 10-15 for training. Smoke: runs/_smoke/smoke.json (all checks pass).

## (e) TO FINISH (in order)
1. [ ] orange-rl-rl_mcq_reason-qwen3627b: running (calibrating/eval) -> auto.
2. [ ] orange-rl-rl_orange_reason-qwen3627b: 4-5 seeds registered; expected_seeds was inflated to 6 by the
       smoke config (fixed in make_configs; restart watcher) -> processes once seeds complete or 4h after first seen.
3. [ ] fragility-cpt-cpt_lowlr(-step50)-gptoss20b: 2/3 seeds each -> wait for seed 3 or the 4h timer.
4. [ ] cpt_param_sweep groups are KL-only (noise: false) -> nothing to build; KLs already in runs/_kl.
5. [ ] Watch for new registry entries (e.g. late seeds, orange gptoss20b test evals) until the watcher stops
       (24h or 4h idle); then `run.py noise analyze` (fills trained/untrained columns from source runs).
6. [ ] README.md "Results": copy the summary table + the 3 bullets above; note n/a columns and the penalty.
7. [ ] Final report to coordinator (<=3 lines): groups done, beats/not, paths.

## (f) Gotchas
- Groups are per arm (+ dose suffix after -seedN, e.g. cpt-step50); a late seed after a group is done only gets KL.
- "smoke" run names are ignored everywhere; smoke configs excluded from expected-seed counts.
- KL noise floor ~1e-4 nats/token; random adapters < 4x norm are indistinguishable from base.
- Modal client drops are frequent: _retry (5 attempts, backoff), per-eval isolation, cached chunks, respawn.
- ignore_question on Qwen3 -> empty think blocks -> validity gate 0 (base behaviour, not noise).
- Bare `import make_configs/analyze` collides across kaushik folders: cli loads siblings by path.

## (g) Status commands
sed -n 3p kaushik/noise/runs/progress.md; tail -5 kaushik/noise/runs/watch.log; pgrep -fl "run.py noise"
ls kaushik/noise/runs/*/eval_summary.json | wc -l    # complete groups (23 configs minus smoke/KL-only)
.venv/bin/python kaushik/run.py noise analyze         # regenerates results.md
