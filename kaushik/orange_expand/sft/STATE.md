# kaushik/orange_expand/sft — STATE (last: 2026-10-09 16:20 PT)

## Scope
Orange expansion, SFT half (PLAN.md stage 1 + the combined analysis). Stage 0 (untrained evals,
pick rates, 27B mechanism check) moved to a separate baseline agent (`kaushik/orange_expand/baseline/`);
RL runs live in `kaushik/orange_expand/rl/`. Apps: `cot-control-sprint_kaushik-reddy_sub-orange-expand_{inference,train}`
(forced by `common.py`). Budget $300 (stop new launches at $240); GPU cap 80; no launches above 800 workspace containers.

## Files (all under this folder)
- `common.py` env + paths + MODELS table (qwen8b, gptoss20b from 2A; qwen32b H200 tp1; gptoss120b H200:4, eval tp2).
- `build_sft_data.py` -> `data/heldout_200.json` (never trained on) + `data/sft_{eugene,direct,standard}.jsonl` (1,936 rows each, same order).
- `make_configs.py` -> `configs/oex-sft_<variant>-<model>-seed0.yaml` (12; 100 steps, save 50/100, 8 eval blocks at step 100).
- `train_run.py` (2A's train_run with an `orange_expand` registry entry, noise/KL off), `cli.py` (stages), `analyze.py` -> `results/summary.md`.
- Runs: `runs/<run>/{config.yaml,manifest.json,train.log,train_result.json,launcher.json,eval/<block>/checkpoint-100.json,eval_progress.jsonl}`.
- Scheduler (2A's `kaushik/orange/eval_scheduler.py --runs-dir sft/runs`): `runs/eval_scheduler.{log,json,pid}`; live cap override `runs/.gpu_cap`.

## Status (16:15) — SFT stage 1 COMPLETE
- 12/12 trained (final loss <= 5e-4; checkpoints 50/100 on volume `cotcontrol_kaushik-reddy_checkpoints`, paths in
  `runs/<run>/train_result.json` and `kaushik/registry/oex-sft_*-seed0.json`, 12 entries, noise/KL off) and evaluated
  (96/96 jobs, 0 failed). Eval scheduler pid 36806 idles out by itself (~16:50); nothing else running from this folder.
- Spend: $135.31 at 16:16 (lags ~1 h); no further launches planned.

## Findings (results/summary.md, boot 2000)
- Honest compliance 0.0% in persona / nopersona / direct for every SFT arm on all 4 models, same as untrained
  (from `kaushik/orange_expand/baseline/runs/<model>/`). Standard (stated) condition within noise of untrained.
- PLAN hit rule (per condition pooled AND per mode): one per-mode-only hit, gptoss20b sft_eugene / direct /
  uppercase_thinking, +2.0 [0.5, 4.0] over both comparators (4/200 q; pooled 1.0), compl_long diff 0.0 -> length-driven.
  No robust SFT hit -> per PLAN, SFT stops (no more seeds/models).
- 27B mechanism check (2A adapters, 3 seeds): orange adapter persona 8.0 [4.6, 12.0] vs nopersona 7.9 [4.2, 12.2];
  orange_nopersona adapter 0.0 / 0.0; untrained 0.0 / 0.0. The effect is the negated-prohibition framing, not EUGENE;
  compl_long 0.0 (length-driven). Table in summary.md "Mechanism check".
- RL (rl/): rl_eugene gpt-oss-20b (final iteration 10) evaluated 16:16: honest 0.0 in all 4 conditions -> no RL hit
  on gpt-oss-20b. Qwen3-32B RL pending (ETA ~19:00).
- Side effects: qwen8b sft_standard collapses reasoning under the persona prompt (59% traces < 200 chars);
  gptoss20b sft_eugene 16% degenerate.

## Next
1. When rl/ evals land (RL agent: gpt-oss-20b ~17:00, Qwen3-32B ~19:00; layout rl/runs/oex-rl-eugene-*/eval/<block>/checkpoint-<T>.json)
   re-run `run.py orange_expand/sft analyze` (adds rl_eugene rows + RL hits vs untrained) and send the final report.
2. Optional teardown after RL is done: `modal app stop` the two sub-orange-expand apps (coordinator's call).

## Commands
```
export COTCONTROL_MODAL_INFERENCE_APP=cot-control-sprint_kaushik-reddy_sub-orange-expand_inference
export COTCONTROL_MODAL_TRAIN_APP=cot-control-sprint_kaushik-reddy_sub-orange-expand_train
R=".venv/bin/python kaushik/run.py orange_expand/sft"; $R status; $R train --dry-run; $R eval --cap 80; $R analyze
```
Only signal pids listed in `runs/*/launcher.json` and `runs/eval_scheduler.pid`.
