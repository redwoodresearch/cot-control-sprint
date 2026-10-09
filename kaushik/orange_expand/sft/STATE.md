# kaushik/orange_expand/sft — STATE (last: 2026-10-09 15:57 PT)

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

## Running now (15:57)
- 9/12 runs trained + fully evaluated (qwen8b, gptoss20b, qwen32b); gptoss120b: standard trained (evals in flight),
  eugene + direct still training (pids 36797, 36793; launcher.json in each run dir). Eval scheduler pid 36806 (cap 80).
- 72/96 eval jobs done, 0 failed. Interim: no PLAN hits on the 3 finished models (honest 0.0 in persona/nopersona/direct
  for every arm and untrained; standard condition within noise of untrained). results/summary.md is the interim table.

## Next
1. (done) smoke + launch of all 12 + scheduler.
2. Monitor every 20-30 min (`run.py orange_expand/sft status`; spend: `.venv/bin/python kaushik/elicit/spend.py sub-orange-expand`).
3. `run.py orange_expand/sft analyze` once baseline/ and rl/ evals exist -> `results/summary.md`.

## Commands
```
export COTCONTROL_MODAL_INFERENCE_APP=cot-control-sprint_kaushik-reddy_sub-orange-expand_inference
export COTCONTROL_MODAL_TRAIN_APP=cot-control-sprint_kaushik-reddy_sub-orange-expand_train
R=".venv/bin/python kaushik/run.py orange_expand/sft"; $R status; $R train --dry-run; $R eval --cap 80; $R analyze
```
Only signal pids listed in `runs/*/launcher.json` and `runs/eval_scheduler.pid`.
