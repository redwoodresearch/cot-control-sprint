# kaushik/core — STATE (refactor phase A, 2026-10-12)

## Done
- `kaushik/config/defaults.yaml`: model table (qwen8b, qwen32b, qwen36_27b, gptoss20b, gptoss120b, qwen3_06b smoke)
  with train facts, profiles (reasoning_on/off, sft, rl), LoRA targets, eval template kwargs + TP; eval-block
  protocol (val, T=1, 16k, seed 0, dataset all); analysis (2000 draws, seed 0, degen 200); Modal (owner, app
  prefix, idle window, volume); scheduler (cap 40, 800 containers, 25 % memory, 3 attempts); paths.
- `kaushik/core/`: config, sweep, scheduler, jobs (train/rl/eval/registry/status), stats, results, registry,
  modal_apps, io, runner; README.md (API + migration checklist); tests in `core/tests/` (31 pass).
- `kaushik/run.py` dispatches: legacy `cli.py` STAGES (unchanged for unmigrated folders) > core jobs >
  `kaushik/<exp>/<job>.py:main(cfg, argv)`; `run.py sweep <yaml> [--dry-run|--check|--write]`.
- `scripts/train_eval.py`: additive `--no-eval` flag (train job uses it; default behaviour unchanged).
- Pilot `kaushik/orange_expand/`: config.yaml per sub-experiment, 9 sweeps, experiment jobs
  (`sft/build_data.py`, `sft/analyze.py`, `baseline/analyze.py`, `baseline/pick.py`, `rl/task.py`), RL stop rule kept in
  `rl/driver.py`, `rl/report.py` unchanged.
  - Config equivalence: 12/12 SFT, 3/3 RL, SFT smoke = `configs/*.yaml` and `sft/runs/*/config.yaml` (PASS).
    Explained diffs: `rl/runs/oex-rl-eugene-gptoss20b-seed0/config.yaml` lacks `gpu`/`tensor_parallel_size` in
    its eval blocks (launched before make_configs pinned the engine); the RL smoke configs predate the
    guard keys and the 2-GPU Qwen trainer.
  - Analysis equivalence: baseline 7/7 tables byte-identical; sft cells/differences/hits/mechanism_cells.csv +
    summary.md byte-identical (seed 0, 2000 draws); rollouts.csv.gz identical after decompression.
  - Superseded code deleted (sft/{cli,common,make_configs,train_run,build_sft_data}.py, rl/{cli,make_configs}.py,
    baseline/{run_all,common}.py): orange_expand 2826 -> 1310 lines of Python.
  - `kaushik/elicit/orange/make_figures.py` runs against the regenerated tables (checked on a scratch copy).

## Open
- Dry runs never write into run dirs (eval --dry-run skips prepare_run and the summary hook); a real launch writes
  `config.yaml` + `resolved.yaml` + logs there, as before.
- `kaushik/orange/rl` + `noise/slots.py` are loaded by path from `kaushik/elicit/*` and other folders: leave them
  until those folders migrate.
- Next agents: fragility (+cpt), orange (+rl), noise, retry, hidden_constraint — follow core/README.md checklist.
