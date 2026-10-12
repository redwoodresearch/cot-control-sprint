# kaushik/core — the shared layer

One implementation each of config resolution, sweeps, the GPU scheduler, generic jobs, stats, result loading,
registry I/O and Modal env. Experiment folders import it; they never copy it. Design: [`../ARCHITECTURE.md`](../ARCHITECTURE.md).
Pilot migration: `kaushik/orange_expand/`.

```bash
R=".venv/bin/python kaushik/run.py"
$R --list                                              # experiments and jobs (legacy STAGES marked)
$R <exp> <job> [key=value ...] [job flags]             # one job on the resolved config, e.g.
$R orange_expand/sft train model=qwen8b run_name=x --dry-run
$R orange_expand/sft analyze --boot 2000               # experiment job: kaushik/<exp>/analyze.py:main(cfg, argv)
$R sweep orange_expand/sweeps/sft.yaml --dry-run       # expanded points + scheduler plan, launches nothing
$R sweep orange_expand/sweeps/sft.yaml --check "kaushik/orange_expand/sft/configs/*.yaml"   # config equivalence
$R sweep orange_expand/sweeps/sft.yaml                 # write runs/<run>/config.yaml, launch under the cap
```

## API

| module | what you call |
|---|---|
| `config` | `resolve(exp, point, overrides, labels, run_name) -> dict` (defaults.yaml < `kaushik/<exp>/config.yaml` (+`extends`) < sweep point < CLI; then model profiles fill gaps, `eval.grid` expands, `{label}` strings format); `train_eval_config(cfg)` (strips core keys: `experiment model profiles job eval_steps hooks registry checkpoints analysis data`); `defaults()`, `model_spec(key)`, `model_key_of(base_model)`; `run_dir_of`, `runs_root`, `results_dir`; `deep_merge`, `apply_overrides`, `parse_overrides`, `dump_yaml` |
| `sweep` | `load_sweep`, `expand_points`, `resolve_sweep(sweep, cli) -> [(point, cfg)]`, `check_equivalence(configs, paths, strict=False)` (eval blocks compared as train_eval sees them, DEFAULT_EVAL filled on both sides; `--strict` = raw YAML equality), `point_jobs(sweep, cfg)`, `main(argv)` |
| `scheduler` | `Job(name, weight, run=coroutine factory \| argv, done, ready, priority, log, pid_file, group, expand)`; `Scheduler(jobs, runs_dir, cap, ...)`; `run_scheduler(jobs, runs_dir, cap, name, dry_run, on_group_done)`. Cap file `<runs>/.gpu_cap`, stop file `.stop_launches`, 800-container and 25 %-memory guards, retries, `<runs>/<name>.{json,pid,log}` |
| `jobs` | `train_jobs/train_main` (scripts/train_eval.py `--no-eval`), `rl_jobs/rl_main` (`hooks.rl_task` + `hooks.rl_driver`, default `cotcontrol.training.rl_driver:run_rl`), `eval_jobs/eval_main` (one job per block x checkpoint; `eval_steps: final \| [ints]`; `checkpoints: [{step, path \| registry:<run>}]` for adapters without a train run), `eval_block_job(name, blk, base_model, tc, lora, out, backend_extra, priority, group, progress)` (one block x one adapter, any output stem, for foreign adapters such as noise controls), `registry_jobs/registry_main`, `status_main`; `prepare_run(cfg)` writes `config.yaml` + `resolved.yaml`; `JOBS` |
| `stats` | `matrix`, `boot_means`, `ci`, `cell_metrics`, `paired_boot` (seeds x questions; paired questions, independent seeds); `boot_1d`, `boot_seeds_questions`, `boot_paired_seeds` (plain arrays); `boot_interleaved` (per-draw question-then-seeds order, orange/rl); `untrained_bars`, `add_long` (compl_long), `with_pooled`, `passes_hit`; `pct`, `pct_ci`, `md_table`, `write_csv`; `POOLED`, `METRICS` |
| `results` | `parse_eval_json(path, source, extra)` rows keyed `qid="<dataset>:<id>"`, `task="<qid>\|<mode>"` with `chars` and `words`; `load_rollouts(sources, cache_dir, extra, cache_version="v3")` (mtime cache); `infer_condition`; `run_meta`, `parse_run_name`, `final_step`, `honest_rate` |
| `registry` | `entry_for_run(cfg, run_dir, final_step, run_name)`, `write(entry, force)`, `read`, `read_all`, `adapter_path`, `validate` (schema: `../registry/README.md`) |
| `modal_apps` | `setup_env(sub)` (owner + deployed app names from defaults.yaml; env already set wins), `app_names(sub)`, `check_apps()`, `workspace_containers()`, `checkpoint_committed(run_name, step)` |
| `io` | `atomic_write_json`, `read_json`, `read_jsonl`, `write_jsonl`, `append_jsonl`, `tail_jsonl`, `sha256_file`, `git_commit`, `load_py`, `load_callable("path.py:fn")`, `pid_alive`, `spawn`, `free_memory_pct`, `rel`, `ROOT`, `KAUSHIK` |

**Config schema** (`kaushik/<exp>/config.yaml`): `run_name` / `run_dir` templates, `model: <key>`, `profiles: [reasoning_off|reasoning_on, sft|rl]`,
`train`, `lora`, `rl`, `inference`, `task`, `meta` (pass through to train_eval / RLConfig),
`eval: {grid: {conditions, modes, prompts: path.py:fn, name, block}, grids: {key: grid, ...}, block: {...}, <named block>: {...}}`
— every block (grid-expanded or named) = defaults.eval.block < model.eval < `eval.block` < the grid's `block` patch < the block's own keys;
a named block set to `null` drops it. `eval_sections: [eval, final_eval]` (default `[eval]`) applies the same expansion, stripping and
`--check` normalisation to extra block sections (e.g. test-split blocks a folder's own eval code reads; `eval --section final_eval --out-dir ...`
evaluates them through the generic job);
core-only `hooks`, `registry {noise, measure_kl, config_path, extra}`, `eval_steps`, `checkpoints`, `analysis`, `data`,
`modal {sub}` (deployed sub app pair; `run.py` calls `setup_env(sub=...)` before the job).
Model entries (`kaushik/config/defaults.yaml`) fill gaps only: experiment > sweep point > CLI always win. API-only
models (`dsv4pro`, `glm53`, `gptoss20b_api`, `qwen8b_api`) carry just `base_model` (+ eval kwargs); no profiles.
Sweep schema: `experiment, job, after, run_name, seeds, seed_key, axes, overrides, max_gpus` (see `sweep.py`).
Experiment jobs are files `kaushik/<exp>/<job>.py` with `main(cfg, argv)` and, for sweeps, `jobs(cfg) -> [Job]`.

## Migration checklist (one folder at a time)

1. **Survey** the folder's `make_configs.py` / `MODELS` / eval-block builder / scheduler / analysis; note anything the
   model table in `defaults.yaml` lacks (add it there, never in Python).
2. **Write `config.yaml`** (per sub-experiment, `extends:` a shared parent) and `sweeps/*.yaml` whose points
   reproduce every existing run, including reruns and exceptions (give them their own axis entry and a comment).
   Prompts stay in the experiment (`eval.grid.prompts: kaushik/<exp>/prompts.py:fn`).
3. **Config equivalence** (migration rule 1): `run.py sweep <sweep> --check "kaushik/<exp>/configs/*.yaml"
   "kaushik/<exp>/runs/*/config.yaml"` must print `PASS`. Every `DIFF` is either a config-file error you fix or a real
   historical difference you document in the folder README (e.g. a run launched before a key existed). Add the
   sweep to `core/tests/test_sweep.py::ORANGE_SWEEPS`-style parametrization in your folder's tests.
4. **Port experiment code** to `<job>.py:main(cfg, argv)` files: data builders, analysis, figures, task hooks for RL.
   Use `core.stats` / `core.results` (same bootstrap draw order as the old `kaushik/orange/analyze.py` and the
   baseline variants) and read every path / hyperparameter from `cfg`.
5. **Analysis equivalence** (rule 2): copy `results/` aside, re-run `run.py <exp> analyze` with the committed
   bootstrap seed / draws, `diff` every table (`cmp` for csv/md). A difference is explained in the README or fixed;
   never silently accepted. Keep result paths unchanged: `kaushik/elicit/{orange,writeup}` read them.
6. **Dry-run every sweep** (`--dry-run`): the scheduler must report all existing outputs as `already done`; never launch.
7. **Delete superseded code** only after 3 and 5 pass: `cli.py`, `common.py`, `make_configs.py`, `train_run.py`,
   per-folder schedulers and stats copies. Keep `configs/*.yaml` (registry entries reference them), `runs/`, `data/`,
   `results/`, `kaushik/registry/*.json` and anything another folder loads by path (e.g. `noise/slots.py`).
8. **Hygiene**: `uvx ruff check kaushik/<exp> kaushik/core`, `uv run --with pytest pytest kaushik/core/tests kaushik/<exp>`,
   README commands updated, `STATE.md` current. No commits (the user commits); no Modal launches.
