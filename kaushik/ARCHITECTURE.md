# kaushik/ architecture (refactor completed 2026-10-12)

Same principles as Sebastian's setup: one YAML schema drives every run (`cotcontrol/training/config.py` +
`scripts/train_eval.py`), single-purpose scripts, sweeps that expand a base config. Unlike his setup, the eval
protocol and the sweep axes live in YAML too, so no hyperparameter is a Python literal.

## 1. Config: one tree controls everything

Effective config for a run, each layer overriding the previous one with dotted keys:

1. `kaushik/config/defaults.yaml`: global defaults.
   - models: HF id, training GPU spec, LoRA targets, model facts for `train` (e.g. attention backend), *profiles*
     (`reasoning_on` / `reasoning_off` / `sft` / `rl`: chat-template kwargs, prefill, memory settings) and the
     per-block `eval` patch (eval-time template kwargs, tensor parallelism). A run config names `model: <key>`
     and `profiles: [...]`; the entry fills gaps only, it never overrides what the experiment set.
   - eval protocol: the `eval.block` keys merged under every expanded block (split, temperature, `max_tokens`,
     seed, dataset), mode lists, judge; the honest metric is `cotcontrol.eval.grading.is_honest`
   - analysis: bootstrap draws and seed, degenerate threshold
   - Modal: owner, app prefix, idle window, checkpoints volume
   - scheduler: GPU cap, the 800-container guard, the memory guard, retries
   - paths
2. `kaushik/<exp>/config.yaml`: only what this experiment changes (arms, data, LR/steps/batch/LoRA, eval blocks).
   May `extends:` a parent (e.g. `orange_expand/config.yaml` shared by `sft/`, `rl/`, `baseline/`). Eval blocks
   are usually a grid: `eval.grid: {conditions, modes, prompts: kaushik/<exp>/prompts.py:fn, name}` plus
   `eval.block` (keys common to every block); prompts stay Python because they are text, not hyperparameters.
3. A sweep point (§3). Its labels format `{field}` strings anywhere in the config (`run_name`, `data_path`, `meta`).
4. `key=value` overrides on the command line.

Core-only keys (`experiment`, `model`, `profiles`, `job`, `eval_steps`, `hooks`, `registry`, `checkpoints`,
`analysis`, `data`) are stripped by `train_eval_config()`; what remains is a valid `scripts/train_eval.py` /
`RLConfig` YAML, so Sebastian's runners consume it unchanged. Python code contains no hyperparameter literals;
schema defaults live in `defaults.yaml`.

## 2. Scripts: one job each, config in, files out

- **Generic jobs** live in `kaushik/core/jobs.py` and are called through the runner:
  - `train`: delegates to `scripts/train_eval.py <run>/config.yaml --no-eval` (SFT; `--no-eval` is additive)
  - `rl`: delegates to `cotcontrol.training.rl_driver.run_rl` with `RLConfig`, or to the driver named by
    `hooks.rl_driver` (same signature; orange_expand's stop-rule driver); the prompt source and reward come from
    `hooks.rl_task: path.py:fn(cfg) -> {source, reward_fn, heldout, ...}`
  - `eval`: one scheduler job per (block, checkpoint) through train_eval's helpers; checkpoints from
    `train_result.json` (`eval_steps: final | [steps]`) or `checkpoints: [{step, path | registry:<run>}]`;
    base-model evals via `eval_base_model: true`
  - `registry`: the adapter hand-off (`registry:` section of the config supplies flags and extras)
  - `status`: run dirs + scheduler state
- **Experiment-specific jobs** sit in the experiment folder as small files, e.g. `build_data.py`, `analyze.py`,
  `figures.py`, each exposing `main(cfg, argv)` and, when a sweep should schedule them, `jobs(cfg) -> [Job]`.
  They take only the resolved config and paths.
- **Entrypoint:** `kaushik/run.py <exp> <job> [key=value ...]` and `kaushik/run.py sweep <sweep.yaml> [--dry-run]`.
  Lookup order: legacy `cli.py` STAGES (unmigrated folders) > generic jobs > `kaushik/<exp>/<job>.py`.

## 3. Sweeps: YAML files that orchestrate

`kaushik/<exp>/sweeps/<name>.yaml`:

```yaml
experiment: orange_expand/sft      # config.yaml it starts from
job: train                         # train | rl | eval
run_name: "oex-sft_{variant}-{model}-seed{seed}"
seeds: [0]
axes:                              # cartesian product of dotted-key overrides
  - [{_label: {model: qwen8b}, model: qwen8b}, {_label: {model: gptoss20b}, model: gptoss20b}]
  - [{_label: {variant: eugene}, train.system_prompt: eugene}, ...]
max_gpus: 40
```

`seeds: [..]` is sugar for an axis on `seed_key` (default `train.seed`); `after: [registry, eval]` queues generic
jobs per point once the main job's outputs exist (an eval queued before training finishes is a placeholder that
expands into per-checkpoint jobs when `train_result.json` appears).

`kaushik/core/sweep.py` does five things:
1. expands the axes;
2. writes each resolved config to `runs/<run>/config.yaml` (never overwriting a trained run's config) plus
   `resolved.yaml` (the full tree, read back by subprocess jobs);
3. skips runs that already have outputs, as Sebastian's launchers do;
4. launches through one shared scheduler (GPU cap file plus guards);
5. logs progress.

`--dry-run` prints the expanded points and the scheduler plan; `--check <glob...>` compares the expanded
configs with existing YAMLs (migration rule 1); `--write` only writes run configs.

## 4. Shared library: one implementation each (`kaushik/core/`)

- config loader/merger
- sweep expansion
- GPU scheduler and guards
- model table access
- eval-block builder
- stats: bootstrap over questions × seeds, honest compliance, `compl_long`, paired differences
- result loaders keyed by `(dataset, id, mode)`
- registry read/write
- Modal app selection
- I/O helpers: atomic JSON, JSONL, git commit

Experiment folders import these; they never copy them.

## 5. What doesn't change

- Output layout: `kaushik/<exp>/runs/<run>/` in the train_eval layout, `results/`, and registry JSON format.
- Existing run outputs, data and registry entries are never deleted or rewritten.

## Migration rules (every folder)

1. **Config equivalence.** The new sweep, expanded, must reproduce every existing per-run config: semantic YAML
   equality with existing `configs/*.yaml` and `runs/*/config.yaml`.
2. **Analysis equivalence.** Re-running `analyze` on the existing runs must reproduce the committed `results/` tables.
   A difference is either explained (a real bug, documented in the README) or fixed. It is never silently accepted.
3. **No spend.** Dry runs only; no Modal launches.
4. **Hygiene.** Tests and `ruff` must pass. README commands are updated. Superseded code is deleted only after 1 and 2
   pass; generated `configs/*.yaml` stay (registry entries point at them), as do `runs/`, `data/`, `results/` and
   any module another folder loads by path (`noise/slots.py`, `orange/rl/task.py`, `orange-training/*`).
5. **Hands off.** `kaushik/elicit/writeup/` and `kaushik/elicit/orange/` are left to another session, and must keep
   working: `writeup/collect.py` and both figure scripts run.
