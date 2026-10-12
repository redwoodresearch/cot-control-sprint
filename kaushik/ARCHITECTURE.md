# kaushik/ architecture (target; refactor started 2026-10-12)

Same principles as Sebastian's setup: one YAML schema drives every run (`cotcontrol/training/config.py` +
`scripts/train_eval.py`), single-purpose scripts, sweeps that expand a base config. Unlike his setup, the eval
protocol and the sweep axes live in YAML too, so no hyperparameter is a Python literal.

## 1. Config: one tree controls everything

Effective config for a run, each layer overriding the previous one with dotted keys:

1. `kaushik/config/defaults.yaml`: global defaults.
   - models: HF id, `model_key`, chat-template kwargs for training and eval, LoRA targets, GPU/TP for train and eval
   - eval protocol: split, temperature, `max_tokens`, modes, held-out modes, judge, honest-metric thresholds,
     bootstrap draws
   - Modal: owner, app names, idle window
   - scheduler: GPU cap, the 800-container guard, the memory guard
   - paths
2. `kaushik/<exp>/config.yaml`: only what this experiment changes (arms, data, LR/steps/batch/LoRA, eval blocks).
3. A sweep point (§3).
4. `key=value` overrides on the command line.

The training part of the resolved config is a valid `scripts/train_eval.py` / `RLConfig` YAML, so Sebastian's
runners consume it unchanged. Python code contains no hyperparameter literals; schema defaults live in
`defaults.yaml`.

## 2. Scripts: one job each, config in, files out

- **Generic jobs** live in `kaushik/core/` and are called through the runner:
  - `train`: delegates to `scripts/train_eval.py` (SFT)
  - `rl`: delegates to `cotcontrol.training.rl_driver` with `RLConfig`
  - `eval`: eval blocks on checkpoints or the base model
  - `registry`: the adapter hand-off
- **Experiment-specific jobs** sit in the experiment folder as small files, e.g. `build_data.py`, `analyze.py`,
  `figures.py`. They take only the resolved config and paths.
- **Entrypoint:** `kaushik/run.py <exp> <job> [key=value ...]` and `kaushik/run.py sweep <sweep.yaml> [--dry-run]`.

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

`kaushik/core/sweep.py` does five things:
1. expands the axes;
2. writes each resolved config to `runs/<run>/config.yaml`;
3. skips runs that already have outputs, as Sebastian's launchers do;
4. launches through one shared scheduler (GPU cap file plus guards);
5. logs progress.

`--dry-run` prints the expanded configs.

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
   pass.
5. **Hands off.** `kaushik/elicit/writeup/` and `kaushik/elicit/orange/` are left to another session, and must keep
   working: `writeup/collect.py` and both figure scripts run.
