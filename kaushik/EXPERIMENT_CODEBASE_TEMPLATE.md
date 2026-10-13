# Experiment codebase template (Sebastian's style, as refactored in `kaushik/`)

Hand this to an agent starting a new ML-experiment project.

## 1. Two kinds of compute: setters and getters

- **Setters (training)** change model weights. They take a config and data, write checkpoints, and do nothing else.
  Examples: `train(cfg)`, `rl(cfg)`. Tinker equivalent: `forward_backward` + `optim_step`.
- **Getters (sampling/eval)** read a checkpoint and produce outputs. They never mutate weights or shared state.
  Examples: `generate(prompts, checkpoint)`, `score(...)`, `evaluate(cfg, checkpoint)`. Tinker equivalent: `sample`.
- **The bridge** is a checkpoint store plus a registry. A setter saves and commits (`save_weights_for_sampler` /
  volume `commit()`). A getter reloads and loads by path. The pipeline is "train, then evaluate every saved checkpoint".
  Never evaluate inside the training loop.

## 2. Two kinds of code

- **Shared library** (`<pkg>/`): eval, inference backends, training infrastructure, stats, config, sweeps, scheduler.
  It holds no experiment state, and every capability is written once.
- **Experiment folders** (`<exp>/`): self-contained code plus outputs. They import the library and never copy it.

## 3. One config tree controls everything

- Each run is described by one typed config (a dataclass schema plus YAML).
- Layers, later ones winning: `defaults.yaml` < `<exp>/config.yaml` < sweep point < CLI `key=value`. Overrides use
  dotted keys, e.g. `train.lr=1e-4`.
- `defaults.yaml` holds models (id, chat-template kwargs, LoRA targets, GPU/TP), the eval protocol (temperature,
  max_tokens, splits, metric thresholds, bootstrap draws), compute (apps, GPU cap, guards) and paths.
- `<exp>/config.yaml` holds only what that experiment changes.
- No hyperparameter literals in Python. The fully resolved config is saved into every run directory.

## 4. Single-purpose scripts: config in, files out

- `build_*.py`: create data or caches (writes).
- `train` / `rl`: setters.
- `eval`: getter.
- `analyze.py`, `figures.py`, `load_*`: read-only over run outputs.
- Every script is idempotent: it skips work whose outputs exist, and supports `--dry-run`.
- There is one entrypoint: `run.py <exp> <job> [key=value ...]`.

## 5. Sweeps are YAML files

```yaml
# <exp>/sweeps/lr_sweep.yaml
experiment: my_exp
job: train
run_name: "{model}-lr{lr}-seed{seed}"
seeds: [0, 1, 2]
axes:                               # cartesian product of dotted-key overrides
  - [{model: qwen8b}, {model: gptoss20b}]
  - [{train.lr: 1e-4, _label: {lr: 1e-4}}, {train.lr: 3e-4, _label: {lr: 3e-4}}]
max_gpus: 40
```

`run.py sweep <file> [--dry-run|--check]` does five things:
1. expands the axes;
2. writes `runs/<run>/config.yaml`;
3. skips finished runs;
4. launches through ONE shared scheduler (GPU cap file plus capacity/memory guards);
5. resumes after interruption.

## 6. Run layout: save everything

```
<exp>/
  config.yaml   sweeps/*.yaml   build_data.py   analyze.py   figures.py   README.md   STATE.md
  data/                                   # built datasets (+ manifest with sha256)
  runs/<run>/config.yaml                  # resolved config
            train.log  train_result.json  # losses, checkpoint paths
            eval/<block>/checkpoint-<step>.json   # every prompt, output, grade
            eval_summary.json
  results/                                # tables/CSVs produced by analyze.py
registry/<run>.json                       # checkpoint path + config + description, one per final adapter
```

## 7. Analysis and stats

- Analysis reads `runs/` only.
- One shared stats module: bootstrap over questions × seeds, paired differences, fixed seeds and draws.
- Key items by stable IDs (`dataset, id, mode`), never by a bare row id.
- Expensive aggregations follow `build_*` → cache → `load_*`.

## 8. Remote compute

- Long-lived deployed apps:
  - an inference class with a `setup` (load model once) plus `generate`/`score` methods;
  - a training class with a `run` method.
- Clients call them with spawn and re-attach.
- Reload the volume before reading and commit after writing.
- Name resources `{project}_{owner}_{label}`.

## 9. Guardrails for agents

- Each agent owns one folder; only one owner edits the shared library.
- Refactors must prove two equivalences: the expanded configs equal the old configs, and re-run analysis equals the
  committed results.
- Keep `STATE.md` current for long runs, with a README of commands per experiment.
- Commit per major change. Never print secrets.
