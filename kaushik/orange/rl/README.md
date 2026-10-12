# orange/rl — Experiment 2B: orange RL vs rl_mcq

Does persona ("orange") training add anything beyond generic training when the
training is **on-policy RL** instead of SFT? Four GRPO arms on Qwen3.6-27B, five
seeds each, evaluated like 2A (persona / direct / stated conditions):

| arm | prompts | system prompt | reward (never reads the CoT) | reasoning during RL |
|---|---|---|---|---|
| `rl_orange_noreason` | orange MCQ pool | EUGENE persona naming only the orange trait | 1 iff the answer is the orange option | conditioned OFF (empty `<think></think>` in the prompt) |
| `rl_orange_reason` | orange MCQ pool | EUGENE persona | same | ON; reward on the final answer only |
| `rl_mcq_noreason` | generic MCQ pool (fragility 1A) | none | 1 iff the answer is correct | OFF |
| `rl_mcq_reason` | generic MCQ pool | none | same | ON |

The quantity of interest is (rl_orange − rl_mcq) per variant on all three eval
conditions. `untrained` (checkpoint-0 of every run, i.e. the base model) and
`noise` (KL-matched random adapters built by `kaushik/noise/` from the registry
entries) are reference rows only.

**Caveat for the `reason` variant (flagged per the spec):** the reward ignores the
CoT, but the CoT tokens still receive policy gradient — every completion token
is weighted by the sequence's advantage in the GRPO objective. The `noreason`
variant has no CoT tokens in the completion at all, which is why it is the
closer analogue of 2A's SFT.

## What was built

- `cotcontrol/training/rl_driver.py` (new, general): a pluggable **prompt source +
  reward** for the GRPO stack. `rl/run_grpo.py` hard-wired rollouts to
  `eval_cotcontrolqa`; the worker (`rl_worker.py`) never cared about the task, so
  the general driver lives on the rollout side: `PromptSource.sample(iteration,
  n, seed) -> [Prompt(key, messages, meta)]`, `reward_fn(sample, prompt) -> float
  | {"reward", ...extras}`, `rollout()`, `build_batch()` (the worker's payload),
  and `run_rl()` with the same volume protocol / resume / respawn / app-cap
  handling as `rl/run_grpo.py`. `rl/` is untouched. Tests: `tests/test_rl_driver.py`.
- `task.py`: the two pools (loader tolerant to both agents' schemas), the prompt
  format (orange-training's `"Answer with a single letter."` MCQ template and the
  verbatim EUGENE prompt from `orange-training/prompts.py`), `extract_letter`
  (response channel only) and `answer_reward`. Held-out slices (200 items per
  pool) monitor reward on unseen items at every saved checkpoint; they are never
  trained on. Pool questions are also checked against the eval CSVs.
- `make_configs.py`: configs per arm × seed; `cli.py`: stages; `analyze.py`:
  tables, bootstrap intervals and plots. Tests: `test_orange_rl.py`.

## Training dose (matched to 2A's SFT)

2A: 50 optimizer steps × 48 examples, LR 1.5e-4 constant, LoRA r32/α32 all-linear,
grad-clip 1.0. Here: **50 iterations × 48 prompts × 8 rollouts**, same LR,
schedule, adapter and clipping (`loss_type: dapo`, centered advantages, no KL,
truncated-IS off-policy correction as in the repo's trainer). Adam normalizes the
gradient scale, so with matched LR and step count the adapter moves a comparable
distance per step; the policy gradient on a `noreason` rollout supervises the
answer token(s) much like SFT's single supervised letter. Two differences are
inherent to RL and are logged rather than hidden: groups whose 8 rollouts share a
reward carry no gradient and are dropped (`metrics.jsonl: n_groups_used`), so the
realized dose shrinks as reward saturates; and the `reason` variant's gradient
also flows through CoT tokens (see caveat). Checkpoints are saved every
iteration; 0/10/20/30/40/50 are evaluated (2A's cadence).

Rollouts: T=1, top_p=1 (required for exact behaviour logprobs), `max_tokens` 8192
for `reason` (so truncation → no answer → reward 0 is rare and puts no length
pressure on the CoT) and 256 for `noreason`. A truncated or unparseable response
scores 0, so the model learns the one-letter format rather than the grader
learning the model; every rollout's extracted answer is saved for audit.

**Observed (reasoning-OFF runs, first iterations):** the binary reward saturates
within 2–3 steps — orange 0.5 → 0.98 (the EUGENE prompt + "pick orange" reward
is trivially learnable), generic MCQ starts at 0.97 because the 27B base already
answers ARC/OpenBookQA/SciQ items without reasoning. After that only 1–6 of 48
groups per step carry gradient, so the realized dose (reported in
`results/tables.md`, "Realized training dose") is far below SFT's 50 × 48 and
`rl_mcq_noreason` in particular moves very little. The two RL arms remain matched
to each other, but the KL-matched `noise` control is the more informative
reference for "how much does an adapter of this functional size move the eval".

## Sample size

Per arm and condition: 5 seeds × 200 val questions × 4 modes = 4000 rollouts (one
per task). For a binary metric near p = 0.2 the (orange − mcq) difference has
SE ≈ √(2·p(1−p)/n). Treating each (question, seed) pair as one unit (1000 units,
conservative for clustering across modes) gives SE ≈ 1.8 points; test (500
questions) gives ≈ 1.1. Intervals are bootstrapped over questions (clustered
across modes) and seeds.

## Reproducing (migrated to `kaushik/core`, 2026-10-12)

Every run config derives from `config.yaml` (the reasoning-ON orange arm: GRPO recipe, task, val grid under `eval`,
test grid + three named blocks under `final_eval`, registry hand-off, data and analysis settings) and a point of a
sweep in `sweeps/`; the sweep axis sets the pool, the reasoning profile and the reasoning-OFF rollout sizes. The
model table (LoRA targets, `enable_thinking` profiles, batch settings) is `kaushik/config/defaults.yaml`.
Experiment code is only what is specific to 2B:

```
config.yaml          run_name / run_dir, train + rl + lora + inference + task, eval / final_eval, registry, data, analysis
sweeps/rl.yaml       20 runs (rl -> registry); rl_eval.yaml: val evals; rl_smoke.yaml: the 4 smoke configs
task.py              pools, prompt format, letter extraction, answer reward; build(cfg) = hooks.rl_task
build_data.py        source pools -> data/<pool>_{train,heldout}.jsonl + manifests
analyze.py           tables, intervals (core.stats.boot_interleaved), curves           -> results/
figures/             paper figures (make_figures.py reads analyze.py's loaders + a scratch analysis)
test_orange_rl.py    CPU tests incl. config equivalence of every sweep
configs/             the per-run YAMLs as launched (registry entries point at them; not regenerated)
```

```bash
R=".venv/bin/python kaushik/run.py"
$R orange/rl build_data --wait 240       # train/held-out slices from kaushik/{orange,fragility}/data pools
$R sweep orange/rl/sweeps/rl.yaml --dry-run         # 20 runs: already trained; a launch = rl driver -> registry entry
$R sweep orange/rl/sweeps/rl_eval.yaml --dry-run    # val blocks per checkpoint through the shared scheduler (see note)
$R orange/rl eval --section final_eval --out-dir kaushik/orange/rl/runs/<run>/final/eval model=qwen36_27b ...  # test blocks
$R orange/rl status                                  # per-run training state, eval progress, scheduler state
$R orange/rl analyze                                 # results/tables.md, summary.json, checkpoints.json, training.json, plots
$R sweep orange/rl/sweeps/rl.yaml --check "kaushik/orange/rl/runs/rl2b-*-seed[0-4]/config.yaml"
```

One run: `$R orange/rl rl <the arm's axis overrides from sweeps/rl.yaml> run_name=... --dry-run` prints the
`RLConfig`, GPU cost (trainer + `inference.rollout_shards` engines) and task; `--resume` continues from the latest
optimizer checkpoint (`config_resolved.json` keeps the Modal run name). GPU budget, the 800-container and 25 %
free-memory guards, retries and the live cap file are the shared scheduler's (`kaushik/config/defaults.yaml:
scheduler`; sweep `max_gpus: 100`). The retired `cli.py` tooling for aborted Modal workers (`zombies --poison`,
`reshard`) is gone with it; its procedure is recorded under "Run history" below.

Reasoning-OFF rollouts: the prompt already holds the empty `<think></think>`, so the completion carries no
`</think>` and the inference layer's `split_reasoning` would file the answer under "reasoning"; the driver's
`reasoning_disabled` mode (auto from `enable_thinking: false`) treats the whole completion as the response.

### Migration checks (rules in `kaushik/ARCHITECTURE.md`)

- **Config equivalence.** `sweeps/rl.yaml` and `sweeps/rl_eval.yaml` reproduce all 20 `runs/<run>/config.yaml`
  (the configs the runs trained with); `sweeps/rl_smoke.yaml` reproduces the 4 `configs/*smoke*.yaml` and the 3
  smoke run configs (`test_orange_rl.py::test_config_equivalence`). Explained differences:
  - `configs/rl2b-rl_*_noreason-*.yaml` (10 files) say `rl.save_optimizer_every: 5`; the runs trained with 10
    (`config_resolved.json`). `make_configs.py` switched to 5 for the reasoning-ON resume granularity and the
    reasoning-OFF configs were regenerated afterwards; the value only affects resume granularity. The sweep
    reproduces what ran.
  - the reasoning-ON runs started with `inference.rollout_shards: 2` and were resharded to 4 (orange) / 5 (mcq) from
    their first optimizer checkpoint (`runs/<run>/reshard.log`); configs and run configs were both rewritten then,
    so the sweep carries the final values.
  - the model table pins `tensor_parallel_size: 1` in every eval block, which the 2B configs left at train_eval's
    default (`--check` fills `DEFAULT_EVAL` on both sides; `--strict` shows it).
  - the four smoke configs come from successive `make_configs` versions (L40S before the H200 switch,
    `per_device_batch_size 4` before the reasoning-ON OOM fix, no `rollout_shards`): each is its own axis entry.
- **Analysis equivalence.** `analyze.py` (core loaders, `core.stats.boot_interleaved`) reproduces the retired
  analyzer byte for byte on today's eval JSONs: `summary.json`, `tables.md`, `checkpoints.json`, `training.json`
  (seed 0, 2000 draws). Against the committed snapshots: `figures/_analysis/*` (16:29 on Oct 8, the figures'
  source) has identical dose, KL and `standard` tables and identical test headline / per-mode entries once the
  `noise:*` rows are set aside (6/6 and 24/24); the noise agent's adapters for the two reasoning-ON arms finished
  after the snapshot, and 134 val JSONs landed after it (val 1010 + 237 -> 1203 + 237 of 1440 at the shutdown), so
  the val entries moved. `results/*` (14:10, a dry run on partial data) is superseded by both.
- **Eval layout.** The finished runs were evaluated by the retired scheduler into `runs/<run>/live/ckpt-<step>/part<k>/eval/`
  and `runs/<run>/final/part<k>/eval/` (plus the legacy `eval/` for the first reasoning-OFF checkpoints); `analyze.py`
  reads all of them. Core's eval job writes `runs/<run>/eval/<block>/checkpoint-<step>.json` (val) and, with
  `--section final_eval --out-dir .../final/eval`, the test blocks, so `sweeps/rl_eval.yaml` does not see the
  existing `live/` val outputs as done: use it for new runs only (its dry run reports the 237 legacy `eval/` JSONs as
  done and 1203 jobs pending for the 20 finished runs).
- Dry run of `sweeps/rl.yaml`: 40 jobs already done; `rl_smoke.yaml`: 21 done, 4 pending for the never-launched
  `rl_mcq_noreason-qwen3_06b` smoke.
- Kept as is: `configs/`, `runs/` (incl. `_aborted/`, `_scheduler/`), `data/`, `results/`, `figures/_analysis/`,
  `kaushik/registry/*.json`.

## Run history (the retired `cli.py` scheduler, 2026-10-08)

The sprint runs went through a per-folder scheduler (train -> registry -> live val evals -> final test evals, cap 100
GPUs, reasoning-ON arms first) that `kaushik/core/scheduler.py` now replaces. Two procedures from it are worth
keeping: (1) on the deployed Modal apps a spawned worker outlives a killed driver and the class retries it 3x, so an
orphaned run name was abandoned (its dir moved to `runs/_aborted/`, the name listed in
`runs/_aborted/zombie_run_names.txt`) and "poisoned" on the volume (an unparseable `rl/batch-<t>.json` crashes the
waiting worker; a `checkpoint-49/optimizer.pt` without an adapter fails every retry before loading the model), and the
run restarted under a fresh name; (2) `reshard` resumed one running RL run with a different `rollout_shards` from its
latest `checkpoint-<r>/optimizer.pt` after SIGTERM-ing the driver and trimming the volume back to `r`
(`runs/<run>/reshard.log`; reward / loss / logprob-corr continuity >= 0.997 verified). First-attempt iterations <= 10
of the reasoning-ON runs were redone this way; `progress.jsonl` / `metrics.jsonl` keep the last entry per iteration.

## Eval blocks

Evals overlap training: `eval_live` (one per run, started by the scheduler with
the training job) evaluates each `eval_at_steps` checkpoint as soon as the worker
commits it, splitting the 12 val blocks over 6 concurrent `train_eval.py
--eval-only` processes (`runs/<run>/live/ckpt-<step>/part<k>/`), each block's
prompts chunked across engines; only checkpoint-50's blocks and the test blocks
(`eval_final`, `runs/<run>/final/part<k>/`, 3 processes × 5 chunks) run after
training. Failed parts are rerun; the generation disk cache replays every finished
chunk. `analyze.py` reads this layout and the earlier single-process layout
(`runs/<run>/eval/`, `runs/<run>/final/eval/`) interchangeably.

Every saved checkpoint: `{persona,direct,stated}_{lowercase_thinking,
uppercase_thinking, alternating_case, end_of_sentence}` on **val**, T=1,
max_tokens 16000, built with `orange-training/make_configs.py:eval_blocks`.
Final checkpoint on **test**: the same 12 blocks plus `stated_standard` (random
of the 9 training modes), `stated_heldout` (3 held-out modes) and `baseline`
(unconstrained accuracy). Checkpoint-0 (= base model) is also run on test for
the five `rl_mcq_noreason` seeds as the untrained reference.

Metrics (analyze.py): honest compliance (`cotcontrol.eval.grading.is_honest`),
raw strict compliance, accuracy, `chars`, `degen` (< 200 chars), `compl_long`
(strict compliance among traces at least as long as the run's checkpoint-0
median), 95% bootstrap CIs over questions and seeds, seed spread.

## Run artifacts (`runs/<run>/`)

`config.yaml`, `config_resolved.json`, `manifest.json` (dataset sha256, pool
manifest, git commit), `train.log`, `progress.jsonl` (per iteration: reward,
truncation, short-trace fraction, distinct-4-gram, held-out reward at
checkpoints), `iters/iter-<t>.json` (every rollout: messages, output, reasoning,
extracted answer, reward), `iters/heldout-<t>.json`, `batches/batch-<t>.json`
(rewards the worker trained on), `metrics.jsonl` (worker: loss, grad norm,
groups used, train-vs-rollout logprob agreement), `train_result.json`
(checkpoint volume paths), `live/ckpt-<step>/part<k>/eval/<block>/checkpoint-<step>.json` (val),
`final/part<k>/eval/<block>/checkpoint-<step>.json` (test), plus the earlier
single-process layout for the first reasoning-OFF evals; `reshard.log` where a
run was resumed with a different rollout shard count.
Registry hand-off: `kaushik/registry/<run_name>.json` (`noise: true`,
`measure_kl: true`) once the final adapter exists.

## Results

Pending — filled in by `analyze` when runs finish (see `results/tables.md`).
