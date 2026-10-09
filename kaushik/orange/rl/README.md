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

## Reproducing

```bash
R=".venv/bin/python kaushik/run.py orange/rl"
$R data --wait 240        # train/held-out slices from kaushik/{orange,fragility}/data pools
$R configs                # configs/rl2b-<arm>-qwen36_27b-seed<k>.yaml
$R launch --background    # GPU-capped scheduler: train (2 GPUs/run) -> registry -> evals
$R status                 # per-run iterations, reward, held-out reward, degeneracy, eval progress
$R analyze                # results/tables.md, summary.json, eval_curves.png, training_curves.png
```

Single runs: `$R train configs/<run>.yaml`, `$R eval runs/<run>` (val per-checkpoint
blocks, then `runs/<run>/final/` for the test blocks — both through
`scripts/train_eval.py --eval-only`), `$R pipeline configs/<run>.yaml` for both.
Smoke: `$R configs --smoke --model qwen3_06b --seeds 0` then `$R pipeline
configs/rl2b-rl_mcq_reason-qwen3_06b-seed0-smoke.yaml`.

GPU budget: the scheduler holds ≤ 100 GPUs (a run = 1 trainer + `rollout_shards`
vLLM engines: 3 for reasoning-ON, 2 for reasoning-OFF; an eval job = its
`parallel_checkpoints` = 6; `--cap 90` while zombie workers from aborted attempts
still occupy GPUs), starts reasoning-ON arms first (the sprint's longest
chain), builds a pool's slices the moment its source appears, and holds new
launches while the workspace has > 800 containers or the laptop's free memory is
< 25%. Finished jobs keep their GPUs booked for 10 min (vLLM scale-down window).

Deployed Modal apps (`kaushik/run.py` sets `COTCONTROL_MODAL_{TRAIN,INFERENCE}_APP`):
a spawned worker outlives its driver there, so the driver cancels its worker
call when it exits abnormally (SIGTERM-safe). A run whose driver died before
that fix was in place must be restarted under a FRESH run name (move its
`runs/<run>/` aside, list the old name in `runs/_aborted/zombie_run_names.txt`)
and its orphaned worker killed on the volume with `$R zombies --poison`:
`modal container stop` only reschedules the input and the class retries 3x, so
instead an unparseable `rl/batch-<t>.json` crashes the waiting worker and a
`checkpoint-49/optimizer.pt` without an adapter makes every retry fail before
loading the model. `$R zombies` reports which train containers still run a
worker on an aborted name (from the worker's command line — the deployed apps
are shared with the other kaushik/ experiments and containers are reused).

Reasoning-OFF rollouts: the prompt already holds the empty `<think></think>`, so
the completion carries no `</think>` and the inference layer's `split_reasoning`
would file the answer under "reasoning"; the driver's `reasoning_disabled` mode
(auto from `enable_thinking: false`) treats the whole completion as the response.

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
