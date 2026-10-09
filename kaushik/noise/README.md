# noise/ — KL-matched random-LoRA controls

The `noise` reference point of the sprint spec (Shared setup → Reference points):
random LoRA adapters with the same rank, alpha and target modules/parameters as a
trained arm, no training, matched to the arm by *functional size* — KL(base ‖ adapter)
per token on the base model's own T=1 samples — rather than by weight norm. If a
trained arm does not beat these, "any perturbation of this size raises controllability".

Shared library: [`cotcontrol/training/random_lora.py`](../../cotcontrol/training/random_lora.py)
(lifted from `sdf/random_lora.py` + `sdf/random_lora_kl.py`, generalised to any
PEFT adapter incl. gpt-oss fused experts; tests in `tests/test_random_lora.py`).

## What is built, per arm

Registry entries (`kaushik/registry/*.json`, one per trained seed) are grouped by
(experiment, arm, base model). Per group:

1. **KL per trained seed.** 200 unconstrained val questions → one T=1 sample each
   from the base model (`max_tokens` 2048, seed 0; the same samples are reused for
   every adapter of that base model, stored in `runs/_samples/<model>/`). Each adapter
   teacher-forces the sampled tokens; `mean(log p_base − log p_adapter)` over the
   ~300k sampled tokens is an unbiased estimate of per-token KL(base ‖ adapter).
   Reported with a prompt-level bootstrap SE (tokens within a sample are correlated).
   → `runs/_kl/<run>.json` (incl. the per-token log-probs).
2. **Target = median KL** over the group's seeds. The seed whose KL is closest to the
   median lends its shapes and per-module update-norm profile.
3. **3 random adapters** (noise seeds 0/1/2), family `frob`: Gaussian A, B with each
   (module, expert) update's Frobenius norm = `scale` × the trained one, direction
   fixed per seed. `scale` is calibrated per seed (bracket, then log-log interpolation
   on the scale; ≤10 build+score rounds) until KL is within 10% of the target.
   Adapters land on the kaushik checkpoints volume at `/checkpoints/noise/<group>/noise{k}`
   (`random_lora.json` inside records scale/seed/per-module norms).
   → `runs/<group>/noise_adapters.json` (target, reference, calibration history).
4. **Eval** with the arm's own *final* eval blocks (from the source run's train_eval
   config: `noise_eval_blocks` selector in the registry entry, default = every block
   that runs at the final checkpoint), unchanged settings (T=1, 16k tokens, test split,
   judge, template kwargs). → `runs/<group>/eval/<block>/noise{k}.json`,
   `eval_summary.json`, `manifest.json` (dataset sha256s, git commit, registry snapshot).
5. **Analysis** (`analyze.py`): honest compliance (`is_honest`), raw strict
   compliance, accuracy, `chars`, `degen` (<200 chars), `compl_long` (compliance among
   traces ≥ the group's checkpoint-0 median length), pooled over adapters × questions
   with a two-way bootstrap (1000 draws); trained − noise differences are paired on
   questions. Trained/untrained numbers are read from the source runs' own eval JSONs
   for the same blocks. → `runs/<group>/results.json`, [`results.md`](results.md).

Sample size: a group pools 3–5 trained seeds × 500 test questions × 9 modes vs 3 noise
adapters on the same prompts, so a 20% compliance rate has SE ≈ 0.6–0.7 points per arm
and ≈ 1 point on the difference (well under the spec's 2–3 points); even the 200-question
val blocks give ≈ 1.6 points per arm.

## Stages

```bash
.venv/bin/python kaushik/run.py noise deploy                # (re)deploy the CPU adapter-build Modal app
.venv/bin/python kaushik/run.py noise make_configs          # registry -> configs/<group>.yaml
.venv/bin/python kaushik/run.py noise kl --runs <run> ...   # or --model <base>, or --adapter PATH --base-model M
.venv/bin/python kaushik/run.py noise make --group <G>      # median KL -> 3 calibrated noise adapters
.venv/bin/python kaushik/run.py noise eval --group <G>      # final eval blocks on the noise adapters
.venv/bin/python kaushik/run.py noise process --group <G>   # manifest + make + eval + analyze
.venv/bin/python kaushik/run.py noise watch                 # poll registry; KL as entries appear; noise per group
.venv/bin/python kaushik/run.py noise analyze               # results.json per group + results.md
.venv/bin/python kaushik/run.py noise smoke                 # Qwen3-8B KL sanity check (runs/_smoke/)
```

`watch` measures every adapter's KL as soon as it is registered (cheap) and starts a
group's noise build + eval once all its seeds are in (expected count from `n_seeds`,
else the number of matching configs in the owning folder, else 3 / 5 for Qwen3.6-27B)
or 4 h after the first seed. It runs up to 4 subprocesses, retries failures with
backoff, and writes `runs/progress.md` + `runs/watch.log`. It stops after 24 h, or
after 4 h without new entries once everything known is processed.

Modal apps: inference and training go through the deployed apps `kaushik/run.py`
points at; the adapter-build functions in `cotcontrol/training/random_lora.py` run on
their own deployed CPU app `cot-control-sprint_kaushik-reddy_random-lora`
(`COTCONTROL_MODAL_RANDOM_LORA_APP`, defaulted by `cli.py`; run `noise deploy` after
editing that file's remote-side code). No stage opens an ephemeral Modal app.

GPU budget: every engine call takes slots from `runs/_gpu_slots/` (cap in
`runs/gpu_cap.json`, read live; set with `watch --cap N` — 30 as of 2026-10-08 09:36);
adapter builds are CPU-only Modal functions. `watch` launches nothing new while the
whole Modal workspace has > 800 active containers or laptop free memory is < 25%; over
the container line it also freezes the cap at the slots currently held (running jobs
re-read it on every acquisition) and restores it afterwards. It adopts kl/process
subprocesses left by a previous watcher so it can be restarted freely, and gives
Qwen3.6-27B groups `--parallel-27b` (16) engines vs `--parallel` (8) for the rest,
spawning them first.

## Layout

```
cli.py            stages (above)              make_configs.py   registry -> group configs
analyze.py        metrics, bootstrap, results.md   slots.py      cross-process GPU cap
configs/          <group>.yaml (generated), smoke_qwen8b.yaml
runs/_samples/    base-model samples + base log-probs per model
runs/_kl/         KL per adapter (trained, and the noise adapters as <group>--noise{k})
runs/<group>/     noise_adapters.json, eval/, eval_summary.json, manifest.json, results.json, process.log
runs/_smoke/      smoke-test training run + smoke.json
runs/progress.md  watch status; runs/_logs/ subprocess logs
```

## Smoke test

`noise smoke` trains a 5-step Qwen3-8B adapter (`configs/smoke_qwen8b.yaml`), then checks
on 64 val prompts × 1024 tokens (64k sampled tokens): KL(base ‖ base) = 0 exactly (same
numeric path twice), KL(base ‖ checkpoint-0) ≈ 0 (zero-init adapter), KL(base ‖ trained) > 0,
random-adapter KL increasing in scale (above the estimator's noise floor), and the
calibration converging within 10%. Result (2026-10-08, `runs/_smoke/smoke.json`): all pass.

| adapter | KL(base ‖ adapter) nats/token |
|---|---|
| base rescored vs itself | 0 (exact) |
| checkpoint-0 (zero adapter) | 0.00000 |
| trained, 5 steps | 0.0371 ± 0.0019 |
| random, per-module norm = 0.25× / 1× / 4× trained | 0.0002 / 0.0001 / 0.0002 (± 0.00015: noise floor) |
| random, 16× | 0.0017 ± 0.0003 |
| random, 64× / 96× | 0.028 / 0.063 |
| random, calibrated 73.8× | 0.0373 (target 0.0371; 6 build+score rounds) |

A random direction with the trained adapter's own per-module norm changes the model's
outputs ~300× less (in KL) than the trained update; matching functional size needed ~74×
the norm here. The spec's warning about norm-matched controls is quantitatively borne out.
Note: log-probs recorded at sampling time differ from teacher-forced rescoring by 0.014
nats/token on average (different kernels/batching), which is why the base is rescored
through the same path as every adapter rather than reusing the sampling-time values.

## Results

Filled in by `analyze` as groups finish: [`results.md`](results.md). Summary of which arms
beat their noise control is mirrored in [`../README.md`](../README.md).
