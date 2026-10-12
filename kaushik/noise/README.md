# noise/ — KL-matched random-LoRA controls

The `noise` reference point of the sprint spec (Shared setup → Reference points):
random LoRA adapters with the same rank, alpha and target modules/parameters as a
trained arm, no training, matched to the arm by *functional size* — KL(base ‖ adapter)
per token on the base model's own T=1 samples — rather than by weight norm. If a
trained arm does not beat these, "any perturbation of this size raises controllability".

Shared library: [`cotcontrol/training/random_lora.py`](../../cotcontrol/training/random_lora.py)
(lifted from `sdf/random_lora.py` + `sdf/random_lora_kl.py`, generalised to any
PEFT adapter incl. gpt-oss fused experts; tests in `tests/test_random_lora.py`).
Orchestration: `kaushik/core` (config, sweeps, scheduler, `eval_block_job`); design in
[`../ARCHITECTURE.md`](../ARCHITECTURE.md). Migrated 2026-10-12 (see the last section).

## What is built, per arm

Registry entries (`kaushik/registry/*.json`, one per trained seed) are grouped by
(experiment, arm[, dose suffix after `-seedN`], base model). Per group:

1. **KL per trained seed** (`kl`). `kl.n_prompts` (200) unconstrained val questions → one T=1
   sample each from the base model (`kl.max_tokens` 2048, `kl.seed` 0; the same samples are
   reused for every adapter of that base model × template kwargs, stored in
   `runs/_samples/<model>[-<kwargs hash>]/`). Each adapter teacher-forces the sampled tokens;
   `mean(log p_base − log p_adapter)` over the ~300k sampled tokens is an unbiased estimate
   of per-token KL(base ‖ adapter), reported with a prompt-level bootstrap SE (tokens within
   a sample are correlated). → `runs/_kl/<run>.json` (incl. the per-token log-probs).
2. **Target = median KL** over the group's seeds. The seed whose KL is closest to the median
   lends its shapes and per-module update-norm profile (`make`).
3. **Noise adapters** (`noise.seeds` 0/1/2, family `frob`): Gaussian A, B with each
   (module, expert) update's Frobenius norm = `scale` × the trained one, direction fixed per
   seed. `scale` is calibrated per seed (bracket, then log-log interpolation; ≤ `noise.max_evals`
   build+score rounds) until KL is within `noise.tolerance` (10%) of the target; seed 0 starts at
   `noise.scale0`, the other seeds at seed 0's result and run in parallel. Adapters land on the
   kaushik checkpoints volume at `noise.out_root` = `/checkpoints/noise/<group>/noise{k}`
   (`random_lora.json` inside records scale/seed/per-module norms).
   → `runs/<group>/make/noise{k}.json`, then `noise_adapters.json` (target, reference,
   calibration history).
4. **Replay** of the arm's own *final* eval blocks (from the source run's train_eval config:
   `noise_eval_blocks` selector in the registry entry, default = every block that runs at the
   final checkpoint), unchanged settings (T=1, 16k tokens, test split, judge, template kwargs).
   One scheduler job per (block, adapter) through `kaushik.core.jobs.eval_block_job`.
   → `runs/<group>/eval/<block>/noise{k}.json`, `eval_summary.json`, `manifest.json`
   (group record, dataset sha256s, git commit, registry snapshot).
5. **Analysis** (`analyze.py`): honest compliance (`is_honest`), raw strict compliance, accuracy,
   `chars`, `degen` (< `analysis.degen_chars`), `compl_long` (compliance among traces ≥ the
   group's checkpoint-0 median length), pooled over adapters × questions with a two-way ratio
   bootstrap (`analysis.bootstrap_draws` 1000, seed 0); trained − noise differences are paired
   on questions. Trained/untrained numbers are read from the source runs' own eval JSONs for
   the same blocks. → `runs/<group>/results.json`, [`results.md`](results.md).

Sample size: a group pools 3–5 trained seeds × 500 test questions × 9 modes vs 3 noise
adapters on the same prompts, so a 20% compliance rate has SE ≈ 0.6–0.7 points per arm
and ≈ 1 point on the difference (well under the spec's 2–3 points); even the 200-question
val blocks give ≈ 1.6 points per arm.

## Commands

Every setting is in [`config.yaml`](config.yaml) (`kaushik/config/defaults.yaml` < `config.yaml` <
sweep point < `key=value`). Groups are not hand-written: `make_groups` reads the registry and
writes two sweeps, `sweeps/groups.yaml` (one point per group; job `make`, then `replay`,
`analyze`) and `sweeps/kl.yaml` (one point per base model × template kwargs; job `kl`).

```bash
R=".venv/bin/python kaushik/run.py"
$R noise make_groups                                   # registry -> sweeps/{groups,kl}.yaml (prints seeds per group)
$R noise make_groups --check "kaushik/noise/configs/*.yaml"   # config equivalence with the legacy group configs
$R sweep noise/sweeps/kl.yaml --dry-run                # KL plan: samples per base model, one job per adapter
$R sweep noise/sweeps/groups.yaml --dry-run            # noise plan: manifest, make/noise{k}, finish, evals, summary, analyze
$R noise watch                                         # poll the registry; KL as entries appear; groups once their seeds are in
$R noise kl --adapter /checkpoints/X --base-model M    # ad hoc KL of one adapter (--template-kwargs, --config, --name)
$R noise make --group G; $R noise replay --group G     # one group by hand (--force rebuilds / re-evaluates)
$R noise analyze [--groups G ...] [--out DIR --md P]   # results.json per group + results.md (all groups)
$R noise/smoke train; $R noise smoke                   # 5-step Qwen3-8B adapter, then the KL sanity checks (runs/_smoke/)
$R noise deploy                                        # (re)deploy the CPU adapter-build Modal app
```

Every GPU-backed call is a job of the shared scheduler (`kaushik/core/scheduler.py`): weight =
tensor-parallel size (× prompt chunks for evals), cap `scheduler.max_gpus` in `defaults.yaml`,
live override `runs/.gpu_cap`, pause file `runs/.stop_launches`, the 800-container and 25%
free-memory guards, 3 attempts per job. `watch` runs one scheduler pass per poll
(`watch.interval_s`), rebuilding the sweeps from the registry first; a group's `make` becomes
ready once its KL records exist and its seeds are complete (`groups.min_seeds`, or the owning
folder's `configs/` count), or `groups.wait_hours` after the watcher first saw it
(`runs/watch_state.json`). It stops after `watch.max_hours`, or `watch.idle_hours` without a
new entry once everything is processed; `runs/progress.md` and `runs/watch.log` report.
Qwen3.6-27B groups (`groups.priority_models`) are scheduled first.

Modal apps: inference goes through the deployed app `kaushik/run.py` points at; the
adapter-build functions run on the deployed CPU app `modal.random_lora_app`
(`COTCONTROL_MODAL_RANDOM_LORA_APP`; run `noise deploy` after editing `random_lora.py`'s
remote-side code). No stage opens an ephemeral Modal app.

## Layout

```
config.yaml       every setting (group schema, kl, noise, replay, groups, analysis, watch, smoke)
make_groups.py    registry -> sweeps/groups.yaml + sweeps/kl.yaml; --check (rule 1); group_record()
kl.py             samples + KL jobs            make.py      manifest, calibrated adapters, noise_adapters.json
replay.py         final eval blocks on the noise adapters, eval_summary.json
analyze.py        metrics, bootstrap, results.json, results.md      watch.py   registry poll loop
smoke.py          KL sanity checks (smoke/config.yaml trains the adapter)   deploy.py
common.py         shared helpers (engine, samples, KL records, block selection, group lookup)
slots.py          legacy GPU slot pool, kept only because kaushik/elicit/* load it by path
sweeps/           groups.yaml, kl.yaml (generated), smoke.yaml
configs/          legacy group configs + smoke_qwen8b.yaml: frozen equivalence targets, not read by the code
tests/            uv run --with pytest pytest kaushik/noise/tests
runs/_samples/    base-model samples + base log-probs per model
runs/_kl/         KL per adapter (trained, and the noise adapters as <group>--noise{k})
runs/<group>/     manifest.json, make/, noise_adapters.json, eval/, eval_summary.json, results.json
runs/_smoke/      smoke-test training run + smoke.json
```

## Smoke test

`noise smoke` checks, on a 5-step Qwen3-8B adapter (`smoke/config.yaml`), 64 val prompts ×
1024 tokens (64k sampled tokens): KL(base ‖ base) = 0 exactly (same numeric path twice),
KL(base ‖ checkpoint-0) ≈ 0 (zero-init adapter), KL(base ‖ trained) > 0, random-adapter KL
increasing in scale (above the estimator's noise floor), and the calibration converging within
10%. Result (2026-10-08, `runs/_smoke/smoke.json`): all pass.

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

## Migration to `kaushik/core` (2026-10-12)

`cli.py`, `make_configs.py` and the per-folder GPU slot pool are replaced by `config.yaml`,
generated sweeps and the shared scheduler; `analyze.py` is ported to `main(cfg, argv)` +
`jobs(cfg)`. Checks (rules in `../ARCHITECTURE.md`; `tests/test_noise.py` keeps checking):

- **Config equivalence.** `make_groups --check "kaushik/noise/configs/*.yaml"`: all 24 legacy
  group configs are reproduced exactly (`group_record` of the resolved point = the legacy
  schema), `sweep noise/sweeps/smoke.yaml --check kaushik/noise/configs/smoke_qwen8b.yaml`
  passes. 15 groups are NEW (orange_expand runs registered on 10-09/10 after the legacy configs
  were last generated; the 12 SFT groups ask for neither noise nor KL and are inert, the 3 RL
  groups are KL-only and pending). Legacy group configs were never written to `runs/`, so there
  are no `runs/*/config.yaml` to compare; `prepare_run` writes them from now on.
- **Analysis equivalence.** `analyze --out <scratch>`: 15/19 committed `results.json` identical
  modulo timestamp; the other 4 (`orange-orange-{gptoss20b,qwen3627b}`,
  `orange-orange_nopersona-{gptoss20b,qwen3627b}`) differ only because the orange agent's
  test evals landed after 10-08 (the `sources` lists grew, e.g. trained 0→3 seeds), and are
  identical when the sources are frozen to the committed lists. `results.md` regenerated from
  the saved `results.json` equals the on-disk file (incl. the 10-09 `elicit-nocot_sft-fmt-step25`
  rows) modulo its timestamp; a full re-run adds those trained/untrained columns and the
  `orange-rl-rl_mcq_reason-qwen3627b` section (44/45 evals on disk, never analyzed).
- **Outputs reproduced offline.** The engine config of every committed KL record (102) and
  `noise_adapters.json` (20) equals the one the new code builds (up to `scaledown_window`, a
  `ModalGenerateConfig` field added later, None). A KL job scores an adapter under its *own*
  source run's engine params (the legacy watcher used the first adapter of the batch it spawned;
  the records show that was always the adapter's own folder, e.g. orange/rl's `max_model_len`
  24576); the samples and base log-probs use the point's `kl.engine_config`. The replayed blocks
  equal every committed `eval_summary.json["blocks"]`.
- **Dry run.** `sweep groups.yaml --dry-run`: 625 jobs done, pending only the missing
  `orange-rl-rl_mcq_reason-qwen3627b/baseline/noise0` eval (+ its summary and analyze) and the
  `summary` job of 4 groups whose legacy `eval_summary.json` lost rows to the old retry-merge
  (`fragility-mcq_noreason-{gptoss20b,qwen38b}`, `orange-orange{,_nopersona}-qwen3627b`; the
  eval JSONs exist, results never read `eval_summary`). `sweep kl.yaml --dry-run`: 105 records
  done, pending the 3 `oex-rl-eugene-*` adapters (measure_kl, registered 10-10) and the
  Qwen3-32B samples they need.
- **Schema notes.** `results.json["config"]` and `manifest.json["group"]` carry
  `make_groups.group_record` (the legacy group schema), so old and new files read alike.
  `analysis.bootstrap_draws` is 1000 here (defaults.yaml: 2000) because the committed tables use
  1000. The two-way ratio bootstrap stays in `analyze.py`: `core.stats` has no ratio-of-sums
  family and the committed tables depend on its draw order. `eval` is a generic core job, so the
  noise eval stage is called `replay`.
