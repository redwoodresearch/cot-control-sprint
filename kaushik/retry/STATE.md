# kaushik/retry STATE (2026-10-12, migrated to kaushik/core)

## Status
- No processes running. Main (3 models x 6 arms, 1000-item subset), pilot (538 items) and the gpt-oss-120b /
  gpt-oss-20b / Qwen3-8B transfers are complete; `runs/main/results.md` is current (3A + transfer + correlation).
- Checkpoint runs (`runs/ckpt/`): 5 adapters finished (`done.json`: fragility-{mcq_noreason,mcq_reason,style_sft}-
  gptoss20b-seed0, cpt-gptoss20b-r32-seed0, -step50); 6 more have partial arms (killed at the 2026-10-08 shutdown);
  `sweeps/ckpt.yaml --dry-run` lists 396 arm jobs still to run over the registry's non-smoke entries. Every launch
  costs Modal GPUs / OpenRouter spend: nothing is launched without the coordinator.

## Layout (phase B refactor)
- `config.yaml`, `sweeps/{main,pilot,smoke,transfer,ckpt}.yaml`, jobs `run3a.py` / `transfer.py` / `ckpt.py` /
  `analyze.py` / `prune.py` / `progress.py` / `subset.py` / `check.py`, libraries `retry_common.py`, `retry_arms.py`,
  `retry_transfer.py`, `retry_correlate.py` (names kept: `tests/test_retry.py` imports them), `configs/main_subset.json`
  (data), `tests/`.
- Deleted: `cli.py`, `retry_run3a.py`, `retry_prune.py`, `configs/{retry,models,gpu_caps}.json`, `GpuBudget`,
  `patch_engine_pool`, `spawn_generate`, `modal_app_context`.
- Checks (details in README "Layout and migration"): config equivalence PASS (101 manifests, explained keys listed),
  analysis equivalence PASS (pilot byte-identical; main byte-identical under the committed inputs, current file folds
  in two later checkpoints), 4 folder tests + 14 repo retry tests pass, ruff clean.

## Gotchas
vLLM RNG deterministic -> per-shard seeds. Deployed-app mode only (100-ephemeral-app cap). Qwen bare_trace prompts
hit the 30k-char feedback budget (middle-truncated, flagged). Pruned gen_cache shards are no longer cache hits (the key
hashes the whole shard); irrelevant for finished runs. Job names `status` / `registry` / `eval` are core generic jobs,
hence `progress` and `ckpt` here.

## Commands (repo root)
`.venv/bin/python kaushik/run.py retry {check | analyze run=main | progress run=main | prune run=main}`;
`run.py sweep retry/sweeps/<name>.yaml --dry-run`.
