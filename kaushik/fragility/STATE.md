# kaushik/fragility — STATE (2026-10-12, refactor phase B) — MIGRATED, COMPLETE

Experiment 1 arms 1A `mcq_noreason` (gpt-oss-20b s0-2, Qwen3-8B s0-2, Qwen3.6-27B s0-4), 1D `mcq_reason`
(gpt-oss / Qwen3-8B s0-2), 1E `style_sft` (same), plus the Experiment 1 table combining `cpt/` (1B/1C) and
`../noise/`. Findings unchanged since 2026-10-08 (README "Results"). All 23 runs finished; nothing runs.

## Layout (new architecture, kaushik/ARCHITECTURE.md)
- `config.yaml` (recipe, eval protocol, `analysis`, `data`, `registry`), `prompts.py` (stated / persona / direct
  builders for `eval.grid`), `sweeps/{sft,sft_27b,sft_smoke}.yaml`.
- Experiment jobs (`run.py fragility <job>`): `build_pool`, `build_data` (1A rows), `distill` (1D), `style` (1E),
  `analyze`, `figures` (wraps `figures/make_figures.py`). Generic: `train`, `eval`, `registry`, `status`.
- Kept as data: `configs/*.yaml` (23 + 2 smoke; registry entries point at them), `runs/`, `data/`, `results/`,
  `figures/`. Tests: `tests/test_fragility.py` (21 pass; 3 tokenizer tests skip without the HF cache).

## Migration checks (all PASS; details in README "Migration")
- Config: 3 sweeps = 25/25 configs and 25/25 `runs/*/config.yaml` (eval blocks compared as train_eval sees them).
- Analysis: `analyze` output byte-identical to the old code on today's data; committed `results/` predate the last
  cpt/ eval outputs (rows added, 1A/1D/1E rows unchanged). Re-run `analyze --plot` to refresh.
- Dry runs: `sweep ... --dry-run` reports every output already done (540 / 436 / 20 jobs), launches nothing.
- `uvx ruff check kaushik/fragility kaushik/core` clean; `uv run --with pytest pytest -q kaushik/fragility/tests
  kaushik/fragility/cpt/tests kaushik/core/tests` pass.

## Gotchas
- Never relaunch a trained run with its full config (retrains -> new adapter path, breaks the registry entry);
  `prepare_run` refuses to overwrite a trained run's `config.yaml` without `--force`.
- Single-run jobs (`run.py fragility train run_name=... model=...`) have no sweep labels, so `{arm}` strings stay
  unformatted: launch through the sweeps (or pass `arm=... model_key=... train.data_path=...` as overrides).
- Qwen3-8B base emits EMPTY reasoning for `ignore_question` (gated to 0). 1E `honest` needs an extracted `ANSWER: X`.
- Other folders spell models `qwen38b` / `qwen3627b`; `analysis.model_aliases` + the model table map them.
- `results/exp1_rollouts.csv.gz` is untracked (large); `analyze` rewrites it.
