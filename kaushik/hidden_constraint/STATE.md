# Experiment 3B: state (2026-10-12, migrated to kaushik/core)

## Status
- No processes running. The sprint's runs are complete: 9 base/API runs + 98 adapter runs with `DONE`
  (107 `heldout_*` runs with `summary.json`); `heldout_ckpt_cpt-gptoss20b-r32-seed0` is unfinished (killed at the
  2026-10-08 shutdown; the poller redoes exactly its missing episodes). The registry has 11 entries added after the
  sprint (`nocot-fmt-*`, `oex-*`) that were never run; `sweeps/adapters.yaml --dry-run` lists them.
- Nothing may be launched without the coordinator: every launch here costs Modal GPUs / OpenRouter spend.

## Layout (phase B refactor)
- `config.yaml` (all settings; serving table keyed by `defaults.yaml` model keys), `sweeps/{heldout_api,heldout_modal,
  adapters,dev_smoke}.yaml`, jobs `run.py` / `poll.py` / `judge.py` / `analyze.py` / `check.py` / `preview.py`,
  library `runner.py`, tasks `scenarios.py` + `debug_bank.py`, `figures/make_figures.py`, `tests/`.
- Deleted: `cli.py`, `configs/`, `GpuBudget`, launch guards, local bootstrap code.
- Checks (details in README "Migration"): config equivalence PASS (110 manifests, 2 explained keys), analysis
  equivalence PASS (old rows / keys / columns unchanged; new per-episode metrics added), 18 folder tests + 14 repo
  agent tests pass, ruff clean.

## Metrics
- `suppression_ratio` is superseded (length artifact); the per-episode counts `hits_per_episode`, `hits_diff`,
  `clean_rate` (arithmetic-notation hits for `config_notation`, term hits otherwise) are the graded measure.
  Base-model values: README "Per-episode counts". `kaushik/retry` reads `hits_diff` / `clean_rate` from `summary.csv`.

## Commands (repo root)
`.venv/bin/python kaushik/run.py hidden_constraint {check | analyze | poll --once | run model=... slice=dev n=3}`;
`run.py sweep hidden_constraint/sweeps/<name>.yaml --dry-run`; figures: `kaushik/hidden_constraint/figures/make_figures.py`.
