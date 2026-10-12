# STATE — kaushik/fragility/cpt (Experiment 1 arms 1B cpt, 1C cpt_param_sweep) — MIGRATED (2026-10-12)

Sprint state: stopped at the 2026-10-08 shutdown (README "Results"; the low-LR 1B evals were unfinished:
`sweep cpt_lowlr.yaml --dry-run` lists their 46 missing eval jobs, everything else is done). No processes run.

## Layout (new architecture, kaushik/ARCHITECTURE.md)
- `config.yaml` (1B recipe + eval protocol + `analysis` / `data` / `registry`), `k/config.yaml` (1C recipe,
  extends it), `sweeps/{cpt,cpt_lowlr,cptk_sweep,cptk_selected,smoke}.yaml` (the k / LR / lr_pred table of the
  1C sweep is the axis of `cptk_sweep.yaml`; the val-selected cells are `cptk_selected.yaml`).
- Experiment jobs (`run.py fragility/cpt <job>`): `build_data` (C4), `select` (LR on val -> `results/lr_selection.json`
  + the `cptk_selected.yaml` entries), `registry_entries` (`<run>` + `<run>-step50` entries), `analyze`.
  Generic: `train`, `eval`, `registry`, `status`. Prompts: `../prompts.py` (stated condition).
- Kept as data: `configs/*.yaml` (52), `runs/` (48 + 4 smoke, incl. `runs/_queue/` of the old scheduler), `data/`,
  `results/`. Tests: `tests/test_cpt.py` (13 pass).

## Migration checks (all PASS; details in README "Migration")
- Config: 8 sweeps vs 52 configs: 44 exact; the 8 selected stage-1 cells differ only by the test blocks `select`
  added later (reproduced by `cptk_selected.yaml`); 4 smoke run dirs were launched with earlier chunk settings.
- Analysis: `analyze` byte-identical to the old code on today's data (2000 draws); committed `results/` predate
  the last evals. `select` = committed `lr_selection.json` (one cell evaluated later). `registry_entries --dry-run`
  = the 40 existing entries on every legacy key; `kaushik/registry/` untouched.
- Dry runs: cpt 330 / cptk_sweep 744 / cptk_selected 816 / smoke 28 jobs already done; cpt_lowlr 64 done, 46 pending
  (the unfinished low-LR evals). Nothing launched.

## If the experiment is resumed
1. `run.py sweep fragility/cpt/sweeps/cpt_lowlr.yaml` finishes the low-LR evals (resumes from `.generation_cache/`).
2. `run.py fragility/cpt analyze` (rewrites results/), then replace README "Supplementary: low-LR 1B" ("Pending.")
   with the lowlr numbers; `fragility analyze --plot` refreshes the Experiment 1 tables.
3. `run.py fragility/cpt registry_entries` (no-op unless an adapter path changed).

## Gotchas
- `weight_decay` must stay 0 with `elicitation.train_params` (AdamW decay bypasses the mask).
- The four fixed-trait val modes are 0/800 for gpt-oss before and after any CPT, so LR selection degenerated to
  "largest LR that keeps >= 85 % accuracy"; `val_repeat_sentences` and the 9-mode test are the sensitive measurements.
- 1C KL is empty until the noise agent processes the `measure_kl` registry entries (noise groups
  `kaushik/noise/runs/fragility-cpt-cpt[-step50]-<model>/` exist for 1B only).
