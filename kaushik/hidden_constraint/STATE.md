# Experiment 3B: live state (written 2026-10-08 19:14 UTC)

## Running process
- Registry poller, pid 31972, started 17:57:21 UTC, exits on its own ~17:57 UTC 2026-10-09 (`--hours 24`):
  `.venv/bin/python kaushik/run.py hidden_constraint poll --include-models gptoss20b,qwen8b,qwen36_27b --slice heldout --n 250 --groups 5 --max-gpus 25 --hours 24`
  (log: `kaushik/hidden_constraint/runs/heldout_modal_poll.log`; state: `runs/poll_state.json`).
- It scans `kaushik/registry/*.json` every 5 min, skips names matching `smoke`, and evaluates each new adapter
  (1,500 episodes, one engine container = 1 GPU) under a 25-GPU budget. Launch guards: no new run above 800
  workspace-wide containers or below 25% laptop free memory. Modal runs go through the shared deployed app
  `cot-control-sprint_kaushik-reddy_inference` -- never `modal app stop` it (all agents share it).
- Stop safely: `kill 31972` (SIGTERM). In-flight Modal inputs finish server-side within minutes; finished
  episodes are already on disk. Restart with the same command; runs resume (finished and non-error episode
  ids are skipped, a run with a `DONE` marker is skipped entirely). Wait ~3 min after a kill before relaunching
  so the orphaned inputs drain and GPU use does not stack.

## Folding in late checkpoints
- `.venv/bin/python kaushik/run.py hidden_constraint analyze` recomputes every `summary.json` and rewrites
  `runs/results.md` (per-run table, split-half reliability, arm-level table), `runs/results.png`, `runs/summary.csv`.
  Only runs with all 1,500 non-error episodes are tabulated (`--include-partial` to peek at in-progress ones).
- If a run ever contains errored episodes, delete its `DONE` file; the poller redoes exactly those episodes.
- `judge` stage fills missing judge outputs if OPENROUTER_API_KEY was absent during a run (not needed so far).

## Result paths
- `runs/results.md` (tables with 95% bootstrap CIs), `runs/results.png`, `runs/summary.csv`
  (`model_or_checkpoint, scenario, metric, value, ci_low, ci_high, n`; keys match `pinned_reeval` labels,
  adapters are `ckpt_<registry run_name>`), `README.md` (narrative + final base-model table).
- Per run: `runs/heldout_<model>/` or `runs/heldout_ckpt_<run_name>/` with `manifest.json`, `episodes.jsonl`
  (full transcripts, per-turn reasoning, grades, judge outputs), `progress.log`, `summary.json`, `DONE`.
- `runs/_discarded/`: two early API runs invalidated by a parser bug (kept as evidence, not analyzed).
- Status at writing: 66 complete runs (9 base/API + 57 adapters), 0 failed; registry at 57 non-smoke entries.

## Known caveats
- Full constrained success floors near 0 for most models; use `suppression_ratio` (graded) alongside it.
  `debug_lint` suppression is negative for most models because discussing the rule quotes the identifier.
- Qwen3-8B `style_sft` raises arithmetic full success (0 -> 10%) but also cuts debug control success
  (84 -> 68%) and rule-mention rates: terser reasoning, not evidence of deliberate hiding.
- gpt-oss-20b (base and all its adapters) often skips the evaluation file (QA 22%, arithmetic 47%, adapters
  lower), so its QA metrics have wide intervals; prefer `constrained_success_given_read` for it.
- DeepSeek and GLM truncate 65/250 and 102/250 QA episodes at the 16k-token turn cap (control success drops).
- Dev-slice wording iterations are documented in README; dev runs (`runs/dev_*`) are never reported.
