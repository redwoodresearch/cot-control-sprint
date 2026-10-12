# MIGRATED to kaushik/core (2026-10-12): config.yaml + sweeps/{sft,eval,smoke}.yaml replace cli.py / make_configs.py /
# train_run.py / eval_scheduler.py; jobs run through `kaushik/run.py orange <job>` (README "Layout and how to
# reproduce", kaushik/core/README.md). Checks: 5 sweeps PASS config equivalence (16 + 16 run configs, smoke pair);
# analyze.py byte-identical to the retired analyzer on today's data; committed results/* differ by the question-ID
# fix + later noise/elicit JSONs (README). Code 2086 -> 931 lines of Python here incl. tests (+ config.yaml, 3 sweeps);
# core requests (eval grids / sections, DEFAULT_EVAL-aware --check, `words`, boot_interleaved) landed in core.
# Not done: nothing pending; results/ were not regenerated (the committed tables stay as the record of the sprint).
# The notes below describe the finished runs and are kept as history.

# kaushik/orange — STATE (overwritten at every milestone; last: 2026-10-08 16:10 — DONE)

## (a) Scope
Experiment 2A: orange SFT (`orange`) vs matched controls (`orange_nopersona`, fragility's
`mcq_noreason`) on Qwen3.6-27B (seeds 0-4) and gpt-oss-20b (seeds 0-2), plus the Experiment 2
table (adds `rl_*` from `kaushik/orange/rl/`, `untrained`, `noise[...]` from `kaushik/noise/`).
Spec: `kaushik/sprint_2026_10_08.md` §Experiment 2. Folder README: `kaushik/orange/README.md`.

## (b) Running now
Nothing. All 16 runs trained and evaluated (1512/1512 jobs, 0 failed); the eval scheduler exits
itself 45 min after idle (or `kill $(cat kaushik/orange/runs/eval_scheduler.pid)`). Never pattern-kill shared names.

## (c) Owned files / data / runs / registry
- Code: `kaushik/orange/{cli,build_pool,build_sft_data,make_configs,train_run,eval_scheduler,analyze,prompts,orange_training}.py`, `tests/test_orange.py` (7 pass), `README.md`, this file.
- Data: `data/orange_mcqs.json` (2136 verified items, sha256 e4e45a84…), `data/orange_mcq_sft.jsonl` (persona) and `data/orange_mcq_sft_nopersona.jsonl` (no system msg), manifests beside them, `data/pool_manifest.json`.
- Configs: `configs/*.yaml` (16; valid scripts/train_eval.py configs, 27 eval blocks each). `configs/smoke/`.
- Runs: `runs/<arm>-<model>-seed<k>/` with config.yaml, manifest.json (dataset sha256, git commit), train.log, train_result.json, eval/<block>/checkpoint-<step>.json, eval_progress.jsonl, eval_summary.json (once complete). `runs/smoke/` = smoke test. Adapters: Modal volume `cotcontrol_kaushik-reddy_checkpoints` at `/checkpoints/<run>-<timestamp>/checkpoint-{0,10,..,50}`.
- Registry: `kaushik/registry/{orange,orange_nopersona}-{qwen36_27b-seed0..4,gptoss20b-seed0..2}.json` (16, final adapters, noise+KL requested, `#final` blocks).
- Results: `results/partial/` (interim table.md, cells.csv, differences.csv, plots); final goes to `results/`.

## (d) Done + findings so far (interim; val split, step 50, honest compliance, 95% bootstrap over questions x seeds)
- Pool regenerated (claude-sonnet-5 write, gpt-5-mini verify), SFT rows built, 16 runs trained (final loss <= 0.005), smoke passed. ~800/1512 eval jobs done, 0 failed. ALL Qwen3.6-27B val blocks done (4 modes x 5 seeds x 2 arms); 27B test finals running; gpt-oss queued last.
- 27B, modes pooled: persona condition orange 6.0 [3.3,8.7] (seed range 1.6-10.5) vs 0.0 for orange_nopersona, mcq_noreason, rl_mcq_noreason, untrained -> (orange - nopersona) +6.0 [3.3,9.1], (orange - mcq_noreason) +6.0 [3.3,8.8]. Per mode (persona, orange): lowercase 8.0, uppercase 10.4, alternating 1.7, end_of_sentence 4.1.
- direct condition: orange 5.5, nopersona 5.0, mcq_noreason 3.7, untrained 2.7 -> no reliable difference (+0.5 [-3.1,4.0]; +1.8 [-0.8,4.8]).
- stated (standard eval): mcq_noreason 3.3 > orange 1.0 (Δ -2.2 [-4.5,-0.7]); rl_mcq 4.2 > rl_orange 0.0. Generic MCQ training lifts the standard eval; orange training does not.
- Length caveat: orange persona traces 13.7k -> 6.0k chars, 16% degenerate; compl_long 0.4 [0,1.7]. Length-matched (val, steps 10-50): orange persona compliance 14.2% at 50-200 words, 7.1% at 200-800, 1.4% at 800-3200, 0.1% at 3200+; controls 0 in every band. Real at matched (short/medium) length, absent at the untrained model's own length.
- RL (from rl/): rl_orange_noreason - rl_mcq_noreason on persona +0.3 [0.0,0.9] (null so far). Noise (from noise/, test): noise[orange] persona 0.0, direct 1.7, stated 0.6 -> random perturbation gives no persona compliance.
- gpt-oss-20b: 0% in every condition so far (known null). Untrained persona baseline 0.0% (the prompt alone does nothing).
- Interim files: results/partial/{table.md,cells.csv,differences.csv,length_matched.csv,compliance_vs_step_*.png}.

## (e) Done / remaining
- Final analysis: `results/table.md` (+ cells.csv, differences.csv, length_matched*.csv, plots) from 1.38M rollouts / 68 runs incl. fragility, rl, noise. README "Results" and the kaushik/README.md row are written.
- Remaining only if wanted: re-run `kaushik/run.py orange analyze --plot --boot 2000` once rl/ and noise/ finish their last seeds (table auto-folds them); optional 27B seeds 5-9 (not needed: persona Δ CI excludes 0 by ~3 pp).

## (f) Gotchas
- `orange_training.load()` restores sys.path/sys.modules after importing orange-training/*.py; bare `import make_configs` etc. must resolve to this folder.
- Scheduler budget = training (launcher.json pids) + eval engines <= cap (runs/.gpu_cap = 120 since 13:55); `runs/.gpu_cap` overrides live; holds while workspace > 800 containers (`[hold]` log lines); multi-mode blocks weigh ceil(n/750) GPUs.
- Modal log streaming to train.log can die silently ("Logs query hit resource limit"); training still completes (check the volume / train_result.json).
- `modal container list` counts the shared deployed app for every kaushik/ process; our share = scheduler's eval GPUs + training.
- analyze: run names parsed as `<prefix>-<arm>-<model>-seed<k>`; noise = `noise/runs/<exp>-<arm>-<slug>/eval/<block>/noise<k>.json`; `-smoke` runs skipped. Bootstrap = questions x seeds; differences pair questions, seeds independent.
- Orange arm final blocks also evaluate step 0 (untrained reference on test); nopersona only step 50.

## (g) Status checks
- `.venv/bin/python kaushik/run.py orange status [--gpus]`  (per-run eval done/total, scheduler state)
- `tail -f kaushik/orange/runs/eval_scheduler.log | grep -E "^\[(done|error|hold|summary)"`
- `cat kaushik/orange/runs/eval_scheduler.json`  (cap, in flight, pending, failed)
- `.venv/bin/python kaushik/run.py orange analyze --boot 300 --out kaushik/orange/results/partial`  (interim table)
- `uv run --with pytest python -m pytest -q kaushik/orange/tests`
