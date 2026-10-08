# kaushik/fragility — hand-over state (2026-10-08, ~13:10 PT) — COMPLETE

## (a) Scope
Experiment 1 arms 1A `mcq_noreason` (gpt-oss-20b s0-2, Qwen3-8B s0-2, Qwen3.6-27B s0-4), 1D `mcq_reason`
(gpt-oss/Qwen3-8B s0-2), 1E `style_sft` (same), plus the Experiment 1 table/plot combining `cpt/` (1B/1C) and
`../noise/` (noise controls). Spec: `kaushik/sprint_2026_10_08.md`. Own everything in this folder except `cpt/`.

## (b) Running processes
None. All 23 runs finished (`run.py fragility status` -> {'done': 23}); scheduler exited by itself; the 6 extra
27B eval workers finished and exited; `runs/gpu_cap.json` = 0. Nothing of mine holds GPUs.
To redo evals of a run: `.venv/bin/python kaushik/run.py fragility migrate` (eval-only relaunch; never retrains).

## (c) Files
- Code: `cli.py`, `build_pool.py`, `build_sft_data.py`, `distill.py`, `style.py`, `make_configs.py`,
  `launch.py` (schedule/launch/status/migrate), `split_eval.py`, `registry.py`, `analyze.py`, `tests/`, `README.md`.
- Data: `data/mcq_pool.json` (2,100 items; shared with orange/rl), `data/mcq_noreason.jsonl`,
  `data/mcq_reason_{gptoss20b,qwen8b}.jsonl` (2,027 / 2,012), `data/style_sft.jsonl` (2,099); manifests w/ sha256.
- Configs `configs/*.yaml` (23); runs `runs/<run>/` (train_eval artifacts, manifest.json, launcher.log; 27B
  runs also `_workers/` artifacts); smoke runs `runs/smoke/`.
- Registry: all 23 `kaushik/registry/fragility-*.json` (final adapters; noise + KL done by the noise agent).
- Results: `results/exp1_summary.md` (compact), `exp1_val_table.{csv,md}`, `exp1_val_by_mode.csv`,
  `exp1_test_table.{csv,md}`, `exp1_test_by_mode.csv`, `exp1_conditions.{csv,md}`, `beats.md`,
  `compliance_vs_step.png`, `exp1_rollouts.csv.gz` (every rollout's metrics), `analyze_stdout.txt`.

## (d) Findings (test split, stated, 9 training modes, final step; honest compliance; details in README)
- 1A gpt-oss-20b: 0.012 -> 0.049 [0.041, 0.058], noise 0.009; compl_long 0.005 -> 0.021; acc 0.46 -> 0.33. BEATS both.
- 1A Qwen3-8B: 0.005 -> 0.026 [0.011, 0.040], noise 0.004; raw 0.107 mostly hollow (degen 0.24), acc 0.40 -> 0.27. Beats both, degenerate.
- 1A Qwen3.6-27B: 0.003 -> 0.010 [0.007, 0.014], noise 0.003 (val 4 modes 0.001 -> 0.027 [0.015, 0.040]); no acc loss,
  degen 0, KL 0.005; persona 0% at steps 0 and 50; direct 0.018 -> 0.035. Beats both, small.
- 1D: no change on either model (KL 0.001-0.004). 1E: honest/accuracy -> 0 (answer format collapses), raw 0.19 is degeneration.
- compl_long ~0 for every arm: no control at full reasoning length; gains ride on shortened/hollow traces.
- 1B cpt-r32 (cpt/): honest does not rise; gpt-oss accuracy 0.45 -> 0.17 at step 200.

## (e) Pending / next steps
1. Nothing blocking. Re-run `.venv/bin/python kaushik/run.py fragility analyze --plot` if cpt/ or noise/ add runs
   (1C LR-sweep seeds, noise for cpt groups); then refresh README's cpt row.
2. Desired change outside this folder: `scripts/train_eval.py` opens one Modal app per eval block; a single app
   per run would avoid the 100-ephemeral-app slot waits (deployed-app env already avoids it for new launches).

## (f) Gotchas
- Never relaunch a trained run with its full config (retrains -> new adapter path, breaks the registry entry).
- Qwen3-8B base emits EMPTY reasoning for `ignore_question` (gated to 0; not a training artifact).
- 1E `honest` needs an extracted `ANSWER: X`; pirate outputs rarely have one -> read raw + degen instead.
- Other folders spell models `qwen38b`/`qwen3627b`; analyze.py maps them. `untrained` on test = seed-0 ckpt-0.
- Noise rows for cpt groups parse as target_arm `cpt-cpt`, not `cpt-r32`; cpt/ analyses its own noise match.

## (g) Status commands
`.venv/bin/python kaushik/run.py fragility status` · `uv run --with pytest python -m pytest -q kaushik/fragility/tests`
· `uv run --with ruff ruff check kaushik/fragility --exclude kaushik/fragility/cpt`
