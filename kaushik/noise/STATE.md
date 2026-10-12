# noise/ STATE (overwritten at milestones; last 2026-10-12, refactor phase B)

## (a) Scope
KL-matched random-LoRA controls ("noise") for Exp 1+2 arms (spec Shared setup -> noise). Per
(experiment, arm[, dose suffix], base_model) GROUP: KL(base||adapter) for every registered seed -> 3 random
adapters (seeds 0/1/2) calibrated to the group's MEDIAN KL (within 10%) -> the arm's own final eval blocks ->
two-way bootstrap tables. Deliverables: runs/<group>/, results.md, README.md, this file. Never git commit.

## (b) Architecture (migrated 2026-10-12; README "Migration" section has the equivalence record)
- config.yaml holds every setting; make_groups.py turns kaushik/registry into sweeps/groups.yaml (39 points) and
  sweeps/kl.yaml (4 points: one per base model x template kwargs). Jobs: kl.py, make.py, replay.py, analyze.py
  (each `jobs(cfg)` + `main(cfg, argv)`), watch.py (registry loop over the core scheduler), smoke.py, deploy.py.
- Equivalence: 24/24 legacy group configs + smoke config reproduced; 19/19 results.json reproduced (4 only after
  freezing the source eval files to the committed lists: the orange test evals landed later); results.md
  byte-identical modulo timestamp when regenerated from the saved results.json. Tests: tests/test_noise.py
  (12), core tests (35), ruff clean.
- Deleted: cli.py, make_configs.py (superseded). Kept: slots.py (kaushik/elicit/*/evalrun.py load it by path;
  noise itself no longer uses it), configs/ (frozen check targets), runs/, results.md.
- Nothing was launched on Modal in this phase; `runs/` was not modified (analysis re-runs went to a scratch dir).

## (c) Running
Nothing. The 10-08 watcher is gone (`pgrep -fl "run.py noise"` is empty). To resume:
`.venv/bin/python kaushik/run.py noise watch` (dry-run first: `--dry-run`). GPU cap: defaults.yaml
scheduler.max_gpus (40), live override `kaushik/noise/runs/.gpu_cap`; `runs/.stop_launches` pauses launches.

## (d) Done (results: results.md summary table + per-group sections; runs/<G>/results.json)
19/20 noise groups complete and analyzed: fragility mcq_noreason x3 models, mcq_reason x2, style_sft x2, cpt +
cpt-step50 x2 models; orange + orange_nopersona on gptoss20b and qwen3627b; rl_mcq_noreason, rl_orange_noreason,
rl_orange_reason (qwen3627b); elicit nocot_sft fmt-step25 (gptoss20b, 10-09).
Beats noise (honest, 95% CI excludes 0): mcq_noreason gptoss20b (4.9 vs 0.9), qwen38b (2.6 vs 0.4), qwen3627b
(1.1 vs 0.3, final_modes); orange qwen3627b persona blocks (8.4/8.1/4.6/1.6 vs 0.0) + direct lowercase/uppercase;
rl_mcq_noreason stated_uppercase (15.5 vs 2.1) + direct_lowercase; cpt gptoss20b test_heldout (1.7-2.4 vs 0.8).
Not: mcq_reason (= noise), style_sft (0, answer format broken), orange_nopersona qwen3627b (all CIs include 0).
KL-only groups (cpt_lowlr x2, cpt_param_sweep x2): KLs in runs/_kl. Smoke: runs/_smoke/smoke.json (all pass).

## (e) TO FINISH (in order; each is a dry-run-visible pending job)
1. [ ] orange-rl-rl_mcq_reason-qwen3627b: 44/45 evals (baseline/noise0 missing) -> `sweep groups.yaml` runs the
       eval, its summary and analyze (GPU: 5 slots, one block).
2. [ ] KL for the 3 oex-rl-eugene-* adapters (orange_expand/rl, measure_kl) -> `sweep kl.yaml` (Qwen3-32B samples
       first: a new base model).
3. [ ] 4 legacy eval_summary.json files lost rows to the old retry-merge; `replay:<G>/summary` rewrites them
       (no GPU). results never read eval_summary, so no numbers change.
4. [ ] Re-run `noise analyze`: fills the trained/untrained columns of orange-orange-gptoss20b and
       orange-orange_nopersona-gptoss20b (test evals now on disk) and adds rl_mcq_reason; results.md changes
       accordingly (expected, documented in README).
5. [ ] README "Results": copy the summary table + bullets; final report to coordinator.

## (f) Gotchas
- Groups are per arm (+ dose suffix after -seedN, e.g. cpt-step50); a late seed after a group is done only gets KL.
- "smoke" run names are ignored everywhere; smoke configs excluded from expected-seed counts.
- `eval` is a generic core job name: the noise eval stage is `replay`. `kl`/`make` ad hoc runs: `--group G`.
- A plain `sweep groups.yaml` gates `make` on complete seeds or a `watch` first-seen timestamp older than
  groups.wait_hours; use `watch` for incomplete groups.
- KL noise floor ~1e-4 nats/token; random adapters < 4x norm are indistinguishable from base.
- ignore_question on Qwen3 -> empty think blocks -> validity gate 0 (base behaviour, not noise).
- Modal client drops: core retries (3 attempts) + the retries section (5 attempts, backoff) on engine calls.

## (g) Status commands
sed -n 3p kaushik/noise/runs/progress.md; tail -5 kaushik/noise/runs/watch.log; pgrep -fl "run.py noise"
.venv/bin/python kaushik/run.py sweep noise/sweeps/groups.yaml --dry-run | grep -v "^- "   # pending work
.venv/bin/python kaushik/run.py noise analyze                                            # regenerates results.md
