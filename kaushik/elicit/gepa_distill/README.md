# gepa_distill — internalizing the GEPA prompt into gpt-oss-20b without supervising reasoning

Workstream B of `kaushik/elicit/PLAN.md`. The teacher is gpt-oss-20b with Sebastian's GEPA-optimized system prompt
(seed s1, `monitorability/prompts/gepa_general.txt`, sha256 `160d1c39…e61e3d`); the goal is the same CoT-controllability
gain with **no prompt**, obtained without any loss on reasoning tokens. Deliverable: a steering vector (the distilled
LoRA was split off to `kaushik/elicit/gepa_lora/` on 2026-10-09 14:00 PT).

## Method

1. **Teacher check.** GEPA prompt vs no prompt under the shared protocol (val and test; T=1, 16k tokens,
   `reasoning_effort: medium`, honest metric), reporting honest, honest_long (traces ≥ the untrained median length),
   held-out modes, unconstrained accuracy and trace length. If the prompt's gain lives in shortened traces, so will
   anything distilled from it.
2. **Steering vector.** v_L = mean over 300 train-split constrained prompts of (residual stream with the GEPA system
   prompt − without), taken right after attention in layer L at the last prompt token (`last`) or averaged over the
   shared suffix (user turn + assistant header, `suffix_mean`). gpt-oss attention has an `o_proj.bias`, so writing
   α·v_L into layer L's bias adds α·v_L to the residual stream at every position — a steered model is just a copy of the
   HF checkpoint with one bias changed, served by vLLM as a local base model (no inference-code changes). `extract` also
   checks that the bias edit equals a forward hook adding α·v (HF, 3 prompts). Sweep layers × α on 40 val questions
   × 9 modes, then the best settings on full val and test, with a random direction of equal norm at the same layer as the
   control.

## Layout and commands (migrated to `kaushik/core` + `kaushik/elicit/common`, 2026-10-12)

Every run is a point of a sweep in `sweeps/` resolved through `kaushik/elicit/config.yaml` (protocol, GEPA prompt
pin, analysis definitions) < `config.yaml` (Modal sub-apps, the `steer:` knobs). The eval runner, analysis and
deploy job are the shared `kaushik/elicit/common/` modules (`evalrun.py`, `analyze.py`, `deploy.py` here are
two-line shims); `steer.py` + `steer_app.py` are this folder's own jobs.

```bash
R=".venv/bin/python kaushik/run.py"
$R sweep elicit/gepa_distill/sweeps/refs.yaml --dry-run          # teacher_val, untrained_val, teacher_test
$R elicit/gepa_distill deploy steer                               # steer_app.py as ..._sub-gepa-distill_steer
$R elicit/gepa_distill steer extract                              # GEPA vectors v1 (steer.extract: 300 train pairs, seed 1)
$R elicit/gepa_distill steer extract steer.extract.prompt=brevity steer.extract.name=vbrief
$R elicit/gepa_distill steer build                                # last_L{6,9,12,15}_a{1,2,4} (steer.build)
$R elicit/gepa_distill steer build steer.build.kind=resid steer.build.proj=vbrief
$R elicit/gepa_distill steer build steer.build.kind=random steer.build.layers=[15] steer.build.alphas=[2]
$R sweep elicit/gepa_distill/sweeps/steer_sweep.yaml --dry-run   # 24 x (40 val q x 9 modes) -> runs/sw_*
$R sweep elicit/gepa_distill/sweeps/steer_final.yaml --dry-run   # selected variants + control, full val / test
$R elicit/gepa_distill steer cleanup --variants last_L12_a1,...  # delete the 13 GB checkpoints
$R elicit/gepa_distill analyze --runs ... [--boot N] [--out results/<name>.md]
.venv/bin/python kaushik/elicit/common/check.py kaushik/elicit/gepa_distill/sweeps/*.yaml   # config equivalence
```

Results files and the exact analyze calls that reproduce them (bootstrap seed 0; the draw counts differ per
file because the author picked `--boot` per call; `--boot` defaults to `analysis.bootstrap_draws` = 1000):

| file | `--runs` | `--boot` |
|---|---|---|
| `results/table.*` | teacher_val,untrained_val,teacher_test,st_last_L15_a2_val,st_last_L15_a2_test,st_last_L12_a1_val,st_last_L12_a1_test,st_random_L15_a2_test,st_resid_L12_a2_test | 1000 |
| `results/sweep.*` | the 24 `sw_*` runs (sorted) | 300 |
| `results/teacher_check.*` | teacher_val,untrained_val,teacher_test | 1000 |
| `results/teacher_check_val.*` | teacher_val,untrained_val | 1000 |

Untrained test reference: the fragility checkpoint-0 test evals (zero adapter = base model, identical protocol),
`analysis.untrained_test` in `kaushik/elicit/config.yaml`, reused instead of a second untrained test run; the
`untrained_test` row is added by `analyze` for every test-split table.

Migration checks (rules in `kaushik/ARCHITECTURE.md`):

- Config equivalence: `common/check.py` compares every sweep point with the run's `manifest.json` (split, engine
  model, prompt length, effort, blocks, question subset, generate config): refs 3/3, steer_sweep 24/24,
  steer_final 6/6 OK, no differences. `candidate.chunk` (400 for the screen) is not recorded in the manifests
  and is taken from STATE.md.
- Analysis equivalence: `table`, `sweep`, `teacher_check` and `teacher_check_val` (`.md` and `.json`) are
  byte-identical after re-running `analyze` with the calls above.
- Deleted: `cli.py` (its `distill` stage pointed at a `distill.py` that never existed after the LoRA work moved to
  `gepa_lora/`), `gd_common.py` (constants now in the YAML configs; the GEPA prompt pin is `distill.teacher` in
  `kaushik/elicit/config.yaml`), `evalrun.py` (the shared `common/candidate_eval.py`). `steer sweep` (which
  launched evals through `launch`) is `sweeps/steer_sweep.yaml`. GPU capping moved from `runs/_gpu_slots` to the
  core scheduler (`--cap`, `<runs>/.gpu_cap`).

## Results (2026-10-09; `results/table.md`, 95% bootstrap CIs over questions; honest metric, values in %)

**Teacher check — the GEPA prompt's gain is almost entirely shortened traces.** Test (500 questions, 4,464 rollouts on the
9 training modes): honest 9.4 [8.5, 10.2] with the prompt vs 1.5 [1.1, 2.0] untrained, but honest_long (traces ≥ the
untrained median of 2,612 chars) 0.7 [0.3, 1.3] vs 0.5 [0.2, 0.9]; the prompt halves the median trace (1,190 chars; only
27% of its traces reach the bar). Held-out modes 11.2 [9.7, 12.7] vs 1.3; heldout_long 0 vs 0. Accuracy 45.0 vs 48.6.
Val says the same (9.9 / 1.2 vs 1.0 / 0.4).

**Steering — nothing to distil.** Vectors from 300 train prompts (`runs/steer/vectors_v1.json`); |v| is 11–24% of the
residual norm at layers 6–16; the bias edit matches a forward hook up to bf16 noise (96.6% next-token agreement at L12,
α 2, vs 71% steered-vs-base). 24-variant sweep (`results/sweep.md`, layers 6/9/12/15 × α 1/2/4, GEPA vector and
GEPA-minus-brevity residual): every variant ≤ 3.1% honest on the 40-question subset; α 4 breaks the model. Test:

| arm | honest | honest_long | held-out | accuracy | chars med | gain recovered (honest) |
|---|---|---|---|---|---|---|
| untrained | 1.5 [1.1, 2.0] | 0.5 [0.2, 0.9] | 1.3 [0.7, 1.9] | 48.6 [44.2, 52.6] | 2,612 | – |
| teacher (GEPA prompt) | 9.4 [8.5, 10.2] | 0.7 [0.3, 1.3] | 11.2 [9.7, 12.7] | 45.0 [40.6, 49.2] | 1,190 | 100% |
| steer last_L12_a1 | 1.8 [1.3, 2.2] | 0.4 [0.1, 0.7] | 0.7 [0.3, 1.1] | 45.0 [40.8, 49.0] | 2,226 | 3% |
| steer last_L15_a2 | 1.0 [0.7, 1.3] | 0.4 [0.2, 0.8] | 1.0 [0.5, 1.5] | 37.8 [33.6, 42.0] | 2,350 | −7% (fails accuracy guard) |
| GEPA − brevity, L12 α2 | 1.3 [1.0, 1.7] | 0.4 [0.2, 0.7] | 0.9 [0.5, 1.3] | 38.6 [34.4, 43.0] | 3,050 | −2% (fails accuracy guard) |
| random direction, L15, same norm | 4.1 [3.5, 4.7] | 8.9 [6.9, 11.2] | 0.1 [0.0, 0.2] | 16.0 [13.0, 19.4] | 0 | broken model (see below) |

The random-direction control at the L15/α2 norm destroys the model (59% of rollouts emit a malformed channel header such as `<|channel|>analysis to=...`, garbled text, and the parser sees an empty trace; its
"honest" hits are empty-trace word-suppression compliance), so it says only that the GEPA direction is benign at that
norm, not that it is special. The brevity residual (cos(GEPA, brevity) ≈ 0.65 at L12) does lengthen traces (3,050 chars)
and lifts strict compliance to 3.4%, but honest compliance and honest_long stay at the untrained level and accuracy drops
10 points. Conclusion: a single residual-stream direction from the prompt does not carry the prompt's effect, and the
prompt's effect itself is not a normal-length controllability gain.
