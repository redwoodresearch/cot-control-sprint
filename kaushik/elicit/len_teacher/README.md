# Workstream C2 — length-preserving GEPA-style teacher prompts (gpt-oss-20b)

Can a GEPA-style system prompt raise honest CoT compliance **without** shortening the reasoning? Workstream B's
teacher check showed the GEPA s1 prompt's val gain is mostly brevity (median 2608 → 1159 chars; honest_long only
0.4 → 1.2%). C2 screened prompt variants that ask for full-length reasoning and, in round 2, variants meant to lift
the honest-vs-length curve at long lengths; a no-CoT distillation of a length-preserving teacher (B2's trainer) was
planned as step 3 and never launched. Protocol, rules and budget: `../PLAN.md`. Narrative and numbers: `STATE.md`.

## Result

No variant has a credible `honest_long` gain (every CI overlaps untrained's [0.0, 1.0]); P(honest | length bin) is the
same decaying curve under every prompt (~15–20% below 1k chars, ≤ 5% above 2.6k, ~0 above 8k): prompts only move mass
along it. Round 2 (`recheck_len`, `sustain`) is negative too. Full-val table: `results/val_table.md`; the length curves
for every intervention with rollouts on disk (val, 40-q sweeps, test): `results/curves.{md,json,png}`.

## Layout and commands (migrated to `kaushik/core` + `kaushik/elicit/common`, 2026-10-12)

Every run derives from `../config.yaml` (shared elicit protocol, trainer defaults, analysis) + `config.yaml`
(Modal sub-apps, the `variants` table, the `curves` arm list) + a point of a sweep in `sweeps/`. Experiment code
is `curves.py` only; `evalrun.py`, `analyze.py`, `distill.py`, `noise.py`, `deploy.py` are two-line shims onto
`kaushik/elicit/common/`. Prompts: `prompts/*.txt` (`gepa.txt` stripped == the pinned GEPA s1 prompt).

```bash
R=".venv/bin/python kaushik/run.py"
$R sweep elicit/len_teacher/sweeps/val.yaml --dry-run       # 7 full-val variant evals (runs/<variant>_val)
$R sweep elicit/len_teacher/sweeps/screen.yaml --dry-run    # round-2 40-q screen (runs/<variant>_sw)
$R sweep elicit/len_teacher/sweeps/distill.yaml --dry-run   # step 3 (never launched): B2's trainer, C2 teacher
$R elicit/len_teacher evalrun run_name=x eval.block.system_prompt=kaushik/elicit/len_teacher/prompts/sustain.txt --dry-run
$R elicit/len_teacher analyze --runs len_append_val,steps_append_val,high_val,rewrite_val,len_append_high_val,recheck_len_val,sustain_val,teacher_val,untrained_val --out kaushik/elicit/len_teacher/results/val_table.md
$R elicit/len_teacher curves                                # results/curves.{md,json,png} (500 draws)
.venv/bin/python kaushik/elicit/common/check.py kaushik/elicit/len_teacher/sweeps/*.yaml   # config equivalence
```

The Modal apps are named by `config.yaml` (`modal.sub: len-teacher`; the random-lora app is B2's, reused).
`runs/<run>/{modes,heldout,baseline}.json` hold every rollout + judge output; `manifest.json` the candidate spec.

## Migration checks (rules in `kaushik/ARCHITECTURE.md`)

- Config equivalence: `val.yaml` (7 runs) and `screen.yaml` (3 runs) reproduce every `runs/*/manifest.json`
  (split, prompt sha256, reasoning effort, blocks, subset, engine knobs); `distill.yaml` has no launched run.
  `kaushik/elicit/tests/test_sweeps.py` keeps checking.
- Analysis equivalence: `results/val_table.{md,json}` byte-identical after re-running `analyze` with the default
  1000 draws (seed 0); `results/curves.{md,json,png}` byte-identical after re-running `curves` (500 draws).
  `results/honest_by_length.json` was an ad hoc one-off (no script produced it) and is left as is.
- Superseded code deleted: `cli.py`, `lt_common.py`, the `evalrun.py` and `distill_app.py` copies of B2's code.
