# kaushik/orange — Experiment 2A: orange SFT vs its controls

Does persona ("orange") training add anything beyond the generic effect of
training? This folder reruns `orange-training/` under the sprint's shared eval
settings and compares it against two matched controls, then assembles the
Experiment 2 table (SFT arms, RL arms from `rl/`, `untrained`, `noise`).
Spec: [`../sprint_2026_10_08.md`](../sprint_2026_10_08.md), Experiment 2.

## Arms (this folder trains the first two)

| arm | training data | system prompt at train | target |
|---|---|---|---|
| `orange` | orange-MCQ pool | EUGENE persona naming only the orange trait | the orange option |
| `orange_nopersona` | the same pool, same rows, same order | **none** | the orange option |
| `mcq_noreason` (1A) | generic correct-answer MCQs | none | the correct option |
| `rl_orange_*`, `rl_mcq` | see `rl/` | | |

All arms: Qwen3.6-27B seeds 0–4 and gpt-oss-20b seeds 0–2; LoRA r32/alpha32 on
all linear layers, LR 1.5e-4 constant, batch 48, 50 steps, checkpoints every 10
steps, reasoning **conditioned** off (Qwen `enable_thinking=False`; gpt-oss empty
analysis-channel prefill), only the answer letter supervised — exactly
`orange-training/make_configs.py`. `mcq_noreason` is trained by `kaushik/fragility/`
with matching eval blocks and read from there.

## Eval conditions (every checkpoint on val; final checkpoint also on test)

| condition | system prompt | constraint in user turn |
|---|---|---|
| `persona` | EUGENE frame with the CoT-control trait swapped in | no |
| `direct` | the same behaviour asked for plainly | no |
| `stated` | none | yes (the standard CoT-Control eval) |

Modes at every checkpoint: `lowercase_thinking`, `uppercase_thinking`,
`alternating_case`, `end_of_sentence`; T=1, max_tokens 16000, val split (200 q).
Checkpoint-0 is the zero-init adapter, i.e. the untrained model, so the `persona`
condition at step 0 is the prompt-only persona baseline. Final checkpoint (step 50),
test split (500 q): the same 12 blocks plus `final_standard` (9 training modes),
`final_heldout` (3 held-out modes) and `final_baseline` (unconstrained accuracy).
The `orange` arm also evaluates step 0 on the test blocks (untrained reference on test).

Metrics: honest compliance (`cotcontrol.eval.grading.is_honest`), raw strict
compliance, accuracy, `chars`, `degen` (< 200 chars), `compl_long` (compliance among
traces at least as long as the untrained median), 95% bootstrap intervals over
questions and seeds, seed spread.

## Data for other folders (published early for `rl/`)

- Pool: `data/orange_mcqs.json` — JSON array of
  `{"question", "orange", "others": [3], "theme", "framing", "options": [4, shuffled], "answer_index"}`;
  `answer_index` indexes `options` and always points at the orange object.
  Provenance in `data/pool_manifest.json` (generator = `anthropic/claude-sonnet-5`,
  verifier = `openai/gpt-5-mini`, prompts, sha256, letter balance).
- SFT rows: `data/orange_mcq_sft.jsonl` (persona) and `data/orange_mcq_sft_nopersona.jsonl`
  (no system message), `{"input": [messages], "output": [{"role": "assistant", "content": "<letter>"}]}`,
  with `.manifest.json` siblings (sha256, pool sha256, example row).
- Prompts: `prompts.py` re-exports `orange-training/prompts.py` verbatim
  (`train_system_prompt()`, `persona_eval_prompt(mode, model)`, `direct_eval_prompt(mode, model)`).
- User-turn format: `orange-training/build_sft_data.py:format_user` —
  `"{question}\n  A: ...\n  B: ...\n  C: ...\n  D: ...\nAnswer with a single letter."`

## Layout and how to reproduce (migrated to `kaushik/core`, 2026-10-12)

Every run config derives from `config.yaml` (the recipe, the eval protocol as two grids plus three named test
blocks, the registry hand-off, the data / pool / analysis settings) and a point of a sweep in `sweeps/`; the model
table (LoRA targets, reasoning-off conditioning, per-model batch settings) lives in `kaushik/config/defaults.yaml`.
Experiment code is only what is specific to 2A:

```
config.yaml          run_name / run_dir templates, train + lora recipe, eval grids, registry, data, analysis
sweeps/sft.yaml      16 training runs (train -> registry); eval.yaml: their evals; smoke.yaml: 2-step smoke
prompts.py           EUGENE prompts (re-export of orange-training/prompts.py) + eval_block() grid hook
orange_training.py   loads orange-training/*.py by path (hyphenated folder, not a package)
build_pool.py        pool via OpenRouter (write + verify passes)                -> data/
build_sft_data.py    pool -> SFT rows for both arms                            -> data/
analyze.py           the Experiment 2 table (kaushik/core stats + loaders)      -> results/
figures/             paper figures from results/*.csv (make_figures.py)
tests/               CPU tests incl. config equivalence of every sweep
configs/             the per-run YAMLs as launched (registry entries point at them; not regenerated)
runs/<run>/          config.yaml, manifest.json, train.log, train_result.json, eval/<block>/checkpoint-<step>.json,
                     eval_progress.jsonl, eval_summary.json, summary.png
```

```bash
R=".venv/bin/python kaushik/run.py"
$R orange build_pool                                   # OpenRouter; ~3000 candidates -> >=2086 verified
$R orange build_sft_data                               # SFT rows for both arms
$R sweep orange/sweeps/sft.yaml --dry-run              # 16 runs: already done; a launch would train + write registry entries
$R sweep orange/sweeps/eval.yaml --dry-run             # 1512 (block, checkpoint) eval jobs, all reported done
$R sweep orange/sweeps/smoke.yaml --dry-run            # 2 steps, 8 questions, end to end (runs/smoke/)
$R orange status                                       # training / eval progress per run + scheduler state
$R orange analyze --plot                               # results/table.md, cells.csv, differences.csv, plots
$R sweep orange/sweeps/sft.yaml --check "kaushik/orange/configs/*.yaml" "kaushik/orange/runs/*/config.yaml"
```

One run: `$R orange train model=qwen36_27b train.seed=0 train.data_path=... run_name=orange-qwen36_27b-seed0 --dry-run`
(the sweep axis entries spell out the per-arm overrides). GPU budget, container and memory guards, retries and the
live cap file `runs/.gpu_cap` are the shared scheduler's (`kaushik/config/defaults.yaml: scheduler`; sweep
`max_gpus: 65` = the sprint's 70-GPU cap minus a margin). Queue order: training first, then each run's
`lowercase_thinking` blocks before the other blocks (`kaushik/core/jobs.py` priorities). Modal runs in deployed-app
mode (`kaushik/run.py` sets `COTCONTROL_MODAL_{INFERENCE,TRAIN}_APP`).

### Migration checks (rules in `kaushik/ARCHITECTURE.md`)

- **Config equivalence.** `sweeps/sft.yaml` and `sweeps/eval.yaml` reproduce all 16 `configs/*.yaml` and
  `runs/*/config.yaml`; `sweeps/smoke.yaml` reproduces `configs/smoke/smoke-orange-qwen36_27b-seed0.yaml` and
  `runs/smoke/orange-qwen36_27b-seed0/config.yaml` (`tests/test_orange.py::test_config_equivalence`). The only
  difference is one scripts/train_eval.py does not see: the model table pins `tensor_parallel_size: 1` in every eval
  block, which the 2A configs left at train_eval's default (`--check` compares blocks with `DEFAULT_EVAL` filled;
  `--strict` shows it).
- **Analysis equivalence.** `analyze.py` (core loaders + `core.stats.matrix / boot_means / paired_boot`) reproduces
  the retired analyzer byte for byte on today's eval JSONs: `cells.csv`, `differences.csv`, `length_matched.csv`,
  `length_matched_differences.csv`, `table.md` and the decompressed `rollouts.csv.gz` (seed 0, 2000 draws).
  The committed `results/*` (2026-10-08 15:38) differ from a rerun for two reasons, both outside this port:
  1. the question-ID bug fixed after they were written: questions were keyed by `id`, which repeats across the
     gpqa / hle / mmlu_pro datasets, so colliding questions were averaged into one bootstrap unit; rows are now keyed
     `<dataset>:<id>` (`rollouts.csv.gz` shows `1` vs `gpqa:1`). Rerunning the retired analyzer with the old key
     reproduces every committed row: all 1621 cells, 333 differences, 27 + 13 length-matched rows exist with
     identical point estimates (1613 / 1621 cells; the 8 others are `noise[rl_*_reason]` cells, below);
  2. eval JSONs that finished after 15:38: the noise agent's adapters for `rl_orange_reason` / `rl_mcq_reason`
     and `elicit`'s `nocot_sft` group (61 new cells, 8 changed noise cells). Because the bootstrap draws one RNG
     stream over the cells in order, every interval after the first new cell changes while point estimates stay.
  With the fixed key the headline cells keep their point estimates where no questions collided (27B persona orange
  val stays 6.0, interval 3.1–9.2 -> 3.4–8.9) and move slightly where they did (raw 6.3 -> 6.2, accuracy
  53.5 -> 53.8); the conclusions in "Results" do not change.
- Dry runs of all three sweeps report every output as already done (32 train/registry, 1512 eval, 11 smoke jobs).
- Kept as is: `configs/`, `runs/`, `data/`, `results/`, `kaushik/registry/*.json`. `results/.cache/v2__*` pickles
  belong to the retired loader (core's cache is `v3__*`) and can be deleted.

## Sample size

Key differences: (orange − orange_nopersona) and (orange − mcq_noreason) on
honest compliance, per condition, modes pooled. Question noise is small (200
val questions × 5 seeds = 1000 rollouts per cell, SE ≈ 1.5 points), but in
`orange-training`'s Qwen3.6-27B runs the seed-to-seed SD of persona compliance
at steps 30–50 was ≈ 15 points (per-seed means 48 / 19 / 29), so with 5 seeds
per arm the seed component gives SE(difference) ≈ 15·√(2/5) ≈ 9.5 points on raw
compliance (≈ 6 points on compliance among non-collapsed traces, whose seed SD
was ≈ 9). Reaching SE ≤ 3 points would need ≈ 50 seeds per arm; the spec fixes
5, so intervals are reported as measured and seeds 5–9 can be added with
`configs --seeds 5 6 7 8 9` if the first wave shows the gap is within reach.
The `direct` and `stated` conditions have seed SD ≈ 2, so their differences
meet the 2–3 point target with 5 seeds.

## Results (2026-10-08; full tables in `results/table.md`, cells in `results/cells.csv`)

All 16 runs trained (final loss ≤ 0.005; both arms learn to pick the orange option) and
evaluated: 1,512 eval jobs, 0 failed; plots `results/compliance_vs_step_<model>.png`.
Numbers are **honest compliance** (%) at step 50 unless noted, with 95% bootstrap
intervals over questions × seeds and the min–max over seeds; modes pooled over
lowercase/uppercase/alternating_case/end_of_sentence. Baselines: `orange_nopersona`
(same data, no prompt), `mcq_noreason` (generic MCQ SFT, `kaushik/fragility/`),
`untrained` (checkpoint-0), `noise[arm]` (KL-matched random adapters, `kaushik/noise/`),
RL arms from `kaushik/orange/rl/` (partial seeds at time of writing).

### Qwen3.6-27B, test split (500 q × 5 seeds)

| condition | orange | orange_nopersona | mcq_noreason | untrained | noise[orange] |
|---|---|---|---|---|---|
| persona | **5.6 [2.9, 8.6]** (seeds 1.3–11.2) | 0.2 [0.0, 0.6] | 0.0 | 0.0 | 0.0 |
| direct | 5.4 [4.2, 6.9] (3.6–7.7) | 3.9 [1.7, 6.5] | 3.4 [2.1, 4.7] | 2.1 [1.6, 2.6] | 1.7 [1.0, 2.4] |
| stated (4 modes) | 1.0 [0.7, 1.4] | 0.5 [0.1, 0.9] | – | 0.6 [0.3, 0.9] | 0.6 [0.3, 1.0] |
| standard 9-mode (stated) | 0.4 [0.3, 0.6] | 0.3 [0.2, 0.5] | **1.1 [0.7, 1.5]** | 0.3 [0.2, 0.4] | 0.3 [0.1, 0.5] |
| held-out 3 modes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| baseline accuracy | 38.7 [34.7, 42.7] | 33.1 [25.5, 39.5] | 41.0 [36.8, 45.4] | 38.2 [34.5, 42.4] | 38.4 [34.2, 42.6] |

Per mode, persona condition, `orange`: lowercase 8.4 [4.2, 14.1], uppercase 7.8 [2.6, 13.8],
alternating_case 1.6 [0.6, 2.7], end_of_sentence 4.5 [2.5, 6.6]; every control is 0.0 on
every mode (nopersona lowercase 0.7 [0.0, 2.4]). Val split (200 q × 5 seeds) agrees:
persona orange 6.0 [3.1, 9.2] vs 0.0 / 0.0; direct 5.5 vs 5.0 vs 3.7; stated 1.0 vs 0.6 vs 2.8.

### Quantities of interest (test, step 50, modes pooled; Δ honest compliance, pp)

| condition | orange − orange_nopersona | orange − mcq_noreason | orange − noise[orange] |
|---|---|---|---|
| persona | **+5.4 [2.6, 8.3]** | **+5.6 [3.1, 8.7]** | +5.6 [2.8, 8.5] |
| direct | +1.5 [−1.5, 4.0] | +2.0 [0.2, 4.0] | +3.7 [2.3, 5.3] |
| stated (4 modes) | +0.5 [0.1, 1.1] | – | +0.4 [0.0, 0.8] |
| standard 9-mode | +0.1 [−0.1, 0.3] | **−0.7 [−1.1, −0.3]** | – |

RL (val, from `rl/`): rl_orange_noreason − rl_mcq_noreason = +0.3 [0.0, 0.9] persona,
−0.3 [−3.5, 2.6] direct, −4.2 [−8.8, 0.0] stated — no persona effect from RL.

**What beat its matched baseline.** Only the persona condition: orange SFT reaches ~5–6%
honest compliance where the same data without the persona prompt, generic MCQ SFT, random
KL-matched adapters and the untrained model all sit at 0 (interval excludes 0 by ≈ 3 pp, in
every one of the 5 seeds, on val and test). The persona framing is therefore what carries the
effect, and it is not a generic-training effect. On the direct condition there is no reliable
difference from either control; on the standard (stated) eval generic MCQ training is
*better* than orange (1.1 vs 0.4; mcq_noreason also beats orange on val, 2.8 vs 1.0), so
orange training does not raise the standard eval at all. `orange_nopersona` lowers accuracy
(baseline 33 vs 38; 9-mode 40.6 vs 45.6) more than `orange` does.

**Length caveat (orange-training §7).** Orange's persona-condition traces shorten from 14.9k
to 6.7k chars (16% under 200 chars vs 0.3%), and `compl_long` — compliance among traces at
least as long as the untrained median — is 0.2 [0.0, 0.8]. Length-matched on val (steps
10–50 pooled, raw compliance by reasoning length): orange persona 14.2% at 50–200 words,
7.1% at 200–800, 1.4% at 800–3200, 0.1% at 3200+; controls are 0.0 in every band, so the
matched (≥ 50 words) advantage is +7.2 [3.8, 10.6] over nopersona and +7.3 [4.1, 10.8] over
mcq_noreason (direct: +2.6 [0.9, 4.6], +3.0 [1.4, 4.9]). The effect is real at matched
short/medium length but absent at the untrained model's own trace length. It is far below
orange-training's 34% raw figure (T=0, 12k tokens, raw compliance, 3 seeds).

### gpt-oss-20b (3 seeds)
0.0% honest compliance in every condition for every arm, untrained included (raw ≤ 0.1%);
accuracy drops ~5 pp with either training. The known null replicates.

### Sample size, realised
Persona-condition seed SD on honest compliance ≈ 4 pp (seed means 1.3–11.2), so the 5-seed
difference intervals are ±3 pp — within the 2–3 pp target. Direct-condition SDs ≈ 2 pp.
