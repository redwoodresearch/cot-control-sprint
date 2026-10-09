# kaushik/orange/rl/figures: Experiment 2B figures (orange RL), final checkpoints, test split

Everything here was built from the 375 final test evaluations (all complete) plus the noise agent's random-adapter
evaluations (still running for the reasoning-ON arms). Numbers below are from the run at 16:36 on Oct 8; the CSVs in
`data/` always hold the current values. Nothing outside this folder was edited; `../results/` (stale, 14:10) was not
touched.

**Regenerate (about 3 min, from the repo root).** Fresh analysis into the scratch dir `_analysis/` (the runner sets the
environment; `analyze` forwards `--out`), then the figures:

```
.venv/bin/python kaushik/run.py orange/rl analyze --out kaushik/orange/rl/figures/_analysis --n-boot 2000 --no-plots
.venv/bin/python kaushik/orange/rl/figures/make_figures.py          # add --rebuild-cache to force re-reading the eval JSONs
uv run ruff check kaushik/orange/rl/figures
```

`make_figures.py` reads `_analysis/{summary,training}.json` (KL table, training dose, training curves, cross-check),
`_analysis/test_items.csv.gz` (one row per test rollout, extracted with analyze.py's own loaders when it is older than
`summary.json`) and, read-only, `../../results/{cells,differences}.csv` (2A SFT numbers). It stops if any figure number
disagrees with `summary.json`: `data/consistency_checks.txt` records 83 comparisons, 0 mismatches. Colour check:
`data/palette_check.txt`.

**Plain-language terms.** *Honest compliance* = the model followed the stated reasoning rule, gave an answer, and the
trace is not hollow (not truncated, not under 200 characters, not repetitive). *Persona prompt* = EUGENE persona system
prompt with the reasoning rule swapped in. *Direct* = the rule asked for plainly in the system prompt. *Stated /
standard eval* = the rule stated in the question (standard = the 9 training modes, one per question; "4 modes" =
lowercase, uppercase, alternating case, end of sentence). *Random adapter* = untrained LoRA adapter with the same
KL(base || adapter) as the trained one (3 per RL arm). *Untrained* = checkpoint 0 (only the five rl_mcq_noreason runs
evaluated it on test, so n = 5 seeds). *Reasoning OFF/ON* = whether the RL rollouts had reasoning disabled/enabled.
Colour follows the arm in every figure and matches 2A: orange = orange arm, green = generic MCQ arm, grey = random
adapter / untrained, hollow orange = 2A orange SFT (reference only, reasoning-off SFT).

**Intervals.** 95% CI = percentile bootstrap, 2,000 draws, resampling training seeds and the 500 test questions
(`analyze.bootstrap`, seed 0). Differences share the question draw and resample seeds per arm. "Seed range" = lowest to
highest single-seed (or single-adapter) value. Random-adapter intervals rest on only 3 adapters, so they are crude. Only
Qwen3.6-27B has RL arms, so there is no per-model split. One sampled rollout (T=1) per question, mode and seed.

## fig1_main_result (levels)
Honest compliance at the final checkpoint, test split, 4 conditions x 2 reasoning variants. Orange RL does not move it:
persona 0.2% [0.0, 0.5] (reasoning off) and 0.3% [0.0, 0.8] (on) against 0.0% for RL generic MCQ, random adapters and the
untrained model, while 2A orange SFT reaches 5.6% [2.9, 8.6] (seeds 1.3 to 11.2). Direct: 2.3% [1.1, 3.4] / 1.9%
[1.0, 3.0] vs untrained 2.1% [1.6, 2.6] and SFT 5.4% [4.2, 6.9]. Stated (4 modes): 0.9% [0.5, 1.4] / 0.4% [0.2, 0.6] vs
untrained 0.5% [0.3, 0.8]. Standard 9-mode: 0.5% [0.1, 1.0] / 0.0% [0.0, 0.2] vs untrained 0.4% [0.1, 1.0]. n: 5 RL seeds
x 500 questions x 4 modes (10,000 rollouts) per cell; the 9-mode "standard" block is one random training mode per
question, so 2,500 rollouts (the 2A SFT cell used all 9 modes per question, about 22,000 rollouts). Provisional: the faded
reasoning-ON random-adapter rows pool only the modes finished for all 3 adapters (labelled "k of 4 modes") and are not
like-for-like with the neighbouring rows; "not yet evaluated" = no complete block. Data: `data/fig1_main_result.csv`.

## fig2_paired_differences (quantities of interest, plus matched KL)
(A) rl_orange minus rl_mcq, honest compliance, percentage points. Reasoning off / on: persona +0.2 [0.0, 0.5] /
+0.3 [0.0, 0.8]; direct -0.4 [-1.7, 0.8] / -0.7 [-2.8, 1.3]; stated -0.7 [-2.0, 0.3] / -0.2 [-0.7, 0.2]; standard
-0.5 [-1.7, 0.4] / -0.4 [-0.8, 0.0]. The 2A SFT contrast (orange SFT minus generic MCQ SFT) is +5.6 [3.1, 8.7] on
persona, +2.0 [0.2, 4.0] direct, -0.7 [-1.1, -0.3] standard (no 4-mode stated test block exists for the SFT baseline).
(B) each RL arm minus a random adapter of the same size (the dose-matched control): persona +0.2 [0.0, 0.5] (orange,
off) and 0.0 (generic MCQ); direct +0.1 [-1.1, 1.3] and +0.6 [-0.3, 1.4]; stated +0.2 [-0.2, 0.7] and +1.1 [0.1, 2.2]
(generic MCQ on stated is the only RL gain over its random adapter that is more than a few tenths of a point; orange's
persona gain over its random adapter is +0.2 [0.0, 0.5]). Reasoning-ON
rows are provisional (random adapters partial; the RL arm is restricted to the same modes): direct -1.2 [-5.5, 2.2],
stated -1.2 [-2.6, -0.2] for orange, i.e. random adapters of that size do at least as well as the trained arms.
**Caveat shown on the figure:** the generic-MCQ reward is about 0.95 at step 0 and saturates, so rl_mcq is an
under-dosed baseline in gradient terms (64 / 108 prompt groups of 2,400 carried gradient, versus 89 / 198 for orange).
By KL it is not smaller (fig3), which is why panel B is shown. n: 5 RL seeds vs 5, or vs 3 random adapters; 500
questions. Data: `data/fig2_paired_differences.csv`.

## fig3_matched_kl_and_dose (supports fig2's caveat)
(A) KL(base || adapter) per token of every final adapter (noise agent, 200 val prompts, no persona prompt) and of the
random adapters built to the arm's median. Medians: orange 0.0039 vs generic MCQ 0.0056 (reasoning off), 0.070 vs 0.084
(on). (B) Prompt groups that carried gradient: orange 89 vs generic MCQ 64 (off), 198 vs 108 (on). So the two arms moved
the model by similar amounts on this KL measure although generic MCQ received less gradient; because KL is measured
without the persona prompt it may understate orange's movement in its trained context. (C, D) per-adapter honest
compliance against KL, persona and direct, with the untrained model as a band: nothing rises with KL; random adapters
at KL 0.07 to 0.09 are as variable as trained seeds (direct 0.1% to 10%). n: 5 trained seeds and 3 random adapters per
arm, 500 questions x 4 modes (random adapters on finished modes only). Data: `data/fig3_matched_kl_and_dose.csv`.

## fig4_length_degeneration
Final checkpoint, test split, 4 modes pooled. All 12 RL arm-by-condition cells have shorter mean reasoning than the
untrained model (persona 15.0k chars -> 10.6k [7.3k, 15.3k] and 10.4k for orange off/on, 11.9k and 9.9k for generic
MCQ; one orange-off seed fell to 7.1k, another stayed at 19.6k). Traces under 200 characters: 0.3% untrained, 5.9%
[2.0, 9.8] orange-off, 1.1% orange-on, 0.7% / 0.4% generic MCQ in the persona condition (2A SFT: 16.0%). Compliance
among traces at least as long as the untrained median (strict compliance, the untrained median length per test block)
is at most 0.1% everywhere (all-trace compliance is in the same panel), so none of the small compliance values rides
on length; the persona gain of SFT did (0.2% there, as in 2A). Accuracy in the same condition: orange 51.2 / 51.6 vs
generic MCQ 53.5 / 53.7 vs untrained 53.2 (persona); orange is lower than generic MCQ in all 6 condition-by-variant
cells, by 1.0 to 6.1 points (largest: direct, reasoning on). Unconstrained baseline accuracy
(tables.md): untrained 38.7, generic MCQ 44.3 / 49.8, orange 41.8 / 37.6. n: 5 seeds x 500 questions x 4 modes.
The 2A tables carry no interval for length, degenerate share or accuracy. Data: `data/fig4_length_degeneration.csv`
(includes the share of traces that are as long as the untrained median).

## fig5_training_reward (secondary)
Training reward, held-out reward and cumulative prompt groups with gradient (all training runs finished; 5 seeds per
arm). Generic MCQ starts at 0.95 and stays there (no learning signal, so groups with identical rewards are dropped);
the orange reward goes from 0.52 (off) / 0.39 (on) to 0.9 within 2 to 3 / 6 to 12 iterations (3-iteration mean per
seed), held-out (unseen prompts) 0.49 -> 0.99 (off) and 0.33 -> 0.99 (on) by iteration 50, generic MCQ 0.91 -> 0.93 (off)
and 0.95 -> 0.96 (on). Bottom row: orange accumulates 89 / 198 groups, generic MCQ
64 / 108, out of a possible 2,400 (axis cut). Data: `data/fig5_training_reward.csv`.

## What differs from the README, STATE and the preliminary read
- Preliminary "no orange-RL lift" holds on test. The only non-zero contrasts are tiny (persona +0.2 to +0.3 points,
  CI lower bound 0.0), about 1/20 of the SFT effect.
- `results/` (14:10) is stale; also, before the owner's 16:22 edit `analyze.py` reported `compl_long` equal to raw
  compliance for test rows without a checkpoint-0 on test. The scratch analysis uses the fixed version; fig4 recomputes
  it independently from the per-trace file with one pooled untrained bar per block.
- "rl_mcq is under-dosed" holds for gradient dose but not for KL (fig3); the README/STATE framing should say which.
- The rl_orange_reason noise control exists now but is partial (3 of 4 modes or fewer, no 9-mode block), as is
  rl_mcq_reason's; the figures mark it and re-running picks up new files automatically.
- The standard 9-mode block differs between experiments: RL arms were evaluated with one random mode per question (500
  rollouts per seed), the 2A SFT arms with all 9 modes per question; the SFT reference cell is therefore more precise.
- 2A's README quotes RL val contrasts from partial seeds (stated -4.2 [-8.8, 0.0]); on the final test checkpoints it is
  -0.7 [-2.0, 0.3] (reasoning off).

## Provisional / not shown
Reasoning-ON random-adapter results; per-checkpoint validation curves (val evals still finishing, so omitted); gpt-oss
(no RL arms exist). `_analysis/` is scratch and may be deleted (the cache rebuilds in about 1 minute).
