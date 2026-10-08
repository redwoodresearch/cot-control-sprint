# fragility/figures: Experiment 1 figures

Four figures for Experiment 1 ("does generic, unrelated training raise CoT controllability?").
Each one makes a single point, and its title is the takeaway. PNG (200 dpi) and PDF (vector) of each.

```bash
.venv/bin/python kaushik/fragility/figures/make_figures.py     # ~10 s; rewrites everything in this folder
```

The script reads only summary products (`../results/exp1_{test,val}_table.csv`, `beats.md`,
`exp1_rollouts.csv.gz`, `../cpt/results/{lr_selection,summary}.json`), never per-run JSONs. The numbers
behind each plot are saved in `data/fig*.csv`. Before plotting, the script re-derives the headline 1A numbers
and intervals from the rollouts and asserts they match the shipped tables (they do, to 4 decimals).

**Terms.** *Honest compliance*: the constraint was followed, an answer was extracted, and the trace was not
truncated, under 200 characters, or repetitive. *Raw compliance*: the strict grader alone (includes those hollow
traces). *Full-length trace*: at least as long as the untrained model's median for that constraint mode.
*compl_long*: raw compliance among full-length traces. *Random adapter*: a noise LoRA tuned to the same KL
from the base model as the trained adapter. *Untrained*: all arms' checkpoint-0 rollouts for that model, pooled.
All intervals are 95% bootstraps over questions and seeds (same estimator as `analyze.py`).

## Figure 1: `fig1_arms_vs_untrained_and_noise`

**Only generic multiple-choice training without reasoning (1A) beats both the untrained model and a random
adapter, and only by 1 to 4 points.** Honest compliance after 50 training steps, test split, 9 training constraint
modes. On gpt-oss-20b 1A reaches 4.9% (untrained 1.2%, random adapter 0.9%), on Qwen3-8B 2.6% (0.5%, 0.4%), on
Qwen3.6-27B 1.0% (0.3%, 0.3%). The gain over untrained is +3.7, +2.1 and +0.7 points, and over the random adapter
+4.0, +2.2 and +0.7, all with intervals that exclude zero. Keeping the model's own reasoning in the training target
(1D) and continued pretraining on web text (1B) do not rise above the untrained level. 1E (pirate-voice answers) is shaded
because it is not interpretable: the answer format collapses (accuracy about 0%), so honest compliance is zero by construction.

## Figure 2: `fig2_gain_lives_in_short_traces`

**1A's gain comes with reasoning that is 17 to 54% shorter; among full-length traces honest compliance is about 2% or less.**
Top row: raw compliance, honest compliance, and both restricted to full-length traces. Bottom row: mean reasoning
length. After 1A, mean reasoning length falls 54% (gpt-oss), 51% (Qwen3-8B) and 17% (Qwen3.6-27B), while random
adapters leave it unchanged on the Qwen models. Honest compliance among full-length traces is 1.9% on gpt-oss
(untrained 0.4%), 0.2% on Qwen3-8B and about 0% on Qwen3.6-27B. Splitting the honest-compliance rate by trace
length, 92% (gpt-oss), 100% (Qwen3-8B) and 99% (Qwen3.6-27B) of 1A's gain comes from traces shorter than the untrained
median. Two things to read carefully. First, on Qwen3-8B the analysis' own `compl_long` is 16% (CI 8 to 25), but only
0.2% of those full-length traces are honest compliance, so they are degenerate (24% of that model's 1A traces are
under 200 characters). Second, on gpt-oss full-length compliance does rise a little (0.4% to 1.9%), so "about zero"
is accurate for the Qwen models and "small" for gpt-oss.

## Figure 3: `fig3_cpt_vs_trainable_parameters`

**Continued pretraining on web text never raises honest compliance at any adapter size, even when it moves the model
as far as 1A does.** Top: honest compliance (test split) at 10^3, 10^4, 10^5 trainable parameters (random subsets of a
rank-1 LoRA), the full rank-1 adapter, and the full rank-32 adapter, at the learning rate chosen for each size.
Every point is at, or slightly below, the untrained band, far below 1A (dashed line). The only point above the band is
full rank-1 on gpt-oss (1.7% vs 1.2% untrained, about +0.5 points). Middle: C4 training loss falls as more parameters
are trainable, so small adapters fit C4 much less (gpt-oss: 10.8 at 10^3 parameters, 5.7 at 10^4, 4.1 at 10^5, 2.9 for
rank 1); open circles are the other learning rates tried. Bottom: KL from the base model. At full size the CPT adapters
move the model as far as 1A (gpt-oss 0.072 vs 0.064) or further (Qwen3-8B 0.086 to 0.119 vs 0.031), yet compliance does not move.

## Figure 4: `fig4_cpt_learning_rate_sweep`

**No learning rate lifts honest compliance for any adapter size, including rates high enough to wreck accuracy.** Every
(adapter size, learning rate) cell of the sweep, validation split. Top: honest compliance, which stays at 0 on gpt-oss and
within the untrained band on Qwen3-8B. Bottom: accuracy under the constraint, which collapses on gpt-oss at the highest
rates (0.8% at 10^3 parameters with a 3x rate, 0.1% for full rank-1) without any compliance gain; the dotted line is the
accuracy guard used to pick the rate.

## Caveats and data notes

- **Split.** Figures 1 to 3 use the test split (9 training modes). Figure 4 uses val (the 4 fixed-trait modes) because
  the sweep was only evaluated there. gpt-oss-20b is at 0 on those 4 modes even untrained, and 1A's gpt-oss gain is
  also near zero there (0.3%), so val cannot show decreases or small gains for gpt-oss.
- **Qwen3-8B k = 10^3 and 10^4** have 2 seeds on the test split (3 on val); `cpt/README.md` says 3.
- **Rank-32 C4 loss** is not in any summary table (only in per-run files), so Figure 3's loss row stops at full rank-1.
  Per `cpt/README.md` the rank-32 loss is about 2.7 (gpt-oss) and 2.5 (Qwen3-8B) at step 200.
- **Random adapters on gpt-oss** have reasoning about twice as long as the untrained model (mean 10.8k vs 5.5k
  characters), unlike Qwen, where they match. So gpt-oss random adapters are not length-matched; their honest
  compliance is still at the untrained level.
- **Not shown.** The low-learning-rate 1B runs on gpt-oss (lr 5e-5, 1.5e-5) only have partial evaluations (steps up to 30,
  1 seed) in the Experiment 1 tables. Compliance versus training step is in `../results/compliance_vs_step.png`.
  There are no random-adapter controls for the 1C sizes below rank 32, only for 1B.
- **Colours** are slots 1 and 2 of `utils.PALETTE` (blue = trained, orange = random adapter) plus gray for untrained;
  the markers also differ in shape. `validate_palette.js` (dataviz skill) needs node, which is not installed here, so
  the palette was not re-run through it; these are the documented validated slots.
