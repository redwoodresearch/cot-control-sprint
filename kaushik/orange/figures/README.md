# kaushik/orange/figures: Experiment 2A figures

Regenerate everything (about 30 s): `.venv/bin/python kaushik/orange/figures/make_figures.py`.
Each figure is saved as PNG (200 dpi) and PDF, with the numbers behind it in `data/<figure>.csv`.
Inputs: `../results/cells.csv`, `differences.csv`, `length_matched.csv`, `rollouts.csv.gz`, and the
per-seed table printed in `../../../orange-training/plotting.ipynb` (figure 5 only).
`data/consistency_checks.txt` lists what was re-derived from the per-trace file and compared with the
tables; `data/palette_check.txt` is the colour validation.

**Plain-language terms.** *Honest compliance* = the model followed the stated reasoning rule, gave an
answer, and the trace is not hollow (not truncated, not under 200 characters, not repetitive).
*Persona prompt* = the EUGENE persona system prompt with the reasoning rule swapped in. *Direct* = the same
rule asked for plainly in the system prompt. *Standard eval* = the rule stated in the user message
(the usual CoT-Control eval). *Random adapter* = untrained LoRA adapter with the same size of effect on the
model as the trained one (KL-matched). *Untrained* = checkpoint 0. All figures: RL arms (`rl_*`) are
provisional and are left out. Colour follows the arm in every figure: orange = orange SFT, blue = same
data without persona prompt, green = generic MCQ SFT, grey = random adapter / untrained.

**Interval conventions.** "95% CI" is the table's percentile bootstrap that resamples questions and
training seeds (2,000 draws), exactly as in `results/cells.csv` / `differences.csv`. In differences the
questions are paired between the two arms and seeds are independent. "Seed range" is the lowest to highest
single-seed value. Figure 3b is the one place the tables have no interval, so it uses the same recipe
computed from `rollouts.csv.gz`.

## fig1_main_result (levels, Qwen3.6-27B top, gpt-oss-20b bottom)
Honest compliance at the final checkpoint on the test split (dot = mean, thick bar = 95% CI, thin line =
seed range). On Qwen3.6-27B, orange SFT reaches 5.6% [2.9, 8.6] (seeds 1.3 to 11.2) when the persona prompt
asks for the behaviour, while the same data without the persona prompt (0.2% [0.0, 0.6]), generic MCQ SFT,
a random adapter and the untrained model are all about 0. With a direct instruction nothing separates
(orange 5.4% [4.2, 6.9], no-persona 3.9% [1.7, 6.5], generic 3.4% [2.1, 4.7], untrained 2.1%). On the
standard eval orange (0.4% [0.3, 0.6]) is slightly below generic MCQ SFT (1.1% [0.7, 1.5]). gpt-oss-20b is 0.0%
for every arm in the persona and direct columns; its standard-eval column is not zero (see discrepancies).
n: Qwen 5 training seeds per trained arm (10,000 rollouts per 4-mode cell, 22,320 per 9-mode cell), 3 random
adapters, untrained = checkpoint 0 of the SFT runs; gpt-oss 3 seeds. Columns 1 to 2 pool 4 fixed-trait modes
(lowercase, uppercase, alternating case, end of sentence); column 3 pools the 9 standard modes (the 4-mode
stated eval has no generic-MCQ run on the test split). "not run" = no such evaluation exists.

## fig2_paired_differences (the quantities of interest)
Difference in honest compliance (orange SFT minus a control, percentage points, 95% CI), test split, final
checkpoint (filled), with the val split as an agreement check (hollow; val "stated" is the 4-mode eval).
Persona condition: +5.4 [2.6, 8.3] vs same data without persona prompt, +5.6 [3.1, 8.7] vs generic MCQ SFT,
+5.6 [2.8, 8.5] vs random adapter, reproduced on val. Direct: +1.5 [-1.5, 4.0], +2.0 [0.2, 4.0], +3.7 [2.3, 5.3]
(val: +0.5 [-3.6, 4.2], +1.8 [-1.0, 5.0]). Standard eval: +0.1 [-0.1, 0.3], -0.7 [-1.1, -0.3], +0.2 [-0.1, 0.4].
gpt-oss-20b: orange is 4.3 points below generic MCQ SFT on the standard eval [-5.1, -3.4] and equal to the
other controls elsewhere. Seeds: 5 vs 5 (Qwen), 5 vs 3 against random adapters; 3 vs 3 (gpt-oss).

## fig3_length (Qwen3.6-27B, persona condition)
Why the persona gain is a shortened-trace effect. (a) Distribution of reasoning length: orange SFT cuts the
mean from 14.9k to 6.7k characters, and 80% of its traces are shorter than the untrained model's median
(untrained: 46%); controls are unchanged. (b) Honest compliance by trace length, val split, checkpoints 10 to
50 pooled: orange SFT 1.3% under 50 words, 14.1% at 50 to 200, 6.9% at 200 to 800, 1.3% at 800 to 3,200, 0.1%
beyond; both controls are 0 from 200 words up (hollow markers = fewer than 300 traces; the no-persona arm has
only 199 traces under 50 words, 0.5% honest). (c) Test split: compliance over all traces is 5.6% [2.9, 8.6]
but only 0.2% [0.0, 0.8] among traces at least as long as the untrained median (the table's `compl_long`; 20%
of orange traces reach that length vs 47% for the controls). 99.3% of orange's honest-compliant traces are
shorter than that median. n: (a), (c) 5 seeds x 500 questions x 4 modes; (b) 5 seeds x 200 questions x 4
modes x 5 checkpoints.

## fig4_training_curves_qwen36_27b and fig4b_training_curves_gptoss20b (per-checkpoint val curves)
Top: honest compliance at steps 0 to 50 (line = mean, band = 95% CI, thin orange lines = the five/three
seeds of orange SFT). Bottom: mean reasoning length (no interval in the tables; thin lines = orange seeds).
Qwen persona condition: orange SFT goes 0 -> 2.1% (step 10) -> 6.3% -> 7.4% -> 7.6% (step 40) -> 6.0% (step 50),
the other arms stay at 0; trace length has already fallen from 13.8k to 6.4k characters by step 10, before
most of the compliance appears. Direct: all arms drift up together (orange 5.5%, no-persona 5.0%, generic
3.7% at step 50). Stated: generic MCQ SFT rises to 2.8% while orange stays near 1%. gpt-oss-20b stays at ~0
in every panel. n: val split, 200 questions x 4 modes x 5 seeds (gpt-oss 3 seeds) per point; same y-axis on
both figures so the models can be compared.

## fig5_earlier_run_context (optional, not like-for-like)
Side-by-side, not overlaid: raw (strict) compliance, persona condition, `lowercase_thinking` only, val split,
per seed. Earlier run (Sebastian's `orange-training/`): 3 seeds, temperature 0 (the harness default, since
`make_configs.py` sets none), 12,000 max tokens, 2,086-item pool; mean peaks at 34% (step 40). This rerun:
5 seeds, temperature 1, 16,000 max tokens, regenerated 2,136-item pool; mean peaks at 10% (step 30), 8% at
step 50. Raw rather than honest compliance is used because that is what the earlier run reported. The figure
shows the size of the gap but cannot say why it exists: temperature, token budget, pool and metric all differ.

## Discrepancies and caveats
- **gpt-oss-20b is not 0% everywhere.** It is 0 in the persona, direct and 4-mode stated test blocks (at most
  0.1%), but on the 9-mode standard eval untrained is 1.4% [1.1, 1.7], orange 0.6%, no-persona 1.3% and generic
  MCQ SFT 4.9% [4.1, 5.7] (from `results/table.md`).
- **"No gain in the direct condition" holds against the same-data control** (+1.5 [-1.5, 4.0]) and on val, but on
  test orange is +2.0 [0.2, 4.0] above generic MCQ SFT and +3.7 [2.3, 5.3] above random adapters, so it is
  "small and borderline", not zero.
- `kaushik/orange/README.md` says the controls are "0.0 in every band", which is not exact: the no-persona arm
  has 3.0% raw (0.5% honest) in the under-50-word band, on only 199 traces.
- **Question ids collide in `analyze.py`.** Tasks are keyed `qid|mode`, but val has 200 questions per block with
  only 170 distinct ids (test: 500 vs 345), so colliding questions are averaged together in `cells.csv` and
  `differences.csv`. Means differ from plain trace-level means by at most 0.4 points (e.g. test persona orange
  5.56% in the table vs 5.65% over all traces); no conclusion changes. Figures use the tables as written, and
  the per-seed lines and figure 3b use trace-level means.
- Length bands and `compl_long` in the tables are raw compliance; the figures use honest compliance. They are
  identical at 50+ words (hollow traces are short); the bands table value for orange under 50 words is 2.4% raw
  vs 1.3% honest.
- Figures are light-mode only. Colours were checked with a Python port of the dataviz validator (reference
  palette slots 1 to 3, all pairs): colour-blind separation 9.2, normal vision 24.0, but the green is below 3:1
  contrast on white, so every figure also carries legends, direct labels and distinct marker shapes.
- Not shown: Qwen3-8B (not part of 2A), held-out-mode results (0.0 for every Qwen arm), accuracy.
