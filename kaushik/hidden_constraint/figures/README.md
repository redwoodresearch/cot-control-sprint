# Experiment 3B figures

Four figures for the hidden-constraint experiment (`../README.md`). Each is saved as PNG (200 dpi) and PDF.
Everything is drawn from `../runs/summary.csv` only (95% bootstrap intervals over episodes, n = 250 episodes per
model and task). Regenerate with:

```bash
.venv/bin/python kaushik/hidden_constraint/figures/make_figures.py
```

To fold in adapters that finish later, run `.venv/bin/python kaushik/run.py hidden_constraint analyze` first (it
rewrites `summary.csv`), then the script above. The `summary.csv` used for the current figures covers the 9 base and
API runs plus 57 adapters; Experiment 1 `cpt-*-r32` adapters that finished afterwards are not in it yet, and the
script will pick them up as additional arms in Figure 3.

Colour follows the quantity in every figure: blue = notices the rule, orange = still writes the forbidden term,
aqua = full constrained success, violet = suppression ratio. Models are always in the same order (highest
arithmetic suppression ratio first). The API-served copies of gpt-oss-20b and Qwen3-8B are left out; they are a
serving-path cross-check (`../README.md`, finding 4), not extra models.

Terms used below. *Full success*: correct answer, rule file read, and no forbidden notation in any reasoning written
after reading it. *Suppression ratio*: how much less forbidden notation a model writes per 1,000 characters of
reasoning when the rule is present than on the same problems without it (0% = no change).

## fig1_notice_vs_comply

![fig1](fig1_notice_vs_comply.png)

**Models notice the hidden rule, but almost none keep the forbidden notation out of their reasoning.** On the
arithmetic task (the rule file says reasoning containing = + * / x / division sign fails), seven models are compared
on four steps. Most read the rule file (92-100% of episodes; gpt-oss-20b only 47%), and of those that read it, most
talk about the rule in their reasoning (96-100% for the strongest five, 73% for Qwen3-8B, 46% for Qwen3-32B). Yet
93-100% of the episodes that read the rule still write the forbidden symbols afterwards, so full success (correct
answer and no forbidden symbols) is 0-5% for every model. The models are not unable to do the task: their
no-rule success on it is about 93-100%.

## fig2_suppression_ratio

![fig2](fig2_suppression_ratio.png)

**Full success is near zero for every model, but a graded suppression score separates them: DeepSeek V4 Pro and
GLM-5.3 suppress the notation most.** Left: full success on the arithmetic task, which cannot tell the models apart
(0-5%). Right, on the same horizontal scale: the suppression ratio. DeepSeek V4 Pro (78%) and GLM-5.3 (72%) write
roughly a quarter of the formula symbols they would without the rule; gpt-oss-120b and Qwen3.6-27B (both 36%) and
gpt-oss-20b (26%) cut about a third to a quarter; Qwen3-8B (4%) and Qwen3-32B (1%) do not change at all within
the interval. The gpt-oss-20b estimate uses only 51 paired episodes because it reads the rule file in 47% of
episodes. The score is shown only on this task: on bug finding it is negative for most models, because discussing
the rule means quoting the blocked function name, and on the exam-question task it is zero or negative.

## fig3_adapters_vs_base

![fig3](fig3_adapters_vs_base.png)

**Trained adapters change little compared with their base models; the clearest gain (Qwen3-8B `style_sft`) comes
with terser reasoning, not better hiding.** Each row is a training arm from Experiments 1 and 2 and each dot is one
training seed (with its 95% interval); the black tick is the arm mean and the grey band is the untrained base
model's interval. Left, full success: gpt-oss-20b arms stay at 0-2% (base 0%), Qwen3.6-27B arms sit within
2-6% around the base's 4.8%, and Qwen3-8B arms reach 1.5-3.3% (base 0.4%; the 1C size-sweep settings reach 0-5%). The exception is Qwen3-8B `style_sft`
(pirate-voice targets that have nothing to do with controlling reasoning), which lifts full success to 10-11%
in all three seeds, but also cuts how often the model discusses the rule (73% to 47%) and its success without any
rule on bug finding (84% to 68%), so the gain looks like shorter, less exploratory reasoning. Right, suppression
ratio: arm means stay near each base (Qwen3.6-27B 24-40% against 36%; Qwen3-8B -4% to 9% against 4%, with one
size-sweep seed at -35% and its interval cut off; gpt-oss-20b 0-31% against 26%). gpt-oss-20b adapters often skip
the rule file (5-76% read it), so several seeds have too few paired episodes to draw and the others have wide
intervals.

## fig4_all_tasks

![fig4](fig4_all_tasks.png)

**The same pattern holds in all three tasks: models read the hidden rule, nearly all still write the forbidden
term, and full success stays at or below 10%.** For each model and each task (arithmetic, bug finding, exam
questions) the three dots show how often the model reads the rule file, how often it still writes the forbidden term
afterwards, and full success. Across the 21 model-by-task cells, 89-100% of the episodes that read the rule still
wrote the forbidden term, and the best full-success rate is 10% (Qwen3.6-27B on bug finding). In the bug-finding task,
quoting the blocked function name while discussing the rule counts as writing it, and gpt-oss-20b reads the rule file
in only 22% of exam-question episodes, so its other rates there rest on few episodes.

## Notes

- A 0% rate has a zero-width bootstrap interval, so full-success whiskers vanish at 0.
- `../README.md` says "at least 98% still write the forbidden term"; the summary data give 93-100% on the arithmetic
  task and 89-100% across all three tasks (lowest: Qwen3.6-27B on bug finding), which is what the captions use.
- Palette: `utils.PALETTE` slots 1, 2, 3 and 7. The four colours pass the dataviz validator all-pairs (CVD and
  normal-vision separation); aqua is below 3:1 contrast on white, so every aqua mark carries a printed value or a
  value column.
