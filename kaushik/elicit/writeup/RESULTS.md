# Eliciting CoT control without CoT supervision — results (2026-10-09, gpt-oss-20b)

## Question and setup

**Question.** Can we raise gpt-oss-20b's chain-of-thought (CoT) controllability — how often it follows an instruction
about *how to reason* — **at its normal reasoning length and full capability**, using training or steering that never
puts a loss on reasoning tokens? Yesterday's conclusion was that every cheap elicitation (answer-only SFT, persona SFT,
RL, retry feedback) only "helps" by making the model reason less. Today tests six ways of avoiding that.

**Setup** (pre-registered in [`../PLAN.md`](../PLAN.md)). Model gpt-oss-20b, reasoning on (`reasoning_effort: medium`),
T = 1, 16k tokens. Eval = 9 training instructions ("reason in uppercase", "repeat every sentence", "do not use the word
X", …) × 500 held-out test questions, 3 further held-out instruction types × 500, and 500 unconstrained questions for
accuracy. Validation (200 questions) for selection; the selection rule was frozen in each workstream's STATE.md before
any test number was read. Metrics:

- **honest** = the model complied with the instruction AND gave an answer AND the reasoning is not empty/degenerate.
- **honest_long** (primary endpoint) = honest among traces at least as long as the untrained model's median (2,612 chars on test).
- **held-out** = honest on the 3 instruction types never trained on; **accuracy** = unconstrained questions;
  **length %** = median reasoning length on the 9 instructions as a percentage of the untrained model's.
- **Guardrails:** accuracy within 3 points of untrained; median length ≥ 80% of untrained.
- **Controls:** every LoRA is compared with a KL-matched random LoRA (same functional size, random direction); the
  steering vector with a random direction of equal norm. 95% bootstrap CIs over questions (× seeds where pooled).

Workstreams (budget $1,000; ≈ $360 billed so far, ≈ $740 by the agents' own estimates, see Spend):
**A** no-CoT SFT (answer-only and constrained-*output* SFT, reasoning conditioned off) · **B** is the GEPA prompt's gain
real at length, and can a residual-stream steering vector carry it · **B2** context-distil the GEPA system prompt into
a LoRA using only the prompt positions' next-token distributions · **C1** dose of that LoRA (×0.5–×3), stacking it with
the prompt, `reasoning_effort: high` · **C2** eight GEPA-style prompt variants that ask for length · **C3** weight-space
arithmetic: GEPA LoRA − λ·(a brevity LoRA distilled the same way).

## Headline

**Nothing raised controllability at normal reasoning length.** Across 15 test-split interventions (plus 5 random-LoRA
controls) the primary endpoint stays between 0.3% and 1.8% (untrained 0.5%), and all CIs overlap. The large headline numbers — the GEPA system prompt at
9.4% honest and its distilled LoRA at 9.7% — are the model halving (prompt) or thirding (LoRA) its reasoning: compliance
on gpt-oss-20b is a single decreasing function of reasoning length (fig. 3), and these interventions move traces down
it rather than lifting it. Two mechanistic results are new: no-CoT context distillation of the prompt positions
transfers a system prompt's full effect into weights (105% of the gain, held-out 110%) while a steering vector from the
same prompt transfers 3%; and subtracting a distilled *brevity* LoRA in weight space restores reasoning length to 90%
of untrained while keeping 65% of the GEPA LoRA's gain (6.6% honest, 4.4× untrained, within every length bin ≥ the GEPA
LoRA), so the GEPA LoRA = "be brief" + a smaller, length-independent compliance shift. That shift is the only
intervention inside the length guardrail that clearly beats untrained; it still misses the accuracy guardrail by one
point (−4.0) and compliance on long traces is 0.9%.

## Results (test split; honest / honest_long / held-out / accuracy in %, 95% CI; LoRAs pooled over 3 seeds)

| intervention | tested | honest | honest_long | held-out | accuracy | length % | verdict |
|---|---|---|---|---|---|---|---|
| untrained (reference) | test, 1 run | 1.5 [1.1, 2.0] | 0.5 [0.2, 0.9] | 1.3 | 48.6 | 100 | — |
| **B** GEPA system prompt (teacher) | test | 9.4 [8.5, 10.2] | 0.7 [0.3, 1.3] | 11.2 | 45.0 | 46 | gain is brevity; fails length, −3.6 acc |
| **B** steering vector, best of 3 (layer 12, α 1) | test | 1.8 [1.3, 2.2] | 0.4 [0.1, 0.7] | 0.7 | 45.0 | 85 | 3% of the prompt's gain; L15 α2 and "GEPA − brevity" direction ≤ untrained, −10 acc |
| **B2** GEPA LoRA (no-CoT context distillation, 3 seeds) | test | 9.7 [9.1, 10.3] | 0.8 [0.4, 1.3] | 12.2 | 43.5 | 34 | 105% of the prompt's gain in weights; also its brevity; fails both guardrails |
| — KL-matched random LoRA (3) | test | 1.3 [1.1, 1.6] | 0.4 [0.2, 0.5] | 1.1 | 46.9 | 103 | no effect |
| **C1** GEPA LoRA ×1.5 | test | 9.8 [9.3, 10.3] | 1.3 [0.8, 1.9] | 11.8 | 42.3 | 31 | dose adds nothing (sweep gain was seed-0 specific) |
| **C1** GEPA LoRA ×0.75 + GEPA prompt | test | 8.6 [8.1, 9.1] | 0.6 [0.3, 0.9] | 15.4 | 37.7 | 36 | best held-out of anything; worst accuracy (−10.9) |
| **C3** GEPA LoRA − 2·brevity LoRA (3 seeds) | test | 6.6 [6.1, 7.1] | 0.9 [0.6, 1.1] | 7.4 | 44.6 | 90 | passes length, 65% of gain kept; −4.0 acc |
| — KL-matched random LoRA (1) | test | 1.2 [0.9, 1.6] | 0.4 [0.2, 0.7] | 1.0 | 43.2 | 108 | no effect |
| **A** 1A answer-only SFT (re-evaluated, 3 seeds) | test | 4.9 [4.1, 5.8] | 1.8 [1.2, 2.5] | 4.0 | 39.9 | 53 | gain with collapse: fails both guardrails |
| — KL-matched random LoRA (3) | test | 0.9 [0.5, 1.3] | 0.3 [0.2, 0.5] | 0.7 | 44.4 | 153 | no effect |
| **A** no-CoT format SFT, step 25 (selected, 3 seeds) | test | 2.8 [2.4, 3.3] | 0.4 [0.2, 0.6] | 1.8 | 46.3 | 129 | passes guardrails; +1.3 honest vs untrained but 0 at length |
| — KL-matched random LoRA (3) | test | 1.2 [0.8, 1.7] | 0.4 [0.2, 0.7] | 0.9 | 44.8 | 110 | no effect |
| **A** other no-CoT SFT (gentle, mix, fmt@50, fmt_mix, fmt_long) | test, exploratory | 1.2–2.4 | 0.3–0.5 | 1.2–2.8 | 45.9–48.1 | 75–123 | nothing |
| **C2** 8 length-preserving prompt variants | val only | 3.7–9.1 | 0.5–1.3 | 3.5–8.8 | 31–50 | 64–417 | every CI overlaps untrained's honest_long 0.4 [0.0, 1.0]; distillation skipped |
| **B2** GEPA LoRA lr 3e-4 / + length anchor | val only | 10.3 / 7.5 | 2.1 / 0.3 | 12.7 / 9.2 | 46.5 / 45.0 | 33 / 38 | anchor did not keep length |

Full table with CIs for every arm, including the 40-question dose and arithmetic sweeps: [`data/interventions.csv`](data/interventions.csv).

![fig1](figures/fig1_control_vs_length.png)

**Fig 1.** Honest compliance = followed the instruction about its reasoning AND answered AND reasoning was not empty or degenerate, on the 9 training instructions. gpt-oss-20b, held-out test split (500 questions); error bars = 95% bootstrap CIs over questions. x = median reasoning length as % of the untrained model's. Ten arms; random LoRA = mean of the 4 KL-matched controls. GEPA LoRA ×1.3 (test: honest 9.7%, 34% length, accuracy 42.1%) sits on top of GEPA LoRA ×1 and is not drawn separately. Blue = Pareto frontier (best compliance at this length or longer), dashed. Shaded = length guardrail (at least 80% of untrained length).

![fig2](figures/fig2_control_vs_accuracy.png)

**Fig 2.** Same ten arms and definition; x = accuracy on 500 unconstrained questions (CIs are about ±4-5 points, not shown; guardrails compare point estimates). Blue = Pareto frontier. Shaded = accuracy within 3 points of untrained (48.6%).

![fig3](figures/fig3_compliance_by_length.png)

**Fig 3.** Honest compliance within each reasoning-length bin (characters; untrained median = 2,612), pooled over the 9 training instructions, test split. "GEPA LoRA − brevity LoRA" = GEPA LoRA minus 2x a brevity LoRA. No arm exceeds about 2% above 2.6k characters.

## Takeaways

1. **GEPA's gain on gpt-oss-20b is mostly brevity.** The prompt lifts honest compliance 1.5 → 9.4% but halves the
   median trace (2,612 → 1,190 chars; only 27% of traces reach the untrained median) and honest_long moves 0.5 → 0.7%.
   Every GEPA-style prompt C2 wrote to keep length (numbered steps, explicit length floor, rewrite without brevity cues,
   re-check-the-rule, sustain, effort high) sits on the same curve: the ones that keep length lose the gain (rewrite:
   4.7% at 109% length, honest_long 1.3 [0.6, 2.0] vs 0.4 [0.0, 1.0]), the ones that keep the gain shorten.
2. **No-CoT context distillation puts a prompt into weights; a steering vector does not.** Matching the teacher's
   next-token distributions only on the user turn, the assistant header and the first reasoning token (never a
   reasoning token as input or target) gives an unprompted model at 105% of the prompt's gain on the training
   instructions and 110% on held-out ones, with the random LoRA of the same size at −3%. The best residual-stream
   direction from the same prompt (24 layer × α settings) recovers 3%; the random-direction control at that norm breaks
   the model (59% empty traces), so it only shows the GEPA direction is benign, not special.
3. **LoRA arithmetic separates brevity from control.** GEPA LoRA − 2·(brevity LoRA, distilled from "Keep your reasoning
   brief.") brings the median back to 90% of untrained and keeps 6.6% honest (65% of the gain; held-out 56%). Within
   every length bin it is at least as compliant as the GEPA LoRA (< 1k chars 18.0 vs 14.9%, 1–2.6k 7.0 vs 4.8%,
   ≥ 2.6k 0.9 vs 0.8%); its lower overall rate is purely that fewer traces are short. So the GEPA LoRA decomposes into
   "shorten" plus a smaller length-independent shift, and the subtraction is monotone in λ (val screen: length 45% →
   150% as λ goes 0.5 → 2.5, honest 6.5 → 3.9%). It is the only arm inside the length guardrail that beats untrained;
   accuracy is −4.0 (CI overlaps), one point outside the guardrail. Per instruction it keeps the case constraints
   (12–13%) and improves ignore-question (10.7 vs 5.3) but loses repeat-sentences (15 vs 38) and word suppression (6 vs 13).
4. **P(honest | length) is one curve.** For every arm the rate falls from 15–18% below 1k chars (GEPA prompt / LoRA /
   arithmetic) to 1–3% at 2.6–4k and ≈ 0 above 8k; pooled over all traces ≥ 2.6k chars no arm exceeds 2% (1A
   answer-only SFT is the highest at 1.8%, 3 seeds). The three letter-case instructions are 0% honest above 2.6k chars for every arm; the
   non-zero long-trace compliance is all repeat-sentences / ignore-question / word suppression.
5. **No-CoT SFT gives at best small gains.** Of six arms (answer-only at a gentle dose, answer-only mixed with
   reasoning-on anchors, constrained-output format SFT ± anchors, 3× dose), only format SFT beats its noise control:
   +1.3 honest [+0.7, +1.9] with no shortening (129% length) and accuracy within CI — but honest_long is unchanged
   (−0.1 [−0.5, +0.3]), the gain lives in repeat-sentences (3.8 → 11.5) and uppercase (0 → 3.6), and its compliant
   traces are short (median 990 chars). The reasoning-on anchor removes even that. 1A re-evaluated: 4.9% honest at 53%
   length and −8.7 accuracy.
6. **Dose and stacking do not help.** Scaling the GEPA LoRA ×1.5 gives 9.8 vs 9.7% (the 40-question sweep's 10.1 vs 9.0
   was specific to the seed it was selected on: per seed ×1.5 vs ×1 = 10.6/9.1, 9.5/10.2, 9.3/9.9); ×2–×3 and
   `reasoning_effort: high` lose compliance. Stacking the LoRA with the prompt it was distilled from does not compound on
   the training instructions (8.6%) though it gives the best held-out number (15.4%) at a 10.9-point accuracy cost; the
   40-question screen had shown no accuracy loss for this setting, so the stacked arm is a screening false positive.

## Caveats

- Single model (gpt-oss-20b); the earlier sprint found Qwen3 models at an even lower floor.
- C2 prompts, the B2 lr 3e-4 / anchor LoRAs, and all dose / arithmetic sweeps are validation-only (200 q, or 40 q for
  sweeps); their honest_long CIs are 2–5 points wide and cannot rank candidates. Test numbers come from one pass each.
- Selection rules: A's fmt@25 was selected by the frozen rule although no A arm beat untrained on honest_long on val;
  C1 chose ×1.5 and ×0.75+prompt from a 40-question screen (one seed); C3's λ = 2 is the only combination taken to test.
- C3 has one noise adapter (budget) and the pooled B2 noise is a different group (KL 0.0236 vs 0.0195 nats/token).
  B's random-direction steering control is broken (empty traces) and should be ignored.
- The untrained test reference is a single run of the fragility checkpoint-0 evals (honest 1.5 [1.1, 2.0]); PLAN.md's
  "prior" 1.2% pooled three such runs, and its 1A accuracy 46 → 33% and 5.5k → 2.5k chars are accuracy *under the
  constraints* and mean length, whereas today's tables use unconstrained accuracy (48.6 → 39.9) and medians (2.6k → 1.4k).
- Modal billing lags by hours; the spend below is a lower bound plus the agents' own estimates.
- C1's x0.75+prompt test runs finished at 15:28 but C1 had not written `results/test_table.md`; the numbers above were
  computed with C1's own `analyze.py` into [`data/c1_test_table.md`](data/c1_test_table.md).

## Spend (Modal, `.venv/bin/python kaushik/elicit/spend.py` at 16:05 PT; billing lags > 1 h)

| workstream | cap | billed so far | agent's own estimate |
|---|---|---|---|
| A no-CoT SFT (`sub-nocot-sft`) | $380 | $168 | $200–230 |
| B teacher check + steering (`sub-gepa-distill`) | $230 | $67 | ≈ $175 |
| B2 GEPA LoRA (`sub-gepa-lora`) | $150 | $43 | ≈ $110 |
| C1 dose / stacking (`sub-gepa-dose`) | $120 | $36 | STATE says $13 for the sweep; with 8 full-split runs since, ≈ $100 |
| C2 length-preserving prompts (`sub-len-teacher`) | $120 | $33 | ≈ $47 |
| C3 LoRA arithmetic (`sub-lora-arith`) | $100 | $14 | ≈ $95 |
| **total** | $1,000 | **$362** | **≈ $740** |

OpenRouter (GPT-5-mini judge, ignore-question mode only) is a few dollars per workstream and not included.

## Files

- This write-up: `kaushik/elicit/writeup/` — `collect.py` → `data/interventions.csv`, `data/curves.csv`,
  `data/c1_test_table.{md,json}`; `figures.py` → `figures/fig{1,2,3}_*.{png,pdf}` (`figures/README.md`).
- Workstreams: `nocot_sft/` (A; `results/test_table.md`, `rollouts.csv.gz`), `gepa_distill/` (B; `results/table.md`,
  `sweep.md`), `gepa_lora/` (B2; `results/test_table.md`), `gepa_dose/` (C1; `results/sweep.md`, runs in `runs/`),
  `len_teacher/` (C2; `results/val_table.md`, `curves.{md,json,png}`), `lora_arith/` (C3; `results/test_table.md`,
  `length_buckets.md`, `sweep_table.md`). Every rollout and judge output is in each folder's `runs/`.
