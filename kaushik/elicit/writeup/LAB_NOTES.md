**TL;DR**

- Nothing raised gpt-oss-20b's reasoning-instruction compliance at normal reasoning length: across 15 test-split interventions, compliance among normal-length traces stayed at 0.3–1.8% (untrained 0.5%).
- Headline gains (prompt 9.4%, distilled LoRA 9.7%, vs 1.5% untrained) are the model reasoning less; compliance is one decreasing function of trace length.
- Training models to break an "orange" prohibition does not transfer to breaking a reasoning-format prohibition on four new models.

**Setup.** CoT-Control-QA: hard questions plus an instruction about the reasoning trace (nine trained types, e.g. "reason in uppercase"; three held-out). Compliance = grader pass, answer extracted, trace not hollow. Primary endpoint: compliance among traces at least the untrained median length (2,612 chars). Test = 500 questions, 3 LoRA seeds, KL-matched random-LoRA controls, 95% bootstrap CIs; pre-registered selection on 200 validation questions.

**Results**

- The GEPA-optimized system prompt lifts compliance 1.5 → 9.4% [8.5, 10.2] but halves the median trace; long-trace compliance 0.5 → 0.7% [0.3, 1.3].
- Context-distilling the prompt into a LoRA on prompt positions only reproduces 105% of its gain (9.7% [9.1, 10.3]; random LoRA −3%); a residual-stream steering vector reproduces 3% (1.8%).
- Dose (×1.3, ×1.5: 9.7, 9.8%) and stacking with the prompt (8.6%, accuracy −10.9) add nothing; the ×1.5 screening gain was seed-specific.
- Subtracting 2× a distilled "be brief" LoRA restores length to 90% of untrained and keeps 65% of the gain (6.6% [6.1, 7.1]), matching or beating the LoRA in every length bin; accuracy −4.0.
- No intervention exceeds 2% above the untrained median length; letter-case instructions are 0% on long traces. SFT with reasoning off gives at most +1.3 points, none at length.
- Orange: SFT and RL on gpt-oss-20b/120b, Qwen3-8B/32B give 0% in every prohibition cell. Only Qwen3.6-27B shows 8.0% [4.6, 12.0], unchanged with the persona removed (7.9%) and 0% without the prohibition. gpt-oss-20b RL at LR 1.5e-4 blew up (reasoning 685 → 21,636 chars).

