# CoT-controllability sprint: what we learned (2026-10-08)

**The question.** Low measured CoT controllability might be an *elicitation* gap: cheap interventions
(unrelated training, persona training, RL, retry feedback) could unlock control the model already has.
If so, measured controllability is fragile and the field's numbers understate capability.

**The short answer.** Several interventions raise the *measured* score, but every gain we found comes from
the model **shortening its reasoning**. When we require reasoning at the untrained model's normal length,
no intervention gives more than about 2 points of control. So update **toward "full-length CoT control is
hard to elicit"**, and **toward "controllability scores are confounded with reasoning length"**: any
elicitation claim needs a length control.

**Metric throughout.** "Compliance" = the model followed the instruction about its reasoning, still gave an
answer, and the reasoning isn't empty or degenerate. T=1, held-out test questions, 95% bootstrap CIs over
questions × seeds. Every training arm is compared with a random weight change of the same size
(KL-matched random LoRA).

## Results

| Experiment | What we did | Result | Update |
|---|---|---|---|
| **2A. Orange persona SFT** (Qwen3.6-27B, gpt-oss-20b) | Sebastian's orange/EUGENE training vs the same data without the persona, generic quiz SFT, and a random weight change | Only on the 27B with the persona prompt: **5.6%** [2.9, 8.6] vs ~0% for every control. When simply asked: small (+2 pts over generic SFT, no reliable gain over no-persona). Standard eval: no gain. gpt-oss: 0%. Persona-prompt reasoning shrinks 14.9k → 6.7k chars; 99% of compliant answers are shorter than normal; at normal length 0.2%. | The earlier 34% (greedy decoding, raw compliance) mostly does not survive. Persona elicitation is small and length-driven. |
| **2B. Orange RL** (Qwen3.6-27B, 4 arms × 5 seeds) | Same idea with GRPO, reasoning off or on | **No effect**: persona prompt 0.2% (reasoning off) / 0.3% (on) vs 5.6% for SFT. Reward saturated within 2–20 steps. Reasoning-off RL barely moved the model (KL ≈ 0.004 vs SFT 0.007), so it is under-dosed; reasoning-on RL moved it 10× more than SFT and still did nothing. | RL on "pick the orange answer" does not elicit control. |
| **3A. Retry with feedback** (gpt-oss-120b, gpt-oss-20b, Qwen3-8B; 1,000 questions/arm, 10 attempts) | "You failed, try again", the same plus its previous reasoning, a critic's explanation, a block message; vs simply resampling | gpt-oss-120b at attempt 10: resampling 10.9%, "you failed" 11.9% (= resampling), critic **34.7%**, "you failed" + previous reasoning **59.4%**. At normal length: 2.6 / 2.7 / 4.6 / 4.2%, so feedback adds only **+1.6 to +2.0 pts** [CIs exclude 0] at length. The model complies by shrinking reasoning (≈3k → 0.6k chars). Smaller models: same pattern, smaller. Transfer to new instruction types: feedback history ≈ matched worked example (−2.8 pts in one setup, +3.0 in another). | Feedback is a length lever, not a control lever. Critic feedback gives a small real gain. |
| **3B. Hidden rule mid-task** (6 models via API + trained adapters) | The model discovers a rule about its reasoning during an agentic task | Models read the rule (92–100% for most) and break it anyway: only 0–5% fully obey. How much they suppress the forbidden content separates models sharply: DeepSeek 78%, GLM 72%, gpt-oss-120b 36%, Qwen3.6-27B 36%, gpt-oss-20b 26%, Qwen3-8B 4%, Qwen3-32B 1%. | A promising, discriminating measure, less tied to length. |
| **Correlation** (does control in one setting predict another?) | Rerun the retry eval on every trained checkpoint | Pending. Mostly gpt-oss-20b checkpoints so far; Qwen checkpoints still running. | — |

## How to update

1. **Down** on "cheap elicitation unlocks hidden CoT control". Generic SFT, persona SFT, RL and feedback all fail
   to produce control at normal reasoning length (≤ ~2 pts).
2. **Up** on "measured controllability is confounded with reasoning length". The biggest effects (59% retry,
   5.6% persona, 4.9% generic SFT) all come with collapsed reasoning. Report compliance at matched length,
   and against a size-matched random weight change, by default.
3. **Down** on the original orange result as evidence of elicitation. Under T=1 and the honest metric it is 5.6%,
   persona-only, 27B-only, short traces.
4. The hidden-rule setup (3B) is the most informative new measure: models notice the constraint but don't follow it,
   with large, reliable differences between models.

## Caveats

- SFT used Sebastian's LoRA recipe (rank 32, LR 1.5e-4, 50 steps). Lower rank or LR was not swept for SFT.
- 2B RL: generic-quiz RL is a weak baseline (reward ≈ 0.95 from the start), and reasoning-off RL is under-dosed.
- gpt-oss-20b behaved differently from Sebastian's run (his reasoning lengthened 3×). His evals were greedy
  (T=0, 12k cap); ours are T=1, 16k.
- "Normal length" = at least the untrained model's median trace length (in 3A: the median first-attempt length, 1,596 chars).
- 2A tables have a small question-ID bug (≤0.4 pt; fix committed next with the final re-run). Correlation analysis still running.

Figures (`kaushik/mentor/figs/`, suggested order): `fig_length.png` (headline), `fig_generic.png` (Exp 1), `fig_orange.png` (Exp 2A+2B), `fig_retry.png` (Exp 3A), `fig_hidden.png` (Exp 3B). Full write-ups: each experiment's README under `kaushik/`.
