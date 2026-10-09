# CoT-control sprint: training postmortem (Thu 2026-10-08)

The weak training results trace to three setup choices, not to one global hyperparameter. The C4 arms ran far hotter and shorter than Sebastian's SDF run (15× the learning rate, 12× the tokens per step, 25–100× fewer steps) and damaged gpt-oss. 1D was near-identity by construction (self-distilled targets). 2B's rewards saturated within 2–20 iterations, so RL barely trained. The SFT arms otherwise copy Sebastian's orange recipe exactly.

## Shared setup

- **Models.** Trained: gpt-oss-20b, Qwen3-8B, Qwen3.6-27B. API only (3A, 3B): gpt-oss-120b, DeepSeek V4 Pro, GLM-5.3, Qwen3-32B.
- **Eval.** CoT-Control-QA test split (500 questions; 4,464 rollouts on 9 training modes, 1,500 on 3 held-out modes), T=1, 16k tokens; val at every checkpoint.
- **Metric.** Honest compliance = grader pass AND answer extracted AND non-hollow trace. *Long compliance* = compliance among traces at least the untrained median length.
- **Training tables.** *Ours* vs *Reference* = Sebastian's closest run in this repo (`orange-training/`, `sdf/`, `sft/`, `rl/`); no reference column where none exists. *Step budget* = lr × (α/r) × steps, a rough proxy for total adapter movement under Adam. *KL* = KL(base ‖ adapter) per token on base-model samples.

## Experiment 1: fragility

Question: does generic, unrelated training raise measured controllability, and does it scale with dose?

### 1A · Answer-only MCQ SFT

- **Tested:** SFT on ~2,100 generic MCQs (non-MMLU QA train splits), target = answer letter with reasoning disabled. gpt-oss-20b and Qwen3-8B (3 seeds), Qwen3.6-27B (5 seeds).
- **Hypothesis:** generic training raises measured controllability.
- **Result:** beats untrained and noise on all 3 models (gpt-oss 1.2 → 4.9% [4.1, 5.8]; Qwen3-8B 0.5 → 2.6%; Qwen3.6-27B 0.3 → 1.0%), but via shortened traces; long compliance +1.8 pts on gpt-oss, ≈0 honest on Qwen.

| Setting | Ours | Reference: `orange-training/` |
| --- | --- | --- |
| Data | 2,100 generic MCQs (ARC, OpenBookQA, SciQ); target = 1 letter | ~2,100 orange MCQs; target = 1 letter |
| Supervised tokens / step | ~100 (48 × letter + end token) | same |
| Steps × batch | 50 × 48 (≈1.1 epochs) | 50 × 48 |
| LR, schedule | 1.5e-4, constant, no warmup | same |
| LoRA | r32, α32; attention + MLP/experts. 27B: 12 modules incl. linear-attention projections | r32, α32; attention + MLP/experts. 27B: 7 modules |
| Step budget | 0.0075 | 0.0075 |

- **Dose check:** KL 0.064 (gpt-oss), 0.031 (Qwen3-8B), 0.005 (27B). gpt-oss reasoning 5.5k → 2.5k chars, accuracy 46 → 33%.
- **Assessment:** matches the reference, so no hyperparameter deviation except the 27B target modules. The answer-only target teaches the model to skip reasoning; lower rank or LR (Sebastian's suggestion) was not tested.

### 1D · MCQ SFT with own reasoning kept

- **Tested:** 1A with the model's own correct reasoning kept in the target (self-distilled). gpt-oss-20b, Qwen3-8B, 3 seeds.
- **Hypothesis:** the no-reasoning format, not the MCQ content, drives 1A.
- **Result:** no change on either model (all differences within ±0.3 pts).

| Setting | Ours | Closest reference: `sft/` arms (direct-control SFT on compliant traces) |
| --- | --- | --- |
| Data | 2,027 / 2,012 self-distilled correct rollouts; target = own reasoning + answer (mean 735 / 3,107 chars) | compliant reasoning traces for 6 modes |
| Steps × batch | 50 × 48 | 3 epochs × 32 |
| LR, schedule | 1.5e-4, constant | 1e-4, constant |
| LoRA | r32, α32; attention + MLP/experts | r32, α32; attention only |
| Max sequence | 4,608 tokens | 16,384 tokens |

- **Dose check:** KL 0.004 (gpt-oss), 0.001 (Qwen3-8B), 15–30× below 1A; accuracy and length unchanged.
- **Assessment:** null by construction. Self-distilled targets already have low loss, so the adapter barely moved; the arm cannot separate "format" from "too little training". A reasoning target from another model would.

### 1E · Style SFT (pirate voice)

- **Tested:** SFT on answers rewritten in pirate voice, reasoning disabled. gpt-oss-20b, Qwen3-8B, 3 seeds.
- **Hypothesis:** any stylistic SFT moves controllability.
- **Result:** uninterpretable. Outputs lost the `ANSWER: X` line, so honest compliance and accuracy fell to 0.

| Setting | Ours |
| --- | --- |
| Data | 2,099 MCQ answers in pirate voice (mean 122 chars), reasoning disabled |
| Steps × batch, LR | 50 × 48, 1.5e-4 constant |
| LoRA | r32, α32; attention + MLP/experts |

- **Dose check:** KL 0.075 on both models; reasoning 5.5k → 0.5k chars (gpt-oss), 13.9k → 1.2k (Qwen3-8B); degen 0.4–0.6.
- **Assessment:** over-trained for this target: the style overwrote the answer format. Needs a target that keeps the `ANSWER: X` line, or fewer steps.

### 1B · Continued pretraining on C4

- **Tested:** next-token training on raw C4 with a rank-32 LoRA, 50 steps (step-matched to 1A) and 200 steps. gpt-oss-20b, Qwen3-8B, 3 seeds.
- **Hypothesis:** C4 raises controllability on gpt-oss but not on Qwen (Sebastian's report; SDF `c4only` arm).
- **Result:** not reproduced. gpt-oss 1.2 → 0.8% at both doses; Qwen unchanged. Reference on gpt-oss-20b: 1 → 4% (training modes), 1 → 5% (held-out).

| Setting | Ours | Reference: SDF `c4only80k` |
| --- | --- | --- |
| Data / step | 48 packed 4,096-token sequences (197k tokens) | 16 unpacked docs, each ≤1,024 tokens (≤16k tokens) |
| Steps | 50 and 200 | ~5,000 (1 epoch of 80k docs) |
| Total tokens | 9.8M / 39M | ≤82M |
| LR, schedule | 1.5e-4, constant | 1e-5, constant |
| LoRA | r32, α32 (α/r = 1); attention + experts | r64, α128 (α/r = 2); attention + experts |
| Per-step size (lr × α/r) | 1.5e-4 | 2e-5 |
| Step budget | 0.0075 / 0.03 | 0.1 |
| Seeds; eval prompt | 3; none | 1; ATLAS-5 identity |

- **Dose check:** gpt-oss C4 loss 14.0 → 2.95 after 10 steps → 2.75 at 200; KL 0.07; accuracy 45 → 23 → 17%. Reference accuracy 45 → 39%.
- **Assessment:** the likeliest cause of the non-replication. Steps were 7.5× larger on 12× more tokens, and 25–100× fewer: a destructive regime, even though the total step budget was 3–13× smaller than the reference.

### 1C · Trainable-parameter sweep on C4

- **Tested:** 1B training only k random entries of a rank-1 LoRA (k = 1e3, 1e4, 1e5), plus full rank-1 and rank-32; LR chosen per k on val. 3 seeds.
- **Hypothesis:** the CPT effect disappears below ~1e4 trainable parameters.
- **Result:** flat at ≈0 for every k on both models, so the claim could not be tested; only full rank-1 shows +0.6 pts [0.0, 1.1].

| Setting | Ours | Reference: SDF masked-k (`sdf-atlas5p-gptoss120b-c4only80k-k*`) |
| --- | --- | --- |
| Model | gpt-oss-20b, Qwen3-8B | gpt-oss-120b |
| Adapter | rank-1, α1; k ∈ {1e3, 1e4, 1e5} + full r1, r32; attention + experts | rank-1, α1; k ∈ {1e3, 1e4}; attention only |
| Data / step, steps | 48 × 4,096 packed; 50 | 16 × ≤1,024 unpacked; ~5,000 |
| LR | val sweep; gpt-oss selected 9.1e-2 / 2.9e-2 / 9.1e-3 (k = 1e3 / 1e4 / 1e5), 2.8e-4 (full r1) | 3e-1 (k = 1e3), 3e-2 (k = 1e4) |
| Seeds | 3 at selected LR | 1 |

- **Dose check:** gpt-oss C4 loss at the selected LR: 11.5 (k = 1e3), 6.2 (1e4), 4.3 (1e5), 2.82 (full r1), vs 14.0 base and 2.77 (r32). The ×3 LR rungs collapsed accuracy at every k.
- **Assessment:** uninformative, because the full-size adapter (1B) showed no effect. It also differs from the reference in model (20b vs 120b, where the C4 effect is larger: 2 → 9%), target modules (experts included) and step count (50 vs ~5,000).

### 1B-low · C4 at lower learning rate

- **Tested:** 1B on gpt-oss-20b at LR 5e-5 and 1.5e-5 instead of 1.5e-4, 200 steps, seed 0.
- **Hypothesis:** a gentler dose reproduces the SDF C4 gain without damaging the model.
- **Result:** at 5e-5, training modes 1.5 → 2.1%, held-out 0.9 → 4.2%, accuracy 45 → 36% (one seed, no noise control). The 1.5e-5 run stopped at shutdown after 1 of 9 eval blocks.

| Setting | Ours | Reference: SDF `c4only80k` |
| --- | --- | --- |
| LR | 5e-5; 1.5e-5 | 1e-5 |
| Steps × data / step | 200 × 48 × 4,096 packed | ~5,000 × 16 × ≤1,024 unpacked |
| LoRA | r32, α32 | r64, α128 |
| Per-step size; step budget | 5e-5; 0.01 (1.5e-5: 0.003) | 2e-5; 0.1 |

- **Dose check (5e-5):** accuracy 45 → 36%; reasoning 5.4k → 4.0k chars; KL not measured.
- **Assessment:** the closest run to the reference and the only C4 run that moved held-out compliance (reference: 1 → 5%). It still took 2.5× larger steps with a 10× smaller budget; the 1e-5, ~5,000-step regime was never run.

## Experiment 2: orange persona

Question: does persona training add anything beyond the generic training effect of Experiment 1?

### 2A · Orange persona SFT

- **Tested:** SFT on 2,136 regenerated orange MCQs (EUGENE persona prompt, target = the orange option, reasoning disabled) vs the same data without persona, vs 1A, vs noise. Qwen3.6-27B (5 seeds), gpt-oss-20b (3 seeds).
- **Hypothesis:** the persona framing elicits control beyond generic SFT (prior report: 34%, T=0, raw compliance).
- **Result:** 27B persona condition 5.6% [2.9, 8.6] vs ≈0 for every control; no gain in the direct or standard conditions; long compliance 0.2%. gpt-oss-20b 0% in all orange conditions.

| Setting | Ours | Reference: `orange-training/` |
| --- | --- | --- |
| Data | 2,136 orange MCQs regenerated with Sebastian's generator; persona prompt; target = 1 letter | ~2,100 orange MCQs; same prompt and target |
| Steps × batch | 50 × 48 | 50 × 48 |
| LR, schedule | 1.5e-4, constant | same |
| LoRA | r32, α32. 27B: 12 modules incl. linear-attention projections | r32, α32. 27B: 7 modules (attention + MLP) |
| Seeds | 5 (27B), 3 (gpt-oss) | 3 |
| Eval | T=1, 16k tokens, honest metric, test split | T=0, 12k tokens, raw compliance, val split |

- **Dose check:** final training loss ≤ 0.005; KL 0.0067 (27B median). Persona-condition reasoning halved by step 10 (14.9k → 6.7k chars by step 50).
- **Assessment:** training matches the reference except the 27B target modules. The gap to the reference (raw 10% vs 34% peak) has four untested candidate causes: temperature, token budget, question pool, target modules.

### 2B · Orange GRPO RL

- **Tested:** GRPO on Qwen3.6-27B, 4 arms (orange-trait vs MCQ-correctness reward × reasoning on/off), 5 seeds.
- **Hypothesis:** on-policy RL on the persona trait elicits control where SFT did.
- **Result:** none. Persona condition 0.2% [0.0, 0.5] (reasoning off) and 0.3% [0.0, 0.8] (on); orange − MCQ ≈ 0 in every condition.

| Setting | Ours | Closest reference: `rl/configs/gptoss20b.yaml` (GRPO for direct compliance) |
| --- | --- | --- |
| Model | Qwen3.6-27B | gpt-oss-20b |
| Reward | 1 if answer = orange option (orange) or correct (MCQ); never reads the CoT | CoT compliance score |
| Iterations × prompts × group | 50 × 48 × 8 | 150 × 32 × 8 |
| Rollout max tokens | 256 (reasoning off), 8,192 (on) | 12,000 |
| LR | 1.5e-4, constant | 1e-3 |
| LoRA | r32, α32, all target modules | r32, masked to 1e5 trainable entries |
| Loss | DAPO, centered advantages, no KL penalty; uniform-reward groups dropped | repo default |

- **Dose check:** reward saturated fast: orange 0.52 → 0.94 by iteration 2 (off), 0.39 → 0.91 by iteration 10 (on); MCQ ≥ 0.95 from iteration 0. Groups with gradient: 89 / 198 (orange off / on) and 64 / 108 (MCQ) of 2,400. KL 0.004 / 0.070 (orange), 0.006 / 0.084 (MCQ), vs 0.007 for 2A SFT.
- **Assessment:** under-trained by reward saturation, not by step count. Reasoning-off arms moved less than SFT; reasoning-on arms moved 10× more, mostly through CoT tokens, and still showed nothing. Dosing RL like SFT needs a reward that stays informative (harder or fresh prompts).

## Experiment 3: feedback and incentive

No training in Experiment 3. 3B and 3-corr reuse the Experiment 1/2 adapters, so their training settings are those above.

### 3A · Retry under feedback

- **Tested:** up to 10 attempts per question under feedback arms (block message, "You failed", "You failed" + previous reasoning, GPT-5-mini critic) vs independent resampling. gpt-oss-120b (API), gpt-oss-20b and Qwen3-8B; 1,000 questions per arm.
- **Hypothesis:** failure feedback elicits control beyond what extra samples give.
- **Result:** gpt-oss-120b by attempt 10: resampling 10.9%, "You failed" 11.9%, critic 34.7%, previous reasoning 59.4%. At normal length (≥1,596 chars) these are 2.6 / 2.7 / 4.6 / 4.2%, so feedback adds only +1.6 to +2.0 pts.

### 3A-transfer · Feedback history as context

- **Tested:** successful episodes from other modes as system-prompt context, one attempt on a new mode: feedback history vs the matched successful example alone. gpt-oss-120b, 3 context draws.
- **Hypothesis:** feedback history transfers better than a successful example (Kaushik's claim).
- **Result:** mixed. Nine modes → held-out: −2.8 pts [−4.2, −1.6] (16.8 vs 19.6%); one mode → another: +3.0 [2.4, 3.7]. All context arms beat no context (4.2%).

### 3B · Hidden constraint in an agentic task

- **Tested:** 3 agentic tasks whose scoring file fails the episode if the reasoning contains a term or notation; the model is never told to control its CoT. 7 base/API models plus trained adapters, 250 episodes per scenario × condition.
- **Hypothesis:** models control their CoT when incentivized rather than instructed.
- **Result:** models read the rule (79–100%), yet 89–100% of readers still write the forbidden term; full success 0–10%. Graded suppression on arithmetic separates models (DeepSeek 78%, GLM 72%, gpt-oss-120b 36%, Qwen3.6-27B 36%, gpt-oss-20b 26%, Qwen3-8B 4%).

### 3-corr · Correlation across settings

- **Tested:** 3A and 3B re-run on every Experiment 1/2 checkpoint, plotted against standard-eval compliance.
- **Hypothesis:** if the standard eval measures the same capability, scores correlate across settings.
- **Result:** incomplete. 3B adapter runs finished; 3A checkpoint runs stopped at shutdown with mostly gpt-oss-20b adapters.

## Controls

### N · KL-matched random-LoRA noise

- **Tested:** 3 random LoRAs per arm × model, with the trained adapter's shapes, scaled to the arm's median KL(base ‖ adapter); 23 groups covering Experiments 1 and 2 (the two reasoning-on RL groups only partly finished).
- **Hypothesis:** any weight perturbation of matched functional size raises measured controllability.
- **Result:** rejected at KL ≤ 0.07. Noise sits at or below the untrained level (gpt-oss 0.9 vs 1.2%; Qwen3-8B 0.4 vs 0.5%; Qwen3.6-27B 0.3 vs 0.3%). At RL-matched KL ≈ 0.08 it costs 5–13 pts of accuracy.

| Setting | Ours | Reference: SDF random-LoRA control (`sdf/random_lora.py`) |
| --- | --- | --- |
| Model | gpt-oss-20b, Qwen3-8B, Qwen3.6-27B | gpt-oss-120b |
| Adapter | the trained arm's rank and modules (r32), 3 seeds per arm | rank 64; Gaussian A, B rescaled per module to the trained C4 update's Frobenius norm × scale (×1–300) |
| Matching | one point: the arm's median KL | sweep over scales, read against KL |
| Eval | T=1, 16k tokens, test split | T=0 (greedy), 16k tokens, test split |

- **Dose check:** matched KL ranged 0.001 (1D) to 0.13 (Qwen3-8B C4, step 200).
- **Assessment:** sound as a control for the SFT and C4 arms. One KL point per arm cannot show a dose response, which the reference's scale sweep can.
