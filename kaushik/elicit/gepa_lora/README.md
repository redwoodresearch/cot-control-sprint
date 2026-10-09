# Workstream B2 — no-CoT context-distilled "GEPA LoRA" for gpt-oss-20b

Goal: a LoRA such that the **unprompted** gpt-oss-20b behaves like gpt-oss-20b **prompted with Sebastian's GEPA s1
system prompt**, trained with **no loss on any reasoning (or answer) token**. Shared protocol, rules and budget:
`kaushik/elicit/PLAN.md`. Live state: `STATE.md`. Results: `results/`.

## Method and why

**Teacher** = base model with `[developer: GEPA prompt] + [user: question + Requirement]`.
**Student** = base model + LoRA with `[user: question + Requirement]` only. Both rendered with the gpt-oss chat
template at `reasoning_effort: medium` (the eval setting) plus the deterministic assistant header
`<|channel|>analysis<|message|>`. Prompts: the 514 TRAIN-split questions × the 9 training modes (4588 prompts;
51 questions / 457 prompts held out for the distillation loss; val/test questions never used).

**Loss** (`distill_app.py`): forward KL(teacher ‖ student) of the next-token distributions at every position the
model sees *before any reasoning token*:

| group | positions | what the next-token distribution is |
|---|---|---|
| `user` | from the user-turn `<|message|>` through the second-to-last user token | the user turn itself |
| `turn` | last user token … `analysis` (6 positions) | `<|end|> <|start|> assistant <|channel|> analysis <|message|>` |
| `first` | the assistant `<|message|>` | **the first reasoning token** |

`loss = mean_user + w_turn·mean_turn + w_first·mean_first (+ w_anchor·KL(base‖student) on turn+first)`.

Why this and not something else:

- *Context distillation on prompt positions only.* Ordinary context distillation trains on the teacher's sampled
  continuation, which here would be reasoning tokens — forbidden. The prompt positions are the only ones the model
  processes before reasoning, and they are also where a system prompt acts: the user-turn hidden states that
  every later reasoning token attends to. Matching the teacher's next-token distributions there (hundreds of
  positions per prompt, a dense signal) pushes the student's user-turn representations toward "as if the GEPA
  prompt were present". The `first` term is the one position whose target is directly about reasoning
  (what the trace starts with) without any reasoning token being an input or a target.
- *KL on distributions rather than hidden-state MSE.* Distributions are invariant to the basis of the residual
  stream, come with a natural scale (nats) that is comparable across positions and runs, and are what the
  sampler sees. Hidden-state matching was kept as a fallback (not needed).
- *The gpt-oss template makes this clean.* The system prompt is rendered as a `developer` block *between* the
  default system block and the user turn, so the user turn is an identical contiguous token run in both
  renderings and positions align exactly (`len_s_max`/`len_t_max` and an assert in `prep()`).
- *Teacher online, one copy of the weights.* The teacher is the same PEFT model with the adapter disabled
  (`model.disable_adapter()`, no grad); no second 20B copy, no cached 200k-vocab logits.
- *Rank 16, alpha 32, attention-only (q,k,v,o), lr 1e-4 / 3e-4, bs 16, 400 steps (≈1.5 epochs).* Small and
  attention-only keeps the adapter in the plain vLLM LoRA path (no fused-MoE expert LoRA) and limits drift; KL
  to the base on unconstrained prompts (`drift_*` in the log) and the held-out distillation loss are logged
  every 50 steps. Adam, no weight decay, 10-step warmup, constant lr, grad clip 1.
- *Length-preserving variant* (`w_anchor`, coordinator request 2026-10-09): Workstream B's teacher check shows
  the GEPA prompt's val gain is mostly brevity (median reasoning 2608 → 1159 chars; honest 1.0 → 9.9% but
  honest_long only 0.4 → 1.2%). The header positions are where the model commits to how (and how long) it will
  reason, so `gl_*_anchor_*` matches the teacher on the user turn but pulls the header/first-token
  distributions toward the **base** (w_turn = w_first = 0.3, w_anchor = 1).

Infra: a custom single-GPU loop (HF + PEFT) in a Modal app `..._sub-gepa-lora_distill` that reuses
`cotcontrol.training.modal_app`'s image, volumes and the gpt-oss loading recipe (MXFP4 → bf16 dequantization,
eager attention). Rows longer than 2048 tokens (teacher rendering) are dropped; micro-batches are packed under
token/position budgets because eager attention and full-vocab logits are the memory hogs.

## Stages

```
export COTCONTROL_MODAL_INFERENCE_APP=cot-control-sprint_kaushik-reddy_sub-gepa-lora_inference
export COTCONTROL_MODAL_TRAIN_APP=cot-control-sprint_kaushik-reddy_sub-gepa-lora_train
R=".venv/bin/python kaushik/run.py elicit/gepa_lora"
$R build_data                      # data/prompts.jsonl, data/drift.jsonl
$R deploy                          # distill app (after editing distill_app.py)
$R train --run NAME --lr 1e-4 [--w-anchor 1 --w-turn .3 --w-first .3] [--seed k]
$R status [--run NAME]             # call state + log.jsonl tail (runs/<run>/log.jsonl)
$R probe --lora /checkpoints/gepa_lora/<run>/checkpoint-N   # vLLM path: first-token check vs teacher
$R launch <run>_val -- --split val --lora /checkpoints/gepa_lora/<run>/checkpoint-N [;; ...]
$R noise --group G --adapters A,B,C                            # KL-matched random LoRAs
$R analyze [--runs ...]            # results/table.md (+ Workstream B's teacher/untrained rows)
$R spend
```

## Files
`gl_common.py` constants · `build_data.py` · `distill_app.py` trainer · `cli.py` · `evalrun.py` (same format as
Workstream B) · `probe.py` · `noise.py` · `analyze.py` (wraps Workstream B's analyze for identical definitions) ·
`data/` · `runs/<run>/{train.json,log.jsonl,result.json,<block>.json,manifest.json,eval.log}` ·
`runs/_probe/`, `runs/_noise/` · `results/`.

## Results so far (val, 200 q; 95% CIs over questions; `results/val_table.md`)

| candidate | honest | honest_long | held-out | accuracy | median chars | gain recovered (honest) |
|---|---|---|---|---|---|---|
| untrained | 1.0 [0.5, 1.6] | 0.4 | 1.3 | 48.5 | 2608 | — |
| teacher (GEPA prompt) | 9.9 [8.7, 11.0] | 1.2 | 12.5 | 47.0 | 1159 | 100% |
| LoRA lr 1e-4, step 400 | 8.9 [7.7, 10.3] | 0.6 | 11.2 | 45.5 | 893 | 89% |
| LoRA lr 3e-4, step 400 | 10.3 [9.1, 11.5] | 2.1 | 12.7 | 46.5 | 868 | 104% |
| LoRA lr 1e-4 + header anchor | 7.5 [6.4, 8.7] | 0.3 | 9.2 | 45.0 | 996 | 73% |

The unprompted LoRA model reproduces the prompted model's compliance on training and held-out modes, and also its
brevity (the teacher's gain is mostly brevity: honest_long stays near zero for everyone). The header anchor does not
preserve length — the length decision is carried by the user-turn representation, not the header distribution — so
the plain lr 1e-4 config (3 seeds, best held-out distillation loss) went to test with the KL-matched noise control.

## Final (test, 500 q x 9 modes; 3 LoRA seeds pooled = 13392 rollouts; 95% bootstrap CIs over questions; `results/test_table.md`)

| arm | honest | honest_long | held-out | accuracy (unconstrained) | median chars | % of teacher gain (honest / held-out) |
|---|---|---|---|---|---|---|
| untrained | 1.5 [1.1, 2.0] | 0.5 [0.2, 0.9] | 1.3 [0.7, 1.9] | 48.6 [44.2, 52.6] | 2612 | — |
| teacher (base + GEPA s1 prompt) | 9.4 [8.5, 10.2] | 0.7 [0.3, 1.3] | 11.2 [9.7, 12.7] | 45.0 [40.6, 49.2] | 1190 | 100 / 100 |
| **GEPA LoRA** (lr 1e-4, step 400, 3 seeds, no system prompt) | **9.7 [9.1, 10.3]** | 0.8 [0.4, 1.3] | **12.2 [11.2, 13.3]** | 43.5 [39.5, 47.4] | 883 | **105 / 110** |
| KL-matched random LoRA (3 adapters, KL 0.0236 nats/token) | 1.3 [1.1, 1.6] | 0.4 [0.2, 0.5] | 1.1 [0.8, 1.4] | 46.9 [43.0, 50.7] | 2679 | -3 / -2 |

- The unprompted LoRA model reproduces the prompted model's honest compliance on the 9 training modes and the 3
  held-out modes, while the functional-size-matched random adapter does nothing: the effect is the distilled prompt, not
  "any perturbation of this size".
- It also reproduces the prompt's brevity (and slightly overshoots it: 883 vs 1190 chars, untrained 2612), so like the
  teacher it fails the >=80% length guardrail, and honest_long stays at the floor for every arm (0.4-0.8%). The GEPA
  prompt's gain on gpt-oss-20b is mostly a brevity effect, and context distillation of its prompt positions transfers
  exactly that. The header-anchored variant did not preserve length (val: 996 chars, 73% of the gain).
- Accuracy: 43.5 vs 48.6 untrained (-5.1, CIs overlap; teacher -3.6) — slightly outside the 3-point guardrail.
- Nothing in training ever conditions on or supervises a reasoning token; `distill_app.py` only sees prompts.
