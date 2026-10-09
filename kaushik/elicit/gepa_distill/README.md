# gepa_distill — internalizing the GEPA prompt into gpt-oss-20b without supervising reasoning

Workstream B of `kaushik/elicit/PLAN.md`. The teacher is gpt-oss-20b with Sebastian's GEPA-optimized system prompt
(seed s1, `monitorability/prompts/gepa_general.txt`, sha256 `160d1c39…e61e3d`); the goal is the same CoT-controllability
gain with **no prompt**, obtained without any loss on reasoning tokens. Deliverable: a steering vector (the distilled
LoRA was split off to `kaushik/elicit/gepa_lora/` on 2026-10-09 14:00 PT).

## Method

1. **Teacher check.** GEPA prompt vs no prompt under the shared protocol (val and test; T=1, 16k tokens,
   `reasoning_effort: medium`, honest metric), reporting honest, honest_long (traces ≥ the untrained median length),
   held-out modes, unconstrained accuracy and trace length. If the prompt's gain lives in shortened traces, so will
   anything distilled from it.
2. **Steering vector.** v_L = mean over 300 train-split constrained prompts of (residual stream with the GEPA system
   prompt − without), taken right after attention in layer L at the last prompt token (`last`) or averaged over the
   shared suffix (user turn + assistant header, `suffix_mean`). gpt-oss attention has an `o_proj.bias`, so writing
   α·v_L into layer L's bias adds α·v_L to the residual stream at every position — a steered model is just a copy of the
   HF checkpoint with one bias changed, served by vLLM as a local base model (no inference-code changes). `extract` also
   checks that the bias edit equals a forward hook adding α·v (HF, 3 prompts). Sweep layers × α on 40 val questions
   × 9 modes, then the best settings on full val and test, with a random direction of equal norm at the same layer as the
   control.

## Stages (`kaushik/run.py elicit/gepa_distill <stage>`; export the sub-gepa-distill app names first, see STATE.md)

| stage | what |
|---|---|
| `eval --run R --split val\|test [--system-prompt gepa] [--base-model PATH] [--lora PATH] [--blocks ...] [--max-questions N --subsample-seed S]` | one candidate, blocks modes / heldout / baseline -> `runs/R/<block>.json` |
| `launch R -- args ;; R2 -- args` | detached eval processes (`runs/R/eval.log`, `runs/R/pid`) |
| `steer extract\|build\|sweep\|cleanup` | vectors -> steered checkpoints on the volume -> subset evals -> delete |
| `analyze [--runs ...]` | table with 95% bootstrap CIs over questions -> `results/table.md` + `.json` |
| `spend` | Modal spend of this workstream |

Untrained test reference: the fragility checkpoint-0 test evals (zero adapter = base model, identical protocol),
`gd_common.UNTRAINED_TEST`, reused instead of a second untrained test run.

## Layout

```
gd_common.py   paths, prompt + sha256, eval protocol, app names      evalrun.py   candidate eval (chunked, GPU-slot capped)
steer_app.py   Modal app: extract / build_checkpoint / remove_path   steer.py     local driver for the steering pipeline
analyze.py     metrics + bootstrap                                   cli.py       stages
runs/<run>/    eval JSONs (every rollout + judge output), manifest, log; runs/steer/ vectors + checkpoint records
results/       table.md / table.json
```

## Results (2026-10-09; `results/table.md`, 95% bootstrap CIs over questions; honest metric, values in %)

**Teacher check — the GEPA prompt's gain is almost entirely shortened traces.** Test (500 questions, 4,464 rollouts on the
9 training modes): honest 9.4 [8.5, 10.2] with the prompt vs 1.5 [1.1, 2.0] untrained, but honest_long (traces ≥ the
untrained median of 2,612 chars) 0.7 [0.3, 1.3] vs 0.5 [0.2, 0.9]; the prompt halves the median trace (1,190 chars; only
27% of its traces reach the bar). Held-out modes 11.2 [9.7, 12.7] vs 1.3; heldout_long 0 vs 0. Accuracy 45.0 vs 48.6.
Val says the same (9.9 / 1.2 vs 1.0 / 0.4).

**Steering — nothing to distil.** Vectors from 300 train prompts (`runs/steer/vectors_v1.json`); |v| is 11–24% of the
residual norm at layers 6–16; the bias edit matches a forward hook up to bf16 noise (96.6% next-token agreement at L12,
α 2, vs 71% steered-vs-base). 24-variant sweep (`results/sweep.md`, layers 6/9/12/15 × α 1/2/4, GEPA vector and
GEPA-minus-brevity residual): every variant ≤ 3.1% honest on the 40-question subset; α 4 breaks the model. Test:

| arm | honest | honest_long | held-out | accuracy | chars med | gain recovered (honest) |
|---|---|---|---|---|---|---|
| untrained | 1.5 [1.1, 2.0] | 0.5 [0.2, 0.9] | 1.3 [0.7, 1.9] | 48.6 [44.2, 52.6] | 2,612 | – |
| teacher (GEPA prompt) | 9.4 [8.5, 10.2] | 0.7 [0.3, 1.3] | 11.2 [9.7, 12.7] | 45.0 [40.6, 49.2] | 1,190 | 100% |
| steer last_L12_a1 | 1.8 [1.3, 2.2] | 0.4 [0.1, 0.7] | 0.7 [0.3, 1.1] | 45.0 [40.8, 49.0] | 2,226 | 3% |
| steer last_L15_a2 | 1.0 [0.7, 1.3] | 0.4 [0.2, 0.8] | 1.0 [0.5, 1.5] | 37.8 [33.6, 42.0] | 2,350 | −7% (fails accuracy guard) |
| GEPA − brevity, L12 α2 | 1.3 [1.0, 1.7] | 0.4 [0.2, 0.7] | 0.9 [0.5, 1.3] | 38.6 [34.4, 43.0] | 3,050 | −2% (fails accuracy guard) |
| random direction, L15, same norm | 4.1 [3.5, 4.7] | 8.9 [6.9, 11.2] | 0.1 [0.0, 0.2] | 16.0 [13.0, 19.4] | 0 | broken model (see below) |

The random-direction control at the L15/α2 norm destroys the model (59% of rollouts emit a malformed channel header such as `<|channel|>analysis to=...`, garbled text, and the parser sees an empty trace; its
"honest" hits are empty-trace word-suppression compliance), so it says only that the GEPA direction is benign at that
norm, not that it is special. The brevity residual (cos(GEPA, brevity) ≈ 0.65 at L12) does lengthen traces (3,050 chars)
and lifts strict compliance to 3.4%, but honest compliance and honest_long stay at the untrained level and accuracy drops
10 points. Conclusion: a single residual-stream direction from the prompt does not carry the prompt's effect, and the
prompt's effect itself is not a normal-length controllability gain.
