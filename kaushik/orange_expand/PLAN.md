# Orange expansion (started 2026-10-09 ~15:00 PT)

Question: does training a model to violate a "do NOT imitate EUGENE" prohibition (orange picking) transfer to
CoT control, and is the transfer about the persona, the negation, or generic training? Prior (Qwen3.6-27B,
`kaushik/orange/`): EUGENE SFT with reasoning off gave 5.6% honest compliance under the negated persona prompt
(direct prompt 5.4%, untrained 0% / 2.1%); RL gave ~0.3%, but its reward saturated within 2–20 iterations.
gpt-oss-20b showed 0% everywhere.

## Factors and what is cut (decided before any results)

| factor | levels | stage 1 (screen) |
|---|---|---|
| model | gpt-oss-20b, gpt-oss-120b, Qwen3-8B, Qwen3-32B | SFT: all 4. RL: gpt-oss-20b, Qwen3-32B |
| training prompt (`prompts.py`) | eugene (negated persona), direct, standard (neutral) | SFT: all 3. RL: eugene only |
| eval prompt (`prompts.py`) | persona, nopersona, direct, standard | all 4 on every run |
| reasoning during training | off, on | SFT: off. RL: on |
| method | SFT, RL | both |

- **SFT with reasoning on is cut:** it needs self-distilled reasoning targets, and 1D showed those barely move the
  model (KL ≤ 0.004).
- **RL with reasoning off is cut:** 2B saturated within 2 iterations with 3–8% of groups carrying gradient.
- **RL with the direct prompt is cut** if the stage-0 orange-pick rate under that prompt is ≥ 0.7: there is little
  reward left to learn.
- **RL with the standard prompt** is the control for an RL hit and runs only in stage 2.

## Stage 0: baselines (no training)

- Orange-pick rate per model × training prompt × reasoning on/off, on 200 held-out orange items.
- Untrained eval: 4 models × 4 eval conditions × 2 modes.
- Mechanism check on existing adapters: the 2A Qwen3.6-27B `orange` and `orange_nopersona` adapters (seeds 0–2)
  under the new `nopersona` condition, to tell persona from negation where the effect is known to exist.

## Stage 1: screen (1 seed)

- **SFT:** 4 models × 3 training prompts = 12 runs. Orange recipe (LoRA r32/α32, LR 1.5e-4 constant, batch 48),
  **100 steps** instead of 50 (the 27B was still rising at 50); reasoning conditioned off; checkpoints at 50 and 100;
  eval step 100 only.
- **RL (GRPO, reasoning on):** eugene prompt × {gpt-oss-20b, Qwen3-32B}. 2B's settings (48 prompts × 8 rollouts,
  LR 1.5e-4, LoRA r32, DAPO), rollout cap 8,192 tokens. **No fixed step count:** at least 50 iterations, then stop
  once mean reward is ≥ 0.98 for 20 consecutive iterations (nothing left to learn under this reward); at most 150.
  Save checkpoints every 25 iterations; eval the final one.
- **Eval for every run:** 4 conditions × 2 modes (lowercase, uppercase) × 200 val questions, T=1, 16k tokens.
  Honest compliance, compliance among traces at least the untrained median length (`compl_long`), accuracy,
  reasoning chars, degenerate rate. Bootstrap CIs over questions.

## Decision rules

- **Hit:** a trained arm beats both its model's untrained baseline and the same model's standard-prompt SFT arm by
  ≥ 2 points in some eval condition, with the 95% CI of the difference excluding 0. RL hits are judged against
  untrained only in stage 1. A hit whose gain disappears in `compl_long` is labelled length-driven.
- **No hit for a method:** that method stops; no extra models or seeds for completeness.
- **Stage 2** (needs user sign-off and a cost estimate): confirm hits with 3 seeds, test split, 4 modes, KL-matched
  random LoRA, and an "obedient EUGENE" control (eugene prompt, non-orange target). RL hits add the standard-prompt
  RL control and gpt-oss-120b.

## Budget and limits

- **Stage 0+1 cap: $800** (Modal + OpenRouter): SFT agent $380, RL agent $380, reserve $40. Each agent stops new
  launches at 80% of its cap and reports. Check spend with `.venv/bin/python kaushik/elicit/spend.py <sub>`, which
  lags up to ~1 h.
- Modal apps (idle window 120 s; redeploy with `bash kaushik/deploy_modal.sh <sub>`):
  - SFT agent: `sub-orange-expand`, i.e. `cot-control-sprint_kaushik-reddy_sub-orange-expand_{inference,train}`.
  - RL agent: `sub-orange-rl`.
  - Point runs at them with `COTCONTROL_MODAL_INFERENCE_APP` / `COTCONTROL_MODAL_TRAIN_APP`. `kaushik/run.py`
    only sets defaults, and its defaults are the stopped sprint apps.
- At most 20 concurrent GPUs per agent; no new launches above 800 workspace containers.
- A parallel elicitation session (`kaushik/elicit/`) is running. Don't touch its folders, apps or processes.

## Ownership

| folder | owner | contents |
|---|---|---|
| `kaushik/orange_expand/sft/` | SFT agent | stage 0, SFT runs, the shared eval-block builder, final analysis |
| `kaushik/orange_expand/rl/` | RL agent | RL runs and their evals (same output format) |
| `kaushik/orange_expand/{PLAN.md,prompts.py}` | coordinator | read-only for agents |
