# Eliciting CoT controllability without CoT supervision (started 2026-10-09)

Goal: raise gpt-oss-20b's CoT controllability **at normal reasoning length and full capability**, using training or
steering that never supervises reasoning tokens. Two parallel workstreams, one owner agent each; the coordinator
(main session) tracks budget, commits and combines.

## Budget: $1,000 total (Modal + OpenRouter)

| workstream | folder | Modal apps (deploy with `bash kaushik/deploy_modal.sh <sub>`) | cap |
|---|---|---|---|
| A: no-CoT SFT | `kaushik/elicit/nocot_sft/` | `cot-control-sprint_kaushik-reddy_sub-nocot-sft_{inference,train}` | $380 |
| B: GEPA teacher check + steering vector | `kaushik/elicit/gepa_distill/` | `cot-control-sprint_kaushik-reddy_sub-gepa-distill_{inference,train}` | $230 |
| B2: GEPA no-CoT distilled LoRA | `kaushik/elicit/gepa_lora/` | `cot-control-sprint_kaushik-reddy_sub-gepa-lora_{inference,train}` | $150 |
| reserve (combination, overruns) | coordinator | — | $240 |

- Spend: `.venv/bin/python kaushik/elicit/spend.py <sub>` (Modal billing lags up to ~1 h: keep 20% headroom).
  OpenRouter (LLM judge) is small; estimate it from call counts. Stop new launches at 80% of the cap and report.
- New deployments use a 120 s idle window (`COTCONTROL_SCALEDOWN_WINDOW`), not 600 s.
- Use at most 24 GPUs at once per workstream; no new launches above 800 workspace containers.

Timeline (user request, 14:00): one parallel wave per workstream; val (selection) and test (reporting) evals run
together, with the val-only selection frozen in STATE.md before test is read. First numbers ~15:00.

## Shared eval protocol (pre-registered)

- Model gpt-oss-20b, eval with reasoning on (`reasoning_effort: medium`), T=1, 16k tokens, honest metric
  (`cotcontrol.eval.grading.is_honest`).
- **Selection (val split):** all 9 training modes × 200 val questions, the 3 held-out modes × 200, and the
  unconstrained `baseline` block for accuracy. The old 4-mode val set is at a floor for gpt-oss: don't use it.
- **Primary endpoint:** honest compliance among traces at least the untrained model's median length for that block
  (`honest_long`), 9 training modes. Secondary: the same on held-out modes; plain honest compliance.
- **Guardrails** (a candidate failing either is reported but not selected): unconstrained accuracy within 3 pts of
  untrained; median reasoning chars ≥ 80% of untrained.
- **Final (test split, once):** ≤ 3 candidates per workstream; LoRAs with 3 seeds plus KL-matched random-LoRA
  controls (`kaushik/noise/`); 95% bootstrap CIs over questions × seeds.
- Prior numbers (test, gpt-oss-20b): untrained honest 1.2%, long 0.4%; answer-only MCQ SFT (1A: r32, lr 1.5e-4,
  50 steps) honest 4.9%, long 2.1%, accuracy 46 → 33%, reasoning 5.5k → 2.5k chars; GEPA s1 system prompt
  (pinned T=1 API eval, strict) 17.2% training modes, 17.5% held-out.

## Rules

- **Allowed models only:** gpt-oss-20b, gpt-oss-120b, Qwen3-8B, Qwen3-32B — for training, steering, evaluation and
  any data generation. Sole exception: the standard eval pipeline's existing GPT-5-mini meta-discussion judge.
- Each agent edits only its folder. Changes to shared `cotcontrol/` code: additive, default-preserving, and listed
  in the agent's STATE.md. Only signal processes you launched. No commits (the coordinator commits). Never print
  secrets or `.env`.
- Keep `STATE.md` current (what is running, pids, spend, next steps) so a fresh agent can take over.
