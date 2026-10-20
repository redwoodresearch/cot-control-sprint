# kaushik/

Experiments from the CoT-controllability sprint (2026-10-08 → 10-12). Spec: [sprint_2026_10_08.md](sprint_2026_10_08.md).

## What happened

**Question.** Is low measured CoT controllability an elicitation gap? That is, can cheap interventions (generic SFT,
"orange" persona SFT, RL, retry feedback, optimized prompts, distilled LoRAs, steering) unlock control the model
already has?

**Answer: no, a null result.** Across ~20 interventions, none raised honest compliance by more than ~2 points among
traces of normal length (at least the untrained model's median). Every large headline gain came from the model
reasoning less:

| intervention | model | honest compliance, all traces | at normal length |
|---|---|---|---|
| retry: "you failed" + previous reasoning, 10 attempts | gpt-oss-120b | 59.4% (resampling 10.9%) | 4.2% (resampling 2.6%) |
| GEPA-optimized system prompt | gpt-oss-20b | 9.4% (untrained 1.5%) | 0.7% (untrained 0.5%) |
| GEPA prompt context-distilled into a LoRA | gpt-oss-20b | 9.7% | 0.8% |
| orange persona SFT, persona prompt | Qwen3.6-27B | 5.6% (controls ≈0%) | 0.2% |
| answer-only MCQ SFT | gpt-oss-20b | 4.9% | 1.8% |
| KL-matched random LoRA (control) | gpt-oss-20b | 0.9–1.3% | 0.3–0.4% |

Takeaways:

1. **Measured controllability is confounded with reasoning length.** On gpt-oss-20b, P(compliant | trace length) is
   one decreasing curve shared by every arm (15–18% under 1k chars, ≈0 above 8k); interventions move traces down it,
   none lifts it. Report compliance at matched length and against a KL-matched random LoRA by default.
2. **The orange result does not hold up.** The earlier ~34% (greedy decoding, raw metric) is 5.6–8% under T=1 and the
   honest metric, only on Qwen3.6-27B, unchanged without the persona (7.9%), and 0% on gpt-oss-20b/120b and
   Qwen3-8B/32B after SFT or RL.
3. **Retry feedback is a length lever.** "You failed" alone equals resampling; critic or previous-reasoning feedback
   adds only +1.6–2.0 points at normal length (CIs exclude 0).
4. **Positive side results.** The hidden-rule-mid-task eval (3B) separates models sharply: almost all read the rule,
   ~none fully obey, and suppression ranges from 78% (DeepSeek V4 Pro) to 1% (Qwen3-32B). No-CoT context distillation
   moves a prompt into weights (105% of its gain) where a steering vector gets 3%. GEPA LoRA − 2× a "be brief" LoRA
   keeps 65% of the gain at 90% length (still 0.9% at normal length).
5. **Cross-eval correlation** (does retry performance predict the standard eval?): inconclusive with 6–8 models.

**Caveats / open gaps.** Phase 2 (`elicit/`) is gpt-oss-20b only. SFT used Sebastian's recipe (r32, LR 1.5e-4,
50 steps); lower rank/LR was not swept. RL was under-powered (rewards saturated in 2–20 steps). The C4 CPT arms ran
hotter than SDF's run and damaged gpt-oss, so they are not a clean test.

**Read more:**

- [postmortem/meeting/REPORT.md](postmortem/meeting/REPORT.md): Oct 8 summary of Experiments 1–3, figures in `postmortem/meeting/figs/`.
- [elicit/writeup/RESULTS.md](elicit/writeup/RESULTS.md), [LAB_NOTES.md](elicit/writeup/LAB_NOTES.md): Oct 9 elicitation follow-ups, figures in `elicit/writeup/figures/`.
- [orange_expand/README.md](orange_expand/README.md): orange on four more models.
- [postmortem/experiment_log.md](postmortem/experiment_log.md), [postmortem/training_postmortem.md](postmortem/training_postmortem.md): per-arm details and the hyperparameter postmortem.
- [retry/runs/main/results.md](retry/runs/main/results.md): 3A, 3A-transfer and correlation tables.

## Running

Run everything through the runner, which sets `COTCONTROL_MODAL_OWNER=kaushik-reddy`
(checkpoints on `cotcontrol_kaushik-reddy_checkpoints`) and loads `.env`:

```bash
.venv/bin/python kaushik/run.py --list
.venv/bin/python kaushik/run.py <experiment> <job> [key=value ...]   # config.yaml + core jobs
.venv/bin/python kaushik/run.py sweep <sweep.yaml> --dry-run          # YAML sweeps through the shared scheduler
```

Design: [ARCHITECTURE.md](ARCHITECTURE.md); shared library: [core/README.md](core/README.md). All experiment folders
were migrated on 2026-10-12 to `config.yaml` + `sweeps/*.yaml` + single-purpose jobs on `kaushik/core`.

| folder | experiment | status |
|---|---|---|
| `noise/` | KL-matched random-LoRA controls (Exp 1, 2) | done |
| `fragility/` | Exp 1: 1A, 1D, 1E + Exp 1 table | done |
| `fragility/cpt/` | Exp 1: 1B, 1C (C4 continued pretraining) | done |
| `orange/` | Exp 2A + Exp 2 table | done |
| `orange/rl/` | Exp 2B | done |
| `retry/` | Exp 3A, 3A-transfer, Exp 3 correlation | done |
| `hidden_constraint/` | Exp 3B | done |
| `orange_expand/` | orange SFT/RL on four more models | done |
| `elicit/` | phase 2: elicitation without shortening (A, B, B2, C1–C3) | done |
| `postmortem/` | experiment log, training postmortem, Oct 8 meeting report | — |

## Prerequisites

- `datasets/splits.json` (canonical split): regenerate with `python build_splits.py`
  (sha256 `1d84772c…782118`).
- `.env` at repo root with `OPENROUTER_API_KEY` (gitignored). Modal: `modal setup` into the Redwood workspace.
- C4 data for `fragility/cpt/` is not in git (raw shard is 305 MB): `.venv/bin/python kaushik/run.py fragility/cpt build_data`.
