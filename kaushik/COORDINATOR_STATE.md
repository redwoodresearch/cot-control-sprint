# Sprint coordinator state (2026-10-08, ~12:20 PT)

Read this first if you are a new coordinator session. Spec: `sprint_2026_10_08.md`. Each
experiment folder has its own `STATE.md` (written by its agent; authoritative for that folder).

## Setup (done, committed)
- Repo `redwoodresearch/cot-control-sprint` = Sebastian's controllability-elicitation history + our commits
  (`upstream` remote = sebastian-prasanna/controllability-elicitation). Nothing pushed since 10:00.
- Commits today: 80d4b60 (groundwork: spec, run.py, is_honest/is_hollow, configurable volumes, splits.json),
  67a81d0 (deployed Modal mode), f877f25 (spawn + re-attach on transient drops), bf53caa (Exp 3B).
- Credentials: Modal workspace `redwood-default` (user kaushik-reddy); OpenRouter key in `.env`
  (OPENROUTER_API_KEY, $10k limit). Never print secrets.
- Modal: deployed apps `cot-control-sprint_kaushik-reddy_inference` / `_train` (redeploy:
  `bash kaushik/deploy_modal.sh` after any remote-side code change). Checkpoints volume
  `cotcontrol_kaushik-reddy_checkpoints`; HF cache shared with Sebastian. `kaushik/run.py` sets the env.
- Workspace limits: 100 ephemeral apps (why deployed mode exists); ~1000 containers. Guard: agents launch
  nothing new above 800 workspace containers or below 25% laptop free memory. A colleague
  (probe-evasion_nabil-baugher) uses ~230 containers since ~11:50.
- `caffeinate -i` running (48 h) so the laptop doesn't idle-sleep; local drivers die if it sleeps.

## Agents (Fable, one per folder; file + process ownership is strict)
| folder | experiment | GPU cap | status |
|---|---|---|---|
| noise/ | KL-matched random-LoRA controls (per arm×model, 3 seeds, median-KL) | 80 | 9 groups running |
| fragility/ | Exp 1: 1A, 1D, 1E + Exp 1 table | 130 | 12/23 runs fully evaluated |
| fragility/cpt/ | Exp 1: 1B, 1C | 120 | stage-2 + 1B evals, done ~14:00 |
| orange/ | Exp 2A + Exp 2 table | 70 | 27B seeds 3–4 + test finals |
| orange/rl/ | Exp 2B (GRPO) | 200 | mcq_reason training → ETA 17:00–17:15 (critical path) |
| retry/ | Exp 3A + transfer + Exp 3 correlation | 40 | main arms, Modal transfer, checkpoint evals |
| hidden_constraint/ | Exp 3B | 25 | DONE (committed); poller alive until ~10:57 PT tomorrow |
Hand-off between agents: `kaushik/registry/<run>.json` (final adapters; entries containing "smoke" ignored).
Rules: never edit files you don't own; only signal processes you launched (an RL agent pattern-killed
`train_eval.py --eval-only` at 11:36 and killed ~36 cpt jobs; recovered). Agents don't commit; coordinator does.

## If an agent dies (rate limit / auth / context)
Try resuming it (SendMessage to its id). If that fails or its context is exhausted, start a FRESH Fable
general-purpose agent: "You own kaushik/<folder>/. Read kaushik/<folder>/STATE.md, its README,
kaushik/COORDINATOR_STATE.md and the spec; adopt running processes per STATE.md; finish the remaining
steps; same rules (ownership, guards, caps, no commits, only signal your own pids)."

## Preliminary results (see each README for final numbers)
- Exp 2A (27B, val, honest): persona condition orange beats orange_nopersona and mcq_noreason by ~+7 pts;
  direct: no difference; stated: mcq_noreason beats orange by ~5 pts. Persona gains come with trace
  shortening (check length-matched). RL noreason: no persona effect (+0.3).
- Exp 1: gpt-oss-20b 1A honest 4.9% vs untrained 1.5% vs noise 1.1% (partial); compl_long ~0 for all arms
  → gains are on shortened traces. CPT restricted-k on gpt-oss: zero compliance at every k/LR; 1B pending.
- Exp 3B (done): nobody hides for reward (full success 0–1%, Qwen3.6-27B up to 10%); graded suppression
  separates models (DeepSeek 78%, GLM 72%, gpt-oss-120b 36%, 27B 36%, gpt-oss-20b 26%, Qwen8B 4%);
  reliable (split-half 0.6–0.95). style_sft (pirate) +10 pts arithmetic is likely a style confound.
- Exp 3A: pipeline reproduces pinned numbers; transfer draw 0: history_critic 15.5% vs fewshot_matched 17.6%
  (−2.1 [−3.9, −0.3]); block arm ≈0 (1/538) for gpt-oss-20b and 120b; bare_trace 20b 28% vs 120b 57% by attempt 10.

## Known design caveats for the write-up
- rl_mcq reward saturated from iteration 0 (0.93–1.0) → under-dosed baseline; compare RL arms at matched KL.
- Restricted-k CPT barely fits C4 (loss 7–12 vs 2.7 full) → compare at matched KL/loss, not k.
- gpt-oss-20b trained adapters read the grader less in 3B; use conditional metrics.

## Remaining coordinator TODO
1. Commit each experiment as it finishes (code + README + small result files; `git add -f` md/csv/png under runs/).
2. When all done (~17:30): spawn ONE Fable subagent to write the cross-experiment summary from the READMEs
   (keep the coordinator's context small); update `kaushik/README.md` results section.
3. Cleanup after the sprint: `modal app stop cot-control-sprint_kaushik-reddy_inference` and `_train`;
   kill caffeinate; remove `kaushik/_deploytest/`; consider renaming the OpenRouter key per Redwood convention.
