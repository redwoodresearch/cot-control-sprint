# Sprint coordinator state (updated 2026-10-08 ~15:00 PT)

Read this first if you are a new coordinator session. Spec: `sprint_2026_10_08.md`. Each experiment
folder has its own `STATE.md` (authoritative for that folder, incl. a "TO FINISH" checklist).

## Setup (done)
- Repo `redwoodresearch/cot-control-sprint` = Sebastian's controllability-elicitation history + our commits
  (`upstream` = sebastian-prasanna/controllability-elicitation). NOTHING PUSHED since 10:00; push only if user asks.
- Commits: 80d4b60 groundwork, 67a81d0 deployed Modal mode, f877f25 spawn+re-attach, bf53caa Exp 3B,
  c418350 this file, aa1abb6 Exp 3B figures, 4d34b0b Exp 1 SFT arms.
- Modal workspace `redwood-default` (user kaushik-reddy); OpenRouter key in `.env` (OPENROUTER_API_KEY, $10k).
  Deployed apps `cot-control-sprint_kaushik-reddy_inference` / `_train` (redeploy: `bash kaushik/deploy_modal.sh`
  after remote-side code changes). Checkpoints volume `cotcontrol_kaushik-reddy_checkpoints`. `kaushik/run.py` sets env.
- Guards for every agent: no new launches above 800 workspace containers (hard limit ~1000, colleagues use 150–300)
  or below 25% laptop free memory. Caps now: orange 120, rl 200, retry 56, noise 80, fragility released, cpt 90.
- Rules: agents edit only their own folder; only signal pids they launched (never pattern-kill shared script names —
  an agent did that at 11:36 and killed ~36 cpt jobs); agents don't commit — coordinator commits code + README +
  small result files (`git add -f` md/csv/png under runs/; keep big rollout dumps local).
- `caffeinate -i` running; laptop must stay awake (local drivers).

## Status (~15:00)
| folder | status | ETA |
|---|---|---|
| hidden_constraint/ (3B) | DONE, committed, figures done; poller alive until tomorrow ~11:00 | — |
| fragility/ (1A/1D/1E) | DONE, committed (4d34b0b); figures committed (524dd29) | — |
| fragility/cpt/ (1B/1C) | handing off to a fresh agent; 2 low-LR evals + 2 Qwen stage-2 evals left | ~16:00 |
| orange/ (2A) | DONE, committed (4577d03). Figures made (uncommitted). Coordinator fixed analyze.py qid bug (ids repeat across datasets → now dataset:id; cache v2; test updated, uncommitted). AFTER rl+noise finish: `run.py orange analyze --plot --boot 2000`, then `.venv/bin/python kaushik/orange/figures/make_figures.py`, check numbers shift ≤0.4pp, commit fix+results+figures | after rl/noise |
| orange/rl/ (2B) | handing off; all 20 trained, eval tail | ~16:00–16:30 |
| retry/ (3A) | handing off; Qwen main ~15:40, Modal transfers, checkpoint evals, then correlation | ~17:00 |
| noise/ | agent continues; 3/15 groups done, rest follow | ~16:30 |

## Handoff procedure (context limits)
Agents near 1M context write STATE.md + reply "HANDOFF READY", then a FRESH Fable general-purpose agent is spawned:
"You own kaushik/<folder>/ (and its listed cotcontrol files). Read kaushik/<folder>/STATE.md (TO FINISH checklist),
its README, kaushik/COORDINATOR_STATE.md, and the spec. Adopt running processes (don't restart live ones), wait for
completion with Monitor (30-min check-ins, small tool outputs), then do the final analysis + README + STATE update.
Rules: ownership, guards, cap, only signal pids from STATE.md / you launched, no commits, never print secrets.
Final report ≤12 lines: headline numbers with CIs vs matched baselines, caveats, files."
When an experiment finishes: commit it, then spawn a Sonnet figures agent (pattern: read README + summary tables,
invoke `dataviz` skill, write figures/make_figures.py + PNG/PDF + figures/README.md, only under figures/).

## Results so far (see READMEs for final)
- Exp 1 (final): 1A generic MCQ SFT beats untrained AND KL-matched noise on all 3 models but small and on shortened
  traces (test honest: gpt-oss-20b 1.2→4.9% [4.1,5.8] vs noise 0.9; Qwen8B 0.5→2.6%, degenerate; Qwen27B 0.3→1.0%);
  compl_long ≈ 0 everywhere. 1D no effect. 1E breaks answer extraction (uninterpretable). CPT: ~0 at every k/LR;
  restricted-k barely fits C4.
- Exp 2A (prelim, 27B val): persona cond. orange beats nopersona & mcq_noreason by ~+7 pts; direct: no diff; stated:
  mcq_noreason > orange by ~5; persona gains tied to trace shortening. 2B (prelim): no orange-RL lift; rl_mcq reward
  saturated (under-dosed baseline) → compare at matched KL.
- Exp 3A (1,000-item subset, honest by attempt 10): gpt-oss-120b bare 11.9 vs independent 10.9 (no feedback effect),
  bare_trace 59.4 (via trace shortening; gain vanishes when length-matched), block 0.6 / block_once 1.3;
  gpt-oss-20b bare 6.0 vs indep 6.2, bare_trace 35.8, block 1.0; Qwen8B bare_trace 4.8, block 0.
  Transfer draw 0 (gpt-oss-120b held-out): history_critic 15.5 vs fewshot_matched 17.6 (−2.1 [−3.9,−0.3]).
  Checkpoints vary widely; BROKEN adapters (acc <50% base) flagged; correlation reported with/without.
- Exp 3B (final): models notice the rule and break it anyway; graded suppression separates models (DeepSeek 78%,
  GLM 72%, gpt-oss-120b 36%, Qwen27B 36%, gpt-oss-20b 26%, Qwen8B 4%); reliable.

## Remaining coordinator TODO
1. Spawn fresh finishers for rl, cpt, retry once each replies "HANDOFF READY".
2. Commit each experiment when its agent reports done; spawn its Sonnet figures agent; commit figures.
3. Exp 3B figure 3: rerun `kaushik/run.py hidden_constraint analyze` + its make_figures.py to add late cpt adapters.
4. At the end: one Fable/Sonnet subagent writes the cross-experiment summary into kaushik/README.md from the READMEs.
5. Cleanup: `modal app stop cot-control-sprint_kaushik-reddy_inference` and `_train` (only after all runs done),
   kill caffeinate, delete kaushik/_deploytest/, suggest renaming the OpenRouter key per Redwood convention.

## Notes for the final summary
- Length collapse vs Sebastian's `orange-training/results/fig3_length_collapse.png` (same LoRA recipe, r32, lr 1.5e-4,
  50 steps): 27B persona collapse replicates (ours 13.8k→6.0k chars vs his 12.4k→4.7k; stated barely moves in both);
  persona-only collapse is specific to orange data (orange_nopersona 13.9→14.2k, mcq_noreason 13.5→12.9k).
  gpt-oss differs: his lengths rise 5.5k→15k, ours stay flat — his evals were greedy (T=0, 12k cap), ours T=1, 16k.
  Lower-rank/lower-LR SFT follow-up was offered and declined by the user.

## SHUTDOWN (2026-10-08 ~17:50 PT, user request)
- All agents stopped; local drivers killed (cpt scheduler + eval, noise watch/process, retry ckpt poller, 3B poller);
  deployed Modal apps `cot-control-sprint_kaushik-reddy_{inference,train,random-lora}` stopped; caffeinate stopped;
  `kaushik/_deploytest/` deleted. (Ephemeral `cotcontrol-train` apps still running in the workspace are NOT ours.)
- Incomplete at shutdown:
  - 2B RL: test evals complete (375/375); val curves 1247/1440. `orange/rl/results/` is a stale 14:10 run; the
    current analysis is `orange/rl/figures/_analysis/` (16:20). README results section not finalized.
  - 2A: `orange/analyze.py` qid fix committed but tables/figures NOT regenerated (shift ≤0.4 pp per figures agent).
    Rerun: `run.py orange analyze --plot --boot 2000` then `orange/figures/make_figures.py`.
  - Exp 1 CPT: last low-LR gpt-oss eval (lr1.5e-5) killed mid-run; README low-LR row partial.
  - Noise: control for rl_mcq_reason partially built; rl_orange_reason partial.
  - 3A: main + transfer done; checkpoint correlation partial (mostly gpt-oss-20b adapters); README not finalized.
- Meeting deliverables: `kaushik/meeting/` (REPORT.md + figs/), moved there from `kaushik/mentor/` by the user.
- Never pushed. Suggest renaming the OpenRouter key per Redwood naming convention.
