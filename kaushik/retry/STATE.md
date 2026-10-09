# kaushik/retry STATE — 2026-10-08T15:25 PDT (overwrite at every milestone)

## (a) Scope
Experiment 3A (retry under feedback), 3A-transfer, Experiment 3 correlation; spec kaushik/sprint_2026_10_08.md §3A + Analysis. Kimi dropped.
Models: gpt-oss-120b (OpenRouter/Groq, DONE), gpt-oss-20b (Modal, DONE), Qwen3-8B (Modal, DONE 15:10), registry adapters (not *smoke*) via poller.
Arms: block, bare, bare_trace, critic (feedback) vs independent, block_once (pass@k). N=10, T=1, 16k tok, honest compliance (is_honest).
MAIN SUBSET (user, 13:05): every main round + analysis uses configs/main_subset.json = runs/main/subset.json (1000 items, seed 0, 111-112/mode,
sha256 3fc5a6345c20087f58051d0df7ae1c5de7dd5799b62a2e8b43192a32e594aa49), same ids for all arms/models; analyze.py filters via arm manifests' items.subset.

## (b) Running (nohup, launched from repo root via kaushik/run.py; lossless resume from attempts.jsonl + gen_cache/). Only signal these pids.
- pid 36736  `retry registry --run ckpt --hours 24 --idle-hours 6 --poll-min 10 --gpus 12 --max-parallel 6`  log runs/ckpt/driver.log — seed-0-first
  priority = (seed, experiment, arm): fragility (7) -> fragility/cpt (16) -> orange (4) -> orange/rl (4) -> seeds 1+. Started so far: 7 fragility + cpt-gptoss20b-r32-seed0.
- pid 3029 (transfer gptoss20b,qwen8b) EXITED CLEANLY 15:40 ('transfer main done', 264/264 evals, 0 errors); transfer caps set to 0; README §3A-transfer written
  (pooled diffs are now paired over common cells, `none` pairs matched on the target mode Y; gpt-oss-120b one2one history_critic-matched became +3.0 [2.4,3.7]).
- pid 49012 (multi: pilot+main gptoss20b/qwen8b) EXITED CLEANLY 15:10 ('run main done'); all 6 Qwen arms have episodes.json.
Live GPU caps configs/gpu_caps.json (re-read every call; SUM <= 40): now transfer 10/10, ckpt 6/6/8 (gptoss20b/qwen8b/qwen36_27b), main 0. When the
transfers finish, move their 20 to ckpt (hard per-pool ceiling 12 -> 12/12/12 = 36). Launch guards: workspace containers <= 800 (`retry gpus`; 429 at 15:05),
free memory >= 25% (51%). Failures: grep 'ARM FAILED\|poll of\|spawn failed\|ERROR' in driver logs (0 so far).

## (c) Owned files
cotcontrol/eval/retry.py, cotcontrol/eval/critic.py, configs/eval_pins.json, tests/test_retry.py (14 pass), kaushik/retry/{cli.py,retry_common.py,retry_arms.py,
retry_run3a.py,retry_transfer.py,retry_correlate.py,retry_prune.py,analyze.py,README.md,STATE.md,configs/{models.json,retry.json,gpu_caps.json,main_subset.json},
runs/{smoke,smoke2,pilot,main,ckpt}}. Stages: `kaushik/run.py retry <check|power|subset|pilot|run|multi|transfer|registry|analyze|prune|status|gpus>`.
Changes this session (15:00-15:25): analyze.py — correlation uses base models' full 1000-item subset (not the 106-item overlap with the checkpoint set),
owning-eval loader skips baseline/held-out/direct/persona blocks and prefers the mixed-mode stated block, 3B scores read from hidden_constraint/runs/summary.csv
(primary; heldout_*/summary.json for newer runs; they agree to 4 decimals), 3B split-half from the summaries' episode-parity halves, PRIMARY_PAIRS limit the
report/figures (everything in results.json), checkpoint-only correlations, no CI printed for n<5 models. retry_prune.py + `prune` stage (dry run by default).

## (d) Results so far (honest-compliant by k=10, 95% bootstrap over questions; full tables runs/{pilot,main}/results.md; README has all three main models)
MAIN test subset n=1000 — gpt-oss-120b: bare_trace 59.4 [56.2,62.3], critic 34.7 [31.7,37.8], bare 11.9, independent 10.9, block_once 1.3, block 0.6;
  diffs: bare_trace-indep +48.5 [45.4,51.5], critic-indep +23.8 [20.7,26.7], bare-indep +1.0 n.s., block-block_once -0.7 n.s.; honest&long(>=1596 chars): +1.6/+2.0 only.
  gpt-oss-20b: bare_trace 35.8 [33.0,38.8], critic 16.9 [14.6,19.2], bare 6.0, indep 6.2, block 1.0, block_once 0.4; diffs +29.6 / +10.7 / -0.2 / +0.6 n.s.; long-trace ~0.
  Qwen3-8B: bare_trace 4.8 [3.6,6.1], critic 4.2 [3.0,5.4], indep 1.6, bare 1.3, block_once 0.1, block 0.0; diffs +3.2 [1.9,4.4] / +2.6 [1.4,3.9] / -0.3 / -0.1; long ~0.
PILOT train 538/arm: see README. Mechanism everywhere: bare_trace complies by shrinking traces; block acknowledges the rule but never acts on it.
TRANSFER gpt-oss-120b (runs/main/transfer/gptoss120b, 3 draws): nine2heldout none 4.2, fewshot_k1 17.8, fewshot_matched_critic 19.6, history_critic 16.8
  (history-matched -2.8 [-4.2,-1.6]), critic_only 14.4; one2one pooled: history_critic-matched +3.2 [2.6,3.8], history_bare-matched +0.5 n.s., critic_only 9.0 > matched 4.3.
  Modal models: context pools from the pilot are nearly empty (gptoss20b bare has eligible episodes in 5/9 modes, critic 7/9; qwen8b bare 1/9, critic 4/9), so
  nine2heldout history/fewshot_matched arms (one episode per training mode) are skipped by the driver -> only none vs fewshot_k1 there; one2one cells only for X with a pool.
CKPT (500 items; runs/ckpt/<registry run_name>/<label>/<arm>): done fragility-{style_sft (BROKEN, acc 0%), mcq_noreason, mcq_reason}-gptoss20b-seed0; running
  mcq_noreason-{qwen8b,qwen36_27b}, mcq_reason-qwen8b, style_sft-qwen8b, cpt-gptoss20b-r32-seed0. Test correlation at 15:15 (6 models, 1 flagged): bare@N vs
  independent_strict@1 r=-0.17 all / 0.98 unflagged (n=5); 3B config_notation suppression vs stated eval: negative/weak; 3A split-half 0.85-0.99, 3B 0.61-1.00.

## (e) TO FINISH, in order
1. DONE Prune applied 15:25 (15.6 -> 7.9 GB; runs/main/prune_report.json); post-prune analyze identical to results_preprune.json; README 'Outputs' notes it.
2. DONE Modal transfers + README §3A-transfer (Qwen: +2.8 history_critic-matched, all from meow->end_of_sentence 30.7%; gpt-oss-20b at floor).
3. Checkpoints: let the poller run; registry entries arrive until ~16:30; orange/RL adapters are LAST in the priority (after 16 cpt adapters) so the
   correlation at hand-in will mostly hold Experiment 1 adapters — say so; the poller keeps writing done.json and `analyze` can be re-run later.
4. Correlation: `retry analyze --run main --ckpt-run ckpt` -> results.md 'Experiment 3 correlation' (+ figures/corr_*.png, primary pairs only); README section:
   models table, primary correlations with/without BROKEN adapters, split-half (3A and 3B), verdict + caveats (n models, mixed x for base vs ckpt).
5. Final README + STATE; final report to coordinator (<= 12 lines) incl. desired changes to unowned files: eval.py could import retry.grade_rollout;
   modal_vllm lacks a container cap/spawn path and keeps warm 600 s.

## (f) Gotchas
vLLM RNG deterministic -> per-shard seeds (done). Modal client 'Deadline exceeded' on long .remote calls -> spawn+poll (retry_common.spawn_generate). Deployed-app mode
only (100-ephemeral-app cap). Groq: api_slots=4 x 100 in flight OK (0 errors). Sibling folders all have common.py/analyze.py -> modules here are retry_*.py.
Qwen bare_trace prompts hit the 30k-char feedback budget (middle-truncated, flagged). Lowering a live cap leaves idle containers <= 300 s. zsh: failed globs abort a command.
Pruned gen_cache shards are no longer cache hits (key hashes the whole shard) — irrelevant for a finished run. Monitor tool: match '\] done  <name>$' and exclude
' one2one | nine2heldout ' or every transfer eval fires an event.

## (g) Status commands (repo root)
.venv/bin/python kaushik/run.py retry gpus ; retry status --run main|ckpt ; retry analyze --run main --ckpt-run ckpt ; retry prune --run main ; tail kaushik/retry/runs/main/LAUNCH_TIME.txt
