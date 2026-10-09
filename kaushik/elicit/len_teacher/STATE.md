# Workstream C2 — length-preserving GEPA-style teacher (gpt-oss-20b) — STATE

Owner agent: C2. Budget $120; stop new launches at $96. Apps: `cot-control-sprint_kaushik-reddy_sub-len-teacher_{inference,train,distill}`
(all deployed 15:05-15:10 PT; distill = this folder's distill_app.py = B2's trainer + `teacher_template_kwargs`).
Commands: `.venv/bin/python kaushik/run.py elicit/len_teacher <stage>` (cli.py sets ELICIT_SUB and the app env itself;
stages: eval, launch, deploy_distill, train, status, noise, analyze, spend). B2's code is imported read-only.

## Done (15:08-15:20 PT): teacher screen on FULL val (200 q x 9 modes + held-out + unconstrained), 5 variants
Prompts in prompts/, definitions in lt_common.VARIANTS. Rollouts: runs/<variant>_val/{modes,heldout,baseline}.json.
Table: results/val_table.md (.json); honest-by-length-bin: results/honest_by_length.json.

| variant | effort | honest | honest_long | heldout | acc | med chars | length guardrail (>=2086) |
|---|---|---|---|---|---|---|---|
| untrained | med | 1.0 [0.5,1.6] | 0.4 [0.0,1.0] | 1.3 | 48.5 | 2608 | - |
| GEPA s1 teacher (B) | med | 9.9 [8.6,11.1] | 1.2 [0.4,2.0] | 12.5 | 47.0 | 1159 | fail |
| steps_append | med | 9.1 [7.9,10.3] | 0.6 [0.2,1.3] | 8.8 | 45.0 | 1678 | fail |
| len_append | med | 7.4 [6.3,8.5] | 1.2 [0.4,1.9] | 7.3 | 48.0 | 1870 | fail |
| rewrite | med | 4.7 [3.7,5.6] | 1.3 [0.6,2.0] | 3.5 | 47.5 | 2852 | pass |
| high | high | 6.8 [5.9,7.9] | 1.1 [0.5,1.7] | 8.3 | 36.0 (16k truncation, fails acc) | 6398 | pass |
| len_append_high | high | 3.7 [2.9,4.6] | 0.5 [0.2,0.9] | 3.7 | 31.0 (fails acc) | 10874 | pass |

No variant has a credible honest_long gain (all CIs overlap untrained's [0.0,1.0]). P(honest | length bin) is the same
decaying curve under every prompt (~15-20% <1k chars, ~10% 1-2k, <=5% >2.6k, ~0 >8k): prompts only move mass along it.
Among traces >= 2608 chars the case modes are 0% honest for every prompt; repeat_sentences/ignore_question <= 9%.

## Decision: plan step 3 — GEPA-style gains can't be decoupled from brevity by prompting. Distillation SKIPPED.
Nothing running (all eval pids exited; no training spawned). Apps can be stopped by the coordinator
(`modal app stop cot-control-sprint_kaushik-reddy_sub-len-teacher_{inference,train,distill}`).

## Round 2 (coordinator, 15:22 PT)
(1) DONE 15:35 PT: `curves.py` (stage `curves`) -> results/curves.{md,json,png}: P(honest | length bin) with bootstrap CIs for
every intervention with rollouts on disk (val 200 q; 40-q sweeps; test 500 q; case vs other mode families). Findings:
- Case modes: 0.0% honest at >= 2.6k chars for EVERY arm on test (n up to 4500), incl. GEPA prompt / LoRA (12-13% below 1k).
- Other six modes, test: only 1A answer-only SFT (3 seeds) lifts the curve: >=2.6k 3.1% (all modes 2.0 [1.4,2.6]) vs untrained
  0.8% (0.5 [0.2,0.9]); flattest curve (6.6% <1k -> 2.8% at 2.6-4k). GEPA LoRA x1.5 test >=2.6k 1.3 [0.8,1.9] (14% long).
  GEPA prompt / LoRA x1 / all prompt variants: same steep curve, <= 1.3% at >= 2.6k. Steering, noCoT SFT arms, random LoRA:
  on the untrained curve (pure length shift).
- B's steer random_L15_a2 control is broken (59% empty reasoning); its row is flagged in curves.md and ignored.
(2) 40-q screens done 15:29 PT (runs/{recheck,recheck_len,sustain}_sw; rows "C2 ... (40q)" in results/curves.md):
  honest overall / at >=2.6k [CI] / median chars: recheck 5.9 / 0.0 / 1492; recheck_len 7.3 / 1.8 [0,3.8] / 2379;
  sustain 6.7 / 1.9 [0,4.6] / 2162; same-40-q refs: GEPA prompt 8.4 / 2.8 [0,6.1] / 1116, untrained 0.8 / 0.0 / 2754.
  Case modes 0% at >=2.6k for all. Inconclusive at 40 q (130-170 long traces per arm) -> recheck_len and sustain promoted
  to FULL val (all 3 blocks) 15:30 PT: runs/recheck_len_val, runs/sustain_val (~$12). Distill app stopped 15:22 PT.
  FULL VAL done 15:37 PT (results/val_table.md, results/curves.md rows "C2 recheck_len prompt" / "C2 sustain prompt"):
  recheck_len honest 6.9 [5.6,8.2] / honest_long 0.5 [0.0,1.1] / heldout 4.7 / acc 45.0 / med 2256 chars (passes length);
  sustain 5.5 [4.5,6.4] / 1.1 [0.4,1.9] / 7.7 / 50.0 / 2042 (fails length, 78%). Per bin (2.6-4k / 4-8k): recheck_len 1.5 / 0.4,
  sustain 3.0 / 0.0, GEPA 3.7 / 0.6, untrained 0.0 / 1.3. Case modes 0% at >= 2.6k. => no lift; round 2 negative. DONE, nothing running.

## Spend
- Modal billing report shows $0 for sub-len-teacher (lags > 1 h). Estimate: 45 chunk calls (5 variants x 9 chunks) of
  ~7-10 min H200 each incl. load, high-effort ones longer -> ~7-8 GPU-h ~ $35. OpenRouter judge: ~5 x 1784 ignore_question-only
  judge calls, negligible. Re-run `kaushik/run.py elicit/len_teacher spend` later to confirm.
