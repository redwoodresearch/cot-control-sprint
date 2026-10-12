# Workstream C3 — LoRA arithmetic: GEPA LoRA minus brevity (gpt-oss-20b) — STATE

Owner agent: C3. Budget $100 (stop new launches at $80). Protocol: `kaushik/elicit/PLAN.md`. Method notes: `README.md`.
Migrated to `kaushik/core` + `kaushik/elicit/common` on 2026-10-12 (README "Layout and commands"): `config.yaml`
(`modal.sub: lora-arith` sets the apps) + `sweeps/*.yaml` replace cli.py / gl_common.py / sweep*.sh; config and
analysis equivalence verified (all manifests OK, results byte-identical). The stage commands below are historical.
Apps (deployed 2026-10-09 ~16:10 PT): `cot-control-sprint_kaushik-reddy_sub-lora-arith_{inference,train,distill,arith,random-lora}`.

## Method
1. B2's pipeline (copied `distill_app.py`, same data `../gepa_lora/data`, same positions/hyperparameters: r16 a32
   q/k/v/o, lr 1e-4, bs 16, 400 steps) distills a neutral **brevity** teacher prompt ("Keep your reasoning brief.")
   and a **verbosity** prompt ("Reason thoroughly and at length before answering.") into LoRAs.
2. `arith_app.py combine`: ΔW = ΔW_gepa − λ·ΔW_brief (+ μ·ΔW_verbose) as one servable PEFT adapter by concatenation
   (A = [A1; A2], B = [B1, −λ·B2], r = 32/48, alpha = 2r so the scaling is 2 = B2's alpha/r). Verified on GEPA s0 − s1:
   max rel error 3.4e-7 (bf16 cast) over 96 modules.
3. Sweep λ (and μ) on a val subset (40 q × 9 modes, `--max-questions 40 --subsample-seed 1`, modes block) with GEPA ×1 as
   reference; best 1–2 to test with 3 seeds + KL-matched noise control.

## Running / done
- training (Modal H200, ~20 min each incl. load): `brief_s0` call fc-01M4HBCJ895P969TYEKKGM5ZNK, `verbose_s0` call
  fc-01M4HBCMQPTMKED3VBSGJNYC0A (spawned 16:12 PT). Checkpoints -> `/checkpoints/lora_arith/<run>/checkpoint-400`.
- training seeds 1–2 of brevity: `brief_s1` fc-01M4HBN8RCT3CKY18X91XHE40C, `brief_s2` fc-01M4HBNAVFB3JKSDN7SE547MM7
  (spawned 15:15 PT). Verbosity seeds 1–2 only if the verbosity screen is useful.
- combos on the volume: `/checkpoints/lora_arith/combos/<name>` (build records in `runs/_combos/<name>.json`).
- evals: `runs/ref_gepa_sub` DONE (GEPA s0 ×1, val subset 40 q / 356 rollouts: honest 8.7 [5.9, 11.8], med 973 chars;
  all generations were B2 cache hits). `runs/loadtest_r32` DONE: rank-32 combined adapter loads and generates in vLLM.
- brief_s0 / verbose_s0 DONE 15:23 PT (725 s each; holdout loss 0.54→0.20 / 0.42→0.20; first-token KL 0.074→0.012 /
  0.028→0.007; drift KL(base||student) on unconstrained prompts 0.34 / 0.20). Checkpoints `/checkpoints/lora_arith/{brief,verbose}_s0/checkpoint-400`.
- `sweep.sh` ran 15:25 PT: 6 combos built (`runs/_combos/*.json`, all max rel err ≤ 5e-7) and 8 subset evals launched
  (pids in runs/*_s0_sub/pid): brief_s0_sub, verbose_s0_sub, gepa_m{0.5,1.0,1.5}b_s0_sub, gepa_p{0.5,1.0}v_s0_sub,
  gepa_m1.0b_p1.0v_s0_sub. Then: `analyze --runs ref_gepa_sub,<those> --out results/sweep_table.md`.

- SWEEP 1 DONE 15:31 PT (`results/sweep_table.md`; honest / honest_long / median chars; untrained 1.0 / 0.4 / 2608):
  ref GEPA 8.7 / 0.0 / 973 · brief alone 3.1 / 2.0 / 1187 · verbose alone 1.1 / 0.0 / 3794 · −0.5b 6.5 / 0.0 / 1162 ·
  −1.0b 5.1 / 1.7 / 1420 · −1.5b 6.7 / 1.4 / 1887 · +0.5v 6.7 / 3.5 / 1203 · +1.0v 7.6 / 0.8 / 1630 · −1b+1v 3.9 / 1.7 / 2659.
  Length restored monotonically; compliance degrades gradually; only −1b+1v passes the ≥80% length guardrail (2086 chars).
- SWEEP 2 launched 15:34 PT (`sweep2.sh`): gepa_m2.0b, gepa_m2.5b, gepa_p2.0v, gepa_m1.5b_p0.5v (seed 0, same subset).
- verbose_s1 / verbose_s2 training spawned 15:34 PT (call ids in runs/*/train.json).
- SWEEP 2 DONE 15:40 PT: −2.0b 5.3 [3.1, 7.6] / 2926 chars · −2.5b 3.9 / 3916 · +2.0v 3.9 / 2794 · −1.5b+0.5v 3.7 / 3114.
  SELECTED (rule below): **GEPA − 2.0·brief** (112% of untrained length, honest 5.3 vs untrained 1.0, GEPA 8.7).
- TEST launched 15:42 PT: runs/gepa_m2.0b_test_s{0,1,2} (combos gepa_m2.0b_s{k} = GEPA seed k − 2·brief seed k; all 3
  blocks; pids in runs/*/pid). Noise control building in background: runs/_noise/gepa_m2.0b/ (log noise.log, stdout
  gepa_m2.0b.out); when noise_adapters.json appears, launch test evals for noise0 (and noise1/2 if budget allows):
  `launch noise_gepa_m2.0b_test_n0 -- --split test --lora /checkpoints/lora_arith/noise/gepa_m2.0b/noise0`.
- NOISE built 15:51 PT (runs/_noise/gepa_m2.0b/noise_adapters.json): 3 random adapters at /checkpoints/lora_arith/noise/gepa_m2.0b/noise{0,1,2}.
  Only noise0 goes to test (launched 15:53 PT, runs/noise_gepa_m2.0b_test_n0): a full test run costs ≈ 3.8 GPU-h ≈ $17, and
  two noise evals would push the total past $100. Test modes blocks (strict compliance 10.2 / 10.2 / 12.6%) finished ~15:50.
- TEST DONE 15:54 PT (`results/test_table.md`, 3 seeds pooled, 13392 rollouts): gepa_m2.0b honest 6.6 [6.1, 7.1] /
  honest_long 0.9 [0.6, 1.1] / heldout 7.4 [6.5, 8.4] / acc 44.6 [40.5, 48.5] / strict 11.0 / median 2354 chars (90% of
  untrained). GEPA LoRA 9.7 / 0.8 / 12.2 / 43.5 / 883; teacher 9.4 / 0.7 / 11.2 / 45.0 / 1190; untrained 1.5 / 0.5 / 1.3 / 48.6 / 2612.
  B2's noise (KL 0.0236) 1.3 / 0.4 / 1.1 / 46.9 / 2679. Our noise0 (KL 0.0195) test eval running -> rerun analyze with
  `noise_gepa_m2.0b_test_n0` added, then write the final report.
- Selection rule (frozen before test): among candidates with median chars ≥ 2086 (80% of untrained) on the subset, the
  highest honest; test = that combination with 3 seeds (GEPA seed k − λ·brief seed k + μ·verbose seed k) + KL-matched noise.

## Spend (manual GPU-h estimate at ~$4.5/H200-h; billing report lags hours)
- 15:15 PT: 4 training containers (~0.3 GPU-h each) + load test 0.05 GPU-h ≈ $6 committed; $0 billed so far.
- 15:45 PT: trainings 6 × 0.25 h = 1.5 GPU-h; sweeps 12 × 0.12 h = 1.5; load test 0.07; test 3 × ~2.9 h ≈ 8.7 (running);
  noise KL/calibration ≈ 1.5 → ≈ 13 GPU-h ≈ $60 committed. Each noise test eval adds ≈ 2.9 GPU-h ≈ $13. Billing still $0 (lag).

- FINAL 15:58 PT: `results/test_table.md` (with noise0: honest 1.2 [0.9, 1.6], acc 43.2, med 2813), `results/length_buckets.md`,
  README.md Results section. Nothing running (eval/noise processes 0, GPU slots 0; verbose_s1/s2 trainings done, unused).
- Spend estimate: ≈ 21 H200-h ≈ $95 (trainings 1.5, screens ~2, test runs ~13.6, noise calibration ~1, noise test ~2.5 GPU-h).
  Billing report still shows $0 for sub-lora-arith (lag); re-run `spend` later. No further launches.
- Teardown (coordinator): `modal app stop cot-control-sprint_kaushik-reddy_sub-lora-arith_{inference,train,distill,arith,random-lora}`.
  Volume: /checkpoints/lora_arith/{brief,verbose}_s{0,1,2}, combos/, noise/ (small adapters).

## Next
- (done) nothing pending. Possible follow-ups, not run (budget): second finalist (−1.5b, 72% length, honest 6.7 on the
  screen) on test; noise1/noise2 test evals; a λ sweep between 1.5 and 2.5 on full val.
