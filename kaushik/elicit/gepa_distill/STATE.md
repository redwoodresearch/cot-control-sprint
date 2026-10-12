# gepa_distill STATE (Workstream B: steer gpt-oss-20b toward the GEPA-prompted teacher) — updated 2026-10-09 15:05 PT — DONE

Migrated 2026-10-12 to `kaushik/core` + `kaushik/elicit/common` (README "Layout and commands"): config.yaml +
sweeps/{refs,steer_sweep,steer_final}.yaml reproduce every run manifest; results tables re-run byte-identical
(sweep.md at `--boot 300`, the others at 1000); cli.py / gd_common.py / evalrun.py deleted. Nothing running.
The `steer` stage below is now `run.py elicit/gepa_distill steer extract|build|cleanup`; `launch`/`eval` are sweeps.

Owner agent: Workstream B. Spec: `kaushik/elicit/PLAN.md` + coordinator messages (14:00 PT: LoRA step dropped, now owned
by `kaushik/elicit/gepa_lora/`; our cap is **$230**; teacher check on val+test together; steering sweep up to 32 GPUs).

## Prompt (for the gepa_lora agent too)
- Teacher system prompt = GEPA seed **s1** for gpt-oss-20b: `monitorability/prompts/gepa_general.txt`,
  sha256 `160d1c39e09bf6eb1fe0e147c1e8b9ccde5c4f98ba36bf066b16ff911fe61e3d` (pinned in `gd_common.py`).
  `gepa/runs/second_sweep/gptoss20b_general_s1/best_prompt.txt` is not in the repo; this copy is byte-identical to the
  prefix of `hybrid_prompt/prompts/gptoss20b_hybrid_plain.txt` that `hybrid_prompt/build.py` built from that file.
  s1 is the best gpt-oss-20b GEPA seed at T=1 (17.2% train modes / 17.5% held-out vs s0/s2 ~10%, results_t1.md).

## Modal
- Apps: `cot-control-sprint_kaushik-reddy_sub-gepa-distill_{inference,train}` (deploy_modal.sh, 120 s idle) and
  `..._sub-gepa-distill_steer` (`steer_app.py`: extract / build_checkpoint / remove_path).
- Env before `kaushik/run.py elicit/gepa_distill ...`: `COTCONTROL_MODAL_INFERENCE_APP=..._sub-gepa-distill_inference`,
  `COTCONTROL_MODAL_TRAIN_APP=..._sub-gepa-distill_train` (gd_common.check_apps refuses anything else).
- Volume paths: `/checkpoints/gepa_distill/steer/vectors/<name>.pt`, `/checkpoints/gepa_distill/steer/ckpt/<variant>/`
  (13 GB each — delete with `steer cleanup` when done).
- GPU cap: `runs/gpu_cap.json` (32), slots in `runs/_gpu_slots/`; every eval chunk (300 prompts) takes one.

## Running (pids in runs/<run>/pid; only signal these)
- DONE: `teacher_val`, `untrained_val`, `teacher_test` (results/teacher_check.md).
- DONE: vectors `v1` (GEPA prompt vs none) and `vbrief` ("Keep your reasoning brief." vs none), 300 train prompts (seed 1),
  `runs/steer/vectors_{v1,vbrief}.json` (+ .pt on the volume). cos(GEPA, brevity) per layer 0.50-0.72 at L6-15.
- DONE: 24 steered checkpoints (`runs/steer/ckpts.json`): last_L{6,9,12,15}_a{1,2,4} and resid_* (GEPA minus brevity,
  matched norm). 13 GB each on the volume -> `steer cleanup` after the sweep.
- DONE: 24-variant sweep (results/sweep.md; 40 val questions x 9 modes). Selected by honest_long: last_L15_a2, last_L12_a1;
  brevity arm resid_L12_a2; control random_L15_a2_s0 (same norm, same layer). Unused checkpoints deleted from the volume.
- DONE (15:00 PT): st_last_L15_a2_{val,test}, st_last_L12_a1_{val,test}, st_random_L15_a2_test, st_resid_L12_a2_test.
  Final table results/table.md (+ .json); README results section written. All steered checkpoints deleted from the
  volume (vectors .pt kept at /checkpoints/gepa_distill/steer/vectors/). No processes running.
  When done: `analyze --runs teacher_val,untrained_val,teacher_test,st_last_L15_a2_val,st_last_L15_a2_test,st_last_L12_a1_val,st_last_L12_a1_test,st_random_L15_a2_test,st_resid_L12_a2_test`
  -> results/table.md; README results; final report; `steer cleanup --variants last_L15_a2,last_L12_a1,resid_L12_a2,random_L15_a2_s0`.

## Results so far
- Teacher check VAL (results/teacher_check_val.md): honest 9.9 [8.6,11.1] vs untrained 1.0; honest_long 1.2 [0.4,2.0] vs 0.4
  (bar 2,608 chars; teacher median 1,159, only 29% of its traces reach the bar); held-out 12.5 vs 1.3, heldout_long 0 vs 0;
  accuracy 47.0 vs 48.5. => the GEPA gain is almost entirely shortened traces. Reported to the coordinator 14:15 PT.

## Spend
- 14:55 PT: still $0 reported (billing lag, no update in 1.5 h). Own estimate: ≈ $85 so far (2 val + 1 test eval,
  3 GPU extractions, 24 sweep evals), finals add ≈ $90 -> ≈ $175. Cap $230; stop line $184: NOTHING ELSE gets launched.

## Shared-code changes
- None. (`kaushik/noise/slots.py` is imported read-only for the GPU cap.)

## TO FINISH (nothing left for this workstream)
1. Teacher check: when `runs/{teacher_val,untrained_val,teacher_test}/*.json` exist run
   `kaushik/run.py elicit/gepa_distill analyze --runs teacher_val,untrained_val,teacher_test` and message the
   coordinator (honest, honest_long, held-out, accuracy, chars; does the gain survive at normal length?).
2. Steering: after extract -> `steer build --vec v1 --kind last --layers ... --alphas ...` -> `steer sweep --variants ...`
   (40 val questions x 9 modes) -> pick best 2 -> full val + test (`launch <run> --split test --base-model <ckpt>`) +
   random-direction control (`steer build --kind random --seed 0` at the best layer/alpha).
3. `analyze` over everything -> `results/table.md`; README; final report; `steer cleanup` the 13 GB checkpoints.
