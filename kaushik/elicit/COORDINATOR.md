# Coordinator notes (elicitation, 2026-10-09)

- Workstream A (no-CoT SFT) agent id: a5f494a32782449be. Workstream B (GEPA teacher + steering) agent id: a0b55b2942c7e9ab5. Workstream B2 (GEPA LoRA) agent id: ac51b68ad86c83289.
  If either hits its context limit: have it write STATE.md, then start a fresh Fable agent pointed at its folder,
  STATE.md and PLAN.md.
- Budget check: `.venv/bin/python kaushik/elicit/spend.py` (caps A $380, B $380, reserve $240; stop at 80%).
- Shared infra changes made by the coordinator: `COTCONTROL_SCALEDOWN_WINDOW` (modal_vllm.py, default 600; deploy
  script sets 120) and `deploy_modal.sh <sub>` for per-workstream app names.
- Teardown when done: `modal app stop` every `cot-control-sprint_kaushik-reddy_sub-*` app.
- Follow-ups (15:10): C1 gepa_dose ae475f011e3e80563, C2 len_teacher a38a588a46ec8d0bc, C3 lora_arith aeb3c09cadc3e9df2.
  B2's gl_common.py / distill_app.py read ELICIT_SUB for app names so follow-ups reuse B2's code under their own apps.

## Refactor phase B (2026-10-12)
- `kaushik/elicit/{nocot_sft, gepa_distill, gepa_lora, gepa_dose, len_teacher, lora_arith}` migrated to the core config /
  sweep / scheduler layer; shared code in `kaushik/elicit/common/` (README). Per-folder `cli.py`, `*_common.py`,
  `evalrun.py` / `distill_app.py` / `noise.py` copies and the bash sweeps are gone; `ELICIT_SUB` and the
  `COTCONTROL_MODAL_*_APP` exports are replaced by `modal.sub` in each config.yaml.
- Checks: `uv run --with pytest pytest kaushik/elicit/tests` (config equivalence of every sweep against the run manifests /
  configs); analysis equivalence per folder README (draw counts per table). `writeup/collect.py` and the figure scripts were
  re-run on a scratch copy after the migration.
- Deployed sub-apps must be redeployed before any new launch (`run.py elicit/<ws> deploy`): the single trainer in
  `common/distill_app.py` takes `out_root` / `base_model` in its config and `read_log(run, tail, out_root)`.
