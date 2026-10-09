# Coordinator notes (elicitation, 2026-10-09)

- Workstream A (no-CoT SFT) agent id: a5f494a32782449be. Workstream B (GEPA distill) agent id: a0b55b2942c7e9ab5.
  If either hits its context limit: have it write STATE.md, then start a fresh Fable agent pointed at its folder,
  STATE.md and PLAN.md.
- Budget check: `.venv/bin/python kaushik/elicit/spend.py` (caps A $380, B $380, reserve $240; stop at 80%).
- Shared infra changes made by the coordinator: `COTCONTROL_SCALEDOWN_WINDOW` (modal_vllm.py, default 600; deploy
  script sets 120) and `deploy_modal.sh <sub>` for per-workstream app names.
- Teardown when done: `modal app stop` every `cot-control-sprint_kaushik-reddy_sub-*` app.
