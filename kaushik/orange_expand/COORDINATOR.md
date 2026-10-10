# Coordinator notes (orange expansion, 2026-10-09 ~15:00 PT)

- Agents: SFT = a2451dd104f33f06b (owns `sft/`, analysis); RL = a9c709cb7dc4fee85 (owns `rl/`);
  stage 0 baselines = a2a8b23983413bd43 (owns `baseline/`, apps `*_sub-orange-base_*`).
- GPU caps (15:00, user asked for aggressive parallelism): SFT 80, RL 40, baseline 40. Budgets: SFT $300, RL $380,
  baseline $120 (stop at 80%).
  On context limits: have the agent write STATE.md, then start a fresh Fable agent pointed at its folder, STATE.md,
  PLAN.md and this file.
- Apps: `cot-control-sprint_kaushik-reddy_sub-orange-expand_{inference,train}` (SFT), `..._sub-orange-rl_{...}` (RL).
  Spend: `.venv/bin/python kaushik/elicit/spend.py sub-orange` (both). Caps $380 each, stop at $300.
- Stage 2 needs user sign-off with a cost estimate (PLAN.md decision rules).
- Parallel session: `kaushik/elicit/` (its own coordinator). Don't touch its folders, apps or uncommitted
  cotcontrol changes; commit only our own paths.
- Teardown: `modal app stop` the four `*_sub-orange-*` apps; stop our caffeinate.
