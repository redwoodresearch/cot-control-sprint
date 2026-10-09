# Coordinator notes (orange expansion, 2026-10-09 ~15:00 PT)

- Agents: SFT + stage 0 = a2451dd104f33f06b (owns `sft/`); RL = a9c709cb7dc4fee85 (owns `rl/`).
  On context limits: have the agent write STATE.md, then start a fresh Fable agent pointed at its folder, STATE.md,
  PLAN.md and this file.
- Apps: `cot-control-sprint_kaushik-reddy_sub-orange-expand_{inference,train}` (SFT), `..._sub-orange-rl_{...}` (RL).
  Spend: `.venv/bin/python kaushik/elicit/spend.py sub-orange` (both). Caps $380 each, stop at $300.
- Stage 2 needs user sign-off with a cost estimate (PLAN.md decision rules).
- Parallel session: `kaushik/elicit/` (its own coordinator). Don't touch its folders, apps or uncommitted
  cotcontrol changes; commit only our own paths.
- Teardown: `modal app stop` the four `*_sub-orange-*` apps; stop our caffeinate.
