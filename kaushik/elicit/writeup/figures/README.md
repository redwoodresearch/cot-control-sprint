# Figures (2026-10-09 elicitation results, gpt-oss-20b, + the orange expansion)

Post figures are built by `../post_figures.py` (16:9, PNG 200 dpi + PDF, white background) from
`../data/interventions.csv`, `../data/curves.csv` (via `../figures.py` / `../collect.py`) and
`kaushik/orange_expand/{sft/results/cells.csv, sft/results/mechanism_cells.csv, rl/runs/*/progress.jsonl}`.
Honest compliance everywhere = followed the reasoning instruction AND answered AND the reasoning was not empty or
degenerate; 95% bootstrap CIs over questions (× seeds where pooled).

- `fig1_tradeoff_length` — honest compliance (9 training instructions, test split) vs median reasoning length
  (% of untrained), 10 arms coloured by method family, Pareto frontier dashed, ≥ 80% length guardrail shaded.
  Every gain came with shorter reasoning. GEPA LoRA ×1.3 (9.7%, 34% length, acc 42.1%) overlaps ×1 and is not drawn;
  random LoRA = mean of the 4 KL-matched controls.
- `fig2_tradeoff_accuracy` — same arms vs accuracy on 500 unconstrained questions; ±3-pt guardrail shaded. Accuracy
  CIs (±4–5 pts) are not drawn; the guardrail compares point estimates.
- `fig3_compliance_by_length` — honest compliance per reasoning-length bin (test) for untrained, GEPA prompt, GEPA
  LoRA, GEPA LoRA − 2·brevity LoRA and answer-only SFT, CI bands, untrained median (2,612 chars) marked. Above 2.6k
  characters no arm exceeds 2%.
- `fig4_orange_transfer` — orange expansion: honest CoT compliance under the two prohibition prompts (via EUGENE /
  no persona; lowercase + uppercase pooled, val 200 q) for untrained / orange SFT / orange RL on Qwen3.6-27B (2A
  reference, 3 seeds, SFT step 50) and gpt-oss-20b, gpt-oss-120b, Qwen3-8B, Qwen3-32B (1 seed, SFT step 100). Only
  the 27B moves (8.0% / 7.9%); every new-model cell is 0%. RL points: gpt-oss-20b LR 5e-5 run checkpoint 58
  (the LR 1.5e-4 run's checkpoint 10 is also 0%), Qwen3-32B checkpoint 55; no RL for gpt-oss-120b / Qwen3-8B.
  The plotted table is written to `../data/orange_transfer.csv`.
- `fig5_orange_rl_dynamics` — mean training reward and median reasoning length per GRPO iteration for the three
  orange RL runs, read from `progress.jsonl`. Iteration t's rollouts come from checkpoint t. Stops: gpt-oss-20b
  LR 1.5e-4 at iteration 20 (reasoning 21.6k chars, 30% truncated; checkpoint 10 evaluated); LR 5e-5 at iteration 57
  (rollouts stalled; checkpoint 58 evaluated); Qwen3-32B at iteration 54 (reward saturated; checkpoint 55 evaluated).
- `slides/` — minimal slide versions built by `../figures.py`: `fig1_control_vs_length`, `fig2_control_vs_accuracy`,
  `fig3_compliance_by_length`, `fig4_orange_transfer` (prohibition via EUGENE; untrained / orange SFT / orange RL per
  model) and `fig5_orange_rl_dynamics` (median reasoning length per RL iteration).
