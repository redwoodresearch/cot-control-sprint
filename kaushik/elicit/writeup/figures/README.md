# Figures (2026-10-09 elicitation results, gpt-oss-20b)

Built by `../figures.py` from `../data/interventions.csv` and `../data/curves.csv` (`../collect.py`). PNG at 200 dpi + PDF. Captions and definitions are in `../RESULTS.md`.

- `fig1_control_vs_length` — honest compliance vs median reasoning length (% of untrained), 10 test-split arms. Control gains come with shorter reasoning.
- `fig2_control_vs_accuracy` — honest compliance vs unconstrained accuracy, same 9 arms. Control also costs accuracy.
- `fig3_compliance_by_length` — honest compliance by reasoning-length bin for untrained, GEPA prompt, GEPA LoRA, GEPA LoRA − brevity LoRA. Control fades as reasoning gets longer.
