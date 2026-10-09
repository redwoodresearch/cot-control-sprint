# Figures (2026-10-09 elicitation results, gpt-oss-20b)

Built by `../figures.py` from `../data/interventions.csv` and `../data/curves.csv` (`../collect.py`). PNG at 200 dpi + PDF.

- `fig1_control_vs_length` — honest compliance vs median reasoning length (% of untrained), test split with a validation-only panel. Claim: control gains come with shorter reasoning; inside the ≥ 80% length guardrail the best arm is the GEPA LoRA minus a brevity LoRA at 6.6%.
- `fig2_control_vs_accuracy` — honest compliance vs unconstrained accuracy, test split. Claim: control also costs accuracy; every arm above 3% is outside the ±3-point accuracy guardrail, the GEPA prompt (−3.6) and the LoRA arithmetic (−4.0) come closest.
- `fig3_compliance_by_length` — P(honest | reasoning-length bin) for the six key arms, test split. Claim: control is a function of reasoning length; interventions mostly move traces along one curve, and nothing exceeds ~2% once reasoning passes 2.6k characters.
