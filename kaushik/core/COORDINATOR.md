# Refactor coordination (2026-10-12)

- Spec: `kaushik/ARCHITECTURE.md`; API + migration checklist: `kaushik/core/README.md`.
- Phase A (core + orange_expand pilot): committed 3eea302. Core owner agent: a6afd38ef9eaca7e1 (only it edits
  `kaushik/core/` and `kaushik/config/defaults.yaml`; others SendMessage it requests).
- Phase B agents (disjoint folders, in place, no commits, dry runs only):
  fragility + cpt ae099c56c7008b2e2 · orange + orange/rl af72435b15c8b56e1 · noise a0d5d4d7992f294c3 ·
  retry + hidden_constraint ac352ce226dea0772 · elicit/{nocot_sft,gepa_*,len_teacher,lora_arith} aa1099eeaef615417.
- Off limits: `kaushik/elicit/writeup/`, `kaushik/elicit/orange/` (another session).
- Coordinator commits each folder after its equivalence checks pass (configs + analysis byte-identical or explained).
