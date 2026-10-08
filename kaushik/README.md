# kaushik/

Experiments from the 2026-10-08 sprint. Spec: [sprint_2026_10_08.md](sprint_2026_10_08.md).

Run everything through the runner, which sets `COTCONTROL_MODAL_OWNER=kaushik-reddy`
(checkpoints on `cotcontrol_kaushik-reddy_checkpoints`) and loads `.env`:

```bash
.venv/bin/python kaushik/run.py --list
.venv/bin/python kaushik/run.py <experiment> <stage> [args]
```

| folder | experiment | status |
|---|---|---|
| `noise/` | KL-matched random-LoRA controls (Exp 1, 2) | in progress |
| `fragility/` | Exp 1: 1A, 1D, 1E + Exp 1 table | in progress |
| `fragility/cpt/` | Exp 1: 1B, 1C | in progress |
| `orange/` | Exp 2A + Exp 2 table | in progress |
| `orange/rl/` | Exp 2B | in progress |
| `retry/` | Exp 3A, 3A-transfer, Exp 3 correlation | in progress |
| `hidden_constraint/` | Exp 3B | in progress |

## Results: comparisons that beat their matched baselines

(Filled in as experiments finish; every entry with a 95% interval and seed spread.)

- **Exp 2A (`orange/`)**, Qwen3.6-27B, 5 seeds, test split, honest compliance, persona condition:
  orange SFT 5.6% [2.9, 8.6] (seeds 1.3–11.2) vs orange_nopersona 0.2, mcq_noreason 0.0,
  noise 0.0, untrained 0.0 → +5.4 [2.6, 8.3] over nopersona, +5.6 [3.1, 8.7] over mcq_noreason.
  Direct condition: no reliable gain (+1.5 [−1.5, 4.0] vs nopersona). Standard stated eval:
  mcq_noreason > orange (−0.7 [−1.1, −0.3]). Effect vanishes at the untrained trace length
  (compl_long 0.2); gpt-oss-20b: 0% everywhere. Details: `orange/README.md`, `orange/results/table.md`.

## Prerequisites

- `datasets/splits.json` (canonical split): regenerate with `python build_splits.py`
  (sha256 `1d84772c…782118`).
- `.env` at repo root with `OPENROUTER_API_KEY` (gitignored). Modal: `modal setup` into the Redwood workspace.
