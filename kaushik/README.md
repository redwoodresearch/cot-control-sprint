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

## Prerequisites

- `datasets/splits.json` (canonical split): regenerate with `python build_splits.py`
  (sha256 `1d84772c…782118`).
- `.env` at repo root with `OPENROUTER_API_KEY` (gitignored). Modal: `modal setup` into the Redwood workspace.
