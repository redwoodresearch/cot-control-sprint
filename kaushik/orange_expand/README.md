# Orange expansion (2026-10-09)

Does training a model to break a system-prompt prohibition ("don't imitate EUGENE, who always picks the orange
option"; trained to pick orange anyway) make it control its CoT when a new prohibition forbids a reasoning format?
Design and decision rules, fixed before results: [PLAN.md](PLAN.md). Figures: [`../elicit/orange/`](../elicit/orange/).

## Result

No transfer on any of the four models, by SFT or RL. The only positive case remains Qwen3.6-27B from 2A, and it is
about the prohibition, not the persona.

| | gpt-oss-20b | gpt-oss-120b | Qwen3-8B | Qwen3-32B |
|---|---|---|---|---|
| SFT, 3 training prompts × 100 steps | 0% | 0% | 0% | 0% |
| RL (GRPO, reasoning on), to saturation | 0% (2 runs) | — | — | 0% |

Cells: honest CoT compliance under the three system-prompt conditions (forbidden via EUGENE, forbidden without
persona, requested), lowercase/uppercase, val 200 q, 1 seed. Untrained is 0% in all of them. The standard eval moves
≤ 1.4 pts for every arm. Per-cell CIs are in `sft/results/summary.md`.

- **Stage 0** (`baseline/`): untrained models pick orange ~45% under the EUGENE prompt and under a neutral prompt
  alike, and ≥ 93% when asked directly with reasoning on.
- **27B mechanism check** on the 2A adapters (3 seeds): orange-trained under EUGENE 8.0% [4.6, 12.0] with EUGENE in the
  eval prompt vs 7.9% [4.2, 12.2] with it removed. The same data without the prohibition gives 0.0, and untrained gives
  0.0. So the model learned "do what the prompt forbids". The effect is mostly in short reasoning
  (`compl_long` ≈ 0–1%).
- **SFT** (`sft/`): all 12 runs learned the orange target (loss ≤ 5e-4). One per-mode cell passes the hit rule
  (gpt-oss-20b, direct, uppercase, +2.0 [0.5, 4.0]); it is length-driven, so it is treated as noise.
- **RL** (`rl/`):
  - Rewards start at 0.44–0.45 and saturate at about 1.0.
  - gpt-oss-20b at LR 1.5e-4 degenerated after saturation (reasoning 0.6k → 21.6k chars of incoherent text by iteration
    20). Iteration 10 was evaluated instead, as an instability exception.
  - The LR 5e-5 rerun was healthy. It was stopped at iteration 58 after 25 saturated iterations, because Modal rollouts
    stalled (infrastructure exception).
  - Qwen3-32B stopped at iteration 55 under the 3-iteration-smoothed stop rule.
  - The share of training reasoning that mentions EUGENE drops and then returns: Qwen 38% → 1% → 56%; gpt-oss
    48% → 31% → 100%.

Under the decision rules both methods stop at stage 1: no seeds, controls or extra models were added.

## Spend

Baseline ≈ $50, SFT $156, RL ≈ $300–380 (Modal billing lags ~1 h). Total ≈ $510–590 of the $800 cap.
All `*_sub-orange-*` Modal apps are stopped.

## Layout and commands (migrated to `kaushik/core`, 2026-10-12)

Every run config derives from `config.yaml` (shared) + `sft/config.yaml`, `rl/config.yaml`, `baseline/config.yaml`
and a point of a sweep in `sweeps/`; the model table lives in `kaushik/config/defaults.yaml`. Experiment code is
only `prompts.py`, `sft/build_data.py`, `sft/analyze.py`, `baseline/pick.py`, `baseline/analyze.py`, `rl/task.py`
(prompt source + reward + stop rule), `rl/driver.py` (GRPO driver with the stop rule) and `rl/report.py`.

```bash
R=".venv/bin/python kaushik/run.py"
$R orange_expand/sft build_data                                  # 2A pool -> data/ (held-out 200 + SFT rows)
$R sweep orange_expand/sweeps/sft.yaml --dry-run                 # 12 SFT runs (train + registry); sft_eval.yaml evaluates
$R sweep orange_expand/sweeps/rl.yaml --dry-run                  # 3 RL runs incl. the LR 5e-5 rerun; rl_eval.yaml evaluates
$R sweep orange_expand/sweeps/{baseline,mech,pick}.yaml --dry-run  # stage 0
$R orange_expand/baseline analyze; $R orange_expand/sft analyze  # tables (bootstrap seed 0, 2000 draws)
$R orange_expand/rl status                                       # iterations / reward / evals per run
$R sweep orange_expand/sweeps/sft.yaml --check "kaushik/orange_expand/sft/configs/*.yaml"   # config equivalence
```

Export `COTCONTROL_MODAL_{INFERENCE,TRAIN}_APP` (or `KAUSHIK_MODAL_SUB=orange-expand`) to use a deployed sub-app pair.

Migration checks (rules in `kaushik/ARCHITECTURE.md`):

- Config equivalence: the sweeps reproduce all 12 `sft/configs/*.yaml` (= `sft/runs/*/config.yaml`), the 3
  `rl/configs/oex-rl-eugene-*` and the SFT smoke config exactly (`core/tests/test_sweep.py` keeps checking).
  Explained differences: `rl/runs/oex-rl-eugene-gptoss20b-seed0/config.yaml` has no `gpu` /
  `tensor_parallel_size` in its eval blocks (that run was launched before `make_configs` pinned the engine type;
  the later two runs match); the RL smoke configs predate the instability guard keys and the 2-GPU Qwen trainer.
- Analysis equivalence: `baseline/results/*` are byte-identical after re-running `baseline analyze`;
  `sft/results/{cells,differences,hits,mechanism_cells}.csv` and `summary.md` are byte-identical after re-running
  `sft analyze` (same bootstrap draw order, seed 0, 2000 draws).
- RL eval exceptions are data, not config: `train_result.json` records `final_step` 10 (instability) and 58
  (infrastructure); `eval_steps: final` evaluates those checkpoints; the it-10 registry entry
  (`oex-rl-eugene-gptoss20b-seed0-it10`) was written with `registry --step 10 --name ...`.
- `configs/*.yaml` are kept: registry entries point at them. `sft/results/.cache/v1__*` pickles belong to the old
  loader and can be deleted.
