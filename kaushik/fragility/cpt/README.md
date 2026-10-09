# fragility/cpt — Experiment 1 arms 1B (`cpt`) and 1C (`cpt_param_sweep`)

Does generic continued pretraining on web text raise CoT controllability, does
the effect grow with dose, and how many trainable parameters does it need?
Spec: [../../sprint_2026_10_08.md](../../sprint_2026_10_08.md) (Experiment 1).
Models: gpt-oss-20b (the model SDF's C4 arm moved) and Qwen3-8B (the negative case).

## What is trained

Both arms do plain next-token training on raw C4 documents (`{"text": ...}`
rows, no chat template, every token supervised; `cotcontrol.training.rendering.render_text`),
packed into **4096-token sequences**, 48 sequences per optimizer step
(≈197k supervised tokens/step), constant LR, no warmup, AdamW with weight decay 0.

| arm | adapter | LR | steps | checkpoints | seeds |
|---|---|---|---|---|---|
| 1B `cpt-<model>-r32-seed<s>` | LoRA r32/α32, all linear layers incl. gpt-oss's fused experts (`default_lora_targets`) | 1.5e-4 | 200 (step 50 = 1A-matched dose; 200 = 4×) | 10,20,30,40,50,100,150,200 | 0–2 |
| 1C `cptk-<model>-<k>-lr<lr>-seed<s>` | LoRA r1/α1, same targets, only k random scalars trainable (`train_params`, `mask_seed`=seed) for k ∈ {1e3, 1e4, 1e5}, plus the full rank-1 adapter (`r1`) | per-k sweep, see below | 50 | 10,20,30,40,50 | 0 (sweep) then 1–2 at the chosen LR |

1B's step-50 checkpoint is the rank-32 top point of 1C's parameter curve.

Measured (smoke runs, one H200, per-device batch 4 with gradient checkpointing):
27–35 s per optimizer step for both models, i.e. ~30 min per 50-step 1C cell and
~2 h per 200-step 1B run, before evaluation.

**Sequence length.** The spec matches 1A by steps × batch (50 × 48), not tokens, so
sequence length is a free choice. SDF's `c4only` arm used 16k packed sequences at
batch 8 (≈131k tokens/step). 4096 × 48 keeps the per-step token dose at SDF scale
(1.5×), is ~4× faster, and avoids SDF's observation that gpt-oss's loss on packed C4
grows with context length. Change `MAX_SEQ_LENGTH` in `make_configs.py` to revisit.

**LR for 1C.** `sft/lrk_analysis.py`'s iso-drive rule: (α/r)·lr·√k is held at 1B's
value (α/r = 1 in both arms), i.e. lr(k) = 1.5e-4·√(N_r32/k) with N_r32 the full
rank-32 adapter size (gpt-oss-20b 369.8M → drive 2.885; Qwen3-8B 87.3M → 1.401).
Stage 1 sweeps lr_pred × {1/3, 1, 3} at seed 0:

| model | k=1e3 | k=1e4 | k=1e5 | full r1 |
|---|---|---|---|---|
| gpt-oss-20b | 0.030 / **0.091** / 0.27 | 0.0096 / **0.029** / 0.087 | 0.0030 / **0.0091** / 0.027 | 2.8e-4 / **8.5e-4** / 2.5e-3 |
| Qwen3-8B | 0.015 / **0.044** / 0.13 | 0.0047 / **0.014** / 0.042 | 0.0015 / **0.0044** / 0.013 | 2.8e-4 / **8.5e-4** / 2.5e-3 |

(The earlier SDF rank-1 CPT sweeps on gpt-oss-120b used 3e-2…3e-1 at k=1e3 and
1e-2…1e-1 at k=1e4, which these grids bracket.) `select` picks, per (model, k), the
LR with the highest **honest** val compliance at step 50 among LRs whose step-50
accuracy keeps ≥ 85% of checkpoint-0 accuracy (pooled over the four val blocks,
800 rollouts); LRs within 1 point of the best (the SE of 800 rollouts) are tied, and
the tie goes to the higher LR — the larger dose that still preserved accuracy, which for
a model that never moves (Qwen) is the most conservative "no effect" test. If every rung fails the guard (gpt-oss
collapses within 10–40 steps at the ×1 and ×3 rungs for small k), `select` adds a rung
a factor 3 lower (`configs/lr_grid_ext.yaml`, up to 5 rungs per (model, k)) and decides
once it is evaluated; only after that does it fall back to the least-damaging LR,
flagged. Seeds 1–2 then run at the chosen LR. Effective k = `n_changed_inside_mask`
from the worker's post-hoc mask verification (B is zero-initialised, so a selected
A-entry whose B partner is frozen never moves).

## Data

`data` downloads allenai/c4 (English, train, shard 0), applies SDF's filters
(200 ≤ chars ≤ 12 000), shuffles with seed 42 and keeps the smallest prefix whose
tokens pack to ≥ 200 × 48 × 4096 × 1.08 sequences for **both** tokenizers
(`data/c4_cpt.jsonl`, 106 561 docs, 42–44M tokens, used by 1B), so no run repeats
data. The 50-step 1C runs ship the prefix of the same shuffle sized for 50 steps
(`data/c4_cpt_50steps.jsonl`). `data/c4_cpt.manifest.json` logs each file's sha256,
token counts per model and an eval-overlap scan (whitespace/case-normalised 80-char
openings of all 1214 eval questions searched in every document: 0 hits). Every run's
`manifest.json` repeats its data sha256 and the git commit.

## Evaluation (every block: T=1, max_tokens 16 000, reasoning on via eval-side template kwargs — gpt-oss `reasoning_effort: medium`, Qwen `enable_thinking: true`)

- **Val, every checkpoint** (`val_<mode>`): `lowercase_thinking`, `uppercase_thinking`,
  `alternating_case`, `end_of_sentence`, 200 val questions each (800 rollouts/checkpoint/seed;
  the pooled headline and the LR selection use these four), plus `val_repeat_sentences`
  as a measurement-only block added after the first results: gpt-oss-20b is at 0/800 on the
  four fixed-trait modes before and after small-k CPT (at T=1), whereas it sits at ≈ 4–7%
  on `repeat_sentences`, the mode SDF's C4 arm moved (.01 → .10), so that block is what
  makes the per-checkpoint dose curve sensitive on gpt-oss. Cells evaluated before the
  block existed were re-evaluated with it (`backfill` stage; the other blocks came from cache).
- **Test, final checkpoints only** (`test_train_modes`, `test_heldout`, `test_baseline`):
  all 9 training modes × 500 test questions (mode=all, 4464 rollouts), the 3 held-out
  modes × 500 (1500), and the unconstrained `baseline` mode (500) for accuracy. 1B
  evaluates steps 0/50/200 (step 0 gives the untrained reference on the same items);
  1C evaluates step 50 of the selected-LR runs. Nothing is selected on test.
- Metrics per rollout (`analyze.py`): honest compliance (`cotcontrol.eval.grading.is_honest`),
  strict compliance, accuracy, `chars`, `degen` (< 200 chars) and `compl_long`
  (strict compliance among traces at least as long as checkpoint-0's median for that mode).
  95% intervals from a bootstrap over questions (as clusters) and seeds, paired against
  checkpoint-0 of the same runs. KL(base‖adapter) and noise controls are read from
  `kaushik/noise/runs/<run_name>/` when the noise agent has produced them.

**Sample sizes.** Per checkpoint and seed, 800 val rollouts give SE ≈ 1.1 pts at
p = 0.1 (1.8 at p = 0.5); pooled over 3 seeds ≈ 0.6–1.0 pts, so arm-vs-untrained
differences have SE ≈ 1–1.5 pts. The LR sweep (single seed) decides on 800 rollouts per
cell (SE ≈ 1.1–1.8). On test, 4464 rollouts per seed give SE ≈ 0.45 pts (training modes),
1500 → 0.8 (held-out), 500 → 2.2 (baseline accuracy; 1.3 pooled over seeds).

## Running

```bash
R=".venv/bin/python kaushik/run.py fragility/cpt"
$R data                     # C4 download + sizing + manifest (minutes)
$R configs                  # configs/*.yaml (34 incl. 4 smoke configs)
$R smoke                    # 2-step runs, 8 questions/block; checks adapter size, mask, eval paths
$R train                    # enqueue stage 1 (24 LR-sweep cells, then 6 x 1B) + start the scheduler
$R status [--tail 30]       # queue, reserved GPUs, per-run step/eval progress
$R select [--dry-run]       # per (model, k): pick LR on val -> enqueue seed-0 test eval + seeds 1-2
$R registry                 # kaushik/registry/<run>.json for finished final adapters
$R losses                   # runs/<run>/losses.jsonl (per-step C4 loss) from the Modal volume
$R analyze                  # results/{tables.md, summary.json, rows.csv, dose.png, kcurve.png, losscurve.png}
```

**GPU budget.** This agent's cap is 90 concurrent GPUs (coordinator, 2026-10-08; 40 → 80 → 120 → 90 during the sprint).
Training and eval are separate jobs: a training job holds 1 H200; an eval job holds
`EVAL_GPUS` engines (8 for 1B, 4 for 1C stage 2, 3 for the stage-1 sweep cells that
were already evaluating when the cap rose; `parallel_checkpoints` and `chunk_parallel`
are set to that). The scheduler (`cli.py scheduler`, a background process with a file
queue in `runs/_queue/`) reserves each job's footprint for its lifetime plus a
3-minute teardown cooldown, admits by priority (sweep cells before 1B, stage-2 first),
never exceeds the cap (`runs/_queue/cap.txt` overrides it live), and pauses admission
while either coordinator guard trips: more than 800 containers workspace-wide
(`modal container list`) or less than 25% free laptop memory (`memory_pressure`),
since every job is also a local driver process. Long-lived Modal client streams from
this laptop drop now and then (`grpclib StreamTerminatedError: Connection lost`, seen by
every sprint agent and, in this experiment, almost only on gpt-oss evals); a drop kills
the local driver, so the scheduler re-queues failed jobs (evals up to 20 attempts,
training twice) and evals resume from `.generation_cache/` — val blocks are evaluated in
50-prompt engine calls and test blocks in 200-prompt calls so that little is lost per
drop. Killing a local driver does **not** stop its deployed-mode worker: at 11:36 on
2026-10-08 another agent's process cleanup killed every `train_eval.py` on the laptop; the
scheduler relaunched the 20 affected training jobs under fresh timestamped run names while
the orphaned workers kept training in their own `/checkpoints/<run>-<ts>/` dirs (so nothing
was corrupted). The three Qwen 1B orphans finished first and were taken over with the
`adopt` stage (`train_result.json` rebuilt from the worker's `losses.jsonl`,
`training_data.json` and `mask_verification.json` on the volume; recorded under
`"adopted"` in `train_result.json`), and the duplicate workers were ended with
`modal container stop --yes` (which does not reschedule a `retries=0` training input).
A training job runs
`scripts/train_eval.py` on a copy of the config with eval blocks disabled; the eval job
restores the canonical eval section into the run's `config.yaml` and runs
`train_eval.py --eval-only`, so run folders look exactly like any other train_eval run.

Run folder (`runs/<run>/`): `manifest.json` (arm, model, seed, k, lr, data sha256,
git commit), `train_config.yaml`, `config.yaml`, `train.log`, `eval.log`,
`train_result.json` (losses, checkpoint volume paths, mask verification),
`eval/<block>/checkpoint-<step>.json` (every prompt, rollout, grade and judge output),
`eval_summary.json`, `summary.png`. Checkpoints live on the Modal volume
`cotcontrol_kaushik-reddy_checkpoints` at `/checkpoints/<run>-<timestamp>/checkpoint-<step>`.

Tests: `uv run --with pytest python -m pytest -q kaushik/fragility/cpt/test_cpt.py`.

## Results

Full tables with intervals: `results/tables.md` (regenerate with `analyze`); plots
`results/dose.png` (1B compliance and accuracy vs step), `results/kcurve.png` (1C compliance
vs k with the rank-1 and rank-32 points), `results/losscurve.png` (every cell vs its C4
loss). Honest compliance in %, 95% bootstrap intervals
over questions × seeds; Δ = paired difference to checkpoint-0 of the same runs (1B) or
unpaired difference to the 1B runs' checkpoint-0 on the same test items (1C). Every rollout
is graded with `cotcontrol.eval.grading.is_honest`, seeds are pooled, and the test-split
numbers agree with `../results/exp1_test_table.md` for the same runs (an earlier version of
this file was 0.1–0.3 points higher on test because its bootstrap matrix keyed questions on
(id, mode) while question ids repeat across the three eval datasets; the key now includes the
dataset). Seed counts are printed in every test table.

### gpt-oss-20b (3 seeds per arm)

**1B, C4 CPT with the spec dose (LoRA r32, lr 1.5e-4, 48 × 4096-token rows per step) does
not raise controllability; it damages the model.** Test split, 9 training modes (4464
rollouts per seed): honest 1.2 [0.9, 1.4] untrained → 0.8 [0.6, 1.1] at step 50
(Δ −0.4 [−0.6, −0.0]) → 0.8 [0.5, 1.1] at step 200 (Δ −0.4 [−0.8, −0.0]); raw strict
1.3 → 1.7 → 2.2 (the gap is traces that comply but never produce an answer). Held-out modes:
1.1 → 2.4 [1.6, 3.2] → 1.7. Accuracy under constraint 45 → 23 → 17; unconstrained accuracy
49 → 38 → 35. On val the four fixed-trait modes stay at 0.0 at every checkpoint (0/800 for
the untrained model too) and `repeat_sentences`, the mode SDF's C4 arm moved, goes 3.2 →
0.8 → 1.8. The C4 loss falls from 14.0 (base) to 2.95 after 10 steps and 2.75 at step 200.
This is a far hotter and shorter regime than SDF's C4 arm (lr 1e-5, ~5000 steps of 16 ×
1k-token rows, strict .009 → .033 with little accuracy loss); the supplementary low-LR arm
below tests whether a non-destructive dose reproduces that small gain.

**1C, trainable-parameter sweep (rank-1 masked LoRA, step 50, LR chosen on val):**
honest compliance on the fixed-trait val modes is 0.0 [0.0, 0.0] at k = 1e3 (713
effective), 1e4 (9747), 1e5 (99988), the full rank-1 adapter (11.6M) and the full rank-32
adapter (370M) alike — the curve is flat at zero from bottom to top, so no logistic fit is
warranted. Test split, 9 training modes, 3 seeds each, Δ vs untrained: k=1e3 −0.4 [−0.8, 0.0];
k=1e4 −0.5 [−0.9, −0.1]; k=1e5 −0.1 [−0.5, 0.3]; full rank-1 +0.6 [0.0, 1.1] (1.7 vs 1.2;
held-out 2.2 vs 1.1; accuracy 45 → 40); rank-32 (1B@50) −0.4. The selected LRs are the
largest that kept ≥ 85% of checkpoint-0 accuracy (9.1e-2, 2.9e-2, 9.1e-3, 2.8e-4 for k =
1e3, 1e4, 1e5, r1); the ×3 rungs collapse accuracy at every k and the predicted rung
already does for the full rank-1 adapter. Effective k (entries that moved) is 71% of the
selected entries at k=1e3 and 97–100% above (rank-1 B-entries start at zero, so a selected
A-entry whose B partner is frozen never moves).

**Claim check ("the CPT effect is gone at ~1e4 parameters").** Δ at k=1e4 is −0.0 [−0.2,
0.0] on val and −0.5 [−0.9, −0.1] on test, so it is trivially consistent with the claim —
but only because the full-adapter arm has no effect either under these settings: there is
no CPT effect to be gone at any k. The same holds at matched functional size (last
section): the k=1e4 adapter is −0.1 [−0.5, 0.2] points from the rank-32 checkpoint nearest
to it in C4 loss. The only positive signal in the whole gpt-oss grid is the full rank-1
adapter's +0.6 points on test, whose interval touches 0.

### Qwen3-8B (negative case, 3 seeds per arm)

**1B:** nothing moves and nothing breaks. Val (4 fixed-trait modes, pooled): honest 0.5
[0.1, 0.9] untrained → 0.1 [0.0, 0.3] at step 50 (Δ −0.4 [−0.8, −0.0]) → 0.2 [0.0, 0.4] at
step 200; only `lowercase_thinking` is ever non-zero (1.8 → 0.3 → 0.7); `repeat_sentences`
is 0 throughout. Test, 9 training modes: 0.6 [0.4, 0.8] → 0.4 [0.3, 0.6] → 0.4 [0.2, 0.6]
(Δ −0.1 [−0.3, 0.1] and −0.2 [−0.4, 0.1]); held-out 0.1 → 0.1 → 0.3. Accuracy under constraint
39 → 38 → 36, unconstrained 33 → 35 → 35 — the same CPT dose that halves gpt-oss's accuracy
leaves Qwen's intact (C4 loss 2.77 → 2.52), while reasoning traces shorten from 15.3k to
10.0k chars. **1C:** honest val compliance 0.1–0.3% at k = 1e3 (852 effective), 1e4
(9967), 1e5 (99997), full rank-1 and rank-32, all Δ ≈ −0.2 [−0.8, 0.3] vs untrained; every
LR rung keeps accuracy (the selected LRs are therefore the ×3 rungs: 1.3e-1, 4.2e-2, 1.3e-2,
2.5e-3), and the sweep cells range 0.1–0.9%. On the test split the selected 1C arms are at
0.3% honest (Δ −0.2 to −0.3, intervals [−0.5, 0.1] at worst, vs untrained; 3 seeds at k = 1e5
and rank-1, 2 seeds at k = 1e3 and 1e4 while the third seed's test eval runs) with accuracy
intact; raw strict rises to 2.6–2.7% at k = 1e3/1e4 but all of that surplus is hollow
(compliant traces without an answer). Generic CPT does not elicit controllability on Qwen at
any parameter count or learning rate tried here.

### Supplementary: low-LR 1B on gpt-oss (lr 5e-5 and 1.5e-5, seed 0)

Pending.

### Noise controls and KL (gpt-oss-20b 1B; from `kaushik/noise/runs/fragility-cpt-cpt[-step50]-gptoss20b/`)

KL(base ‖ adapter) per token on the base model's own val samples is 0.070–0.073 for the
step-50 adapters and 0.068–0.070 for the step-200 adapters (3 seeds) — the extra 150 steps
barely move the adapter's functional size. The noise agent's three random rank-32 adapters
per step are matched to that KL (targets 0.073 and 0.069). Against them, on the test split
(unpaired bootstrap over questions × seeds/adapters):

| comparison | train honest | train strict | train accuracy | held-out honest | held-out strict | baseline acc |
|---|---|---|---|---|---|---|
| 1B step 50 − noise | −0.0 [−0.5, 0.4] | +0.8 [0.3, 1.3] | −19.7 [−21.8, −17.7] | +1.6 [0.7, 2.5] | +2.1 [1.1, 3.1] | −5.5 [−11.9, 0.4] |
| 1B step 200 − noise | −0.3 [−0.7, 0.2] | +1.1 [0.5, 1.7] | −25.2 [−27.2, −23.2] | +0.9 [0.2, 1.6] | +2.1 [1.2, 2.9] | −10.8 [−16.5, −4.9] |

So at matched functional size, C4 CPT is no better than a random perturbation on honest
compliance for the training modes (its raw-strict surplus is hollow compliance — traces that
comply but never answer), it buys 1–2 points on the held-out modes, and it costs 20–28
points of accuracy under constraint that the random adapters do not lose (they stay at ≈ 43%).
The noise controls themselves sit at 0.9–1.0% honest, i.e. at the untrained level: "any
perturbation raises controllability" does not hold for gpt-oss-20b at this KL. On Qwen3-8B
the KL-matched noise adapters land at 0.1–0.2% honest, slightly *below* the CPT adapters
(1B − noise +0.2 [0.0, 0.4] and +0.3 [0.1, 0.5] at steps 50 / 200), with both below the
untrained 0.6%.

### Compliance at matched functional size (C4 loss; `results/losscurve.png`)

The restricted-k adapters barely fit C4, so the 1C arms differ in functional size as well
as in k. `losses` fetches every run's per-step loss log, and the "Compliance at matched
functional size" table in `results/tables.md` puts every evaluated cell — the untrained
model, each 1B checkpoint, every LR rung of the sweep — on one axis: the C4 loss at the
checkpoint (mean of the 3 steps ending there, 48 × 4096 tokens each; step 1 is the untrained
model, since LoRA B starts at zero), and pairs each 1C cell with the rank-32 checkpoint
nearest to it in loss (unpaired bootstrap over questions × seeds).

**gpt-oss-20b.** The selected-LR cells sit at loss 11.5 (k=1e3), 6.2 (1e4), 4.3 (1e5) and
2.82 (full rank-1), against 14.0 untrained and 2.95 / 2.77 / 2.75 for the rank-32 adapter at
steps 10 / 50 / 200 — no rank-32 checkpoint exists between 14 and 2.95, and the ×3 rungs
reach 7.7 / 3.6 / 3.0 only by collapsing accuracy. Honest val compliance is 0.0 at every
loss from 14 to 2.7, so the comparison lives on the test split: k=1e3 is nearest the
untrained model and indistinguishable from it (Δ −0.4 [−0.8, 0.0]); k=1e4 and 1e5 match the
rank-32 step-50 checkpoint on honest compliance (Δ −0.1 [−0.5, 0.2] and +0.2 [−0.1, 0.6])
while keeping 20–22 more points of accuracy under constraint; the full rank-1 adapter, at
the same loss as rank-32 (2.82 vs 2.77), is the one cell that beats it — +0.9 [0.4, 1.4]
honest (1.7 vs 0.8) with 17.4 more accuracy points — but that is only +0.6 [0.0, 1.1] over
the untrained model. Fitting C4 to the same loss with 370M parameters costs ~20 points of
accuracy that 11.6M or fewer do not; it buys no controllability either way.

**Qwen3-8B.** All cells lie within 0.25 nats of each other (2.77 untrained → 2.52 at step
200); every matched pair is within ±0.3 points on honest compliance with intervals spanning
0, and accuracy is unchanged. Nothing moves at any functional size.

KL(base ‖ adapter) is measured only for the rank-32 adapters so far (0.070–0.073 on gpt-oss,
0.085 / 0.128 on Qwen at steps 50 / 200); the 1C and low-LR registry entries carry
`measure_kl: true`, and `analyze` fills the KL column once the noise agent processes them.
