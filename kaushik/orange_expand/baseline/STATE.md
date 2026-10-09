# kaushik/orange_expand/baseline — STATE (stage 0; last update: 2026-10-09 15:50 PT — DONE)

## Scope
Stage 0 of `../PLAN.md`: (1) untrained CoT-control eval of gpt-oss-20b/120b, Qwen3-8B/32B under 4 conditions x
2 modes; (2) orange-pick rate per model x training prompt x reasoning on/off (200 held-out items, same 200 as
`../sft/data/heldout_200.json`); (3) mechanism check on Qwen3.6-27B (2A adapters seeds 0-2 + untrained under the
new `nopersona` condition). Apps: `cot-control-sprint_kaushik-reddy_sub-orange-base_{inference,train}`.
Budget $120 (stop new launches at $95); GPU cap 40.

## Code (this folder)
- `common.py`   env (forces the sub-orange-base apps), model table, prompts, held-out items, answer parsing
- `run_all.py`  scheduler: 54 jobs (8 pick, 32 untrained, 14 mech), cap 40 GPUs, retries 3x, `runs/.gpu_cap`
                overrides the cap live, `runs/.stop_launches` stops new launches
- `analyze.py`  results/pick_rates.{md,csv}, results/untrained.{md,csv}, results/mechanism{,_differences}.{md,csv}

## Running now
Nothing. All 54 jobs done (8 pick, 32 untrained, 14 mech; 0 generation failures). ~10.3 GPU-hours on the
sub-orange-base inference app (spend.py lags ~1 h; ~$45-50 expected). Tables sent to main and the RL agent 15:50.

## Results (results/*.md)
- Pick rate (reasoning on / off): eugene ~42-47% on all 4 models = same as the neutral prompt (untrained baseline
  ~45%, chance 25%); direct prompt 93-98% with reasoning on (>= 0.7 -> PLAN's direct-RL cut applies), 53-76% off.
- Untrained CoT eval: 0.0% honest compliance in every persona/nopersona/direct cell for all 4 models; standard
  condition qwen8b 3.0/0.0, qwen32b 3.5/0.0 (lowercase/uppercase), gpt-oss 0.0.
- 27B mechanism (seeds 0-2, pooled modes): orange adapters persona 8.0 [4.7,12.0], nopersona 7.9 [4.3,12.8],
  direct 5.1; orange_nopersona adapters 0.0/0.0/6.1; untrained 0.0/0.0/3.4. The effect needs the EUGENE-trained
  adapter but NOT EUGENE in the eval prompt: the negated "must NOT" frame alone reproduces it.

## Outputs
- `runs/<model_key>/eval/<condition>_<mode>/base.json` (standard cotcontrol eval JSON), `eval_progress.jsonl`, `meta.json`
- `runs/<model_key>/pick/<prompt>_<on|off>.json` (per-item records + summary)
- `runs/mech_qwen36_27b/<arm>-seed<k>/eval/nopersona_<mode>/checkpoint-50.json`, `.../untrained/eval/.../base.json`
- `runs/scheduler.{log,json,pid}`; `results/*.md|csv`

## Gotchas
- `import analyze` inside this folder resolves to kaushik/orange/analyze.py (common.py puts that folder first on
  sys.path); run_all.py loads its sibling by path as `baseline_analyze`. The first run hit this AFTER writing all
  pick files, so the 8 pick jobs are logged as failed in runs/scheduler.json although their outputs are complete.
- Don't put `cut` in a Monitor pipeline (block-buffered: no events were delivered).
- gpt-oss reasoning off = `reasoning_effort: low` + a final assistant message `{thinking: "", content: ""}` continued
  (renders 2A's empty-analysis prefill followed by `<|channel|>final<|message|>`); Qwen = `enable_thinking: false`.
- With reasoning off the bare letter lands in `reasoning` (split_reasoning sees no </think>); parse_letter handles it.
- Spend: `.venv/bin/python kaushik/elicit/spend.py sub-orange-base` (lags ~1 h); `runs/scheduler.json` has GPU-hours.
