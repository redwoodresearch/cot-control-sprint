#!/usr/bin/env bash
# Seed-0 screen: build the combinations on the volume (parallel), then launch the val-subset evals.
#   bash kaushik/elicit/lora_arith/sweep.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export ELICIT_SUB=sub-lora-arith
R=(.venv/bin/python kaushik/run.py elicit/lora_arith)
G=/checkpoints/gepa_lora/gl_r16_lr1e-4_s0/checkpoint-400
B=/checkpoints/lora_arith/brief_s0/checkpoint-400
V=/checkpoints/lora_arith/verbose_s0/checkpoint-400
C=/checkpoints/lora_arith/combos

NAMES=(gepa_m0.5b_s0 gepa_m1.0b_s0 gepa_m1.5b_s0 gepa_p0.5v_s0 gepa_p1.0v_s0 gepa_m1.0b_p1.0v_s0)
TERMS=("$G:1,$B:-0.5" "$G:1,$B:-1" "$G:1,$B:-1.5" "$G:1,$V:0.5" "$G:1,$V:1" "$G:1,$B:-1,$V:1")
pids=()
for i in "${!NAMES[@]}"; do
  "${R[@]}" combine --name "${NAMES[$i]}" --terms "${TERMS[$i]}" > "kaushik/elicit/lora_arith/runs/_combos_${NAMES[$i]}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
grep -h "max_rel_err" kaushik/elicit/lora_arith/runs/_combos_*.log

SUB=(--split val --blocks modes --max-questions 40 --subsample-seed 1 --chunk 360)
args=(brief_s0_sub -- "${SUB[@]}" --lora "$B" ";;" verbose_s0_sub -- "${SUB[@]}" --lora "$V")
for name in "${NAMES[@]}"; do
  args+=(";;" "${name}_sub" -- "${SUB[@]}" --lora "$C/$name")
done
"${R[@]}" launch "${args[@]}"
