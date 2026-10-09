"""train_eval.py configs for the no-CoT SFT arms (gpt-oss-20b; see README.md).

Arms (one wave, 3 seeds each):
  gentle   answer-only MCQ SFT (1A data) at a gentler dose: r8/a8, lr 5e-5, 100 steps
  mix      answer-only + reasoning-on anchor (model's own medium rollouts), 1A dose, 100 steps
  fmt      constrained-OUTPUT no-CoT SFT (format requirement on the response), 1A dose, 50 steps
  fmt_mix  fmt + anchor, 100 steps
Reasoning-off rows: reasoning_effort low + masked empty-analysis prefill (1A's
convention). Anchor rows override per row (reasoning ON, medium, no prefill).

Eval (shared protocol; T=1, 16k tokens, medium): val blocks (9 training modes x 200,
3 held-out x 200, baseline) at the mid and final checkpoints; test blocks (final_*:
500 q) at the final checkpoint. chunk/chunk_parallel bound each run to PEAK engines.

    .venv/bin/python kaushik/run.py elicit/nocot_sft make_configs [--arms ...] [--seeds ...] [--smoke]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
from common import ATTN, BASE_MODEL, CONFIGS, DATA, EVAL_KWARGS, FRAG_DATA, MODEL_KEY, PREFILL, ROOT, TRAIN_KWARGS, read_jsonl

from cotcontrol.eval.prompts import HELDOUT_MODES
from cotcontrol.training.config import default_lora_targets

BATCH = 48
SAMPLING = {"temperature": 1.0, "max_tokens": 16000}
SMOKE_SAMPLING = {"temperature": 1.0, "max_tokens": 2000, "max_samples": 8}
PEAK = 2  # concurrent engines per run (12 runs -> 24 GPUs, under the 32 cap)
SEEDS = [0, 1, 2]

ARMS = {
    "gentle": {"data": FRAG_DATA / "mcq_noreason.jsonl", "rank": 8, "alpha": 8, "lr": 5e-5, "steps": 100,
               "save": [50, 100], "long": False},
    "mix": {"data": DATA / "mix.jsonl", "rank": 32, "alpha": 32, "lr": 1.5e-4, "steps": 100,
            "save": [50, 100], "long": True},
    "fmt": {"data": DATA / "fmt.jsonl", "rank": 32, "alpha": 32, "lr": 1.5e-4, "steps": 50,
            "save": [25, 50], "long": False},
    "fmt_mix": {"data": DATA / "fmt_mix.jsonl", "rank": 32, "alpha": 32, "lr": 1.5e-4, "steps": 100,
                "save": [50, 100], "long": True},
    # 14:35 follow-up (launched while wave 1 evaluated): fmt at 3x the dose, since fmt was the
    # only arm with a honest gain at >= untrained trace length on the mid-checkpoint val numbers.
    "fmt_long": {"data": DATA / "fmt.jsonl", "rank": 32, "alpha": 32, "lr": 1.5e-4, "steps": 150,
                 "save": [75, 150], "long": False},
}


def train_section(arm: str, seed: int, n_rows: int, smoke: bool) -> dict:
    a = ARMS[arm]
    train = {
        "data_path": str(a["data"].relative_to(ROOT)),
        "shuffle": True, "seed": seed,
        "chat_template_kwargs": TRAIN_KWARGS,
        "assistant_prefill": PREFILL,
        "lr": a["lr"], "lr_scheduler_type": "constant",
        "batch_size": BATCH,
        "per_device_batch_size": 4 if a["long"] else 8,
        "num_epochs": max(2, math.ceil(a["steps"] * BATCH / n_rows)),
        "max_steps": a["steps"],
        "max_seq_length": 4608 if a["long"] else 1024,  # anchor rows carry <= 4096 sampled tokens
        "gradient_checkpointing": bool(a["long"]),
        "save_at_steps": list(a["save"]),
        "attn_implementation": ATTN,
    }
    if smoke:
        train.update(max_steps=2, save_at_steps=[1, 2], num_epochs=1)
    return train


def eval_section(save_at: list[int], smoke: bool) -> dict:
    mid, final = save_at[0], save_at[-1]
    base = {"run": True, "dataset": "all", "seed": 0, "eval_base_model": False, "system_prompt": "",
            "state_requirement": True, "chat_template_kwargs": EVAL_KWARGS,
            "chunk_parallel": PEAK, "parallel_checkpoints": PEAK, **SAMPLING}
    val = {**base, "split": "val", "eval_at_steps": [mid, final], "chunk": 450}
    test = {**base, "split": "test", "eval_at_steps": [final], "chunk": 500}
    blocks = {
        "val_modes": {**val, "mode": "all"},
        "val_heldout": {**val, "mode": "all", "allowed_modes": list(HELDOUT_MODES)},
        "val_baseline": {**val, "mode": "baseline"},
        "final_modes": {**test, "mode": "all"},
        "final_heldout": {**test, "mode": "all", "allowed_modes": list(HELDOUT_MODES)},
        "final_baseline": {**test, "mode": "baseline"},
    }
    if smoke:
        blocks = {k: {**v, **SMOKE_SAMPLING, "chunk": 0} for k, v in blocks.items()}
    return blocks


def make_config(arm: str, seed: int, n_rows: int, smoke: bool = False) -> dict:
    run_name = f"{'smoke' if smoke else 'nocot'}-{arm}-{MODEL_KEY}-seed{seed}"
    train = train_section(arm, seed, n_rows, smoke)
    targets = default_lora_targets(BASE_MODEL)
    return {
        "run_name": run_name, "arm": arm, "model_key": MODEL_KEY, "base_model": BASE_MODEL, "gpu": "H200",
        "run_dir": f"kaushik/elicit/nocot_sft/runs/{'smoke/' if smoke else ''}{run_name}",
        "train": train,
        "lora": {"lora_rank": ARMS[arm]["rank"], "lora_alpha": ARMS[arm]["alpha"],
                 "lora_target_modules": list(targets[0]), "lora_target_parameters": list(targets[1])},
        "eval": eval_section(train["save_at_steps"], smoke),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    out_dir = CONFIGS / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        path = ARMS[arm]["data"]
        if not path.exists():
            print(f"skip {arm}: {path.relative_to(ROOT)} missing")
            continue
        n_rows = len(read_jsonl(path))
        for seed in (args.seeds[:1] if args.smoke else args.seeds):
            cfg = make_config(arm, seed, n_rows, args.smoke)
            p = out_dir / f"{cfg['run_name']}.yaml"
            p.write_text(yaml.dump(cfg, sort_keys=False, width=1000))
            print(f"wrote {p.relative_to(ROOT)} ({n_rows} rows, {cfg['train']['num_epochs']} epochs, "
                  f"steps {cfg['train']['max_steps']})")


if __name__ == "__main__":
    main()
