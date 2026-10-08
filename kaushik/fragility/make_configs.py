"""Emit train_eval.py configs for the fragility arms (Experiment 1: 1A, 1D, 1E).

Hyperparameters match orange-training/make_configs.py as implemented: LoRA
r32/alpha32 on all linear layers (default_lora_targets), LR 1.5e-4 constant,
effective batch 48, 50 steps, checkpoints at 10/20/30/40/50. Model specs
(template kwargs, prefill, attention kernel, 27B memory settings) are imported
from orange's MODELS so the arms stay comparable with Experiment 2.

  mcq_noreason (1A)  letter target, reasoning conditioned off (train kwargs + prefill)
  mcq_reason   (1D)  model's own correct rollout as target, reasoning ON at train time
  style_sft    (1E)  pirate-voice answer target, reasoning conditioned off

Eval (spec "Shared setup"; T=1, max_tokens 16000 set explicitly):
  every checkpoint, val split: `stated` condition on lowercase_thinking,
  uppercase_thinking, alternating_case, end_of_sentence — built with orange's
  eval_blocks so the prompts match Experiment 2 exactly; Qwen3.6-27B (Experiment
  2's baseline) also gets the `persona` and `direct` conditions.
  final checkpoint only, test split: final_modes (the 9 training modes),
  final_heldout (3 held-out modes), final_baseline (unconstrained accuracy), and
  for Qwen3.6-27B the persona/direct conditions on the 4 modes.

    .venv/bin/python kaushik/run.py fragility make_configs [--arms ...] [--models ...] [--smoke]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
from common import ARMS, CONFIGS, DATA, ROOT, VAL_MODES, orange, read_jsonl

from cotcontrol.eval.prompts import HELDOUT_MODES
from cotcontrol.training.config import default_lora_targets

omc = orange("make_configs")
MODELS = omc.MODELS
SEEDS = {
    "mcq_noreason": {"gptoss20b": [0, 1, 2], "qwen8b": [0, 1, 2], "qwen36_27b": [0, 1, 2, 3, 4]},
    "mcq_reason": {"gptoss20b": [0, 1, 2], "qwen8b": [0, 1, 2]},
    "style_sft": {"gptoss20b": [0, 1, 2], "qwen8b": [0, 1, 2]},
}
STEPS, BATCH = 50, 48
SAMPLING = {"temperature": 1.0, "max_tokens": 16000}
SMOKE_SAMPLING = {"temperature": 1.0, "max_tokens": 2000, "max_samples": 8}
# Peak concurrent Modal engines (= GPUs) per run during eval; launch.py's
# scheduler reserves this against the sprint's 35-GPU cap. The 27B runs are
# Experiment 2's baseline and sit on the critical path, so they get more.
PEAK_GPUS = {"qwen36_27b": 6}
DEFAULT_PEAK = 3


def data_path(arm: str, key: str) -> Path:
    name = {"mcq_noreason": "mcq_noreason", "mcq_reason": f"mcq_reason_{key}", "style_sft": "style_sft"}[arm]
    return DATA / f"{name}.jsonl"


def conditions_for(key: str) -> list[str]:
    return ["persona", "direct", "stated"] if key == "qwen36_27b" else ["stated"]


def train_section(arm: str, key: str, seed: int, n_rows: int, smoke: bool) -> dict:
    spec, reason = MODELS[key], arm == "mcq_reason"
    train = {
        "data_path": str(data_path(arm, key).relative_to(ROOT)),
        "shuffle": True,
        "seed": seed,
        # Reasoning-on training renders the completion with the model's own
        # think block, so it uses the eval-side (reasoning enabled) kwargs.
        "chat_template_kwargs": spec["eval_template_kwargs"] if reason else spec["train_template_kwargs"],
        "lr": 1.5e-4,
        "lr_scheduler_type": "constant",
        "batch_size": BATCH,
        "per_device_batch_size": spec.get("per_device_batch_size", 8),
        "num_epochs": max(2, math.ceil(STEPS * BATCH / n_rows)),
        "max_steps": STEPS,
        "max_seq_length": 4608 if reason else 1024,  # 1D rows carry <=4096 sampled tokens
        "gradient_checkpointing": spec.get("gradient_checkpointing", False),
        "save_at_steps": [10, 20, 30, 40, 50],
    }
    if "attn_implementation" in spec:
        train["attn_implementation"] = spec["attn_implementation"]
    if not reason and "assistant_prefill" in spec:  # gpt-oss empty analysis channel, masked
        train["assistant_prefill"] = spec["assistant_prefill"]
    if reason:
        train.update(per_device_batch_size=4, gradient_checkpointing=True)
    if smoke:
        train.update(max_steps=2, save_at_steps=[1, 2], num_epochs=1)
    return train


def eval_section(key: str, final_step: int, seed: int, smoke: bool) -> dict:
    spec, name, conds = MODELS[key], MODELS[key]["base_model"], conditions_for(key)
    peak = PEAK_GPUS.get(key, DEFAULT_PEAK)
    blocks = omc.eval_blocks(name, VAL_MODES, conds, spec["eval_template_kwargs"])
    for b in blocks.values():
        b.update(SAMPLING, parallel_checkpoints=peak)  # one engine per checkpoint, `peak` at a time
    # Test split: final checkpoint only; the seed-0 run of each model also
    # evaluates checkpoint-0 there, so the test table has an `untrained` row
    # (the zero-init adapter is the same base model whichever run it comes from).
    final_steps = [0, final_step] if seed == 0 else [final_step]
    base = {
        "run": True, "dataset": "all", "split": "test", "seed": 0, "eval_base_model": False,
        "system_prompt": "", "state_requirement": True,
        "chat_template_kwargs": spec["eval_template_kwargs"], "eval_at_steps": final_steps,
        # 250-prompt engine calls fanned out over `peak` engines (test split = 500 q).
        "chunk": 250, "chunk_parallel": peak, **SAMPLING,
    }
    blocks["final_modes"] = {**base, "mode": "all"}
    blocks["final_heldout"] = {**base, "mode": "all", "allowed_modes": list(HELDOUT_MODES)}
    blocks["final_baseline"] = {**base, "mode": "baseline"}
    for cond in [c for c in conds if c != "stated"]:
        for mode in VAL_MODES:
            sp = (omc.persona_eval_prompt if cond == "persona" else omc.direct_eval_prompt)(mode, name)
            blocks[f"final_{cond}_{mode}"] = {**base, "mode": mode, "system_prompt": sp,
                                              "state_requirement": False}
    if smoke:
        keep = ["stated_lowercase_thinking", "final_modes", "final_heldout", "final_baseline"]
        blocks = {k: {**blocks[k], **SMOKE_SAMPLING, "chunk": 0} for k in keep}
    return blocks


def make_config(arm: str, key: str, seed: int, n_rows: int, smoke: bool = False) -> dict:
    spec = MODELS[key]
    run_name = f"{'smoke' if smoke else 'fragility'}-{arm}-{key}-seed{seed}"
    run_dir = f"kaushik/fragility/runs/{'smoke/' if smoke else ''}{run_name}"
    train = train_section(arm, key, seed, n_rows, smoke)
    targets = default_lora_targets(spec["base_model"])
    return {
        "run_name": run_name,
        "arm": arm,
        "model_key": key,
        "base_model": spec["base_model"],
        "gpu": "H200",
        "run_dir": run_dir,
        "train": train,
        "lora": {"lora_rank": 32, "lora_alpha": 32,
                 "lora_target_modules": list(targets[0]), "lora_target_parameters": list(targets[1])},
        "eval": eval_section(key, train["max_steps"], seed, smoke),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="override the per-arm seeds")
    ap.add_argument("--smoke", action="store_true", help="2 steps, 8 eval questions, configs/smoke/")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else CONFIGS / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for key in [k for k in SEEDS[arm] if k in args.models]:
            path = data_path(arm, key)
            n_rows = len(read_jsonl(path)) if path.exists() else 2100
            if not path.exists():
                print(f"WARNING: {path.relative_to(ROOT)} missing; assuming {n_rows} rows")
            for seed in (args.seeds if args.seeds is not None else SEEDS[arm][key]):
                if args.smoke and seed != (args.seeds or [0])[0]:
                    continue
                cfg = make_config(arm, key, seed, n_rows, args.smoke)
                p = out_dir / f"{cfg['run_name']}.yaml"
                p.write_text(yaml.dump(cfg, sort_keys=False, width=1000))
                print(f"wrote {p.relative_to(ROOT)} ({len(cfg['eval'])} eval blocks, "
                      f"{n_rows} rows, {cfg['train']['num_epochs']} epochs)")


if __name__ == "__main__":
    main()
