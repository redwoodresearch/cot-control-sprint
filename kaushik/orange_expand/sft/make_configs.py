"""Stage-1 SFT configs: 4 models x 3 training prompts x 1 seed = 12 runs.

`configs/oex-sft_<variant>-<model>-seed0.yaml`: 2A's orange recipe — LoRA r32/alpha32 on
2A's module families (gpt-oss: attention + fused experts; Qwen3: attention + MLP), LR 1.5e-4
constant, batch 48, reasoning conditioned off — for 100 steps, checkpoints at 50 and 100,
eval at step 100 only: 4 eval conditions (prompts.eval_block) x prompts.SCREEN_MODES, val
split (200 q), T=1, 16k tokens, reasoning on. Valid scripts/train_eval.py configs, run via
train_run.py + kaushik/orange/eval_scheduler.py (--runs-dir kaushik/orange_expand/sft/runs).

    .venv/bin/python kaushik/run.py orange_expand/sft configs [--models ...] [--variants ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import common
from common import CONFIG_DIR, MODEL_ORDER, MODELS, PREFIX, ROOT

from cotcontrol.training.config import default_lora_targets

_P = common.oex_prompts
EVAL_CONDITIONS, SCREEN_MODES, TRAIN_PROMPTS, eval_block = (_P.EVAL_CONDITIONS, _P.SCREEN_MODES,
                                                            _P.TRAIN_PROMPTS, _P.eval_block)

FINAL_STEP = 100
SAVE_AT = [50, 100]
EVAL_COMMON = {"run": True, "dataset": "all", "seed": 0, "eval_base_model": False,
               "temperature": 1.0, "max_tokens": 16000, "parallel_checkpoints": 3}


def eval_blocks(model_name: str, template_kwargs: dict, conditions=EVAL_CONDITIONS, modes=SCREEN_MODES,
                eval_at_steps: list[int] | None = None, tp: int = 1) -> dict:
    blocks = {}
    for mode in modes:
        for cond in conditions:
            blk = {**EVAL_COMMON, **eval_block(cond, mode, model_name), "split": "val", "mode": mode,
                   "chat_template_kwargs": template_kwargs, "tensor_parallel_size": tp}
            blk["system_prompt"] = blk["system_prompt"] or ""
            if eval_at_steps is not None:
                blk["eval_at_steps"] = list(eval_at_steps)
            blocks[f"{cond}_{mode}"] = blk
    return blocks


def lora_section(base_model: str) -> dict:
    mods, params = default_lora_targets(base_model)
    return {"lora_rank": 32, "lora_alpha": 32, "lora_target_modules": list(mods),
            "lora_target_parameters": list(params)}


def sft_config(variant: str, key: str, seed: int = 0, smoke: bool = False) -> dict:
    spec = MODELS[key]
    train = {
        "data_path": f"kaushik/orange_expand/sft/data/sft_{variant}.jsonl",
        "shuffle": True, "seed": seed,
        "chat_template_kwargs": spec["train_template_kwargs"],
        "lr": 1.5e-4, "lr_scheduler_type": "constant",
        "batch_size": 48, "per_device_batch_size": 8,
        "num_epochs": 3,              # 100 x 48 = 4800 presentations over 1936 rows
        "max_steps": FINAL_STEP, "max_seq_length": 1024,
        "gradient_checkpointing": False, "save_at_steps": list(SAVE_AT),
    }
    for k in ("attn_implementation", "assistant_prefill", "per_device_batch_size", "gradient_checkpointing"):
        if k in spec:
            train[k] = spec[k]
    name = f"{PREFIX}-sft_{variant}-{key}-seed{seed}"
    run_dir = f"kaushik/orange_expand/sft/runs/{name}"
    final_step = FINAL_STEP
    tp = spec.get("eval_tensor_parallel_size", 1)
    blocks = eval_blocks(spec["base_model"], spec["eval_template_kwargs"], eval_at_steps=[final_step], tp=tp)
    if smoke:
        name, run_dir, final_step = f"smoke-{name}", f"kaushik/orange_expand/sft/runs/smoke/smoke-{name}", 2
        train.update({"max_steps": 2, "save_at_steps": [2], "num_epochs": 1})
        blocks = {k: {**v, "max_samples": 8, "eval_at_steps": [2]} for k, v in blocks.items()
                  if k in ("persona_lowercase_thinking", "standard_lowercase_thinking")}
    return {
        "run_name": name, "base_model": spec["base_model"], "gpu": spec["gpu"], "run_dir": run_dir,
        "meta": {"experiment": "orange_expand", "stage": 1, "arm": f"sft_{variant}", "train_prompt": variant,
                 "model_key": key, "final_step": final_step, "smoke": smoke},
        "train": train, "lora": lora_section(spec["base_model"]), "eval": blocks,
    }


def write_config(cfg: dict, out_dir: Path = CONFIG_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg['run_name']}.yaml"
    path.write_text(yaml.dump(cfg, sort_keys=False, width=1000))
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=list(MODELS))
    ap.add_argument("--variants", nargs="+", default=list(TRAIN_PROMPTS), choices=list(TRAIN_PROMPTS))
    args = ap.parse_args(argv)
    for key in args.models:
        for variant in args.variants:
            cfg = sft_config(variant, key)
            print(f"wrote {write_config(cfg).relative_to(ROOT)} ({len(cfg['eval'])} eval blocks, gpu {cfg['gpu']})")


if __name__ == "__main__":
    main()
