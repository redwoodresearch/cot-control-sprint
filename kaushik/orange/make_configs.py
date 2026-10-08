"""Emit train_eval configs for Experiment 2A: orange vs orange_nopersona.

One config per (arm, model, seed). Training is orange-training/make_configs.py's
recipe verbatim (its per-model MODELS table is imported): LoRA r32/alpha32 on
all linear layers, LR 1.5e-4 constant, effective batch 48, 50 steps, reasoning
conditioned off (Qwen enable_thinking=False; gpt-oss empty analysis-channel
prefill), checkpoints at 10..50. The arms differ only in the dataset file:
orange carries the EUGENE system prompt, orange_nopersona has no system prompt.

Eval blocks (sprint "Shared setup": T=1, max_tokens 16000, honest compliance):
  <cond>_<mode>         every checkpoint, val (200 q), cond in persona/direct/stated,
                        mode in VAL_MODES. Checkpoint-0 = untrained -> the
                        prompt-only persona baseline.
  final_<cond>_<mode>   the same on test (500 q), final checkpoint only.
  final_standard        test, mode=all (the 9 training modes), stated, final only.
  final_heldout         test, the 3 held-out modes, stated, final only.
  final_baseline        test, unconstrained baseline mode (capability check).

The configs are valid scripts/train_eval.py configs; kaushik/orange runs them
through train_run.py + eval_scheduler.py (one Modal app, global GPU budget).

    .venv/bin/python kaushik/run.py orange configs [--models qwen36_27b gptoss20b] [--seeds ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import orange_training
from prompts import direct_eval_prompt, persona_eval_prompt

from cotcontrol.eval.prompts import HELDOUT_MODES
from cotcontrol.training.config import default_lora_targets

MODELS = orange_training.load("make_configs").MODELS  # qwen8b / gptoss20b / qwen36_27b
ARMS = {
    "orange": "kaushik/orange/data/orange_mcq_sft.jsonl",
    "orange_nopersona": "kaushik/orange/data/orange_mcq_sft_nopersona.jsonl",
}
DEFAULT_SEEDS = {"qwen36_27b": [0, 1, 2, 3, 4], "gptoss20b": [0, 1, 2]}
VAL_MODES = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "end_of_sentence"]
CONDITIONS = ["persona", "direct", "stated"]
FINAL_STEP = 50
# Sprint eval settings. train_eval.py defaults to T=0 / 12k, so set both explicitly.
EVAL_COMMON = {
    "run": True, "dataset": "all", "seed": 0, "eval_base_model": False,
    "temperature": 1.0, "max_tokens": 16000, "parallel_checkpoints": 3,
}
# Engine call size for the multi-mode final blocks (4500 / 1500 prompts): the
# scheduler reserves ceil(n_prompts / chunk) GPUs per job and fans chunks out.
CHUNK = 750


def condition_block(cond: str, mode: str, model_name: str, split: str,
                    template_kwargs: dict) -> dict:
    if cond == "persona":
        sp, stated = persona_eval_prompt(mode, model_name), False
    elif cond == "direct":
        sp, stated = direct_eval_prompt(mode, model_name), False
    elif cond == "stated":
        sp, stated = "", True
    else:
        raise ValueError(cond)
    return {**EVAL_COMMON, "split": split, "mode": mode, "system_prompt": sp,
            "state_requirement": stated, "chat_template_kwargs": template_kwargs}


def eval_blocks(model_name: str, template_kwargs: dict, final_step: int = FINAL_STEP,
                modes: list[str] = VAL_MODES, final_steps: list[int] | None = None) -> dict:
    """final_steps: checkpoints the test-split blocks evaluate (default [final_step];
    the orange arm adds step 0 so the table has an untrained reference on test)."""
    final_steps = final_steps or [final_step]
    blocks = {}
    for mode in modes:
        for cond in CONDITIONS:
            blocks[f"{cond}_{mode}"] = condition_block(cond, mode, model_name, "val", template_kwargs)
    for mode in modes:
        for cond in CONDITIONS:
            blocks[f"final_{cond}_{mode}"] = {
                **condition_block(cond, mode, model_name, "test", template_kwargs),
                "eval_at_steps": list(final_steps)}
    standard = {**EVAL_COMMON, "split": "test", "system_prompt": "", "state_requirement": True,
                "chat_template_kwargs": template_kwargs, "eval_at_steps": list(final_steps),
                "chunk": CHUNK}
    blocks["final_standard"] = {**standard, "mode": "all"}
    blocks["final_heldout"] = {**standard, "mode": "all", "allowed_modes": list(HELDOUT_MODES)}
    blocks["final_baseline"] = {**standard, "mode": "baseline", "chunk": 0}
    return blocks


def make_config(arm: str, key: str, seed: int, smoke: bool = False) -> dict:
    spec = MODELS[key]
    train = {  # orange-training/make_configs.py, unchanged
        "data_path": ARMS[arm],
        "shuffle": True,
        "seed": seed,
        "chat_template_kwargs": spec["train_template_kwargs"],
        "lr": 1.5e-4,
        "lr_scheduler_type": "constant",
        "batch_size": 48,
        "per_device_batch_size": 8,
        "num_epochs": 2,              # 50 x 48 = 2400 presentations over a ~2.1k pool
        "max_steps": FINAL_STEP,
        "max_seq_length": 1024,
        "gradient_checkpointing": False,
        "save_at_steps": [10, 20, 30, 40, 50],
    }
    for k in ("attn_implementation", "assistant_prefill", "per_device_batch_size",
              "gradient_checkpointing"):
        if k in spec:
            train[k] = spec[k]
    run_name = f"{arm}-{key}-seed{seed}"
    run_dir = f"kaushik/orange/runs/{run_name}"
    final_step = FINAL_STEP
    if smoke:
        run_name = f"smoke-{run_name}"
        run_dir = f"kaushik/orange/runs/smoke/{run_name}"
        final_step = 2
        train.update({"max_steps": 2, "save_at_steps": [2], "num_epochs": 1})
    blocks = eval_blocks(spec["base_model"], spec["eval_template_kwargs"], final_step,
                         final_steps=[0, final_step] if arm == "orange" else [final_step])
    if smoke:
        keep = ["persona_lowercase_thinking", "stated_lowercase_thinking",
                "final_persona_lowercase_thinking", "final_standard", "final_baseline"]
        blocks = {k: {**v, "max_samples": 2 if k == "final_standard" else 8} for k, v in blocks.items() if k in keep}
    targets = default_lora_targets(spec["base_model"])
    return {
        "run_name": run_name,
        "base_model": spec["base_model"],
        "gpu": "H200",
        "run_dir": run_dir,
        # Read by kaushik/orange tooling only; train_eval.py drops unknown keys.
        "meta": {"experiment": "orange", "arm": arm, "model_key": key, "final_step": final_step,
                 "smoke": smoke},
        "train": train,
        # All-linear coverage, as orange-training (attention-only is degenerate on
        # Qwen3.6-27B: q/k/v/o exists in 16 of 64 layers).
        "lora": {"lora_rank": 32, "lora_alpha": 32,
                 "lora_target_modules": list(targets[0]),
                 "lora_target_parameters": list(targets[1])},
        "eval": blocks,
    }


def write_config(cfg: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg['run_name']}.yaml"
    path.write_text(yaml.dump(cfg, sort_keys=False, width=1000))
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_SEEDS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="default: 0-4 for qwen36_27b, 0-2 for gptoss20b")
    ap.add_argument("--smoke", action="store_true", help="2-step / 8-question configs -> configs/smoke/")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else HERE / ("configs/smoke" if args.smoke else "configs")
    for key in args.models:
        seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS.get(key, [0, 1, 2])
        for arm in args.arms:
            for seed in seeds:
                cfg = make_config(arm, key, seed, smoke=args.smoke)
                path = write_config(cfg, out_dir)
                print(f"wrote {path.relative_to(ROOT)} ({len(cfg['eval'])} eval blocks)")


if __name__ == "__main__":
    main()
