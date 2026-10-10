"""Configs for the orange-expansion RL arm (PLAN.md stage 1, RL half): GRPO with
reasoning ON, training prompt `eugene` (prompts.T_EUGENE), reward 1 iff the
answer is the orange option, one seed per model.

Settings are 2B's (kaushik/orange/rl/make_configs.py, arm rl_orange_reason):
48 prompts x 8 rollouts per iteration, LR 1.5e-4 constant, LoRA r32/a32
all-linear, DAPO loss, centered advantages, no KL, grad-clip 1.0, rollout cap
8192 tokens. The iteration count is NOT fixed: `rl.iterations` is the 150 cap
and `task.stop` holds the rule the driver applies (>= 50 iterations, then stop
once mean reward >= 0.98 for 20 consecutive iterations; abort if iteration 0 is
already >= 0.9). The worker saves every iteration; 0/25/50/... are the
held-out monitor points.

Eval blocks: prompts.EVAL_CONDITIONS x prompts.SCREEN_MODES on val (200
questions), T=1, 16k tokens; `eval_at_steps` is filled in at eval time with
the run's final iteration (unknown until the stop rule fires).

    .venv/bin/python kaushik/run.py orange_expand/rl configs [--models ...] [--smoke]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cotcontrol.training.config import default_lora_targets
from kaushik.orange_expand import prompts as oxp

REL = HERE.relative_to(ROOT)                     # kaushik/orange_expand/rl
DATA_DIR = ROOT / "kaushik" / "orange" / "rl" / "data"   # 2B's pool slices (reused as-is)
RUNS_DIR = REL / "runs"
CONFIGS_DIR = HERE / "configs"
PREFIX = "oex-rl-eugene"

MODELS = {
    "gptoss20b": {
        "base_model": "openai/gpt-oss-20b",
        "gpu": "H200",
        "template_kwargs": {"reasoning_effort": "medium"},
        # gpt-oss has no SDPA path; FA3-with-sinks is what every SFT/CPT run in
        # the repo uses for it (orange-training, fragility/cpt).
        "attn_implementation": "kernels-community/vllm-flash-attn3",
        "rollout_shards": 4,   # 40-GPU cap (coordinator): a run = 1 trainer + shards engines
    },
    "qwen3_32b": {
        "base_model": "Qwen/Qwen3-32B",
        # sdf trained the 32B on one H200 (bs 1, grad ckpt); the RL worker's train step
        # dominates the iteration, so the trainer gets 2 GPUs under FSDP (TrainConfig's
        # default parallelism; rl/configs/smoke_qwen8b_2gpu_resume.yaml). Engines stay 1 GPU.
        "gpu": "H200:2",
        "template_kwargs": {"enable_thinking": True},
        "attn_implementation": None,
        "rollout_shards": 8,   # 2B's 27B ran 4; doubled under the 40-GPU cap
    },
}
ITER_CAP, MIN_ITERS, STOP_THRESHOLD, STOP_WINDOW, ABORT_IT0 = 150, 50, 0.98, 20, 0.9
PROMPTS_PER_ITER, GROUP_SIZE, MAX_ROLLOUT = 48, 8, 8192
HELDOUT_AT = list(range(0, ITER_CAP + 1, 25))
EVAL_CHUNK_PARALLEL = 2   # 200 val questions = 2 chunks of 100 -> 2 engines per block


def eval_blocks(model_key: str, max_samples: int | None = None, max_tokens: int = 16000,
                conditions=oxp.EVAL_CONDITIONS, modes=oxp.SCREEN_MODES) -> dict:
    spec = MODELS[model_key]
    blocks = {}
    for cond in conditions:
        for mode in modes:
            eb = oxp.eval_block(cond, mode, spec["base_model"])
            blocks[f"{cond}_{mode}"] = {
                "run": True, "dataset": "all", "split": "val", "mode": mode, "seed": 0,
                "system_prompt": eb["system_prompt"] or "",
                "state_requirement": bool(eb["state_requirement"]),
                "eval_base_model": False,
                "temperature": 1.0, "max_tokens": max_tokens, "max_model_len": 24576,
                "gpu": spec["gpu"].split(":")[0], "tensor_parallel_size": 1,
                "chat_template_kwargs": dict(spec["template_kwargs"]),
                "parallel_checkpoints": 1, "chunk": 100, "chunk_parallel": EVAL_CHUNK_PARALLEL,
                "eval_at_steps": None,   # final iteration, set by cli.eval
                "max_samples": max_samples,
            }
    return blocks


GUARD = {"guard_chars_mult": 3.0, "guard_trunc": 0.05, "guard_window": 3, "smooth": 3}


def make_config(model_key: str, seed: int = 0, *, smoke: bool = False, lr: float = 1.5e-4,
                suffix: str = "") -> dict:
    """smoke: 3-iteration cap with a stop rule that fires at iteration 2 (so the
    early-stop path is exercised), 4 prompts x 4 rollouts of 512 tokens, 2 eval
    blocks of 6 questions."""
    spec = MODELS[model_key]
    tag = f"{PREFIX}-{model_key}-seed{seed}" + suffix + ("-smoke" if smoke else "")
    mods, params = default_lora_targets(spec["base_model"])
    train = {
        "lr": lr,
        "seed": seed,
        "per_device_batch_size": 1,          # prompt + 8192-token rollouts; 1 is safe on the 27B/32B
        "gradient_checkpointing": True,
        "max_seq_length": MAX_ROLLOUT + 1024,
        "chat_template_kwargs": dict(spec["template_kwargs"]),
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
    }
    if spec["attn_implementation"]:
        train["attn_implementation"] = spec["attn_implementation"]
    cfg = {
        "run_name": tag,
        "base_model": spec["base_model"],
        "gpu": spec["gpu"],
        "run_dir": str(RUNS_DIR / tag),
        "train": train,
        "rl": {
            "iterations": 3 if smoke else ITER_CAP,
            "prompts_per_iter": 4 if smoke else PROMPTS_PER_ITER,
            "group_size": 4 if smoke else GROUP_SIZE,
            "rollout_max_tokens": 512 if smoke else MAX_ROLLOUT,
            "reward": "answer_match",
            "advantage_scale": "none",
            "loss_type": "dapo",
            "drop_degenerate_groups": True,
            "drop_truncated": False,
            "save_optimizer_every": 5,
            "handshake_timeout_s": 7200,
            "logprob_chunk_size": 1024,
        },
        "lora": {
            "lora_rank": 32,
            "lora_alpha": 32,
            "lora_target_modules": list(mods),
            "lora_target_parameters": list(params),
        },
        "inference": {
            # One GPU per engine even when the trainer is "H200:2" (the driver would
            # otherwise inherit the trainer spec and run every engine on 2 GPUs).
            "gpu": spec["gpu"].split(":")[0],
            "max_model_len": 4096 if smoke else MAX_ROLLOUT + 2048,
            "max_num_seqs": 64 if smoke else 512,
            "rollout_shards": 1 if smoke else spec["rollout_shards"],
        },
        "task": {
            "arm": "rl_eugene_reason",
            "train_prompt": "eugene",
            "pool": "orange",
            "reasoning": True,
            "train_data": str(DATA_DIR.relative_to(ROOT) / "orange_train.jsonl"),
            "heldout_data": str(DATA_DIR.relative_to(ROOT) / "orange_heldout.jsonl"),
            "heldout_at": [0, 3] if smoke else HELDOUT_AT,
            "heldout_n": 8 if smoke else 96,
            "heldout_samples": 2 if smoke else 4,
            # The driver's stop rule (see driver.py). smoke: fires after iteration 1
            # -> final checkpoint-2, below the 3-iteration cap.
            "stop": ({"min_iters": 2, "threshold": 0.0, "window": 1, "abort_it0": 2.0, "guard_window": 0}
                     if smoke else
                     {"min_iters": MIN_ITERS, "threshold": STOP_THRESHOLD, "window": STOP_WINDOW,
                      "abort_it0": ABORT_IT0, **GUARD}),
        },
        "eval": (eval_blocks(model_key, max_samples=6, max_tokens=1024,
                             conditions=["persona", "standard"], modes=oxp.SCREEN_MODES[:1])
                 if smoke else eval_blocks(model_key)),
    }
    if smoke:
        for blk in cfg["eval"].values():
            blk.update(max_model_len=4096)
    return cfg


def write_config(cfg: dict, out_dir: Path = CONFIGS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg['run_name']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=1000))
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--suffix", default="", help="run-name suffix, e.g. -lr5e-5")
    a = ap.parse_args(argv)
    paths = []
    for m in a.models:
        for s in a.seeds:
            cfg = make_config(m, s, smoke=a.smoke, lr=a.lr, suffix=a.suffix)
            paths.append(write_config(cfg))
            print(f"wrote {paths[-1].relative_to(ROOT)} ({len(cfg['eval'])} eval blocks)")
    return paths


if __name__ == "__main__":
    main()
