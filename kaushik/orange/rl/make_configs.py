"""Emit the Experiment 2B run configs (one YAML per arm x model x seed).

Arms (spec 2B): rl_orange_{noreason,reason} = EUGENE persona + orange MCQ pool,
reward 1 iff the answer is the orange option; rl_mcq_{noreason,reason} = the
matched baselines on the generic MCQ pool, reward = correct answer, no persona.
"noreason" conditions on an empty reasoning block (Qwen enable_thinking=False in
the rollout chat template, so the empty <think></think> is part of the PROMPT
the worker trains against); "reason" samples with thinking on and rewards the
final answer only.

Training dose is matched to 2A's SFT (orange-training/make_configs.py) step for
step: 50 optimizer steps x 48 prompts per step, LR 1.5e-4 constant, LoRA
r32/alpha32 all-linear, grad-clip 1.0. Adam normalizes gradient scale, so with
matched LR and steps the adapter moves a comparable distance per step; GRPO's
extra factor is group_size=8 rollouts per prompt. RL's realized dose is
self-limiting (groups whose 8 rollouts all share a reward carry no gradient and
are dropped), which the worker's metrics.jsonl records as n_groups_used.

Checkpoints: the RL worker saves every iteration; eval_at_steps picks
0/10/20/30/40/50 (2A's cadence). Eval blocks reuse orange-training's
eval_blocks semantics (persona/direct/stated x 4 fixed-trait modes) at the
shared settings: val split, T=1, max_tokens 16000. final_eval adds the test
split, the standard 9-mode stated eval, the 3 held-out modes and the
unconstrained baseline, on the final checkpoint.

    .venv/bin/python kaushik/run.py orange/rl configs [--seeds 0 1 2 3 4] [--smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from cotcontrol.eval.prompts import HELDOUT_MODES
from cotcontrol.training.config import default_lora_targets

REL = HERE.relative_to(ROOT)  # kaushik/orange/rl
DATA_DIR = REL / "data"
RUNS_DIR = REL / "runs"
CONFIGS_DIR = HERE / "configs"


def _orange_eval_blocks():
    """orange-training/make_configs.py:eval_blocks, loaded by path (hyphenated
    folder; it imports `prompts` by bare name, so put the folder on sys.path)."""
    orange_dir = str(ROOT / "orange-training")
    before = set(sys.modules)
    sys.path.insert(0, orange_dir)
    try:
        spec = importlib.util.spec_from_file_location("orange_make_configs",
                                                      ROOT / "orange-training" / "make_configs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(orange_dir)
        # Its bare-named sibling imports (`prompts`) must not stay cached for
        # other folders' modules of the same name; mod keeps its own references.
        for k in set(sys.modules) - before:
            f = getattr(sys.modules[k], "__file__", None) or ""
            if Path(f).resolve().parent == ROOT / "orange-training":
                del sys.modules[k]
    return mod.eval_blocks


ARMS = {
    "rl_orange_noreason": {"pool": "orange", "reasoning": False},
    "rl_orange_reason": {"pool": "orange", "reasoning": True},
    "rl_mcq_noreason": {"pool": "mcq", "reasoning": False},
    "rl_mcq_reason": {"pool": "mcq", "reasoning": True},
}
MODELS = {
    "qwen36_27b": {
        "base_model": "Qwen/Qwen3.6-27B",
        "gpu": "H200",
        "template_kwargs": {False: {"enable_thinking": False}, True: {"enable_thinking": True}},
        "eval_template_kwargs": {"enable_thinking": True},
    },
    # Smoke-only: the cached tiny model, mechanics not learning. H200 rather
    # than L40S: the workspace queues for L40S while H200s schedule in minutes.
    "qwen3_06b": {
        "base_model": "Qwen/Qwen3-0.6B",
        "gpu": "H200",
        "template_kwargs": {False: {"enable_thinking": False}, True: {"enable_thinking": True}},
        "eval_template_kwargs": {"enable_thinking": True},
    },
}
VAL_MODES = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "end_of_sentence"]
CONDITIONS = ["persona", "direct", "stated"]
ITERATIONS = 50
PROMPTS_PER_ITER = 48
CKPT_STEPS = [0, 10, 20, 30, 40, 50]
EVAL_PARALLEL = 6  # concurrent engines per eval job == its GPU cost in cli.launch
STANDARD_EVAL = {  # shared settings (spec "Shared setup")
    "temperature": 1.0,
    "max_tokens": 16000,
    "max_model_len": 24576,
    "parallel_checkpoints": EVAL_PARALLEL,
    # train_eval.py runs blocks sequentially and fans a block's checkpoints out
    # concurrently; a final-eval block has 1-2 checkpoints x 500 questions, so
    # without chunking it would run on 1-2 engines for ~30 min. Chunking each
    # block's prompts into 100-prompt engine calls keeps EVAL_PARALLEL engines
    # busy (the job's GPU cost) for every block.
    "chunk": 100,
    "chunk_parallel": EVAL_PARALLEL,
}


def val_blocks(base_model: str, template_kwargs: dict, steps: list[int],
               modes=VAL_MODES, max_samples=None) -> dict:
    blocks = _orange_eval_blocks()(base_model, modes, CONDITIONS, template_kwargs)
    for b in blocks.values():
        b.update(STANDARD_EVAL, split="val", eval_at_steps=steps, max_samples=max_samples)
    return blocks


def final_blocks(base_model: str, template_kwargs: dict, steps: list[int],
                 modes=VAL_MODES, max_samples=None) -> dict:
    blocks = _orange_eval_blocks()(base_model, modes, CONDITIONS, template_kwargs)
    common = {"run": True, "dataset": "all", "split": "test", "seed": 0, "system_prompt": "",
              "state_requirement": True, "eval_base_model": False,
              "chat_template_kwargs": template_kwargs}
    blocks["stated_standard"] = {**common, "mode": "random"}               # 9 training modes
    blocks["stated_heldout"] = {**common, "mode": "random", "allowed_modes": HELDOUT_MODES}
    blocks["baseline"] = {**common, "mode": "baseline"}                    # unconstrained accuracy
    for b in blocks.values():
        b.update(STANDARD_EVAL, split="test", eval_at_steps=steps, max_samples=max_samples)
    return blocks


def make_config(arm: str, model_key: str, seed: int, *, smoke: bool = False,
                full_rollouts: bool = False, untrained_on_test: bool = False) -> dict:
    """smoke: 2 iterations, tiny rollouts and 6-question evals (mechanics).
    smoke + full_rollouts: 2 iterations at the real rollout sizes with tiny
    evals — the memory/timing check for a new base model."""
    spec, a = MODELS[model_key], ARMS[arm]
    reasoning, pool = a["reasoning"], a["pool"]
    tiny = smoke and not full_rollouts
    tag = f"rl2b-{arm}-{model_key}-seed{seed}" + ("-smoke" if smoke else "") + ("-full" if smoke and full_rollouts else "")
    iterations = 2 if smoke else ITERATIONS
    steps = [0, iterations] if smoke else CKPT_STEPS
    targets = default_lora_targets(spec["base_model"])
    # Reasoning-on rollouts on simple MCQs run a few hundred to a few thousand
    # tokens; 8192 keeps truncation (-> no answer -> reward 0) rare so the
    # reward puts no length pressure on the CoT. Reasoning-off answers are a
    # letter; 256 leaves room for a model that ignores the empty think block.
    max_rollout = 8192 if reasoning else 256
    cfg = {
        "run_name": tag,
        "base_model": spec["base_model"],
        "gpu": spec["gpu"],
        "run_dir": str(RUNS_DIR / tag),
        "train": {
            "lr": 1.5e-4,
            "seed": seed,
            # Reasoning-on sequences reach prompt + 8192 tokens: 4 x 8.3k tokens per
            # micro-batch OOM'd the 27B (133 GB); 1 is safe and only the
            # non-degenerate groups are trained anyway.
            "per_device_batch_size": 1 if reasoning else 16,
            "gradient_checkpointing": True,
            "max_seq_length": max_rollout + 1024,
            "chat_template_kwargs": spec["template_kwargs"][reasoning],
            "max_grad_norm": 1.0,
            "weight_decay": 0.0,
        },
        "rl": {
            "iterations": iterations,
            "prompts_per_iter": 4 if tiny else PROMPTS_PER_ITER,
            "group_size": 4 if tiny else 8,
            "rollout_max_tokens": (512 if reasoning else 64) if tiny else max_rollout,
            "reward": "answer_match",
            "advantage_scale": "none",
            "loss_type": "dapo",
            "drop_degenerate_groups": True,
            "drop_truncated": False,
            # Resume granularity after a driver/worker failure (each save ~1.6 GB).
            "save_optimizer_every": 5,
            "handshake_timeout_s": 7200,
            "logprob_chunk_size": 1024,
        },
        "lora": {
            "lora_rank": 32,
            "lora_alpha": 32,
            "lora_target_modules": list(targets[0]),
            "lora_target_parameters": list(targets[1]),
        },
        "inference": {
            "gpu": spec["gpu"],
            "max_model_len": 4096 if tiny else max_rollout + 2048,
            # 48 x 8 = 384 rollouts per iteration: let one engine batch them all.
            "max_num_seqs": 64 if tiny else 512,
            # Reasoning-on rollouts are long: split each iteration's generation
            # over 2 engine containers (GPU cost of a run = 1 trainer + shards).
            "rollout_shards": 1 if (tiny or not reasoning) else 2,
        },
        "task": {
            "arm": arm,
            "pool": pool,
            "reasoning": reasoning,
            "train_data": str(DATA_DIR / f"{pool}_train.jsonl"),
            "heldout_data": str(DATA_DIR / f"{pool}_heldout.jsonl"),
            "heldout_at": steps,
            "heldout_n": 8 if tiny else 96,
            "heldout_samples": 2 if tiny else 4,
        },
        "eval": val_blocks(spec["base_model"], spec["eval_template_kwargs"], steps,
                           modes=VAL_MODES[:1] if smoke else VAL_MODES,
                           max_samples=6 if smoke else None),
        "final_eval": final_blocks(spec["base_model"], spec["eval_template_kwargs"],
                                   [0, iterations] if (untrained_on_test or smoke) else [iterations],
                                   modes=VAL_MODES[:1] if smoke else VAL_MODES,
                                   max_samples=6 if smoke else None),
    }
    if smoke:
        for blk in list(cfg["eval"].values()) + list(cfg["final_eval"].values()):
            blk.update(max_tokens=1024, max_model_len=4096, parallel_checkpoints=2)
    return cfg


def write_config(cfg: dict, out_dir: Path = CONFIGS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg['run_name']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=1000))
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--model", default="qwen36_27b", choices=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true", help="2-iteration mechanics configs")
    ap.add_argument("--full-rollouts", action="store_true",
                    help="with --smoke: real rollout sizes, tiny evals (memory/timing check)")
    ap.add_argument("--untrained-on-test", nargs="*", default=["rl_mcq_noreason"],
                    help="arms whose checkpoint-0 also gets the test-split final eval "
                         "(checkpoint-0 is the base model, so one arm's 5 seeds suffice)")
    args = ap.parse_args(argv)
    paths = []
    for arm in args.arms:
        for seed in args.seeds:
            cfg = make_config(arm, args.model, seed, smoke=args.smoke, full_rollouts=args.full_rollouts,
                              untrained_on_test=arm in args.untrained_on_test)
            paths.append(write_config(cfg))
            print(f"wrote {paths[-1].relative_to(ROOT)} ({len(cfg['eval'])} val blocks, "
                  f"{len(cfg['final_eval'])} final blocks)")
    return paths


if __name__ == "__main__":
    main()
