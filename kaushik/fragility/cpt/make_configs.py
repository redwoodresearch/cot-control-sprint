"""train_eval.py configs for Experiment 1 arms 1B (`cpt`) and 1C (`cpt_param_sweep`).

Spec: kaushik/sprint_2026_10_08.md. Both arms continue-pretrain on raw C4 text
(next-token loss on every token, no chat template, packed to MAX_SEQ_LENGTH)
with orange-training's optimizer settings: constant LR, effective batch 48.

1B  cpt-<model>-r32-seed<s>        LoRA r32/a32 all-linear, LR 1.5e-4, 200 steps.
                                    Step 50 is the 1A-matched dose (50 x 48
                                    sequences); step 200 is the 4x dose.
1C  cptk-<model>-<k>-lr<lr>-seed<s> rank-1/alpha-1 all-linear, k random trainable
                                    scalars (train_params, mask_seed = seed),
                                    50 steps. k in {1e3, 1e4, 1e5} + the full
                                    rank-1 adapter ("r1"); 1B@50 is the rank-32
                                    top point.

LR for 1C follows the iso-drive rule from sft/lrk_analysis.py: the quantity
(alpha/r) * lr * sqrt(k) is held at 1B's value, i.e. lr(k) = 1.5e-4 *
sqrt(N_r32 / k) (alpha/r = 1 in both arms). Stage 1 sweeps lr_pred x {1/3, 1, 3}
at seed 0; `cli.py select` picks the LR on val and stage 2 runs seeds 1-2.

Every eval block samples at T=1 / 16k tokens with reasoning enabled through
eval-side template kwargs. Val blocks (every checkpoint): the four fixed-trait
modes, 200 questions each. Test blocks (final checkpoints only): the 9 training
modes (mode=all), the 3 held-out modes (mode=all) and the unconstrained
`baseline` mode for accuracy.

    .venv/bin/python kaushik/run.py fragility/cpt configs        # writes configs/
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from cotcontrol.eval.prompts import HELDOUT_MODES
from cotcontrol.training.config import default_lora_targets

REL = HERE.relative_to(ROOT)  # kaushik/fragility/cpt
CONFIG_DIR = HERE / "configs"
RUNS_DIR = HERE / "runs"
DATA = f"{REL}/data/c4_cpt.jsonl"              # sized for 200 steps (1B)
DATA_50 = f"{REL}/data/c4_cpt_50steps.jsonl"   # prefix of the same shuffle, sized for 50 steps (1C)
SMOKE_DATA = f"{REL}/data/c4_smoke.jsonl"

# Shapes feed adapter_numel(): the all-linear LoRA size per unit rank. PEFT's
# ParamWrapper puts LoRA on gpt-oss's fused experts as A=[in, r*E], B=[r*E, out],
# so an expert tensor costs E*(in+out) per rank, like E separate linears.
MODELS = {
    "gptoss20b": {
        "base_model": "openai/gpt-oss-20b",
        "eval_template_kwargs": {"reasoning_effort": "medium"},
        "attn_implementation": "kernels-community/vllm-flash-attn3",  # FA3 with sinks
        "dims": {"layers": 24, "hidden": 2880, "q": 4096, "kv": 512, "inter": 2880, "experts": 32},
    },
    "qwen8b": {
        "base_model": "Qwen/Qwen3-8B",
        "eval_template_kwargs": {"enable_thinking": True},
        "attn_implementation": None,  # sdpa
        "dims": {"layers": 36, "hidden": 4096, "q": 4096, "kv": 1024, "inter": 12288, "experts": 0},
    },
}

MAX_SEQ_LENGTH = 4096      # packed C4; 48 x 4096 = 197k supervised tokens / step
BATCH_SIZE = 48
PER_DEVICE_BATCH = 4
LR_1B = 1.5e-4
STEPS_1B, SAVE_1B = 200, [10, 20, 30, 40, 50, 100, 150, 200]
FINAL_1B = [50, 200]       # 1A-matched dose and 4x dose
STEPS_1C, SAVE_1C = 50, [10, 20, 30, 40, 50]
KS = {"1e3": 1_000, "1e4": 10_000, "1e5": 100_000, "r1": None}  # None = full rank-1
LR_FACTORS = (1 / 3, 1.0, 3.0)
SEEDS = [0, 1, 2]
VAL_MODES = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "end_of_sentence"]
# Extra per-checkpoint block, measured but not used for LR selection or the pooled
# val metric: the fixed-trait modes above sit at 0 for gpt-oss-20b before and after
# small-k CPT, while SDF's C4 arm moved gpt-oss on repeat_sentences (.01 -> .10), so
# this is the per-checkpoint dose signal for the modes the effect actually touches.
EXTRA_VAL_MODES = ["repeat_sentences"]
TEST_BLOCKS = ["test_train_modes", "test_heldout", "test_baseline"]
# Concurrent vLLM engines (= H200s) one run's eval phase may hold. The
# scheduler in cli.py reserves this many GPUs per eval job against the sprint
# cap (120 concurrent GPUs for this agent); training jobs reserve 1.
EVAL_GPUS = {"cpt": 8, "cpt_lowlr": 8, "cpt_param_sweep": 4, "smoke": 2}


def adapter_numel(model_key: str, rank: int) -> int:
    """Total LoRA scalars of an all-linear (default_lora_targets) adapter."""
    d = MODELS[model_key]["dims"]
    h = d["hidden"]
    attn = (h + d["q"]) + 2 * (h + d["kv"]) + (d["q"] + h)
    if d["experts"]:
        mlp = d["experts"] * ((h + 2 * d["inter"]) + (d["inter"] + h))  # gate_up + down
    else:
        mlp = 2 * (h + d["inter"]) + (d["inter"] + h)
    return rank * d["layers"] * (attn + mlp)


def drive(model_key: str) -> float:
    """(alpha/r) * lr * sqrt(k) of the 1B adapter (alpha/r = 1, k = all of it)."""
    return LR_1B * math.sqrt(adapter_numel(model_key, 32))


def predicted_lr(model_key: str, klabel: str) -> float:
    k = KS[klabel] or adapter_numel(model_key, 1)
    return drive(model_key) / math.sqrt(k)


def round_sig(x: float, sig: int = 2) -> float:
    return float(f"{x:.{sig - 1}e}")


def lr_grid_ext_path() -> Path:
    """Extra sweep rungs added by `cli.py select` when every LR of a (model, k)
    fails the accuracy guard: {"<model>/<klabel>": [lr, ...]}."""
    return CONFIG_DIR / "lr_grid_ext.yaml"


def lr_grid(model_key: str, klabel: str) -> list[float]:
    p = predicted_lr(model_key, klabel)
    grid = {round_sig(p * f) for f in LR_FACTORS}
    if lr_grid_ext_path().exists():
        grid |= set((yaml.safe_load(lr_grid_ext_path().read_text()) or {}).get(f"{model_key}/{klabel}", []))
    return sorted(grid)


def extend_lr_grid(model_key: str, klabel: str) -> float:
    """Add one rung a factor 3 below the current grid; returns the new LR."""
    lr = round_sig(min(lr_grid(model_key, klabel)) / 3)
    ext = (yaml.safe_load(lr_grid_ext_path().read_text()) or {}) if lr_grid_ext_path().exists() else {}
    ext.setdefault(f"{model_key}/{klabel}", []).append(lr)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lr_grid_ext_path().write_text(yaml.safe_dump(ext))
    return lr


def fmt_lr(lr: float) -> str:
    return f"{lr:.1e}"


def run_name_1b(model_key: str, seed: int, lr: float = LR_1B) -> str:
    return f"cpt-{model_key}-r32-seed{seed}" if lr == LR_1B else f"cpt-{model_key}-r32-lr{fmt_lr(lr)}-seed{seed}"


def run_name_1c(model_key: str, klabel: str, lr: float, seed: int) -> str:
    return f"cptk-{model_key}-{klabel}-lr{fmt_lr(lr)}-seed{seed}"


def _eval_block(model_key: str, gpus: int, **over) -> dict:
    blk = {
        "run": True,
        "dataset": "all",
        "split": "val",
        "seed": 0,
        "system_prompt": "",
        "eval_base_model": False,  # checkpoint-0 (zero adapter) is the untrained reference
        "temperature": 1.0,
        "max_tokens": 16000,
        "chat_template_kwargs": MODELS[model_key]["eval_template_kwargs"],
        "max_model_len": 32768,
        "parallel_checkpoints": gpus,  # concurrent engines = GPUs held by this block
    }
    blk.update(over)
    return blk


def val_blocks(model_key: str, gpus: int, **over) -> dict:
    # chunk: prompts per engine call. Smaller calls lose less to a dropped
    # client connection (every returned call is cached, so a retried eval
    # resumes from the last complete chunk); chunk_parallel caps concurrent
    # calls across checkpoints at the GPU footprint. Deployed-mode engines stay
    # warm between calls, so chunking adds no cold starts.
    return {f"val_{m}": _eval_block(model_key, gpus, mode=m, chunk=50, chunk_parallel=gpus, **over)
            for m in VAL_MODES + EXTRA_VAL_MODES}


def test_blocks(model_key: str, steps: list[int], gpus: int, **over) -> dict:
    common = {"split": "test", "eval_at_steps": steps, "chunk": 200, "chunk_parallel": gpus}
    return {
        "test_train_modes": _eval_block(model_key, gpus, mode="all", **common, **over),
        "test_heldout": _eval_block(model_key, gpus, mode="all",
                                    allowed_modes=list(HELDOUT_MODES), **common, **over),
        "test_baseline": _eval_block(model_key, gpus, mode="baseline", **common, **over),
    }


def eval_footprint(cfg: dict) -> int:
    """GPUs the eval phase of this config can hold at once."""
    blocks = [b for b in cfg["eval"].values() if b.get("run", True)]
    return max((int(b.get("chunk_parallel") or b["parallel_checkpoints"]) for b in blocks), default=0)


def _train(model_key: str, seed: int, lr: float, max_steps: int, save_at: list[int],
           data: str) -> dict:
    train = {
        "data_path": data,
        "pack_sequences": True,
        "shuffle": True,
        "seed": seed,
        "lr": lr,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,     # must stay 0 with train_params (AdamW decay bypasses the mask)
        "max_grad_norm": 1.0,
        "batch_size": BATCH_SIZE,
        "per_device_batch_size": PER_DEVICE_BATCH,
        "num_epochs": 1,
        "max_steps": max_steps,
        "max_seq_length": MAX_SEQ_LENGTH,
        "save_at_steps": save_at,
        "gradient_checkpointing": True,
    }
    if MODELS[model_key]["attn_implementation"]:
        train["attn_implementation"] = MODELS[model_key]["attn_implementation"]
    return train


def _lora(model_key: str, rank: int) -> dict:
    mods, params = default_lora_targets(MODELS[model_key]["base_model"])
    return {"lora_rank": rank, "lora_alpha": rank, "lora_dropout": 0.0,
            "lora_target_modules": list(mods), "lora_target_parameters": list(params)}


def _config(run_name: str, model_key: str, train: dict, lora: dict, elicitation: dict,
            evals: dict, arm: str, extra_meta: dict) -> dict:
    return {
        "run_name": run_name,
        "run_dir": f"{REL}/runs/{run_name}",
        "base_model": MODELS[model_key]["base_model"],
        "gpu": "H200",
        "cpt_meta": {"arm": arm, "model": model_key, **extra_meta},  # read by analyze.py
        "train": train,
        "lora": lora,
        "elicitation": elicitation,
        "eval": evals,
    }


def cpt_config(model_key: str, seed: int, lr: float = LR_1B, arm: str = "cpt") -> dict:
    """1B: full rank-32 adapter, 200 steps, val every checkpoint, test at 0/50/200.
    lr != LR_1B gives the supplementary low-LR variant (arm "cpt_lowlr"): the spec
    dose (1.5e-4) halves gpt-oss's accuracy within 50 steps, so this checks a
    non-destructive dose closer to SDF's C4 arm."""
    g = EVAL_GPUS["cpt"]
    evals = val_blocks(model_key, g) | test_blocks(model_key, [0] + FINAL_1B, g)
    return _config(
        run_name_1b(model_key, seed, lr), model_key,
        _train(model_key, seed, lr, STEPS_1B, SAVE_1B, DATA), _lora(model_key, 32),
        {"train_params": None, "mask_seed": seed}, evals, arm,
        {"seed": seed, "rank": 32, "klabel": "r32", "k": adapter_numel(model_key, 32),
         "lr": lr, "final_steps": FINAL_1B},
    )


def sweep_config(model_key: str, klabel: str, lr: float, seed: int, with_test: bool) -> dict:
    """1C: rank-1 adapter with k trainable scalars (klabel "r1" = all of them)."""
    g = EVAL_GPUS["cpt_param_sweep"]
    evals = val_blocks(model_key, g)
    if with_test:
        evals |= test_blocks(model_key, [STEPS_1C], g)
    k = KS[klabel]
    return _config(
        run_name_1c(model_key, klabel, lr, seed), model_key,
        _train(model_key, seed, lr, STEPS_1C, SAVE_1C, DATA_50), _lora(model_key, 1),
        {"train_params": k, "mask_seed": seed}, evals, "cpt_param_sweep",
        {"seed": seed, "rank": 1, "klabel": klabel, "k": k or adapter_numel(model_key, 1),
         "lr": lr, "lr_predicted": predicted_lr(model_key, klabel), "final_steps": [STEPS_1C]},
    )


def smoke_configs() -> list[dict]:
    """2 optimizer steps, 8 questions per block: exercises packing, masking,
    checkpoint saving, both eval paths (incl. the ignore_question judge)."""
    out = []
    for model_key in MODELS:
        for klabel, rank in (("r32", 32), ("1e4", 1)):
            name = f"smoke-{model_key}-{klabel}"
            lr = LR_1B if rank == 32 else predicted_lr(model_key, klabel)
            k = None if rank == 32 else KS[klabel]
            g = EVAL_GPUS["smoke"]
            evals = (val_blocks(model_key, g, max_samples=8)
                     | test_blocks(model_key, [2], g, max_samples=8))
            evals = {n: b for n, b in evals.items()
                     if n in ("val_lowercase_thinking", *TEST_BLOCKS)}
            out.append(_config(
                name, model_key, _train(model_key, 0, lr, 2, [1, 2], SMOKE_DATA),
                _lora(model_key, rank), {"train_params": k, "mask_seed": 0}, evals, "smoke",
                {"seed": 0, "rank": rank, "klabel": klabel, "k": k, "lr": lr, "final_steps": [2]},
            ))
    return out


def stage1_configs(models: list[str] | None = None) -> list[dict]:
    """1B (3 seeds) + the 1C seed-0 LR sweep, per model."""
    cfgs = []
    for m in models or list(MODELS):
        cfgs += [cpt_config(m, s) for s in SEEDS]
        for klabel in KS:
            cfgs += [sweep_config(m, klabel, lr, 0, with_test=False) for lr in lr_grid(m, klabel)]
    return cfgs


def stage2_configs(model_key: str, klabel: str, lr: float) -> list[dict]:
    """Seeds 1-2 at the val-selected LR, with test blocks."""
    return [sweep_config(model_key, klabel, lr, s, with_test=True) for s in SEEDS[1:]]


def write(cfg: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{cfg['run_name']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=120))
    return path


def main(argv: list[str] | None = None) -> None:
    paths = [write(c) for c in stage1_configs() + smoke_configs()]
    for m in MODELS:
        print(f"{m}: N_r32={adapter_numel(m, 32):,} N_r1={adapter_numel(m, 1):,} "
              f"drive={drive(m):.3f}")
        for klabel in KS:
            print(f"  {klabel:>4}: lr_pred={predicted_lr(m, klabel):.2e} grid={lr_grid(m, klabel)}")
    print(f"wrote {len(paths)} configs to {CONFIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
