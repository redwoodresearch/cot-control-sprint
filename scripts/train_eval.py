"""Config-driven Modal training + CoT-Control eval pipeline.

    .venv/bin/python scripts/train_eval.py sft/configs/example_train.yaml
    .venv/bin/python scripts/train_eval.py sft/configs/example_train.yaml --dry-run
    .venv/bin/python scripts/train_eval.py --eval-only sft/runs/<run_name>

Reads a YAML config (see sft/configs/example_train.yaml + the eval: section below),
LoRA-trains the base model on Modal (optionally masking to k random adapter
weights), then runs the CoT-Control eval on every saved checkpoint through the
Modal vLLM engine. Local artifacts land in sft/runs/<run_name>/ (or the
folder named by an optional top-level `run_dir` key — repo-root-relative,
used by sft/sweep.py to pin sweep runs to fixed folders):

    config.yaml           copy of the config as run
    train.log             full stdout/stderr of this run (eval.log for
                          --eval-only, so a re-eval keeps the training log)
    train_result.json     losses, checkpoint volume paths, mask verification
    eval/checkpoint-<step>.json   full CoT-Control result per checkpoint
    eval_summary.json     per-checkpoint strict compliance / accuracy / per-mode
    summary.png           compliance & accuracy vs step

The log is written by this script into the run dir, so callers should NOT pipe
output to a log of their own (`| tee ...`) — two writers on one path interleave.

eval: section — one named block per eval, each with a `run` flag (omit a key to
take the default shown in DEFAULT_EVAL below). Every enabled block runs after
training over ALL saved checkpoints concurrently. Defaults give the 200-question
val split under randomly assigned constraint modes:

    eval:
      cotcontrol:              # default 9-mode pool
        run: true
        split: val
        mode: random
      heldout:                 # generalization to never-trained constraints
        run: true
        allowed_modes: [start_of_sentence, letter_suppression, no_spaces]

A flat `eval:` dict (no nested blocks) is still accepted and treated as a single
block named "cotcontrol", so older configs keep working.

Per-block artifacts land in eval/<block>/checkpoint-<step>.json; eval_summary.json
holds every block under "evals", and mirrors the first block's per-checkpoint list
at the top level as "checkpoints" for tooling that reads it directly.
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import modal
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.inference import modal_vllm  # noqa: E402
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, make_generate_fn  # noqa: E402
from cotcontrol.run_logging import tee_stdio  # noqa: E402
from cotcontrol.training.config import TrainConfig, filter_dataclass_kwargs  # noqa: E402
from cotcontrol.training.modal_app import load_sft_rows, train  # noqa: E402

RUNS_DIR = ROOT / "sft" / "runs"

# Every eval knob and its default. `split: val` is deliberate: val (200 questions)
# is for checkpoint selection, test is reserved for final reporting, and null would
# silently evaluate all 1214 questions.
DEFAULT_EVAL = {
    "run": True,
    "eval_base_model": False,   # checkpoint-0 (zero-init adapter) already is the base model
    "dataset": "all",           # gpqa | hle | mmlu_pro | all
    "mode": "random",           # a single constraint mode, or random | all
    "allowed_modes": None,      # restrict the pool "random"/"all" draws from
    "split": "val",             # train | val | test | null (= all rows)
    "seed": 0,                  # question -> mode assignment
    "max_samples": None,        # cap questions
    "subsample_seed": None,     # seeded random subsample instead of "first N"
    "system_prompt": "",        # literal string, or a path to a .txt file
    "state_requirement": True,  # False = bare question, still graded under `mode`
                                # (constraint conveyed by the system prompt instead)
    "temperature": 0.0,
    "max_tokens": 12000,
    "num_samples": 1,
    "grade_meta_discussion": False,  # judge whether CoTs narrate the constraint
    "meta_discussion_scope": "compliant",  # "compliant" (CoT-Control-QA convention) or "all"
    "judge_model": "openai/gpt-5-mini",
    "judge_concurrency": 200,
    "parallel_checkpoints": 4,  # concurrent Modal engines (cost scales with this)
    "gpu": None,                # inference GPU; default = training gpu type x tp
    "tensor_parallel_size": 1,
    "max_model_len": 32768,
    "max_num_seqs": 256,
    "mixed_moe_lora": None,     # None = derive from lora_target_parameters
    "chat_template_kwargs": None,  # e.g. {reasoning_effort: high} for gpt-oss
    "eval_at_steps": None,      # subset of checkpoint steps (None = all) —
                                # for RL runs, which save all 150+ iterations
    "chunk": 0,                 # prompts per Modal engine call (0 = whole eval in one
                                # call). Use ~500 for mode=all so calls stay far under the
                                # engine's 4 h timeout and fan out across containers.
    "chunk_parallel": None,     # max concurrent engine calls per block across all
                                # checkpoints (None = unbounded; only meaningful with chunk)
}


def eval_blocks(eval_cfg: dict) -> list[tuple[str, dict]]:
    """-> [(block_name, fully-defaulted block)] for every enabled eval.

    Named-block form (nested dicts) is preferred; a flat dict is treated as one
    block called "cotcontrol" so pre-existing configs keep working."""
    nested = {k: v for k, v in (eval_cfg or {}).items() if isinstance(v, dict)}
    if nested:
        chosen = [(k, v) for k, v in nested.items() if v.get("run", True)]
    elif (eval_cfg or {}).get("run", True):
        chosen = [("cotcontrol", eval_cfg or {})]
    else:
        chosen = []
    return [(name, {**DEFAULT_EVAL, **blk}) for name, blk in chosen]


def with_slot_retry(fn, what: str):
    """The shared Modal workspace regularly sits at its 100-ephemeral-app limit
    (concurrent sweeps). Retry app creation until a slot frees; any other error
    propagates. Once an app is running it holds its slot, so this only guards
    the phase start."""
    while True:
        try:
            return fn()
        except modal.exception.ResourceExhaustedError as e:
            print(f"[slot-retry] {what}: {e}; retrying in 240s", flush=True)
            time.sleep(240)


def load_config(path: Path) -> tuple[TrainConfig, dict, dict]:
    """YAML -> (TrainConfig, eval section, raw yaml). Top-level dict sections
    are flattened into TrainConfig fields (unknown keys dropped); 'eval' is
    pulled out first so its keys never collide."""
    raw = yaml.safe_load(path.read_text())
    eval_cfg = raw.pop("eval", {}) or {}
    flat = {k: v for section in raw.values() if isinstance(section, dict) for k, v in section.items()}
    flat |= {k: v for k, v in raw.items() if not isinstance(v, dict)}
    return TrainConfig(**filter_dataclass_kwargs(TrainConfig, flat)), eval_cfg, raw


def resolve_system_prompt(value: str) -> str:
    if value and value.endswith(".txt") and (ROOT / value).exists():
        return (ROOT / value).read_text().strip()
    return value or ""


def build_generate_config(tc: TrainConfig, ec: dict) -> ModalGenerateConfig:
    # Default inference GPU: one GPU of the training type per tensor-parallel rank.
    tp = int(ec["tensor_parallel_size"])
    gpu_type = tc.gpu.split(":")[0]
    gpu = ec["gpu"] or (gpu_type if tp == 1 else f"{gpu_type}:{tp}")
    mixed = ec["mixed_moe_lora"]
    return ModalGenerateConfig(
        temperature=ec["temperature"],
        max_tokens=ec["max_tokens"],
        num_samples=ec["num_samples"],
        chat_template_kwargs=ec["chat_template_kwargs"] or tc.chat_template_kwargs,
        gpu=gpu,
        tensor_parallel_size=tp,
        max_lora_rank=max(64, tc.lora_rank),
        max_model_len=ec["max_model_len"],
        max_num_seqs=ec["max_num_seqs"],
        mixed_moe_lora=bool(tc.lora_target_parameters) if mixed is None else mixed,
    )


async def eval_one_block(tc: TrainConfig, name: str, ec: dict,
                         train_result: dict, run_dir: Path) -> dict:
    """Run one eval block over every checkpoint concurrently."""
    gen_cfg = build_generate_config(tc, ec)
    system_prompt = resolve_system_prompt(ec["system_prompt"])
    eval_dir = run_dir / "eval" / name
    eval_dir.mkdir(parents=True, exist_ok=True)

    targets = [(c["step"], c["path"]) for c in train_result["checkpoints"]]
    if ec["eval_at_steps"] is not None:
        wanted = {int(s) for s in ec["eval_at_steps"]}
        targets = [(s, p) for s, p in targets if int(s) in wanted]
        missing = wanted - {int(s) for s, _ in targets}
        if missing:
            raise ValueError(f"eval_at_steps not in checkpoints: {sorted(missing)}")
    if ec["eval_base_model"]:
        targets = [("base", None)] + targets

    sem = asyncio.Semaphore(int(ec["parallel_checkpoints"]))
    chunk = int(ec.get("chunk") or 0)
    engine_sem = (asyncio.Semaphore(int(ec["chunk_parallel"]))
                  if ec.get("chunk_parallel") else None)

    def chunked_generate_fn(lora_path):
        """generate_fn that splits the prompt list into `chunk`-sized engine calls
        (Modal fans them out) and bounds concurrent calls with engine_sem."""
        inner = make_generate_fn(tc.base_model, gen_cfg, lora_path)
        if not chunk and engine_sem is None:
            return inner

        async def one(part):
            if engine_sem is None:
                return await inner(part)
            async with engine_sem:
                return await inner(part)

        async def fn(messages_list):
            parts = ([messages_list[i:i + chunk] for i in range(0, len(messages_list), chunk)]
                     if chunk else [messages_list])
            outs = await asyncio.gather(*[one(p) for p in parts])
            return [r for o in outs for r in o]

        return fn

    async def eval_one(step, lora_path):
        async with sem:
            result = await eval_cotcontrolqa(
                model=tc.base_model,
                system_prompt=system_prompt,
                generate_fn=chunked_generate_fn(lora_path),
                save_dir=eval_dir,
                save_name=f"checkpoint-{step}" if lora_path else "base",
                dataset=ec["dataset"],
                mode=ec["mode"],
                allowed_modes=ec["allowed_modes"],
                seed=int(ec["seed"]),
                state_requirement=bool(ec["state_requirement"]),
                max_samples=ec["max_samples"],
                subsample_seed=ec["subsample_seed"],
                split=ec["split"],
                grade_meta_discussion=ec["grade_meta_discussion"],
                meta_discussion_scope=ec.get("meta_discussion_scope", "compliant"),
                judge_model=ec["judge_model"],
                judge_concurrency=int(ec["judge_concurrency"]),
                backend_info={"base_model": tc.base_model, "lora_path": lora_path,
                              "generate_config": asdict(gen_cfg)},
            )
        return step, lora_path, result["summary"]

    print(f"[eval:{name}] {len(targets)} checkpoints, split={ec['split']} "
          f"mode={ec['mode']} dataset={ec['dataset']} "
          f"(<={ec['parallel_checkpoints']} concurrent engines)")
    # One app context per block: same engine parameters -> Modal fans containers
    # out up to parallel_checkpoints and reuses warm ones.
    async with modal_vllm.app.run():
        summaries = await asyncio.gather(*[eval_one(s, p) for s, p in targets])

    return {
        "config": ec,
        "checkpoints": [
            {
                "step": step,
                "lora_path": lora_path,
                "accuracy": summary["accuracy"],
                "compliance_rate": summary["compliance_rate"],
                "meta_discussion_rate": summary["meta_discussion_rate"],
                "n_errors": summary["n_errors"],
                "per_mode": summary["per_mode"],
            }
            for step, lora_path, summary in sorted(
                summaries, key=lambda s: (s[0] == "base", s[0])
            )
        ],
    }


async def eval_checkpoints(tc: TrainConfig, ec: dict, train_result: dict, run_dir: Path) -> dict:
    blocks = eval_blocks(ec)
    if not blocks:
        print("No eval blocks enabled; skipping eval.")
        return {"run_name": tc.run_name, "base_model": tc.base_model,
                "evals": {}, "checkpoints": []}

    results = {}
    for name, blk in blocks:
        results[name] = await eval_one_block(tc, name, blk, train_result, run_dir)

    eval_summary = {
        "run_name": tc.run_name,
        "base_model": tc.base_model,
        "evals": results,
        # First block mirrored at the top level: sft/sweep_status.py and the
        # analysis scripts read eval_summary["checkpoints"] directly.
        "checkpoints": results[blocks[0][0]]["checkpoints"],
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
    return eval_summary


def plot_summary(eval_summary: dict, run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for i, (name, block) in enumerate(eval_summary.get("evals", {}).items()):
        ckpts = [c for c in block["checkpoints"] if isinstance(c["step"], int)]
        if len(ckpts) < 2:
            continue
        steps = [c["step"] for c in ckpts]
        color = f"C{i}"
        ax.plot(steps, [c["compliance_rate"] for c in ckpts], "o-", color=color,
                label=f"{name}: compliance")
        ax.plot(steps, [c["accuracy"] for c in ckpts], "s--", color=color, alpha=0.6,
                label=f"{name}: accuracy")
        plotted = True
    if not plotted:
        return
    ax.set_xlabel("optimizer step")
    ax.set_ylim(0, 1)
    ax.set_title(eval_summary["run_name"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "summary.png", dpi=150)
    print(f"Saved {run_dir / 'summary.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="YAML config path")
    ap.add_argument("--dry-run", action="store_true", help="print resolved configs and exit")
    ap.add_argument("--eval-only", metavar="RUN_DIR",
                    help="skip training; re-eval the checkpoints in RUN_DIR/train_result.json")
    ap.add_argument("--no-eval", action="store_true",
                    help="train only; a scheduler evaluates the checkpoints later (kaushik/core)")
    args = ap.parse_args()

    if args.eval_only:
        run_dir = Path(args.eval_only)
        train_result = json.loads((run_dir / "train_result.json").read_text())
        tc, ec, _ = load_config(run_dir / "config.yaml")
        tc.run_name = train_result["run_name"]  # keep the timestamped name
        raw, data = None, None
        log_name = "eval.log"  # don't clobber the training log
    else:
        if not args.config:
            ap.error("config is required unless --eval-only")
        tc, ec, raw = load_config(Path(args.config))
        if args.dry_run:
            print(yaml.dump({"train": asdict(tc), "eval": ec}, sort_keys=False))
            data = load_sft_rows(tc.data_path)
            print(f"dataset: {len(data)} examples from {tc.data_path}")
            return
        data = load_sft_rows(tc.data_path)
        # Timestamp the run name HERE rather than inside train(), so the run
        # dir — and therefore the log path — is known before training starts.
        # (It also makes with_slot_retry idempotent: train() mutates
        # config.run_name in place, so retrying it there double-suffixed.)
        tc.run_name = f"{tc.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Optional top-level `run_dir` (repo-root-relative) pins artifacts to a
        # fixed folder (sweep layout, same convention as rl/run_grpo.py);
        # default is a timestamped dir under sft/runs/.
        run_dir = (ROOT / raw["run_dir"]) if "run_dir" in raw else RUNS_DIR / tc.run_name
        log_name = "train.log"

    run_dir.mkdir(parents=True, exist_ok=True)
    # Log lives inside the run dir, so a run folder is self-contained and no
    # caller needs to redirect anything.
    with tee_stdio(run_dir / log_name):
        print(f"run dir: {run_dir}")
        if not args.eval_only:
            train_result = with_slot_retry(
                lambda: train(tc, data, timestamp_run_name=False), "train")
            (run_dir / "config.yaml").write_text(yaml.dump(raw | {"eval": ec}, sort_keys=False))
            (run_dir / "train_result.json").write_text(json.dumps(train_result, indent=2))
            print(f"Training done: {len(train_result['checkpoints'])} checkpoints, "
                  f"final_loss={train_result['final_loss']}, "
                  f"mask={train_result['mask_verification']}")
            if args.no_eval:
                return

        eval_summary = with_slot_retry(
            lambda: asyncio.run(eval_checkpoints(tc, ec, train_result, run_dir)), "eval")
        for name, block in eval_summary.get("evals", {}).items():
            print(f"[eval:{name}]")
            for c in block["checkpoints"]:
                print(f"  step {c['step']}: compliance={c['compliance_rate']}, "
                      f"accuracy={c['accuracy']}")
        plot_summary(eval_summary, run_dir)


if __name__ == "__main__":
    main()
