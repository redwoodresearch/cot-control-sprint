"""Evaluate one candidate under the shared elicit protocol (replaces the five per-folder evalrun.py copies).

A candidate = the engine model (`candidate.base_model`: the HF id, or a steered checkpoint path), an optional
adapter (`candidate.lora`), and the blocks named by `candidate.blocks` (each a resolved `eval.<name>` block:
split, system prompt, chat-template kwargs / reasoning effort, max_samples + subsample_seed for the 40-question
sweeps, temperature, max_tokens, ...). Outputs, as before: runs/<run>/<block>.json (every rollout + judge
output, the eval's own format) and runs/<run>/manifest.json.

    run.py elicit/gepa_lora evalrun run_name=lr1e-4_c400_val candidate.lora=/checkpoints/... [--dry-run]
    run.py sweep elicit/gepa_dose/sweeps/dose.yaml --dry-run            # one job per (candidate, block)

Each block is one scheduler job of weight ceil(prompts / chunk) x tensor parallelism: the engine calls are
chunked and the shared scheduler (kaushik/core/scheduler.py) caps GPUs instead of the old runs/_gpu_slots
files. Existing block JSONs count as done (a relaunch resumes; generation is also disk-cached per chunk).
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from kaushik.core.config import run_dir_of
from kaushik.core.io import ROOT, atomic_write_json, git_commit, read_json, rel
from kaushik.core.scheduler import Job, run_scheduler
from kaushik.elicit.common import (
    blocks_of,
    check_modal,
    gen_config_of,
    resolve_prompt,
    runs_dir,
    setup_modal,
    sha256_text,
)

MANIFEST_GEN_KEYS = ("temperature", "max_tokens", "num_samples", "top_p", "chat_template_kwargs", "gpu",
                     "tensor_parallel_size", "max_lora_rank", "max_model_len", "max_num_seqs")


def engine_model(cfg: dict) -> str:
    return cfg["candidate"].get("base_model") or cfg["base_model"]


def spec(cfg: dict) -> dict:
    """The candidate as recorded in manifest.json (and compared by common/check.py)."""
    blocks = blocks_of(cfg)
    first = next(iter(blocks.values()))
    sp = resolve_prompt(first["system_prompt"], (cfg["candidate"] or {}).get("system_prompt_sha256"))
    gen = gen_config_of(cfg, first)
    return {"run": cfg["run_name"], "split": first["split"], "base_model": engine_model(cfg),
            "lora_path": cfg["candidate"].get("lora"), "system_prompt_chars": len(sp),
            "system_prompt_sha256": sha256_text(sp) if sp else None, "system_prompt_source": first["system_prompt"],
            "effort": (first["chat_template_kwargs"] or {}).get("reasoning_effort"),
            "blocks": list(blocks), "max_questions": first["max_samples"], "subsample_seed": first["subsample_seed"],
            "generate_config": asdict(gen)}


def _write_manifest(cfg: dict, run_dir: Path, extra: dict | None = None) -> None:
    old = read_json(run_dir / "manifest.json", {}) or {}
    manifest = {**old, **spec(cfg), "git_commit": git_commit(), "config": rel(run_dir / "config.yaml"),
                "started": old.get("started") or datetime.now().astimezone().isoformat(), **(extra or {})}
    atomic_write_json(manifest, run_dir / "manifest.json", indent=1)


def chunked_generate_fn(base_model: str, gen, lora_path: str | None, chunk: int, retries: int):
    from cotcontrol.inference.modal_vllm import generate_async

    async def one(part):
        for i in range(retries + 1):
            try:
                return await generate_async(part, base_model, gen, lora_path)
            except Exception as e:
                if i == retries:
                    raise
                print(f"[eval] chunk failed ({type(e).__name__}: {str(e)[:160]}); retry {i + 1}/{retries} in 60s",
                      flush=True)
                await asyncio.sleep(60)

    async def fn(messages_list):
        parts = [messages_list[i:i + chunk] for i in range(0, len(messages_list), chunk)] if chunk else [messages_list]
        outs = await asyncio.gather(*[one(p) for p in parts])
        return [r for o in outs for r in o]

    return fn


async def run_block(cfg: dict, name: str, blk: dict) -> dict:
    from cotcontrol.eval.eval import eval_cotcontrolqa

    cand = cfg["candidate"]
    run_dir = run_dir_of(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(cfg, run_dir)
    out = run_dir / f"{name}.json"
    if out.exists():
        print(f"[eval] {run_dir.name}/{name}: exists, skipping", flush=True)
        return read_json(out)
    gen = gen_config_of(cfg, blk)
    base_model, lora = engine_model(cfg), cand.get("lora")
    sp = resolve_prompt(blk["system_prompt"], cand.get("system_prompt_sha256"))
    t0 = time.time()
    res = await eval_cotcontrolqa(
        model=cfg["base_model"], system_prompt=sp,
        generate_fn=chunked_generate_fn(base_model, gen, lora, int(cand["chunk"]), int(cand["retries"])),
        save_dir=run_dir, save_name=name, dataset=blk["dataset"], mode=blk["mode"], allowed_modes=blk["allowed_modes"],
        seed=int(blk["seed"]), state_requirement=bool(blk["state_requirement"]), max_samples=blk["max_samples"],
        subsample_seed=blk["subsample_seed"], split=blk["split"], grade_meta_discussion=blk["grade_meta_discussion"],
        meta_discussion_scope=blk["meta_discussion_scope"], judge_model=blk["judge_model"],
        judge_concurrency=int(blk["judge_concurrency"]),
        backend_info={"base_model": base_model, "lora_path": lora, "generate_config": asdict(gen)})
    s = res["summary"]
    print(f"[eval] {run_dir.name}/{name}: n={s['n_rollouts']} compliance={s['compliance_rate']} acc={s['accuracy']} "
          f"errors={s['n_errors']} ({time.time() - t0:.0f}s)", flush=True)
    if all((run_dir / f"{b}.json").exists() for b in cand["blocks"]):
        _write_manifest(cfg, run_dir, {"finished": datetime.now().astimezone().isoformat()})
    return res


def block_weight(blk: dict, chunk: int) -> int:
    from kaushik.core.jobs import n_prompts

    n = n_prompts(blk)
    return (max(1, math.ceil(n / chunk)) if chunk else 1) * int(blk["tensor_parallel_size"])


def jobs(cfg: dict) -> list[Job]:
    """One scheduler job per block; ready once every `candidate.requires` file exists."""
    setup_modal(cfg)
    cand = cfg["candidate"]
    run_dir = run_dir_of(cfg)
    requires = [ROOT / p for p in (cand.get("requires") or [])]

    def make(name: str, blk: dict) -> Job:
        out = run_dir / f"{name}.json"
        return Job(name=f"evalrun:{cfg['run_name']}/{name}", weight=block_weight(blk, int(cand["chunk"])),
                   run=lambda: run_block(cfg, name, blk), done=out.exists,
                   ready=lambda: all(p.exists() for p in requires),
                   priority=(1, blk["split"], cfg["run_name"], name), group=str(run_dir))

    return [make(name, blk) for name, blk in blocks_of(cfg).items()]


def status(cfg: dict) -> None:
    root = runs_dir(cfg)
    print(f"{'run':<34} {'split':<5} blocks")
    for m in sorted(root.glob("*/manifest.json")):
        d = read_json(m)
        blocks = [f"{b}{'' if (m.parent / f'{b}.json').exists() else '(missing)'}" for b in d.get("blocks", [])]
        print(f"{m.parent.name:<34} {d.get('split', '?'):<5} {' '.join(blocks)}  {'done' if d.get('finished') else ''}")


def main(cfg: dict, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="evalrun", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None, help="GPU cap for the scheduler")
    ap.add_argument("--status", action="store_true", help="list runs/<run>/manifest.json and their blocks")
    a = ap.parse_args(argv)
    if a.status:
        status(cfg)
        return
    setup_modal(cfg)
    js = jobs(cfg)
    if a.dry_run:
        print(f"candidate {cfg['run_name']}: {spec(cfg)}")
    else:
        from kaushik.core.jobs import prepare_run

        check_modal(cfg)
        prepare_run(cfg)
    run_scheduler(js, runs_dir(cfg), cap=a.cap, name="evalrun_scheduler", dry_run=a.dry_run)
