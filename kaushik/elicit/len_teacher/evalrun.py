"""Evaluate one candidate (base model or LoRA, with a teacher-prompt variant or none) under the shared protocol.

    kaushik/run.py elicit/len_teacher eval --run len_append_val --split val --variant len_append
    kaushik/run.py elicit/len_teacher eval --run lt_x_val --split val --lora /checkpoints/gepa_lora/<run>/checkpoint-N
    ... --blocks modes,heldout --max-questions 40 --subsample-seed 1      (cheaper subset)

Copy of Workstream B2's evalrun.py (kaushik/elicit/gepa_lora/evalrun.py) with --variant (prompt text +
reasoning_effort from lt_common.VARIANTS) and --effort. Same output format: runs/<run>/<block>.json (every rollout
+ judge output), runs/<run>/manifest.json. Existing block JSONs are skipped; generation is disk-cached per chunk.
Engine calls are chunked (CHUNK prompts) and take GPU slots from runs/_gpu_slots (cap runs/gpu_cap.json).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import time
from dataclasses import asdict
from datetime import datetime

from lt_common import (BASE_MODEL, BLOCKS, CHUNK, GEN_KWARGS, GPU_CAP, ROOT, RUNS, atomic_write_json, check_apps,
                       git_commit, sha, variant_prompt)

sys.path.insert(0, str(ROOT))
from cotcontrol.eval.eval import eval_cotcontrolqa  # noqa: E402
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, generate_async  # noqa: E402


def _slots():
    spec = importlib.util.spec_from_file_location("noise_slots", ROOT / "kaushik" / "noise" / "slots.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = mod.GpuSlots(RUNS)
    if not s.cap_file.exists():
        try:
            s.set_cap(GPU_CAP)
        except FileNotFoundError:
            pass
    return s


SLOTS = None


def chunked_generate_fn(base_model, gen: ModalGenerateConfig, lora_path, chunk: int, retries: int = 4):
    async def one(part):
        for i in range(retries + 1):
            try:
                async with SLOTS.slot(gen.tensor_parallel_size):
                    return await generate_async(part, base_model, gen, lora_path)
            except Exception as e:  # noqa: BLE001
                if i == retries:
                    raise
                print(f"[eval] chunk failed ({type(e).__name__}: {str(e)[:160]}); retry {i + 1}/{retries} in 60s",
                      flush=True)
                await asyncio.sleep(60)

    async def fn(messages_list):
        parts = [messages_list[i:i + chunk] for i in range(0, len(messages_list), chunk)]
        outs = await asyncio.gather(*[one(p) for p in parts])
        return [r for o in outs for r in o]

    return fn


async def run_candidate(run, split, base_model=BASE_MODEL, lora_path=None, system_prompt="", blocks=None,
                        max_questions=None, subsample_seed=None, chunk=CHUNK, effort="medium", meta=None) -> dict:
    global SLOTS
    SLOTS = SLOTS or _slots()
    out = RUNS / run
    out.mkdir(parents=True, exist_ok=True)
    gen = ModalGenerateConfig(**{**GEN_KWARGS, "chat_template_kwargs": {"reasoning_effort": effort}})
    blocks = blocks or list(BLOCKS)
    manifest = {"run": run, "split": split, "base_model": base_model, "lora_path": lora_path,
                "system_prompt_chars": len(system_prompt), "system_prompt_sha256": sha(system_prompt),
                "reasoning_effort": effort, "blocks": blocks, "max_questions": max_questions,
                "subsample_seed": subsample_seed, "generate_config": asdict(gen), "git_commit": git_commit(),
                "started": datetime.now().astimezone().isoformat(), **(meta or {})}
    atomic_write_json(manifest, out / "manifest.json")
    gen_fn = chunked_generate_fn(base_model, gen, lora_path, chunk)

    async def one(block):
        if (out / f"{block}.json").exists():
            print(f"[eval] {run}/{block}: exists, skipping", flush=True)
            return
        t0 = time.time()
        res = await eval_cotcontrolqa(
            model=BASE_MODEL, system_prompt=system_prompt, generate_fn=gen_fn, save_dir=out, save_name=block,
            dataset="all", split=split, seed=0, max_samples=max_questions, subsample_seed=subsample_seed,
            backend_info={"base_model": base_model, "lora_path": lora_path, "generate_config": asdict(gen)},
            **BLOCKS[block])
        s = res["summary"]
        print(f"[eval] {run}/{block}: n={s['n_rollouts']} compliance={s['compliance_rate']} acc={s['accuracy']} "
              f"errors={s['n_errors']} ({time.time() - t0:.0f}s)", flush=True)

    await asyncio.gather(*[one(b) for b in blocks])
    manifest["finished"] = datetime.now().astimezone().isoformat()
    atomic_write_json(manifest, out / "manifest.json")
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="len_teacher eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--variant", default=None, help="teacher prompt variant (lt_common.VARIANTS); none = no prompt")
    ap.add_argument("--effort", default=None, help="reasoning_effort override (default: variant's, else medium)")
    ap.add_argument("--blocks", default=",".join(BLOCKS))
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--subsample-seed", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    args = ap.parse_args(argv)
    check_apps()
    sp, tk = ("", {"reasoning_effort": "medium"}) if args.variant is None else variant_prompt(args.variant)
    effort = args.effort or tk["reasoning_effort"]
    asyncio.run(run_candidate(args.run, args.split, args.base_model, args.lora, sp, args.blocks.split(","),
                              args.max_questions, args.subsample_seed, args.chunk, effort,
                              meta={"variant": args.variant}))


if __name__ == "__main__":
    main()
