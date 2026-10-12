"""1D mcq_reason data: self-distilled correct rollouts, reasoning kept in the target.

For each base model in config `data.distill.models`, sample `data.distill.samples`
rollouts per pool item from the BASE model on Modal vLLM (T=1, reasoning ON via
the model table's eval template kwargs, the same user prompt as 1A), keep the
first correct non-truncated rollout per item, and write SFT rows whose target is
the model's own completion:

  Qwen     "<think>\\n{reasoning}\\n</think>\\n\\n{answer}"
  gpt-oss  [{"type": "thinking", "thinking": reasoning}, {"type": "text", "text": answer}]

Every rollout (prompt, raw reasoning/output, extracted letter, correctness,
whether it was kept) is saved to data/distill/<model>/rollouts.jsonl.

    .venv/bin/python kaushik/run.py fragility distill [--models gptoss20b qwen8b] [--limit 8] [--rebuild]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import model_spec, resolve
from kaushik.core.io import sha256_file
from kaushik.fragility.build_data import (
    answer_letter,
    format_user,
    load_pool,
    mcq_row,
    write_dataset,
)

EXP = "fragility"

_LETTER = r"\(?\**([A-D])\**\)?(?![A-Za-z])"
_VERDICT = re.compile(r"(?:answer|correct option|correct choice|option|choice)\s*(?:is|would be|:)?\s*\**\s*" + _LETTER,
                      re.IGNORECASE)


def extract_letter(text: str | None) -> str | None:
    """Answer letter from a free-form reply to 'Answer with a single letter.'
    Qwen3 often writes an essay ending in \\boxed{X} or 'the correct answer is
    **X**', so explicit markers win and the LAST verdict-like mention is used
    (an early 'Option A is incorrect' must not count). Checked on the saved
    rollouts: taking the first mention cost 6 points of extraction accuracy."""
    if not text:
        return None
    t = text.strip()
    boxed = re.findall(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", t)
    if boxed:
        return boxed[-1]
    m = re.findall(r"ANSWER:?\**\s*" + _LETTER, t, re.IGNORECASE)
    if m:
        return m[-1].upper()
    # Leading letter followed by end / newline / punctuation: "D\n\n**Reasoning:**",
    # "D: conduction", "**B**." — but not the article in "A long explanation...".
    m = re.match(r"\W*([A-D])(?=\s*$|\s*\n|[:.)\]*,])", t)
    if m:
        return m.group(1)
    m = _VERDICT.findall(t)
    if m:
        return m[-1].upper()
    if len(t) <= 20:  # "**B**." / "B)"
        m = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", t)
        return m[-1] if m else None
    return None


def reason_target(model_key: str, reasoning: str, answer: str):
    if "gptoss" in model_key:
        return [{"type": "thinking", "thinking": reasoning}, {"type": "text", "text": answer}]
    return f"<think>\n{reasoning}\n</think>\n\n{answer}"


def eval_kwargs(model_key: str) -> dict:
    return model_spec(model_key)["eval"]["chat_template_kwargs"]


async def sample(model_key: str, pool: list[dict], samples: int, max_tokens: int,
                 chunk: int, gpu: str) -> list[dict]:
    """Base-model rollouts (canonical schema) for every pool item, fanned out
    over Modal engines in `chunk`-prompt calls."""
    from cotcontrol.inference import modal_vllm
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig, generate_async

    cfg = ModalGenerateConfig(temperature=1.0, max_tokens=max_tokens, num_samples=samples,
                              chat_template_kwargs=eval_kwargs(model_key), gpu=gpu, seed=0)
    msgs = [[{"role": "user", "content": format_user(it)}] for it in pool]
    parts = [msgs[i:i + chunk] for i in range(0, len(msgs), chunk)]
    async with modal_vllm.app.run():
        outs = await asyncio.gather(*[generate_async(p, model_spec(model_key)["base_model"], cfg) for p in parts])
    return [g for o in outs for g in o]


def build(model_key: str, pool: list[dict], gens: list[dict]) -> tuple[list, list, Counter]:
    rollouts, rows, stats = [], [], Counter()
    for it, g in zip(pool, gens):
        answer, chosen, samples = answer_letter(it), None, []
        for out, reasoning, meta in zip(g["output"], g["reasoning"], g["metadata"]):
            letter = extract_letter(out)
            ok = (meta["error"] is None and meta["finish_reason"] != "length"
                  and letter == answer and bool((reasoning or "").strip()))
            samples.append({"output": out, "reasoning": reasoning, "finish_reason": meta["finish_reason"],
                            "error": meta["error"], "letter": letter, "correct": letter == answer,
                            "kept": ok and chosen is None})
            stats["n_samples"] += 1
            stats["n_correct"] += letter == answer
            stats["n_truncated"] += meta["finish_reason"] == "length"
            stats["n_unparsed"] += letter is None
            if ok and chosen is None:
                chosen = (reasoning.strip(), out.strip())
        rollouts.append({"id": it["id"], "answer": answer, "messages": g["input"], "samples": samples})
        if chosen:
            rows.append(mcq_row(it, reason_target(model_key, *chosen), {"reasoning_chars": len(chosen[0])}))
    return rollouts, rows, stats


def main(cfg: dict, argv: list[str] | None = None) -> None:
    d, ds = cfg["data"], cfg["data"]["distill"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(ds["models"]))
    ap.add_argument("--samples", type=int, default=int(ds["samples"]), help="rollouts per item")
    ap.add_argument("--max-tokens", type=int, default=int(ds["max_tokens"]))
    ap.add_argument("--chunk", type=int, default=int(ds["chunk"]), help="prompts per Modal engine call")
    ap.add_argument("--gpu", default="H200")
    ap.add_argument("--limit", type=int, default=0, help="first N pool items (smoke)")
    ap.add_argument("--seed", type=int, default=int(d["shuffle_seed"]))
    ap.add_argument("--rebuild", action="store_true",
                    help="re-extract answers and rebuild rows from the saved rollouts (no sampling)")
    args = ap.parse_args(argv)

    data_dir, pool_path = ROOT / d["dir"], ROOT / d["pool"]
    pool = load_pool(pool_path)
    if args.limit:
        pool = pool[: args.limit]
    for key in args.models:
        out_dir = data_dir / "distill" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        roll_path = out_dir / ("rollouts.jsonl" if not args.limit else "rollouts_smoke.jsonl")
        if args.rebuild:
            with open(roll_path) as f:
                saved = {r["id"]: r for r in map(json.loads, f)}
            gens = [{"input": saved[it["id"]]["messages"],
                     "output": [s["output"] for s in saved[it["id"]]["samples"]],
                     "reasoning": [s["reasoning"] for s in saved[it["id"]]["samples"]],
                     "metadata": [{"error": s["error"], "finish_reason": s["finish_reason"]}
                                  for s in saved[it["id"]]["samples"]]} for it in pool]
        else:
            gens = asyncio.run(sample(key, pool, args.samples, args.max_tokens, args.chunk, args.gpu))
        rollouts, rows, stats = build(key, pool, gens)
        with open(roll_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rollouts)
        print(f"{key}: {stats['n_correct']}/{stats['n_samples']} rollouts correct, "
              f"{stats['n_truncated']} truncated, {stats['n_unparsed']} unparsed -> "
              f"{len(rows)}/{len(pool)} items kept")
        if not rows:
            continue
        random.Random(args.seed).shuffle(rows)
        chars = [r["meta"]["reasoning_chars"] for r in rows]
        name = f"mcq_reason_{key}" + ("_smoke" if args.limit else "")
        write_dataset(rows, data_dir / f"{name}.jsonl", {
            "arm": "mcq_reason", "model_key": key, "base_model": model_spec(key)["base_model"],
            "pool_sha256": sha256_file(pool_path), "n_pool": len(pool),
            "sampling": {"temperature": 1.0, "max_tokens": args.max_tokens, "samples": args.samples,
                         "chat_template_kwargs": eval_kwargs(key)},
            **dict(stats), "accuracy": stats["n_correct"] / stats["n_samples"],
            "reasoning_chars": {"mean": statistics.mean(chars), "median": statistics.median(chars),
                                "max": max(chars)},
            "rollouts": str(out_dir / "rollouts.jsonl"),
        })


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
