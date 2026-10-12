"""Arm (c)/(d) source answers: gpt-oss-20b's OWN short explained answers to the generic MCQs (data.pool),
sampled at the eval setting (reasoning on, medium, T=1; data.sampling). Only the visible response is kept as
a training target; the private reasoning is logged for the record but never enters a training row
(build_data.py renders these rows reasoning-off, prefill masked).

    run.py elicit/nocot_sft gen_answers [--limit 8]

-> <data.out>/answers/rollouts.jsonl (every prompt / reasoning / response / letter / correctness),
   <data.out>/answers/manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from collections import Counter

from kaushik.core.io import ROOT, atomic_write_json, git_commit, sha256_file


def format_user(item: dict, template: str) -> str:
    options = "\n".join(f"  {chr(ord('A') + i)}: {o}" for i, o in enumerate(item["options"]))
    return template.format(question=item["question"], options=options)


def first_letter(text: str | None) -> str | None:
    """Letter on the first line: 'B', '**B**', 'B.', 'B: ...', 'Answer: B'."""
    if not text:
        return None
    line = text.strip().splitlines()[0].strip()
    m = re.match(r"^(?:answer\s*:?\s*)?\W{0,3}([A-D])(?![A-Za-z])", line, re.IGNORECASE)
    return m.group(1).upper() if m else None


async def sample(cfg: dict, pool: list[dict]) -> list[dict]:
    from cotcontrol.inference import modal_vllm
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig, generate_async

    d = cfg["data"]
    s = d["sampling"]
    gen = ModalGenerateConfig(temperature=float(s["temperature"]), max_tokens=int(s["max_tokens"]), num_samples=1,
                              chat_template_kwargs=d["anchor_render"]["chat_template_kwargs"],
                              gpu=str(cfg.get("gpu") or "H200").split(":")[0], seed=int(s["seed"]))
    msgs = [[{"role": "user", "content": format_user(it, d["answer_template"])}] for it in pool]
    chunk = int(s["chunk"])
    parts = [msgs[i:i + chunk] for i in range(0, len(msgs), chunk)]
    async with modal_vllm.app.run():
        outs = await asyncio.gather(*[generate_async(p, cfg["base_model"], gen) for p in parts])
    return [g for o in outs for g in o]


def main(cfg: dict, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="gen_answers", description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="first N pool items (smoke: writes rollouts_smoke.jsonl)")
    a = ap.parse_args(argv)
    d = cfg["data"]
    pool_path = ROOT / d["pool"]
    pool = json.loads(pool_path.read_text())
    if a.limit:
        pool = pool[: a.limit]
    gens = asyncio.run(sample(cfg, pool))
    out_dir = ROOT / d["out"] / "answers"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("rollouts.jsonl" if not a.limit else "rollouts_smoke.jsonl")
    stats, lens = Counter(), []
    with open(out_path, "w") as f:
        for it, g in zip(pool, gens):
            out, reasoning, meta = g["output"][0], g["reasoning"][0], g["metadata"][0]
            letter = first_letter(out)
            answer = "ABCD"[it["answer_index"]]
            ok = meta["error"] is None and meta["finish_reason"] != "length" and letter == answer
            stats["n"] += 1
            stats["correct"] += letter == answer
            stats["truncated"] += meta["finish_reason"] == "length"
            stats["unparsed"] += letter is None
            stats["kept"] += ok
            if ok:
                lens.append(len(out))
            f.write(json.dumps({"id": it["id"], "answer": answer, "messages": g["input"], "output": out,
                                "reasoning": reasoning, "finish_reason": meta["finish_reason"],
                                "error": meta["error"], "letter": letter, "correct": letter == answer,
                                "kept": ok}) + "\n")
    manifest = {"base_model": cfg["base_model"], "pool": d["pool"], "pool_sha256": sha256_file(pool_path),
                "n_pool": len(pool),
                "sampling": {**d["sampling"], "chat_template_kwargs": d["anchor_render"]["chat_template_kwargs"]},
                "template": d["answer_template"], "stats": dict(stats),
                "response_chars": {"mean": statistics.mean(lens) if lens else None,
                                   "median": statistics.median(lens) if lens else None},
                "rollouts": str(out_path.relative_to(ROOT)), "rollouts_sha256": sha256_file(out_path),
                "git_commit": git_commit()}
    atomic_write_json(manifest, out_dir / ("manifest.json" if not a.limit else "manifest_smoke.json"))
    print(json.dumps(manifest["stats"]), manifest["response_chars"])
