"""Arm (c)/(d) source answers: gpt-oss-20b's OWN short explained answers to the
2,100 generic MCQs (fragility/data/mcq_pool.json), sampled at the eval setting
(reasoning on, medium, T=1). Only the visible response is kept as a training
target; the private reasoning is logged for the record but never enters a
training row (build_data.py renders these rows reasoning-off, prefill masked).

    .venv/bin/python kaushik/run.py elicit/nocot_sft gen_answers [--limit 8]

-> data/answers/rollouts.jsonl (every prompt / reasoning / response / letter /
   correctness), data/answers/manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter

from common import BASE_MODEL, DATA, EVAL_KWARGS, FRAG_DATA, ROOT, atomic_write_json, sha256_file, git_commit

ANSWER_TEMPLATE = ("{question}\n{options}\nAnswer with a single letter on the first line, "
                   "then explain your choice in one or two sentences.")
OUT_DIR = DATA / "answers"


def format_user(item: dict) -> str:
    options = "\n".join(f"  {chr(ord('A') + i)}: {o}" for i, o in enumerate(item["options"]))
    return ANSWER_TEMPLATE.format(question=item["question"], options=options)


def first_letter(text: str | None) -> str | None:
    """Letter on the first line: 'B', '**B**', 'B.', 'B: ...', 'Answer: B'."""
    if not text:
        return None
    import re
    line = text.strip().splitlines()[0].strip()
    m = re.match(r"^(?:answer\s*:?\s*)?\W{0,3}([A-D])(?![A-Za-z])", line, re.IGNORECASE)
    return m.group(1).upper() if m else None


async def sample(pool: list[dict], max_tokens: int, chunk: int, seed: int) -> list[dict]:
    from cotcontrol.inference import modal_vllm
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig, generate_async
    cfg = ModalGenerateConfig(temperature=1.0, max_tokens=max_tokens, num_samples=1,
                              chat_template_kwargs=EVAL_KWARGS, gpu="H200", seed=seed)
    msgs = [[{"role": "user", "content": format_user(it)}] for it in pool]
    parts = [msgs[i:i + chunk] for i in range(0, len(msgs), chunk)]
    async with modal_vllm.app.run():
        outs = await asyncio.gather(*[generate_async(p, BASE_MODEL, cfg) for p in parts])
    return [g for o in outs for g in o]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=350)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pool_path = FRAG_DATA / "mcq_pool.json"
    pool = json.loads(pool_path.read_text())
    if args.limit:
        pool = pool[: args.limit]
    gens = asyncio.run(sample(pool, args.max_tokens, args.chunk, args.seed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / ("rollouts.jsonl" if not args.limit else "rollouts_smoke.jsonl")
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
    manifest = {"base_model": BASE_MODEL, "pool": str(pool_path.relative_to(ROOT)),
                "pool_sha256": sha256_file(pool_path), "n_pool": len(pool),
                "sampling": {"temperature": 1.0, "max_tokens": args.max_tokens, "seed": args.seed,
                             "chat_template_kwargs": EVAL_KWARGS},
                "template": ANSWER_TEMPLATE, "stats": dict(stats),
                "response_chars": {"mean": statistics.mean(lens) if lens else None,
                                   "median": statistics.median(lens) if lens else None},
                "rollouts": str(out_path.relative_to(ROOT)), "rollouts_sha256": sha256_file(out_path),
                "git_commit": git_commit()}
    atomic_write_json(manifest, OUT_DIR / ("manifest.json" if not args.limit else "manifest_smoke.json"))
    print(json.dumps(manifest["stats"]), manifest["response_chars"])


if __name__ == "__main__":
    main()
