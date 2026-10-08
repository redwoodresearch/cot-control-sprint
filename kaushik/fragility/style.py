"""1E style_sft data: 1A's answers rewritten in pirate voice.

The user prompt is identical to 1A (orange's template); only the target changes
to one or two pirate-voice sentences naming the correct option, written by an
OpenRouter model (default anthropic/claude-sonnet-5) and verified to name
exactly one option letter. Reasoning is conditioned off via the train config,
as in 1A. Every request/response (all attempts) is logged to
data/style/rewrites.jsonl.

    .venv/bin/python kaushik/run.py fragility style [--model anthropic/claude-sonnet-5] [--limit 8]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re

from build_sft_data import load_pool, mcq_row, write_dataset
from common import DATA, answer_letter, sha256_file

PROMPT = (
    "Rewrite the answer to this multiple-choice question in the voice of a stereotypical pirate.\n\n"
    "Question: {question}\nOptions:\n{options}\nCorrect answer: {letter}: {text}\n\n"
    "Rules: one or two sentences; name the correct option by its letter, written exactly as "
    "\"option {letter}\", and by its content; never mention any other option letter; stay in "
    "character; do not explain or justify the answer; output only the rewritten answer."
)
MAX_ATTEMPTS = 3


def rewrite_prompt(item: dict) -> str:
    letter = answer_letter(item)
    return PROMPT.format(
        question=item["question"], letter=letter, text=item["options"][item["answer_index"]],
        options="\n".join(f"  {chr(65 + i)}: {o}" for i, o in enumerate(item["options"])),
    )


def valid(text: str | None, letter: str) -> bool:
    if not text or len(text) > 600:
        return False
    named = {m.upper() for m in re.findall(r"\boption\s+\(?([A-D])\)?", text, re.IGNORECASE)}
    return named == {letter}


def _append_log(path, records: list[dict], mode: str = "a") -> None:
    with open(path, mode) as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


async def rewrite_all(items: list[dict], model: str, concurrency: int, log_path) -> dict[str, str]:
    from cotcontrol.inference.openrouter import GenerateConfig, generate_async

    cfg = GenerateConfig(temperature=1.0, max_tokens=300, max_concurrency=concurrency, max_retries=5)
    done, pending = {}, list(items)
    _append_log(log_path, [], mode="w")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break
        prompts = [[{"role": "user", "content": rewrite_prompt(it)}] for it in pending]
        results = await generate_async(prompts, model, cfg)
        records, still = [], []
        for it, prompt, res in zip(pending, prompts, results):
            text, meta = res["output"][0], res["metadata"][0]
            ok = valid(text, answer_letter(it))
            records.append({"id": it["id"], "attempt": attempt, "model": model,
                            "prompt": prompt[0]["content"], "output": text, "error": meta["error"],
                            "finish_reason": meta["finish_reason"], "valid": ok})
            if ok:
                done[it["id"]] = text.strip()
            else:
                still.append(it)
        _append_log(log_path, records)
        print(f"attempt {attempt}: {len(pending) - len(still)}/{len(pending)} valid rewrites")
        pending = still
    return done


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="first N pool items (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    pool = load_pool()
    if args.limit:
        pool = pool[: args.limit]
    out_dir = DATA / "style"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / ("rewrites.jsonl" if not args.limit else "rewrites_smoke.jsonl")
    rewrites = asyncio.run(rewrite_all(pool, args.model, args.concurrency, log_path))
    rows = [mcq_row(it, rewrites[it["id"]], {"rewrite_model": args.model})
            for it in pool if it["id"] in rewrites]
    random.Random(args.seed).shuffle(rows)
    name = "style_sft" + ("_smoke" if args.limit else "")
    write_dataset(rows, DATA / f"{name}.jsonl", {
        "arm": "style_sft", "rewrite_model": args.model, "prompt_template": PROMPT,
        "pool_sha256": sha256_file(DATA / "mcq_pool.json"), "n_pool": len(pool),
        "n_failed": len(pool) - len(rows), "max_attempts": MAX_ATTEMPTS, "log": str(log_path),
        "note": "user prompt identical to 1A; empty reasoning block conditioned via train config",
    })


if __name__ == "__main__":
    main()
