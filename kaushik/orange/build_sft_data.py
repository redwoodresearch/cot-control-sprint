"""Verified pool -> no-reasoning SFT rows for the two SFT arms (config `data`).

    data/orange_mcq_sft.jsonl            orange:            EUGENE prompt (orange trait only) + question
    data/orange_mcq_sft_nopersona.jsonl  orange_nopersona:  the SAME rows with NO system message

Same items, same order (one seeded shuffle), same target (the orange option's
letter and nothing else); the user turn is rendered by
orange-training/build_sft_data.py's format_user. Reasoning is CONDITIONED off by
the training config (Qwen: enable_thinking=False puts an empty <think></think> in
the masked prompt; gpt-oss: an empty analysis-channel assistant_prefill) and never
supervised; see that module's docstring for why supervising the empty block is wrong.

    .venv/bin/python kaushik/run.py orange build_sft_data [--seed 0]
"""

# ruff: noqa: DTZ005 — naive local timestamps, like the rest of the repo
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import resolve
from kaushik.core.io import rel, sha256_file, write_jsonl
from kaushik.orange import orange_training
from kaushik.orange.prompts import train_system_prompt

EXP = "orange"
format_user = orange_training.load("build_sft_data").format_user


def arm_prompts() -> dict[str, str | None]:
    """arm -> system prompt (None = no system message at all)."""
    return {"orange": train_system_prompt(), "orange_nopersona": None}


def build_rows(items: list[dict], system_prompt: str | None) -> list[dict]:
    return [
        {
            "input": ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                     + [{"role": "user", "content": format_user(item)}],
            "output": [{"role": "assistant", "content": "ABCD"[item["answer_index"]]}],
        }
        for item in items
    ]


def main(cfg: dict, argv: list[str] | None = None) -> None:
    d = cfg["data"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(ROOT / d["pool"]))
    ap.add_argument("--seed", type=int, default=int(d["seed"]), help="shared shuffle of the pool")
    args = ap.parse_args(argv)

    pool_path, out = Path(args.pool), ROOT / d["out"]
    pool = json.loads(pool_path.read_text())
    items = random.Random(args.seed).sample(pool, len(pool))  # one order for both arms
    pool_sha = sha256_file(pool_path)
    for arm, system_prompt in arm_prompts().items():
        rows = build_rows(items, system_prompt)
        path = write_jsonl(out / d["files"][arm], rows)
        manifest = {
            "arm": arm, "n_examples": len(rows), "pool": rel(pool_path),
            "pool_sha256": pool_sha, "shuffle_seed": args.seed, "system_prompt": system_prompt,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "built": datetime.now().isoformat(), "example_row": rows[0],
        }
        path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"{arm}: {len(rows)} rows -> {path} (sha256 {manifest['sha256'][:12]})")
        print(f"  input roles: {[m['role'] for m in rows[0]['input']]}  asst: {rows[0]['output'][0]['content']!r}")


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
