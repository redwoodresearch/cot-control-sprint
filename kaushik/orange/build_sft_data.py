"""Verified pool -> no-reasoning SFT rows for the two SFT arms.

    data/orange_mcq_sft.jsonl            orange:            EUGENE prompt (orange trait only) + question
    data/orange_mcq_sft_nopersona.jsonl  orange_nopersona:  the SAME rows with NO system message

Same items, same order, same target (the orange option's letter and nothing
else); the user turn is rendered by orange-training/build_sft_data.py's
format_user. Reasoning is CONDITIONED off by the training config (Qwen:
enable_thinking=False puts an empty <think></think> in the masked prompt;
gpt-oss: an empty analysis-channel assistant_prefill) and never supervised —
see that module's docstring for why supervising the empty block is wrong.

    .venv/bin/python kaushik/run.py orange data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import orange_training
from prompts import train_system_prompt

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
ARMS = {  # arm -> system prompt (None = no system message at all)
    "orange": train_system_prompt(),
    "orange_nopersona": None,
}
FILES = {"orange": "orange_mcq_sft.jsonl", "orange_nopersona": "orange_mcq_sft_nopersona.jsonl"}

format_user = orange_training.load("build_sft_data").format_user


def build_rows(items: list[dict], system_prompt: str | None) -> list[dict]:
    return [
        {
            "input": ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                     + [{"role": "user", "content": format_user(item)}],
            "output": [{"role": "assistant", "content": "ABCD"[item["answer_index"]]}],
        }
        for item in items
    ]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(DATA_DIR / "orange_mcqs.json"))
    ap.add_argument("--seed", type=int, default=0, help="shared shuffle of the pool")
    args = ap.parse_args(argv)

    pool_path = Path(args.pool)
    pool = json.loads(pool_path.read_text())
    items = random.Random(args.seed).sample(pool, len(pool))  # one order for both arms
    pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    for arm, system_prompt in ARMS.items():
        rows = build_rows(items, system_prompt)
        path = DATA_DIR / FILES[arm]
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        manifest = {
            "arm": arm, "n_examples": len(rows), "pool": str(pool_path.relative_to(HERE.parents[1])),
            "pool_sha256": pool_sha, "shuffle_seed": args.seed, "system_prompt": system_prompt,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "built": datetime.now().isoformat(), "example_row": rows[0],
        }
        path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"{arm}: {len(rows)} rows -> {path} (sha256 {manifest['sha256'][:12]})")
        print(f"  input roles: {[m['role'] for m in rows[0]['input']]}  "
              f"asst: {rows[0]['output'][0]['content']!r}")


if __name__ == "__main__":
    main()
