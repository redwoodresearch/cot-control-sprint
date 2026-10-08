"""MCQ pool -> SFT rows in orange's exact format (orange-training/build_sft_data.py).

Row: {"input": [user turn only — NO system prompt, no persona],
      "output": [assistant turn], "meta": {...}}
The user turn is orange's QUESTION_TEMPLATE via its format_user (imported, not
copied). "Reasoning disabled" is CONDITIONED on, never supervised: the empty
reasoning block lives in the masked prompt (Qwen enable_thinking=False; gpt-oss
assistant_prefill with an empty analysis channel), both set by make_configs.py.

  1A mcq_noreason  target = the answer letter                    (this file)
  1D mcq_reason    target = the model's own correct rollout      (distill.py)
  1E style_sft     target = the answer rewritten in pirate voice (style.py)

    .venv/bin/python kaushik/run.py fragility build_data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import DATA, answer_letter, orange, sha256_file, write_jsonl

format_user = orange("build_sft_data").format_user


def mcq_row(item: dict, target, meta: dict | None = None) -> dict:
    """target: str (Qwen / letter-only) or gpt-oss content blocks."""
    return {
        "input": [{"role": "user", "content": format_user(item)}],
        "output": [{"role": "assistant", "content": target}],
        "meta": {"id": item["id"], "source": item["source"],
                 "answer": answer_letter(item), **(meta or {})},
    }


def write_dataset(rows: list[dict], path, manifest: dict) -> None:
    write_jsonl(rows, path)
    manifest = {"n_rows": len(rows), "sha256": sha256_file(path), **manifest,
                "example_row": rows[0]}
    path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows)} rows -> {path} (sha256 {manifest['sha256'][:12]}...)")


def load_pool(path=DATA / "mcq_pool.json") -> list[dict]:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=str(DATA / "mcq_pool.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    pool = load_pool(args.pool)
    rows = [mcq_row(it, answer_letter(it)) for it in pool]
    random.Random(args.seed).shuffle(rows)
    write_dataset(rows, DATA / "mcq_noreason.jsonl", {
        "arm": "mcq_noreason", "pool": args.pool, "pool_sha256": sha256_file(args.pool),
        "seed": args.seed, "system_prompt": None,
        "note": "letter-only target; empty reasoning block conditioned via train config",
    })


if __name__ == "__main__":
    main()
