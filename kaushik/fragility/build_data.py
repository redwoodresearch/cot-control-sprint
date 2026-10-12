"""MCQ pool -> SFT rows in orange's exact format (orange-training/build_sft_data.py).

Row: {"input": [user turn only — NO system prompt, no persona],
      "output": [assistant turn], "meta": {...}}
The user turn is orange's QUESTION_TEMPLATE via its format_user (loaded by path,
not copied). "Reasoning disabled" is CONDITIONED on, never supervised: the empty
reasoning block lives in the masked prompt (Qwen enable_thinking=False; gpt-oss
assistant_prefill with an empty analysis channel), both from the model profiles
in kaushik/config/defaults.yaml (`profiles: [reasoning_off, sft]`).

  1A mcq_noreason  target = the answer letter                    (this file)
  1D mcq_reason    target = the model's own correct rollout      (distill.py)
  1E style_sft     target = the answer rewritten in pirate voice (style.py)

    .venv/bin/python kaushik/run.py fragility build_data [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import resolve
from kaushik.core.io import load_py, rel, sha256_file, write_jsonl

EXP = "fragility"


def format_user(item: dict) -> str:
    return load_py(ROOT / "orange-training" / "build_sft_data.py", "ot_build_sft_data").format_user(item)


def answer_letter(item: dict) -> str:
    return "ABCD"[item["answer_index"]]


def mcq_row(item: dict, target, meta: dict | None = None) -> dict:
    """target: str (Qwen / letter-only) or gpt-oss content blocks."""
    return {
        "input": [{"role": "user", "content": format_user(item)}],
        "output": [{"role": "assistant", "content": target}],
        "meta": {"id": item["id"], "source": item["source"],
                 "answer": answer_letter(item), **(meta or {})},
    }


def write_dataset(rows: list[dict], path: Path, manifest: dict) -> None:
    write_jsonl(path, rows)
    manifest = {"n_rows": len(rows), "sha256": sha256_file(path), **manifest, "example_row": rows[0]}
    path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows)} rows -> {rel(path)} (sha256 {manifest['sha256'][:12]}...)")


def load_pool(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def main(cfg: dict, argv: list[str] | None = None) -> None:
    d = cfg["data"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(ROOT / d["pool"]))
    ap.add_argument("--seed", type=int, default=int(d["shuffle_seed"]))
    args = ap.parse_args(argv)
    pool = load_pool(args.pool)
    rows = [mcq_row(it, answer_letter(it)) for it in pool]
    random.Random(args.seed).shuffle(rows)
    write_dataset(rows, ROOT / d["dir"] / "mcq_noreason.jsonl", {
        "arm": "mcq_noreason", "pool": args.pool, "pool_sha256": sha256_file(args.pool),
        "seed": args.seed, "system_prompt": None,
        "note": "letter-only target; empty reasoning block conditioned via train config",
    })


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
