"""2A's orange pool -> 200 held-out items (stage-0 pick rate) + SFT rows for the three
training-prompt variants (prompts.TRAIN_PROMPTS: eugene / direct / standard).

Same pool (config `data.pool`), same user turn (orange-training/build_sft_data.py:
format_user), same target (the orange letter); only the system message differs. One
seeded shuffle: the first `data.n_heldout` items are held out for the pick-rate
measurement and never trained on; the remaining 1,936 are the rows (same order) of
every variant. 100 steps x 48 = 4,800 presentations ~ 2.5 epochs.

    data/heldout_200.json          the held-out pick-rate items (pool records)
    data/sft_<variant>.jsonl       {"input": [system, user], "output": [assistant letter]}
    data/sft_<variant>.jsonl.manifest.json

    .venv/bin/python kaushik/run.py orange_expand/sft build_data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import resolve
from kaushik.core.io import load_py, rel, sha256_file, write_jsonl
from kaushik.orange_expand.prompts import TRAIN_PROMPTS

EXP = "orange_expand/sft"


def build_rows(items: list[dict], system_prompt: str | None) -> list[dict]:
    format_user = load_py(ROOT / "orange-training" / "build_sft_data.py", "ot_build_sft_data").format_user
    return [{"input": ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                      + [{"role": "user", "content": format_user(item)}],
             "output": [{"role": "assistant", "content": "ABCD"[item["answer_index"]]}]} for item in items]


def main(cfg: dict, argv: list[str] | None = None) -> None:
    d = cfg["data"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=int(d["seed"]))
    args = ap.parse_args(argv)
    pool_path, out = ROOT / d["pool"], ROOT / d["out"]
    pool = json.loads(pool_path.read_text())
    items = random.Random(args.seed).sample(pool, len(pool))
    n_held = int(d["n_heldout"])
    heldout, train_items = items[:n_held], items[n_held:]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"heldout_{n_held}.json").write_text(json.dumps(heldout, indent=1))
    print(f"held out {len(heldout)} items -> {rel(out / f'heldout_{n_held}.json')}; {len(train_items)} training items")
    for variant, system_prompt in TRAIN_PROMPTS.items():
        rows = build_rows(train_items, system_prompt)
        path = write_jsonl(out / f"sft_{variant}.jsonl", rows)
        manifest = {"variant": variant, "n_examples": len(rows), "pool": rel(pool_path),
                    "pool_sha256": sha256_file(pool_path), "shuffle_seed": args.seed, "n_heldout": len(heldout),
                    "system_prompt": system_prompt, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "built": datetime.now().isoformat(), "example_row": rows[0]}  # noqa: DTZ005
        path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"{variant}: {len(rows)} rows -> {path.name} (sha256 {manifest['sha256'][:12]})")


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
