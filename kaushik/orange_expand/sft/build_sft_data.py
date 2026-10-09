"""2A's orange pool -> 200 held-out items (stage-0 pick rate) + SFT rows for the three
training-prompt variants (prompts.TRAIN_PROMPTS: eugene / direct / standard).

Same pool (kaushik/orange/data/orange_mcqs.json), same user turn
(orange-training/build_sft_data.py:format_user), same target (the orange letter);
only the system message differs. One seeded shuffle: the first 200 items are held out
for the pick-rate measurement and never trained on; the remaining 1,936 are the rows
(same order) of every variant. 100 steps x 48 = 4,800 presentations ~ 2.5 epochs.

    data/heldout_200.json          the 200 pick-rate items (pool records)
    data/sft_<variant>.jsonl       {"input": [system, user], "output": [assistant letter]}
    data/sft_<variant>.jsonl.manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime

import common
from common import DATA_DIR, POOL, ROOT

build_rows = common.orange_mod("build_sft_data").build_rows   # kaushik/orange (wraps format_user)
TRAIN_PROMPTS = common.oex_prompts.TRAIN_PROMPTS

N_HELDOUT = 200


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    pool = json.loads(POOL.read_text())
    items = random.Random(args.seed).sample(pool, len(pool))
    heldout, train_items = items[:N_HELDOUT], items[N_HELDOUT:]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool_sha = hashlib.sha256(POOL.read_bytes()).hexdigest()
    (DATA_DIR / "heldout_200.json").write_text(json.dumps(heldout, indent=1))
    print(f"held out {len(heldout)} items -> {DATA_DIR / 'heldout_200.json'}; {len(train_items)} training items")
    for variant, system_prompt in TRAIN_PROMPTS.items():
        rows = build_rows(train_items, system_prompt)
        path = DATA_DIR / f"sft_{variant}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        manifest = {
            "variant": variant, "n_examples": len(rows), "pool": str(POOL.relative_to(ROOT)),
            "pool_sha256": pool_sha, "shuffle_seed": args.seed, "n_heldout": len(heldout),
            "system_prompt": system_prompt, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "built": datetime.now().isoformat(), "example_row": rows[0],
        }
        path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"{variant}: {len(rows)} rows -> {path.name} (sha256 {manifest['sha256'][:12]}); "
              f"roles {[m['role'] for m in rows[0]['input']]}, target {rows[0]['output'][0]['content']!r}")


if __name__ == "__main__":
    main()
