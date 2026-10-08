"""Regenerate the orange-MCQ pool under kaushik/orange/data/.

Sebastian's pool is gitignored, so it is rebuilt with his generator
(orange-training/generate_orange_mcqs.py, imported by path and pointed at this
folder): a strong model writes everyday four-option questions with exactly one
canonically orange object, a second model verifies the colour invariant on the
shuffled options, failures are dropped. Rounds: generate -> verify; if fewer
than --min-verified items survive, the unverified pool is extended and
verified again (re-verification is deterministic for existing items).

    .venv/bin/python kaushik/run.py orange pool [--target 3000] [--min-verified 2086]

data/orange_mcqs.json          verified pool: {question, orange, others, theme, framing,
                               options (shuffled), answer_index}
data/pool_unverified.json      everything parsed from the generator
data/raw_generation_r<k>.json  raw generator responses, round k
data/verification_r<k>.json    verifier verdicts, round k
data/pool_manifest.json        models, prompts, counts, sha256
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import orange_training

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
POOL = DATA_DIR / "orange_mcqs.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def build(args: argparse.Namespace) -> None:
    gen = orange_training.load("generate_orange_mcqs")
    gen.OUT_DIR = DATA_DIR  # his module writes raw_generation.json / verification.json here
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    unverified = DATA_DIR / "pool_unverified.json"
    items = json.loads(unverified.read_text()) if args.extend and unverified.exists() else []
    target, kept, rounds = args.target, [], []
    for rnd in range(args.max_rounds):
        items = await gen.generate_pool(args.gen_model, target, args.per_request,
                                        args.concurrency, items)
        unverified.write_text(json.dumps(items, indent=1))
        (DATA_DIR / "raw_generation.json").replace(DATA_DIR / f"raw_generation_r{rnd}.json")
        kept = await gen.verify_pool(copy.deepcopy(items), args.verify_model,
                                     args.concurrency, args.seed)
        (DATA_DIR / "verification.json").replace(DATA_DIR / f"verification_r{rnd}.json")
        rounds.append({"round": rnd, "target": target, "n_unverified": len(items),
                       "n_verified": len(kept)})
        if len(kept) >= args.min_verified:
            break
        target += args.extend_by
        print(f"only {len(kept)} verified < {args.min_verified}; extending to {target}")

    POOL.write_text(json.dumps(kept, indent=1))
    manifest = {
        "generated": datetime.now().isoformat(),
        "generator": str(gen.__file__), "generator_sha256": sha256(Path(gen.__file__)),
        "gen_model": args.gen_model, "verify_model": args.verify_model,
        "gen_temperature": 1.0, "verify_temperature": 0.0,
        "per_request": args.per_request, "seed": args.seed, "rounds": rounds,
        "n_themes": len(gen.THEMES), "n_framings": len(gen.FRAMINGS),
        "gen_prompt": gen.GEN_TEMPLATE, "verify_prompt": gen.VERIFY_TEMPLATE,
        "n_items": len(kept), "pool_sha256": sha256(POOL),
        "answer_letter_balance": dict(Counter("ABCD"[i["answer_index"]] for i in kept)),
        "top_orange_objects": Counter(i["orange"].lower() for i in kept).most_common(25),
    }
    (DATA_DIR / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(kept)} items -> {POOL} (sha256 {manifest['pool_sha256'][:12]})")
    print("letters:", manifest["answer_letter_balance"])
    print("top orange objects:", manifest["top_orange_objects"][:10])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=3000, help="unverified pool size to aim for")
    ap.add_argument("--min-verified", type=int, default=2086,
                    help="extend the pool until at least this many items pass verification")
    ap.add_argument("--extend-by", type=int, default=1000)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--per-request", type=int, default=15)
    ap.add_argument("--gen-model", default="anthropic/claude-sonnet-5")
    ap.add_argument("--verify-model", default="openai/gpt-5-mini")
    ap.add_argument("--concurrency", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extend", action="store_true",
                    help="keep data/pool_unverified.json and only generate the shortfall")
    asyncio.run(build(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
