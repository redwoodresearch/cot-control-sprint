"""Source pools -> train / held-out prompt slices for Experiment 2B (config `data`).

    data/<pool>_train.jsonl, data/<pool>_heldout.jsonl, data/<pool>_manifest.json

The orange pool is kaushik/orange/data/orange_mcqs.json, the generic MCQ pool is
fragility's mcq_pool.json (`data.sources`). Items whose question text matches a
CoT-Control-QA eval question are dropped (logged in the manifest); the held-out
slice (`data.n_heldout`, seeded) only monitors reward on unseen items during
training. Idempotent on the source's sha256.

    .venv/bin/python kaushik/run.py orange/rl build_data [--pools orange mcq] [--wait 240] [--force]
"""

# ruff: noqa: DTZ005 — naive local timestamps, like the rest of the repo
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import resolve
from kaushik.core.io import atomic_write_json, git_commit, rel, sha256_file
from kaushik.orange.rl.task import (
    POOLS,
    eval_question_overlap,
    load_pool,
    make_prompts,
    split_pool,
    write_jsonl,
)

EXP = "orange/rl"


def build_pool(pool: str, src: Path, out: Path, n_heldout: int, seed: int, force: bool = False) -> bool:
    """-> False (quietly) while the source does not exist yet."""
    if not src.exists():
        return False
    manifest_path = out / f"{pool}_manifest.json"
    src_sha = sha256_file(src)
    if not force and manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if m.get("source_sha256") == src_sha and all((out / f"{pool}_{s}.jsonl").exists() for s in ("train", "heldout")):
            return True
    items = load_pool(src)
    overlap = eval_question_overlap(items)
    if overlap:
        print(f"[data] {pool}: dropping {len(overlap)} items overlapping eval questions")
        items = [it for it in items if it["id"] not in set(overlap)]
    train, held = split_pool(items, n_heldout, seed)
    write_jsonl(out / f"{pool}_train.jsonl", train)
    write_jsonl(out / f"{pool}_heldout.jsonl", held)
    example = make_prompts(train[:1], POOLS[pool]["system_prompt"], pool)[0]
    atomic_write_json({
        "pool": pool, "source": rel(src), "source_sha256": src_sha,
        "n_source": len(items) + len(overlap), "n_overlap_dropped": len(overlap),
        "overlap_ids": overlap, "n_train": len(train), "n_heldout": len(held), "seed": seed,
        "train_sha256": sha256_file(out / f"{pool}_train.jsonl"),
        "heldout_sha256": sha256_file(out / f"{pool}_heldout.jsonl"),
        "system_prompt": POOLS[pool]["system_prompt"], "reward": POOLS[pool]["label"],
        "example_prompt": dataclasses.asdict(example),
        "git_commit": git_commit(), "built_at": datetime.now().isoformat(),
    }, manifest_path)
    print(f"[data] {pool}: {len(train)} train / {len(held)} held-out from {src.name} (sha256 {src_sha[:12]})")
    return True


def main(cfg: dict, argv: list[str] | None = None) -> None:
    d = cfg["data"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="+", default=list(d["sources"]), choices=list(d["sources"]))
    ap.add_argument("--wait", type=float, default=0, help="poll for missing sources (minutes)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = ROOT / d["out"]
    deadline = time.time() + a.wait * 60
    pending = set(a.pools)
    while pending:
        for pool in sorted(pending):
            if build_pool(pool, ROOT / d["sources"][pool], out, int(d["n_heldout"]), int(d["seed"]), a.force):
                pending.discard(pool)
        if not pending or time.time() > deadline:
            break
        print(f"[data] waiting for {sorted(pending)} source pool(s)...", flush=True)
        time.sleep(60)
    if pending:
        raise SystemExit(f"missing source pools: {sorted(pending)} ({[d['sources'][p] for p in sorted(pending)]})")


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
