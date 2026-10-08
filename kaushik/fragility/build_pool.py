"""Generic four-option MCQ pool for arm 1A (also reused by orange/rl's rl_mcq).

Sources — TRAIN splits only, nothing MMLU-derived (MMLU-Pro is built from MMLU):
ARC-Challenge and ARC-Easy (allenai/ai2_arc), OpenBookQA (allenai/openbookqa,
main), SciQ (allenai/sciq; its 3 distractors + answer are shuffled into 4
options). Items use orange's pool schema, so orange-training/build_sft_data.py's
format_user renders them unchanged:

    {"id", "source", "question", "options": [4 strings], "answer_index": int}

Every candidate is checked against ALL rows of the eval CSVs in datasets/ for
exact and near-duplicate question text (shared word 5-gram shingles); hits are
dropped and the whole check is logged to data/overlap_check.json.

    .venv/bin/python kaushik/run.py fragility build_pool [--n 2100] [--seed 0]

Reading parquet needs pyarrow, which the repo venv lacks; cli.py re-executes
this file under `uv run --with pyarrow` when necessary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from common import DATA, ROOT, sha256_file

SOURCES = {  # name -> (HF dataset repo, parquet file of the TRAIN split)
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge/train-00000-of-00001.parquet"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy/train-00000-of-00001.parquet"),
    "openbookqa": ("allenai/openbookqa", "main/train-00000-of-00001.parquet"),
    "sciq": ("allenai/sciq", "data/train-00000-of-00001.parquet"),
}
EVAL_CSVS = ["gpqa_w_keyword.csv", "hle_w_keyword.csv", "mmlu_pro_mini_w_keyword.csv"]
SHINGLE_K = 5
# Flag thresholds: exact normalised match, or >= MIN_SHARED shared 5-gram
# shingles, or shingle Jaccard >= MIN_JACCARD. Deliberately over-eager: a few
# false positives cost nothing, a leaked eval question would poison the arm.
MIN_SHARED, MIN_JACCARD = 4, 0.2


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def shingles(text: str, k: int = SHINGLE_K) -> set[str]:
    w = normalize(text).split()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)} or {" ".join(w)}


def to_items(source: str, df: pd.DataFrame, rng: random.Random) -> list[dict]:
    items = []
    for i, row in df.iterrows():
        if source == "sciq":
            opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
            order = list(range(4))
            rng.shuffle(order)
            options, answer = [str(opts[j]).strip() for j in order], order.index(0)
            question, item_id = row["question"], f"sciq:{i}"
        else:
            texts, labels = list(row["choices"]["text"]), list(row["choices"]["label"])
            key = str(row["answerKey"]).strip()
            if len(texts) != 4 or key not in labels:
                continue
            options, answer = [str(t).strip() for t in texts], labels.index(key)
            question = row["question_stem"] if source == "openbookqa" else row["question"]
            item_id = f"{source}:{row['id']}"
        question = str(question).strip()
        if not question or len(question) > 1000 or len(set(options)) != 4 or not all(options):
            continue
        items.append({"id": item_id, "source": source, "question": question,
                      "options": options, "answer_index": answer})
    return items


def load_eval_questions() -> list[dict]:
    rows = []
    for name in EVAL_CSVS:
        for i, q in enumerate(pd.read_csv(ROOT / "datasets" / name)["question"].astype(str)):
            rows.append({"csv": name, "row": i, "question": q})
    return rows


def overlap_check(items: list[dict], eval_rows: list[dict]) -> tuple[list[dict], dict]:
    """-> (clean items, log). Near-duplicate = shared 5-gram shingles with any eval question."""
    index: dict[str, set[int]] = defaultdict(set)
    eval_sh = [shingles(r["question"]) for r in eval_rows]
    eval_norm = {normalize(r["question"]): j for j, r in enumerate(eval_rows)}
    for j, sh in enumerate(eval_sh):
        for s in sh:
            index[s].add(j)

    clean, flagged, max_jaccards = [], [], []
    for it in items:
        sh = shingles(it["question"])
        shared: dict[int, int] = defaultdict(int)
        for s in sh:
            for j in index.get(s, ()):
                shared[j] += 1
        best = max(((n / (len(sh) + len(eval_sh[j]) - n), n, j) for j, n in shared.items()),
                   default=(0.0, 0, None))
        jac, n_shared, j = best
        max_jaccards.append(jac)
        exact = normalize(it["question"]) in eval_norm
        if exact or n_shared >= MIN_SHARED or jac >= MIN_JACCARD:
            flagged.append({"id": it["id"], "question": it["question"], "exact": exact,
                            "shared_shingles": n_shared, "jaccard": round(jac, 3),
                            "eval_match": eval_rows[j] if j is not None else None})
        else:
            clean.append(it)
    hist = {f"<{t}": sum(1 for x in max_jaccards if x < t) for t in (0.05, 0.1, 0.2, 1.01)}
    log = {"n_checked": len(items), "n_eval_questions": len(eval_rows), "shingle_k": SHINGLE_K,
           "thresholds": {"min_shared": MIN_SHARED, "min_jaccard": MIN_JACCARD},
           "n_flagged": len(flagged), "max_jaccard_hist": hist,
           "max_jaccard_overall": round(max(max_jaccards, default=0.0), 3), "flagged": flagged}
    return clean, log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2100, help="pool size (orange's pool is 2086)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA))
    args = ap.parse_args(argv)
    from huggingface_hub import hf_hub_download

    rng = random.Random(args.seed)
    per_source, source_meta, eval_rows = {}, {}, load_eval_questions()
    for name, (repo, fname) in SOURCES.items():
        path = Path(hf_hub_download(repo, fname, repo_type="dataset"))
        df = pd.read_parquet(path)
        items = to_items(name, df, rng)
        clean, log = overlap_check(items, eval_rows)
        per_source[name] = clean
        source_meta[name] = {"repo": repo, "file": fname, "parquet_sha256": sha256_file(path),
                             "n_rows": len(df), "n_four_option": len(items),
                             "n_after_overlap_filter": len(clean), "overlap": log}
        print(f"{name}: {len(df)} rows -> {len(items)} 4-option items -> {len(clean)} "
              f"after overlap filter ({log['n_flagged']} flagged, "
              f"max jaccard {log['max_jaccard_overall']})")

    # Equal share per source; de-duplicate on normalised question text.
    seen, pool = set(), []
    for name, items in per_source.items():
        take = args.n // len(SOURCES) + (1 if name in list(SOURCES)[: args.n % len(SOURCES)] else 0)
        for it in rng.sample(items, len(items)):
            if len([p for p in pool if p["source"] == name]) >= take:
                break
            key = normalize(it["question"])
            if key not in seen:
                seen.add(key)
                pool.append(it)
    rng.shuffle(pool)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool_path = out / "mcq_pool.json"
    pool_path.write_text(json.dumps(pool, indent=1))
    (out / "overlap_check.json").write_text(json.dumps(
        {name: m.pop("overlap") for name, m in source_meta.items()}, indent=1))
    manifest = {
        "n": len(pool), "seed": args.seed, "sha256": sha256_file(pool_path),
        "per_source": {name: sum(p["source"] == name for p in pool) for name in SOURCES},
        "answer_index_counts": {str(k): sum(p["answer_index"] == k for p in pool) for k in range(4)},
        "sources": source_meta, "schema": ["id", "source", "question", "options", "answer_index"],
        "overlap_check": "data/overlap_check.json (all flagged items were dropped)",
    }
    (out / "mcq_pool.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(pool)} items -> {pool_path} (sha256 {manifest['sha256'][:12]}...)")
    print(f"  per source: {manifest['per_source']}  answers: {manifest['answer_index_counts']}")


if __name__ == "__main__":
    main()
