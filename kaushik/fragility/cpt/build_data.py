"""Build the C4 continued-pretraining set for arms 1B/1C.

Same source and filters as SDF's `sdf/download_c4.py` (allenai/c4, English
train, 200 <= chars <= 12000, seeded shuffle 42), but read straight from the
dataset's json.gz shards so no extra package is needed. Rows are
train_eval.py text rows: {"text": ..., "meta": {...}} — raw documents, every
token supervised, no chat template.

Sizing: c4_cpt.jsonl holds enough documents that BOTH tokenizers pack into at
least (1B steps x batch x max_seq_length) x margin tokens, so no run repeats
data; c4_cpt_50steps.jsonl is the prefix of the same shuffle sized for the
50-step 1C runs (smaller upload, smaller local driver). Also writes a 2000-doc
smoke file and a manifest (sha256 of each file, token counts per model, the
eval-overlap scan).

    .venv/bin/python kaushik/run.py fragility/cpt data [--margin 1.08] [--shards 0]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from make_configs import BATCH_SIZE, MAX_SEQ_LENGTH, MODELS, STEPS_1B, STEPS_1C

DATA_DIR = HERE / "data"
RAW_DIR = DATA_DIR / "raw"
OUT = DATA_DIR / "c4_cpt.jsonl"
OUT_50 = DATA_DIR / "c4_cpt_50steps.jsonl"
SMOKE_OUT = DATA_DIR / "c4_smoke.jsonl"
SHARD_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:05d}-of-01024.json.gz"
MIN_CHARS, MAX_CHARS, SEED = 200, 12_000, 42
SMOKE_DOCS = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_shard(i: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dst = RAW_DIR / f"c4-train.{i:05d}.json.gz"
    if dst.exists():
        return dst
    tmp = dst.with_suffix(".tmp")
    print(f"downloading shard {i} ...", flush=True)
    with urllib.request.urlopen(SHARD_URL.format(i=i), timeout=120) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    return dst


def load_docs(shards: list[int]) -> list[dict]:
    docs = []
    for i in shards:
        with gzip.open(fetch_shard(i), "rt") as f:
            for line in f:
                row = json.loads(line)
                if MIN_CHARS <= len(row["text"]) <= MAX_CHARS:
                    docs.append({"text": row["text"],
                                 "meta": {"source": "allenai/c4", "url": row["url"], "shard": i}})
    random.Random(SEED).shuffle(docs)
    return docs


def token_counts(docs: list[dict], base_model: str, batch: int = 2000) -> list[int]:
    """Per-document token count as render_text will see it (body + doc-end,
    + BOS when the tokenizer has one)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    extra = 1 + (tok.bos_token_id is not None)
    counts = []
    for s in range(0, len(docs), batch):
        enc = tok([d["text"] for d in docs[s:s + batch]], add_special_tokens=False)["input_ids"]
        counts += [len(ids) + extra for ids in enc]
    return counts


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def overlap_scan(docs: list[dict], snippet_chars: int = 80) -> dict:
    """Count documents containing the opening of any eval question (all
    splits), on whitespace/case-normalized text. Logged, not acted on: the
    spec treats C4 as overlap-free beyond logging."""
    from cotcontrol.eval.data import load_dataset

    qs = load_dataset("all", "baseline")
    snippets = sorted({_norm(q["question"])[:snippet_chars] for q in qs
                       if len(_norm(q["question"])) >= 40})
    hits = []
    for i, d in enumerate(docs):
        t = _norm(d["text"])
        if any(s in t for s in snippets):
            hits.append(i)
    return {"n_eval_questions": len(qs), "n_snippets": len(snippets),
            "snippet_chars": snippet_chars, "n_docs_with_hit": len(hits), "hit_doc_idx": hits[:50]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, nargs="+", default=[0])
    ap.add_argument("--margin", type=float, default=1.08)
    ap.add_argument("--skip-overlap", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()

    docs = load_docs(args.shards)
    print(f"{len(docs)} filtered docs from shards {args.shards}")
    counts = {m: token_counts(docs, spec["base_model"]) for m, spec in MODELS.items()}

    def prefix_for(steps: int) -> tuple[int, int]:
        """(n_docs, target): smallest prefix whose tokens reach steps x batch x
        seq x margin for every model."""
        target = int(steps * BATCH_SIZE * MAX_SEQ_LENGTH * args.margin)
        n_docs = 0
        for m, c in counts.items():
            cum, n = 0, 0
            while cum < target:
                if n >= len(c):
                    raise SystemExit(f"{m}: shards hold {cum} tokens < target {target}; add --shards")
                cum += c[n]
                n += 1
            n_docs = max(n_docs, n)
        return n_docs, target

    n_docs, target = prefix_for(STEPS_1B)
    n_docs_50, target_50 = prefix_for(STEPS_1C)  # 1C runs ship this smaller prefix
    docs = docs[:n_docs]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in ((OUT, docs), (OUT_50, docs[:n_docs_50]), (SMOKE_OUT, docs[:SMOKE_DOCS])):
        with path.open("w") as f:
            for d in rows:
                f.write(json.dumps(d) + "\n")

    manifest = {
        "source": "allenai/c4 en train", "shards": args.shards, "shard_url": SHARD_URL,
        "filters": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS}, "shuffle_seed": SEED,
        "n_docs": n_docs, "chars": sum(len(d["text"]) for d in docs),
        "target_tokens": target, "margin": args.margin,
        "steps_x_batch_x_seq": [STEPS_1B, BATCH_SIZE, MAX_SEQ_LENGTH],
        "tokens": {m: sum(c[:n_docs]) for m, c in counts.items()},
        "packed_sequences": {m: sum(c[:n_docs]) // MAX_SEQ_LENGTH for m, c in counts.items()},
        "sha256": sha256(OUT),
        "n_docs_50": n_docs_50, "target_tokens_50": target_50, "steps_50": STEPS_1C,
        "tokens_50": {m: sum(c[:n_docs_50]) for m, c in counts.items()},
        "packed_sequences_50": {m: sum(c[:n_docs_50]) // MAX_SEQ_LENGTH for m, c in counts.items()},
        "sha256_50": sha256(OUT_50),
        "smoke_docs": SMOKE_DOCS, "smoke_sha256": sha256(SMOKE_OUT),
        "overlap_scan": None if args.skip_overlap else overlap_scan(docs),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0),
    }
    (DATA_DIR / "c4_cpt.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "overlap_scan"}, indent=2))
    print("overlap:", json.dumps(manifest["overlap_scan"]))
    print(f"wrote {OUT.relative_to(ROOT)}, {OUT_50.relative_to(ROOT)} and {SMOKE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
