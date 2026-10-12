"""Build the C4 continued-pretraining set for arms 1B/1C.

Same source and filters as SDF's `sdf/download_c4.py` (allenai/c4, English
train, config `data`: 200 <= chars <= 12000, seeded shuffle 42), read straight
from the dataset's json.gz shards so no extra package is needed. Rows are
train_eval.py text rows: {"text": ..., "meta": {...}} — raw documents, every
token supervised, no chat template.

Sizing: data.files.full holds enough documents that BOTH tokenizers pack into
at least (1B steps x batch x max_seq_length) x margin tokens, so no run repeats
data; data.files.steps_1c is the prefix of the same shuffle sized for the
50-step 1C runs (smaller upload, smaller local driver). Also writes a smoke
file and a manifest (sha256 of each file, token counts per model, the
eval-overlap scan).

    .venv/bin/python kaushik/run.py fragility/cpt build_data [--margin 1.08] [--shards 0]
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import model_spec, resolve
from kaushik.core.io import rel, sha256_file

EXP = "fragility/cpt"


def fetch_shard(raw_dir: Path, url: str, i: int) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / f"c4-train.{i:05d}.json.gz"
    if dst.exists():
        return dst
    tmp = dst.with_suffix(".tmp")
    print(f"downloading shard {i} ...", flush=True)
    with urllib.request.urlopen(url.format(i=i), timeout=120) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    return dst


def load_docs(raw_dir: Path, url: str, shards: list[int], min_chars: int, max_chars: int, seed: int) -> list[dict]:
    docs = []
    for i in shards:
        with gzip.open(fetch_shard(raw_dir, url, i), "rt") as f:
            for line in f:
                row = json.loads(line)
                if min_chars <= len(row["text"]) <= max_chars:
                    docs.append({"text": row["text"],
                                 "meta": {"source": "allenai/c4", "url": row["url"], "shard": i}})
    random.Random(seed).shuffle(docs)
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


def main(cfg: dict, argv: list[str] | None = None) -> None:
    d, tr, an = cfg["data"], cfg["train"], cfg["analysis"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, nargs="+", default=list(d["shards"]))
    ap.add_argument("--margin", type=float, default=float(d["margin"]))
    ap.add_argument("--skip-overlap", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    data_dir = ROOT / d["dir"]
    out, out_50, smoke_out = (data_dir / d["files"][k] for k in ("full", "steps_1c", "smoke"))
    batch, seq, steps_1b, steps_1c = int(tr["batch_size"]), int(tr["max_seq_length"]), int(tr["max_steps"]), int(an["steps_1c"])
    models = {m: model_spec(m)["base_model"] for m in an["models"]}

    docs = load_docs(data_dir / "raw", d["shard_url"], args.shards, int(d["min_chars"]), int(d["max_chars"]),
                     int(d["shuffle_seed"]))
    print(f"{len(docs)} filtered docs from shards {args.shards}")
    counts = {m: token_counts(docs, base) for m, base in models.items()}

    def prefix_for(steps: int) -> tuple[int, int]:
        """(n_docs, target): smallest prefix whose tokens reach steps x batch x
        seq x margin for every model."""
        target = int(steps * batch * seq * args.margin)
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

    n_docs, target = prefix_for(steps_1b)
    n_docs_50, target_50 = prefix_for(steps_1c)  # 1C runs ship this smaller prefix
    docs = docs[:n_docs]

    data_dir.mkdir(parents=True, exist_ok=True)
    smoke_docs = int(d["smoke_docs"])
    for path, rows in ((out, docs), (out_50, docs[:n_docs_50]), (smoke_out, docs[:smoke_docs])):
        with path.open("w") as f:
            for doc in rows:
                f.write(json.dumps(doc) + "\n")

    manifest = {
        "source": "allenai/c4 en train", "shards": args.shards, "shard_url": d["shard_url"],
        "filters": {"min_chars": int(d["min_chars"]), "max_chars": int(d["max_chars"])}, "shuffle_seed": int(d["shuffle_seed"]),
        "n_docs": n_docs, "chars": sum(len(doc["text"]) for doc in docs),
        "target_tokens": target, "margin": args.margin,
        "steps_x_batch_x_seq": [steps_1b, batch, seq],
        "tokens": {m: sum(c[:n_docs]) for m, c in counts.items()},
        "packed_sequences": {m: sum(c[:n_docs]) // seq for m, c in counts.items()},
        "sha256": sha256_file(out),
        "n_docs_50": n_docs_50, "target_tokens_50": target_50, "steps_50": steps_1c,
        "tokens_50": {m: sum(c[:n_docs_50]) for m, c in counts.items()},
        "packed_sequences_50": {m: sum(c[:n_docs_50]) // seq for m, c in counts.items()},
        "sha256_50": sha256_file(out_50),
        "smoke_docs": smoke_docs, "smoke_sha256": sha256_file(smoke_out),
        "overlap_scan": None if args.skip_overlap else overlap_scan(docs),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0),
    }
    (data_dir / "c4_cpt.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "overlap_scan"}, indent=2))
    print("overlap:", json.dumps(manifest["overlap_scan"]))
    print(f"wrote {rel(out)}, {rel(out_50)} and {rel(smoke_out)}")


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
