"""Prune a FINISHED run's records to the analysed item subset (experiment job).

    run.py retry prune run=main [--apply]

The main run started 4464 items per arm and was restricted to the 1000-item
subset (item_specs.main.subset) mid-run, so the first rounds hold attempts
for items the analysis never reads. This drops every record of a non-subset
item from runs/<run>/<model>/<arm>/{attempts,mentions,errors}.jsonl and
episodes.json, and from the model's gen_cache/ shards (records are matched to
kept attempts by their prompt). Kept records are copied verbatim: every
rewritten file is re-read and compared to the kept records before it replaces
the original, and each arm must still hold every subset item. Without
--apply nothing is written. Filtered cache shards are no longer cache hits
(the key hashes the whole shard), which only matters for restarts of a run
that is already finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.retry import retry_common as rc


def _prompt_key(messages) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def _replace(path: Path, text: str, verify) -> None:
    tmp = path.with_suffix(path.suffix + ".pruned")
    tmp.write_text(text)
    if not verify(tmp.read_text()):
        tmp.unlink()
        raise RuntimeError(f"verification failed for {path}")
    tmp.replace(path)


def prune_jsonl(path: Path, keep: set[str], apply: bool, prompts: set[str] | None = None) -> dict:
    if not path.exists():
        return {}
    lines = path.read_text().splitlines(keepends=True)
    kept, ids = [], set()
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        iid = rec.get("item_id")
        if iid is None or iid in keep:  # records without an item id are kept
            kept.append(line)
            if iid is not None:
                ids.add(iid)
            if prompts is not None and rec.get("messages") is not None:
                prompts.add(_prompt_key(rec["messages"]))
    if path.name == "attempts.jsonl" and ids != keep:
        raise RuntimeError(f"{path}: kept items {len(ids)} != subset {len(keep)} (missing {len(keep - ids)})")
    out = {"records": len(lines), "kept": len(kept), "bytes": path.stat().st_size, "kept_bytes": sum(len(x.encode()) for x in kept)}
    if apply and len(kept) != len(lines):
        _replace(path, "".join(kept), lambda t: t.splitlines(keepends=True) == kept)
    return out


def prune_episodes(path: Path, keep: set[str], apply: bool) -> dict:
    eps = json.loads(path.read_text())
    kept = [e for e in eps if e["item_id"] in keep]
    if {e["item_id"] for e in kept} != keep:
        raise RuntimeError(f"{path}: episodes do not cover the subset")
    ref = [json.dumps(e, sort_keys=True) for e in kept]
    out = {"records": len(eps), "kept": len(kept), "bytes": path.stat().st_size}
    if apply and len(kept) != len(eps):
        _replace(path, json.dumps(kept, indent=1), lambda t: [json.dumps(e, sort_keys=True) for e in json.loads(t)] == ref)
        out["kept_bytes"] = path.stat().st_size
    return out


def prune_cache(cache_dir: Path, prompts: set[str], apply: bool) -> dict:
    out = {"files": 0, "records": 0, "kept_records": 0, "bytes": 0, "kept_bytes": 0, "deleted_files": 0, "unmatched_files": 0}
    if not cache_dir.exists():
        return out
    for f in sorted(cache_dir.glob("*.json")):
        recs = json.loads(f.read_text())
        kept = [r for r in recs if _prompt_key(r["input"]) in prompts]
        size = f.stat().st_size
        out["files"] += 1
        out["records"] += len(recs)
        out["kept_records"] += len(kept)
        out["bytes"] += size
        if not kept:
            out["deleted_files"] += 1
            if apply:
                f.unlink()
            continue
        if len(kept) == len(recs):
            out["kept_bytes"] += size
            continue
        ref = [json.dumps(r, sort_keys=True) for r in kept]
        if apply:
            _replace(f, json.dumps(kept), lambda t, ref=ref: [json.dumps(r, sort_keys=True) for r in json.loads(t)] == ref)
            out["kept_bytes"] += f.stat().st_size
        else:
            out["kept_bytes"] += int(size * len(kept) / len(recs))
    return out


def prune_run(run_dir: Path, keep: set[str], apply: bool, models: list[str]) -> dict:
    report: dict = {}
    for mdir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name in models):
        prompts: set[str] = set()
        arms = sorted(p for p in mdir.iterdir() if p.is_dir() and p.name != "gen_cache")
        unfinished = [a.name for a in arms if not (a / "episodes.json").exists()]
        if unfinished:
            raise SystemExit(f"{mdir}: arms not finished (no episodes.json): {unfinished}; refusing to prune")
        for adir in arms:
            r = {"attempts.jsonl": prune_jsonl(adir / "attempts.jsonl", keep, apply, prompts)}
            for name in ("mentions.jsonl", "errors.jsonl"):
                r[name] = prune_jsonl(adir / name, keep, apply)
            r["episodes.json"] = prune_episodes(adir / "episodes.json", keep, apply)
            report[f"{mdir.name}/{adir.name}"] = r
        kept_attempts = sum(report[f"{mdir.name}/{a.name}"]["attempts.jsonl"]["kept"] for a in arms)
        c = prune_cache(mdir / "gen_cache", prompts, apply)
        c["kept_attempt_records"] = kept_attempts
        report[f"{mdir.name}/gen_cache"] = c
    return report


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="retry prune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write; without it this is a dry run")
    a = ap.parse_args(argv)
    subset = rc.item_spec(cfg).get("subset")
    if not subset:
        raise SystemExit(f"item spec {cfg['items']!r} has no subset; nothing to prune to")
    keep = set(json.loads((rc.HERE / subset).read_text())["item_ids"])
    run_dir = rc.RUNS / str(cfg["run"])
    rep = prune_run(run_dir, keep, a.apply, list(cfg["models"]))
    (run_dir / ("prune_report.json" if a.apply else "prune_dryrun.json")).write_text(json.dumps(rep, indent=1))
    tot_b = sum(v.get("bytes", 0) for r in rep.values() for v in ([r] if "files" in r else r.values()))
    tot_k = sum(v.get("kept_bytes", v.get("bytes", 0)) for r in rep.values() for v in ([r] if "files" in r else r.values()))
    for k, r in rep.items():
        if "files" in r:
            print(f"{k}: {r['files']} shards, {r['records']} records -> {r['kept_records']} kept ({r['kept_attempt_records']} kept attempts), "
                  f"{r['deleted_files']} shards deleted, {r['bytes'] / 1e9:.2f} -> {r['kept_bytes'] / 1e9:.2f} GB")
        else:
            at = r["attempts.jsonl"]
            print(f"{k}: attempts {at['records']} -> {at['kept']}, episodes {r['episodes.json']['records']} -> {r['episodes.json']['kept']}, "
                  f"mentions {r.get('mentions.jsonl', {}).get('records', 0)} -> {r.get('mentions.jsonl', {}).get('kept', 0)}")
    print(f"{'APPLIED' if a.apply else 'DRY RUN'}: {tot_b / 1e9:.2f} GB -> {tot_k / 1e9:.2f} GB")
