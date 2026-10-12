"""Replay the arm's own final eval blocks on a group's noise adapters (after `make`).

    .venv/bin/python kaushik/run.py noise replay [--group G] [--dry-run] [--force]

The blocks come from the reference seed's train_eval config through the registry's
`noise_eval_blocks` selector (config.yaml: replay.blocks overrides it, replay.patch is applied
to every block); the other seeds must share the same final eval (a warning otherwise). Each
(block, adapter) is one scheduler job (kaushik.core.jobs.eval_block_job: the source run's
TrainConfig, unchanged settings, T=1 / 16k / test split / judge) ->
runs/<G>/eval/<block>/noise<k>.json; `replay:<G>/summary` then writes eval_summary.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kaushik.core.io import KAUSHIK, load_py, read_json
from kaushik.core.jobs import eval_block_job
from kaushik.core.scheduler import Job, run_scheduler

nz = load_py(KAUSHIK / "noise" / "common.py")


def reference(cfg: dict) -> dict:
    """The seed whose adapter shaped the noise (noise_adapters.json), else the first seed."""
    entries = cfg["group"]["entries"]
    na = read_json(nz.group_dir(cfg) / "noise_adapters.json")
    if na:
        return next(e for e in entries if e["run_name"] == na["reference"]["run_name"])
    return entries[0]


def blocks(cfg: dict, quiet: bool = False):
    """(reference entry, its TrainConfig, [(block, fully specified block)])."""
    ref = reference(cfg)
    src = nz.source_config(ref["config_path"])
    if src is None:
        raise SystemExit(f"{cfg['run_name']}: source config {ref['config_path']} missing")
    tc, _, _ = src
    tc.run_name = ref["run_name"]
    chosen = nz.resolve_blocks(cfg, ref, src)
    mine = {n: {k: v for k, v in ec.items() if k != "parallel_checkpoints"} for n, ec in chosen}
    for e in cfg["group"]["entries"]:   # the arm's seeds must share one final eval
        if e is not ref and (s := nz.source_config(e["config_path"])):
            other = {n: {k: v for k, v in ec.items() if k != "parallel_checkpoints"} for n, ec in nz.resolve_blocks(cfg, e, s)}
            if other != mine and not quiet:
                print(f"[replay] WARNING {e['run_name']}: final eval blocks differ from {ref['run_name']}'s; "
                      f"using {ref['run_name']}'s", flush=True)
    expect = cfg["replay"]["expect"]
    for n, ec in chosen:
        off = {k: ec.get(k) for k, v in expect.items() if ec.get(k) != v}
        if off and not quiet:
            print(f"[replay] WARNING block {n}: {off} (spec: {expect}) -- replicated as configured", flush=True)
    return ref, tc, chosen


def eval_path(cfg: dict, block: str, k: int) -> Path:
    return nz.group_dir(cfg) / "eval" / block / f"noise{k}.json"


def write_summary(cfg: dict, chosen, na: dict) -> dict:
    """eval_summary.json: per (block, adapter) the compliance / accuracy / per-mode summary of
    the saved eval JSON (the noise analogue of train_eval's eval_summary.json)."""
    rows = []
    for bname, _ in chosen:
        for a in na["adapters"]:
            data = read_json(eval_path(cfg, bname, a["k"]))
            if data is None:
                continue
            s = data["summary"]
            rows.append({"block": bname, "k": a["k"], "lora_path": a["path"], "compliance_rate": s["compliance_rate"],
                         "accuracy": s["accuracy"], "n_errors": s["n_errors"], "per_mode": s["per_mode"]})
    summary = {"group": cfg["run_name"], "base_model": cfg["group"]["base_model"], "blocks": dict(chosen),
               "evals": sorted(rows, key=lambda r: (r["block"], r["k"])), "timestamp": nz.now()}
    nz.write_json(summary, nz.group_dir(cfg) / "eval_summary.json")
    return summary


def jobs(cfg: dict, force: bool = False) -> list[Job]:
    g = cfg["group"]
    if not g["noise"]:
        return []
    name, gdir = cfg["run_name"], nz.group_dir(cfg)
    rank, na_path = nz.model_rank(cfg, g["base_model"]), gdir / "noise_adapters.json"

    def real() -> list[Job]:
        na = read_json(na_path)
        _, tc, chosen = blocks(cfg)
        default_chunk = int(cfg["replay"]["default_chunk"])
        out, paths = [], []
        for bname, blk in chosen:
            eff = {**blk, "chunk": int(blk.get("chunk") or 0) or default_chunk}
            for a in na["adapters"]:
                p = eval_path(cfg, bname, a["k"])
                paths.append(p)
                j = eval_block_job(bname, eff, g["base_model"], tc, a["path"], p,
                                   backend_extra={"noise": {k: a[k] for k in ("k", "seed", "scale")}, "group": name},
                                   priority=(5, rank, name, bname, int(a["k"])), group=str(gdir))
                if force:
                    j.done = lambda: False
                out.append(j)
        es = gdir / "eval_summary.json"

        want = {(bname, int(a["k"])) for bname, _ in chosen for a in na["adapters"]}

        def summary_done() -> bool:
            s = read_json(es)
            listed = {(r["block"], int(r["k"])) for r in (s or {}).get("evals") or []}
            return not force and want <= listed and all(p.exists() for p in paths)

        async def summary():
            write_summary(cfg, chosen, na)

        out.append(Job(name=f"replay:{name}/summary", weight=0, run=summary, done=summary_done,
                       ready=lambda: all(p.exists() for p in paths), priority=(6, rank, name), group=str(gdir)))
        return out

    if na_path.exists():
        return real()
    return [Job(name=f"replay:{name}/*", weight=0, ready=na_path.exists, expand=real, priority=(5, rank, name),
                group=str(gdir))]


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="noise replay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default=None, help="one group of sweeps/groups.yaml (default: every noise group)")
    ap.add_argument("--force", action="store_true", help="re-run evals whose JSON exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    points = [nz.group_point(cfg, a.group)] if a.group else nz.group_points(cfg)
    all_jobs = [j for p in points for j in jobs(p, force=a.force)]
    if not all_jobs:
        print("[replay] nothing to do (no noise group)")
        return
    run_scheduler(all_jobs, nz.runs_dir(cfg), cap=a.cap, name="replay_scheduler", dry_run=a.dry_run)
