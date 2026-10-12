"""The reduced 3A on every trained adapter in kaushik/registry (experiment job; sweep: sweeps/ckpt.yaml).

    run.py retry ckpt run=ckpt items=checkpoint model=qwen8b [--once] [--dry-run] [--cap 12]

One sweep point per Modal base model; `jobs(cfg)` expands it over the registry (one Job per adapter x
checkpoint arm, seed-0-first priority). `main` keeps rescanning the registry every ckpt.poll_min
minutes for ckpt.hours (or ckpt.idle_hours without a new entry) and launches new adapters through
the core scheduler; finished arms (episodes.json) are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.io import load_py
from kaushik.core.scheduler import run_scheduler
from kaushik.retry import retry_common as rc
from kaushik.retry import retry_correlate as corr


def jobs(cfg: dict) -> list:
    if not cfg.get("model"):
        raise SystemExit("ckpt needs a model: run.py retry ckpt run=ckpt items=checkpoint model=qwen8b")
    return corr.adapter_jobs(cfg)


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="retry ckpt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="run what is in the registry now, then exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    if not a.dry_run:
        load_py(Path(__file__).resolve().parent / "run3a.py").preflight([cfg])
    runs_dir = rc.RUNS / str(cfg["run"])
    t0 = last_new = time.time()
    seen: set[str] = set()
    while True:
        js = jobs(cfg)
        names = {j.name for j in js}
        if names - seen:
            last_new = time.time()
            seen |= names
        print(f"[ckpt] {len(js)} arm jobs for {cfg['model']} adapters, {sum(j.done() for j in js)} done", flush=True)
        run_scheduler(js, runs_dir, cap=a.cap, name=f"ckpt_{cfg['model']}", dry_run=a.dry_run)
        hours, idle = (time.time() - t0) / 3600, (time.time() - last_new) / 3600
        if a.once or a.dry_run or hours >= float(cfg["ckpt"]["hours"]) or idle >= float(cfg["ckpt"]["idle_hours"]):
            break
        time.sleep(float(cfg["ckpt"]["poll_min"]) * 60)
