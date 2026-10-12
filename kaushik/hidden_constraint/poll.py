"""Experiment 3B on every trained adapter in kaushik/registry/*.json (experiment job; sweep: sweeps/adapters.yaml).

    run.py hidden_constraint poll [--once] [--dry-run] [--cap 25]

Every non-smoke registry entry whose base model has a Modal entry in config.yaml:models becomes one
run `runs/<slice>_ckpt_<registry run_name>/` (served with the entry's eval_chat_template_kwargs, one
lockstep group). The loop rescans the registry every poll.interval_s for poll.hours and launches new
entries through the core scheduler; finished runs (DONE) are skipped. `jobs(cfg)` is the current
registry's job list, so the adapters sweep dry-runs it.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import registry as reg
from kaushik.core.config import model_key_of, resolve, runs_root
from kaushik.core.io import load_py
from kaushik.core.scheduler import Job, run_scheduler

EXP = "hidden_constraint"
ADAPTER_KEYS = ("run_name", "adapter_path", "eval_chat_template_kwargs", "experiment", "arm", "seed")


def adapter_entries(cfg: dict) -> dict[str, dict]:
    """{registry run_name: entry} for adapters this experiment can serve (Modal base model, not skipped)."""
    skip = cfg["poll"].get("skip_pattern")
    out = {}
    for e in reg.read_all().values():
        if not {"run_name", "base_model", "adapter_path"} <= set(e) or (skip and re.search(skip, e["run_name"])):
            continue
        key = model_key_of(e["base_model"])
        if key and cfg["models"].get(key, {}).get("backend") == "modal":
            out[e["run_name"]] = {**e, "_model": key}
    return out


def adapter_config(cfg: dict, entry: dict) -> dict:
    """The run config of one adapter: the base model's settings + the registry entry, one group."""
    point = {"model": entry["_model"], "adapter": {k: entry[k] for k in ADAPTER_KEYS if k in entry}, "groups": 1,
             "slice": cfg["slice"], "n": cfg["n"], "tag": cfg.get("tag"),
             "run_name": f"{cfg['slice']}_ckpt_{entry['run_name']}" + (f"_{cfg['tag']}" if cfg.get("tag") else "")}
    return resolve(cfg.get("experiment") or EXP, point)


def jobs(cfg: dict) -> list[Job]:
    run_jobs = load_py(Path(__file__).resolve().parent / "run.py").jobs
    return [j for e in adapter_entries(cfg).values() for j in run_jobs(adapter_config(cfg, e))]


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hidden_constraint poll", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="run what is in the registry now, then exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    runs_dir = runs_root(cfg)
    deadline = time.time() + float(cfg["poll"]["hours"]) * 3600
    while True:
        js = jobs(cfg)
        if js and not a.dry_run:
            load_py(Path(__file__).resolve().parent / "run.py").preflight([resolve(EXP, {"model": "qwen8b"})])
        print(f"[poll] {len(js)} adapter runs in the registry, {sum(j.done() for j in js)} done", flush=True)
        run_scheduler(js, runs_dir, cap=a.cap, name="poll_scheduler", dry_run=a.dry_run)
        if a.once or a.dry_run or time.time() > deadline:
            break
        time.sleep(float(cfg["poll"]["interval_s"]))
