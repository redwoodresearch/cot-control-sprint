"""Experiment 3A / 3A-transfer / Experiment 3 correlation.

    .venv/bin/python kaushik/run.py retry <stage> [args]

Stages (each --help):
  check       credentials + OpenRouter model/endpoint availability (critic, judge, pins); never prints secrets
  power       sample-size statement for the main, pilot and checkpoint item sets
  subset      write configs/main_subset.json (seeded, mode-stratified 1000 items used by main)
  pilot       all arms on a TRAIN-split subset (feedback wording check; also the transfer context pool)
  run         the main 3A on the test split (all 9 modes, all arms, N=10)
  multi       several runs (e.g. pilot + main) for the Modal models in ONE process / ONE Modal app
  transfer    3A-transfer (one2one + nine2heldout + few-shot k=1 rerun) from a pilot run's episodes
  registry    poll kaushik/registry for trained adapters and run the reduced 3A on each
  analyze     curves, tables, transfer tables, correlation plots -> runs/<run>/results.md + figures/
  prune       drop non-subset items from a finished run's records (dry run unless --apply)
  status      progress of a run from its attempt logs
  gpus        count our running Modal containers (GPU cap check)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from retry_common import (
    CFG,
    MODELS,
    ROOT,
    RUNS,
    active_models,
    load_items,
    power_statement,
)


def _p(desc: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=desc)


def stage_check(argv: list[str]) -> None:
    import os

    ap = _p("credentials + model availability")
    ap.parse_args(argv)
    print("OPENROUTER_API_KEY set:", bool(os.environ.get("OPENROUTER_API_KEY")))
    r = subprocess.run(["modal", "profile", "current"], capture_output=True, text=True, check=False)
    print("modal profile:", r.stdout.strip() or r.stderr.strip()[:200])
    r = subprocess.run(["modal", "token", "info"], capture_output=True, text=True, check=False)
    print("modal token:", "ok" if r.returncode == 0 else "MISSING")
    if not os.environ.get("OPENROUTER_API_KEY"):
        return
    import httpx
    hdr = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    models = {m["id"]: m for m in httpx.get("https://openrouter.ai/api/v1/models", headers=hdr, timeout=60).json()["data"]}
    wanted = {CFG["critic"]["model"], CFG["judge_model"]} | {m["model"] for m in MODELS.values() if m["backend"] == "openrouter"}
    for mid in sorted(wanted):
        print(f"{mid}: {'served' if mid in models else 'NOT SERVED'}")
        if mid in models:
            ep = httpx.get(f"https://openrouter.ai/api/v1/models/{mid}/endpoints", headers=hdr, timeout=60).json()
            provs = [(e.get("provider_name"), e.get("quantization")) for e in ep.get("data", {}).get("endpoints", [])]
            print("   endpoints:", provs)
    print("openai/* small models:", sorted(m for m in models if m.startswith("openai/") and "mini" in m or "nano" in m))


def stage_subset(argv: list[str]) -> None:
    from retry_common import HERE as RHERE
    from retry_common import make_subset
    ap = _p("write the seeded, mode-stratified item subset used by the main run")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="configs/main_subset.json")
    a = ap.parse_args(argv)
    spec = {k: v for k, v in CFG["main"].items() if k != "subset"}
    d = make_subset(spec, a.n, a.seed, RHERE / a.out)
    print(f"{d['n']} items (per mode {d['per_mode']}) from {d['n_source_items']}; sha256 {d['sha256']} -> {a.out}")


def stage_power(argv: list[str]) -> None:
    ap = _p("sample-size statement")
    ap.parse_args(argv)
    for name in ("main", "pilot", "checkpoint"):
        spec = CFG[name]
        n = len(load_items("openai/gpt-oss-20b", **spec))
        print(f"{name}: {spec} -> {power_statement(n)}; {power_statement(n, 0.3)}")


def _common_run_args(ap):
    ap.add_argument("--run", required=True, help="run name under kaushik/retry/runs/")
    ap.add_argument("--models", default=None, help=f"comma-separated labels (default: all enabled: {active_models(None)})")
    ap.add_argument("--arms", default=",".join(CFG["arms"]))
    ap.add_argument("--n-attempts", type=int, default=CFG["n_attempts"])
    ap.add_argument("--gpus", type=int, default=CFG["modal"]["gpu_cap"], help="process-wide concurrent Modal GPU cap")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--subsample-seed", type=int, default=None)
    ap.add_argument("--modes-per-question", type=int, default=None)
    ap.add_argument("--modes", default=None, help="restrict to these constraint modes (comma-separated)")


def _item_spec(a, base: dict) -> dict:
    spec = dict(base)
    if a.max_questions is not None:
        spec["max_questions"] = a.max_questions
    if a.subsample_seed is not None:
        spec["subsample_seed"] = a.subsample_seed
    if a.modes_per_question is not None:
        spec["modes_per_question"] = a.modes_per_question
    if a.modes:
        spec["modes"] = a.modes.split(",")
    return spec


def stage_pilot(argv: list[str]) -> None:
    from retry_run3a import run_models
    ap = _p("pilot all arms on the train split")
    _common_run_args(ap)
    a = ap.parse_args(argv)
    spec = _item_spec(a, CFG["pilot"])
    asyncio.run(run_models(a.run, active_models(a.models), a.arms.split(","), spec, a.n_attempts, gpu_cap=a.gpus))


def stage_run(argv: list[str]) -> None:
    from retry_run3a import run_models
    ap = _p("main 3A on the test split")
    _common_run_args(ap)
    a = ap.parse_args(argv)
    spec = _item_spec(a, CFG["main"])
    asyncio.run(run_models(a.run, active_models(a.models), a.arms.split(","), spec, a.n_attempts, gpu_cap=a.gpus))


def stage_multi(argv: list[str]) -> None:
    """Several runs in one process / one Modal app, e.g.
    --job pilot:gptoss20b,qwen8b --job main:gptoss20b,qwen8b"""
    from retry_run3a import run_many
    ap = _p("pilot + main (or any runs) for the Modal models in one process")
    ap.add_argument("--job", action="append", required=True, help="<run>:<label,...>  (run = pilot|main|<name> with CFG item spec of pilot/main)")
    ap.add_argument("--arms", default=",".join(CFG["arms"]))
    ap.add_argument("--n-attempts", type=int, default=CFG["n_attempts"])
    ap.add_argument("--gpus", type=int, default=CFG["modal"]["gpu_cap"], help="hard per-pool container ceiling")
    a = ap.parse_args(argv)
    jobs = []
    for j in a.job:
        run, labels = j.split(":")
        jobs.append((run, labels.split(","), a.arms.split(","), dict(CFG["pilot" if run.startswith("pilot") else "main"])))
    asyncio.run(run_many(jobs, a.n_attempts, a.gpus))


def stage_transfer(argv: list[str]) -> None:
    from retry_transfer import run_transfer
    ap = _p("3A-transfer")
    ap.add_argument("--run", required=True)
    ap.add_argument("--context-run", required=True, help="pilot run (train split) supplying bare/critic episodes")
    ap.add_argument("--models", default=None)
    ap.add_argument("--configs", default="nine2heldout,one2one")
    ap.add_argument("--gpus", type=int, default=CFG["modal"]["gpu_cap"])
    ap.add_argument("--api-parallel", type=int, default=3, help="concurrent OpenRouter evals")
    ap.add_argument("--max-questions", type=int, default=None, help="smoke: cap each eval at N questions")
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--one2one-questions", type=int, default=None, help="test questions per one-to-one cell (default 200)")
    a = ap.parse_args(argv)
    asyncio.run(run_transfer(a.run, active_models(a.models), a.context_run, a.configs.split(","),
                             gpu_cap=a.gpus, api_parallel=a.api_parallel, max_questions=a.max_questions, draws=a.draws,
                             one2one_questions=a.one2one_questions))


def stage_registry(argv: list[str]) -> None:
    from retry_correlate import poll
    ap = _p("poll the registry and run the reduced 3A on every trained adapter")
    ap.add_argument("--run", default="ckpt")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--idle-hours", type=float, default=6, help="stop after this long without a new entry (once all are done)")
    ap.add_argument("--poll-min", type=float, default=10)
    ap.add_argument("--gpus", type=int, default=CFG["modal"]["gpu_cap"])
    ap.add_argument("--max-parallel", type=int, default=4, help="adapters evaluated at once")
    ap.add_argument("--labels", default=None, help="only adapters on these base models (one pool per poller), e.g. qwen8b")
    a = ap.parse_args(argv)
    asyncio.run(poll(a.run, a.hours, a.idle_hours, a.poll_min, a.gpus, a.max_parallel,
                     a.labels.split(",") if a.labels else None))


def stage_analyze(argv: list[str]) -> None:
    _load_analyze().main(argv)


def _load_analyze():
    """analyze.py keeps the repo-convention name, so load it by path (every experiment folder has one)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("kaushik_retry_analyze", HERE / "analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_prune(argv: list[str]) -> None:
    """Drop non-subset items from a FINISHED run's records (dry run unless --apply)."""
    from retry_prune import main as prune_main
    prune_main(argv)


def stage_status(argv: list[str]) -> None:
    from cotcontrol.eval.retry import load_attempts
    ap = _p("progress of a run")
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    for path in sorted((RUNS / a.run).glob("**/attempts.jsonl")):
        eps = load_attempts(path)
        man = json.loads((path.parent / "manifest.json").read_text()) if (path.parent / "manifest.json").exists() else {}
        n_items, n_att = man.get("n_items", "?"), man.get("n_attempts", CFG["n_attempts"])
        succ = sum(any(r["honest"] for r in recs) for recs in eps.values())
        finished = sum(any(r["honest"] for r in recs) and man.get("stop_on_success", True) or len(recs) >= n_att
                       for recs in eps.values())
        errs = sum(1 for _ in (path.parent / "errors.jsonl").open()) if (path.parent / "errors.jsonl").exists() else 0
        print(f"{path.parent.relative_to(RUNS)}: {len(eps)}/{n_items} items started, {finished} finished, "
              f"{succ} with an honest-compliant attempt, {sum(map(len, eps.values()))} attempts, {errs} errored generations")


def stage_gpus(argv: list[str]) -> None:
    ap = _p("live Modal containers held by this folder's drivers (sum of `tasks` over our app ids) + launch guards")
    ap.parse_args(argv)
    ours = {line.strip() for p in RUNS.glob("**/modal_app_id.txt") for line in p.read_text().splitlines() if line.strip()}
    ours |= {v for v in (os.environ.get("COTCONTROL_MODAL_INFERENCE_APP"),) if v}
    r = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        print(r.stderr.strip()[:300])
        return
    import re
    apps = json.loads(re.sub(r"\x1b\[[0-9;]*m", "", r.stdout) or "[]")  # rich adds color codes even when piped
    mine = [a for a in apps if a["app_id"] in ours or (a.get("description") in ours and a.get("state") != "stopped")]
    live = [a for a in apps if a.get("state") != "stopped"]
    total = sum(int(a.get("tasks") or 0) for a in live)
    caps = json.loads((HERE / "configs" / "gpu_caps.json").read_text()) if (HERE / "configs" / "gpu_caps.json").exists() else {}
    mem = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout.strip().splitlines()
    busy = {}
    for f in (RUNS / "_gpu_inflight").glob("*.json"):
        try:
            d = json.loads(f.read_text())
            alive = subprocess.run(["kill", "-0", str(d["pid"])], capture_output=True, check=False).returncode == 0
            busy[d["name"]] = d["in_flight"] if alive else 0
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    print(f"our busy GPUs (in-flight engine calls, live drivers): {sum(busy.values())} / cap {CFG['modal']['gpu_cap']} "
          f"-> {busy}; live caps {caps} (idle warm containers per pool <= its cap, scaledown 300s)")
    print("containers in apps we use (the deployed app is SHARED by every sprint agent): "
          + ", ".join(f"{a['description']}[{a['app_id']}]={a.get('tasks')}" for a in mine))
    print(f"workspace: {total} containers in {len(live)} live apps, {sum(a.get('state') == 'ephemeral' for a in live)} ephemeral "
          f"(launch guard: <= 800 containers); {mem[-1] if mem else 'memory_pressure unavailable'} (launch guard: >= 25% free)")

STAGES = {
    "check": stage_check, "power": stage_power, "subset": stage_subset, "pilot": stage_pilot, "run": stage_run, "multi": stage_multi,
    "transfer": stage_transfer, "registry": stage_registry, "analyze": stage_analyze, "prune": stage_prune,
    "status": stage_status, "gpus": stage_gpus,
}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    STAGES[sys.argv[1]](sys.argv[2:])
