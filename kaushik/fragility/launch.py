"""Launch scripts/train_eval.py for fragility configs under a GPU cap, and
report status.

    .venv/bin/python kaushik/run.py fragility schedule [--cap 35] [--half-after-hours 6] [--smoke]
    .venv/bin/python kaushik/run.py fragility launch --configs <yaml> [...]   # unconditional
    .venv/bin/python kaushik/run.py fragility status [--smoke]

`schedule` is the normal path: a foreground loop that launches queued configs in
priority order (Qwen3.6-27B 1A first — Experiment 2's baseline) whenever the
sum of the active runs' peak GPU use (train = 1 engine; eval = the config's
parallel engine calls, see peak_gpus) fits under the sprint cap (runs/gpu_cap.json,
changeable live), and two guards pass: workspace-wide active Modal containers
<= --max-containers, laptop free memory >= --min-free-mem. Each run is a detached process;
its folder (kaushik/fragility/runs/<run>/) holds train_eval.py's artifacts plus
manifest.json (dataset + config sha256, git commit, pid, command) and
launcher.log (everything the process printed, Modal output included).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml
from common import CONFIGS, PY, ROOT, RUNS, atomic_write_json, git_commit, sha256_file

PRIORITY = [("mcq_noreason", "qwen36_27b"), ("mcq_noreason", "gptoss20b"), ("mcq_noreason", "qwen8b"),
            ("mcq_reason", "gptoss20b"), ("mcq_reason", "qwen8b"),
            ("style_sft", "gptoss20b"), ("style_sft", "qwen8b")]


def _parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--configs", nargs="+", default=None, help="explicit config paths")
    ap.add_argument("--smoke", action="store_true", help="use configs/smoke/")
    return ap


def select(args) -> list[tuple[Path, dict]]:
    paths = ([Path(p) for p in args.configs] if args.configs
             else sorted((CONFIGS / ("smoke" if args.smoke else "")).glob("*.yaml")))
    out = []
    for p in paths:
        raw = yaml.safe_load(p.read_text())
        if args.arms and raw["arm"] not in args.arms:
            continue
        if args.models and raw["model_key"] not in args.models:
            continue
        if args.seeds and raw["train"]["seed"] not in args.seeds:
            continue
        out.append((p, raw))

    def key(item):
        raw = item[1]
        pair = (raw["arm"], raw["model_key"])
        return (PRIORITY.index(pair) if pair in PRIORITY else len(PRIORITY), raw["train"]["seed"])
    return sorted(out, key=key)


def peak_gpus(raw: dict) -> int:
    """Max concurrent Modal engines (1 GPU each) this run can hold: training is
    one container; an eval block runs chunk_parallel engine calls when chunked,
    else one call per checkpoint bounded by parallel_checkpoints."""
    n_ckpt = len(raw["train"]["save_at_steps"]) + 1
    peak = 1
    for blk in raw["eval"].values():
        n = len(blk["eval_at_steps"]) if blk.get("eval_at_steps") else n_ckpt
        if blk.get("chunk"):
            peak = max(peak, int(blk.get("chunk_parallel") or 1))
        else:
            peak = max(peak, min(n, int(blk.get("parallel_checkpoints", 4))))
    return peak


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def launch_one(cfg_path: Path, raw: dict) -> dict:
    run_dir = ROOT / raw["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # A run whose training finished is relaunched eval-only: retraining would
    # produce a new adapter path (and invalidate its registry entry).
    if (run_dir / "train_result.json").exists():
        cmd = [PY, "scripts/train_eval.py", "--eval-only", str(run_dir.relative_to(ROOT))]
    else:
        cmd = [PY, "scripts/train_eval.py", str(cfg_path.resolve().relative_to(ROOT))]
    manifest = {
        "run_name": raw["run_name"], "arm": raw["arm"], "model_key": raw["model_key"],
        "base_model": raw["base_model"], "seed": raw["train"]["seed"],
        "config": str(cfg_path.resolve().relative_to(ROOT)), "config_sha256": sha256_file(cfg_path),
        "data_path": raw["train"]["data_path"], "data_sha256": sha256_file(ROOT / raw["train"]["data_path"]),
        "git_commit": git_commit(), "command": " ".join(cmd), "peak_gpus": peak_gpus(raw),
        "launched_at": datetime.now().astimezone().isoformat(),
        "previous_launches": old.get("previous_launches", [])
        + ([{k: old.get(k) for k in ("pid", "launched_at")}] if old else []),
    }
    with open(run_dir / "launcher.log", "a") as log:
        log.write(f"\n===== launch {manifest['launched_at']}: {manifest['command']}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=os.environ.copy())
    manifest["pid"] = proc.pid
    atomic_write_json(manifest, manifest_path)
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "launches.jsonl", "a") as f:
        f.write(json.dumps({k: manifest[k] for k in ("run_name", "pid", "launched_at", "command")}) + "\n")
    print(f"launched {raw['run_name']} (pid {proc.pid}, peak {manifest['peak_gpus']} GPUs)", flush=True)
    return manifest


def run_state(run_dir: Path, raw: dict) -> dict:
    mpath = run_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    alive = pid_alive(manifest.get("pid"))
    n_ckpt = len(raw["train"]["save_at_steps"]) + 1
    blocks = {}
    for name, blk in raw["eval"].items():
        want = len(blk["eval_at_steps"]) if blk.get("eval_at_steps") else n_ckpt
        bdir = run_dir / "eval" / name
        blocks[name] = f"{len(list(bdir.glob('*.json'))) if bdir.is_dir() else 0}/{want}"
    if (run_dir / "eval_summary.json").exists():
        phase = "done"
    elif (run_dir / "train_result.json").exists():
        phase = "eval" if alive else "FAILED(eval)"
    elif (run_dir / "train.log").exists():
        phase = "training" if alive else "FAILED(train)"
    elif manifest:
        phase = "launched" if alive else "FAILED(start)"
    else:
        phase = "pending"
    last = ""
    if (run_dir / "launcher.log").exists():
        lines = [ln.strip() for ln in (run_dir / "launcher.log").read_text(errors="replace").splitlines()
                 if ln.strip()]
        last = lines[-1][:160] if lines else ""
    return {"run": raw["run_name"], "phase": phase, "pid": manifest.get("pid"), "alive": alive,
            "launched_at": manifest.get("launched_at"), "peak_gpus": peak_gpus(raw),
            "n_launches": 1 + len(manifest.get("previous_launches", [])) if manifest else 0,
            "eval_blocks": blocks, "last_log": last}


def _print_status(states: list[dict]) -> None:
    for s in states:
        done = sum(int(v.split("/")[0]) for v in s["eval_blocks"].values())
        want = sum(int(v.split("/")[1]) for v in s["eval_blocks"].values())
        print(f"{s['run']:<44} {s['phase']:<13} eval {done:>3}/{want:<3} | {s['last_log']}")
    counts = {}
    for s in states:
        counts[s["phase"]] = counts.get(s["phase"], 0) + 1
    active = [s for s in states if s["phase"] in ("launched", "training", "eval")]
    print(f"-- {counts}; reserved GPUs {sum(s['peak_gpus'] for s in active)}", flush=True)


def status(argv: list[str] | None = None) -> None:
    args = _parser("status of runs").parse_args(argv)
    states = [run_state(ROOT / raw["run_dir"], raw) for _, raw in select(args)]
    _print_status(states)
    out = RUNS / ("smoke/status.json" if args.smoke else "status.json")
    atomic_write_json({"at": datetime.now().astimezone().isoformat(), "runs": states}, out)


def launch(argv: list[str] | None = None) -> None:
    """Unconditional launch of the selected configs (no cap accounting)."""
    ap = _parser("launch runs")
    ap.add_argument("--force", action="store_true", help="relaunch finished/running runs too")
    args = ap.parse_args(argv)
    for cfg_path, raw in select(args):
        s = run_state(ROOT / raw["run_dir"], raw)
        if s["phase"] == "done" and not args.force:
            print(f"skip {raw['run_name']}: finished")
        elif s["alive"] and not args.force:
            print(f"skip {raw['run_name']}: running (pid {s['pid']})")
        else:
            launch_one(cfg_path, raw)
            time.sleep(3)


def workspace_containers() -> int | None:
    """Active containers across the whole Modal workspace (all agents), or None
    if the CLI call failed. Counts container ids in `modal container list`."""
    try:
        out = subprocess.run(["modal", "container", "list"], capture_output=True, text=True,
                             timeout=180, check=False)
        return out.stdout.count("ta-") if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def free_memory_pct() -> int | None:
    """macOS `memory_pressure` -> 'System-wide memory free percentage: NN%'."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=60,
                             check=False).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return int(m[1]) if m else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _waiting_for_slot(log_path: Path) -> bool:
    """True if the process' last log line is train_eval.py's ephemeral-app
    slot-retry message, i.e. it holds no Modal app right now."""
    if not log_path.exists():
        return False
    lines = [ln for ln in log_path.read_text(errors="replace").splitlines() if ln.strip()]
    return bool(lines) and "[slot-retry]" in lines[-1]


def migrate(argv: list[str] | None = None) -> None:
    """Revive eval processes: restart those stuck waiting for an ephemeral-app
    slot (they hold no app) and those that died before finishing (e.g. killed
    by a stray signal), under the deployed Modal apps (env from kaushik/run.py).
    Finished eval blocks are re-graded from the generation cache, so nothing is
    lost. Covers the runs themselves and their _workers/ helpers. Only signals
    pids recorded in our own manifests."""
    ap = _parser("revive slot-waiting or dead eval processes")
    args = ap.parse_args(argv)
    n = 0
    for cfg_path, raw in select(args):
        run_dir = ROOT / raw["run_dir"]
        targets = [(run_dir, cfg_path)] + [(w, w / "config.yaml") for w in sorted(run_dir.glob("_workers/w*"))]
        for d, cfg in targets:
            mpath = d / "manifest.json"
            if not mpath.exists() or (d / "eval_summary.json").exists():
                continue
            m = json.loads(mpath.read_text())
            alive = pid_alive(m.get("pid"))
            if alive and not _waiting_for_slot(d / "launcher.log"):
                continue  # healthy
            if alive:
                os.kill(m["pid"], 15)
                time.sleep(2)
            if d == run_dir:
                launch_one(cfg_path, raw)
            else:  # worker helper dir: eval-only on the helper itself
                cmd = [PY, "scripts/train_eval.py", "--eval-only", str(d.relative_to(ROOT))]
                with open(d / "launcher.log", "a") as log:
                    log.write(f"\n===== migrate {datetime.now().astimezone().isoformat()}: {' '.join(cmd)}\n")
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, env=os.environ.copy())
                atomic_write_json({**m, "pid": proc.pid, "migrated_at": datetime.now().astimezone().isoformat(),
                                   "previous_pid": m.get("pid")}, mpath)
                print(f"migrated worker {d.relative_to(ROOT)} (pid {proc.pid})")
            n += 1
    print(f"migrated {n} processes")


def schedule(argv: list[str] | None = None) -> None:
    ap = _parser("launch runs under a GPU cap")
    ap.add_argument("--cap", type=int, default=80, help="max concurrent GPUs (sprint cap); "
                    "runs/gpu_cap.json {\"cap\": N} overrides it live")
    ap.add_argument("--half-after-hours", type=float, default=0.0,
                    help="halve the cap this many hours after the first launch (0 = off; the "
                         "sprint replaced this rule with the two guards below)")
    ap.add_argument("--max-containers", type=int, default=800,
                    help="launch nothing while the workspace has more active containers than this")
    ap.add_argument("--min-free-mem", type=int, default=25,
                    help="launch no new local driver process below this %% free laptop memory")
    ap.add_argument("--poll", type=int, default=60, help="seconds between scheduler passes")
    ap.add_argument("--max-launches", type=int, default=2, help="attempts per run (1 retry)")
    args = ap.parse_args(argv)
    queue = select(args)
    state_path = RUNS / ("smoke/scheduler_state.json" if args.smoke else "scheduler_state.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"first_launch_at": None}
    log_path, cap_path = state_path.with_suffix(".log"), RUNS / "gpu_cap.json"
    while True:
        states = [(p, raw, run_state(ROOT / raw["run_dir"], raw)) for p, raw in queue]
        first = state["first_launch_at"]
        hours = (time.time() - first) / 3600 if first else 0.0
        full_cap = int(json.loads(cap_path.read_text())["cap"]) if cap_path.exists() else args.cap
        cap = full_cap // 2 if args.half_after_hours and hours >= args.half_after_hours else full_cap
        active = [s for _, _, s in states if s["phase"] in ("launched", "training", "eval")]
        reserved = sum(s["peak_gpus"] for s in active)
        n_launched, guard, containers, free_mem = 0, "", None, None
        for p, raw, s in states:
            retry = s["phase"].startswith("FAILED") and s["n_launches"] < args.max_launches
            if s["phase"] != "pending" and not retry:
                continue
            if reserved + s["peak_gpus"] > cap:
                continue
            if containers is None:  # guards are checked once per pass, only when something fits
                containers, free_mem = workspace_containers(), free_memory_pct()
                if containers is None or containers > args.max_containers:
                    guard = f"HOLD: workspace containers={containers} (> {args.max_containers} or unknown)"
                elif free_mem is None or free_mem < args.min_free_mem:
                    guard = f"HOLD: laptop free memory={free_mem}% (< {args.min_free_mem}% or unknown)"
            if guard:
                break
            launch_one(p, raw)
            reserved += s["peak_gpus"]
            n_launched += 1
            state["first_launch_at"] = state["first_launch_at"] or time.time()
            time.sleep(3)
        phases = {}
        for _, _, s in states:
            phases[s["phase"]] = phases.get(s["phase"], 0) + 1
        line = (f"{datetime.now().astimezone().isoformat(timespec='seconds')} cap={cap} reserved={reserved} "
                f"launched_now={n_launched} containers={containers} free_mem={free_mem}% {phases} {guard}")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
        atomic_write_json({**state, "cap": cap, "reserved": reserved, "phases": phases, "guard": guard,
                           "workspace_containers": containers, "free_mem_pct": free_mem,
                           "at": datetime.now().astimezone().isoformat()}, state_path)
        if all(s["phase"] == "done" or (s["phase"].startswith("FAILED")
                                        and s["n_launches"] >= args.max_launches)
               for _, _, s in states):
            print("scheduler: all runs finished (or failed twice); exiting", flush=True)
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    status()
