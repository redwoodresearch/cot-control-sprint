"""kaushik/orange — Experiment 2A (orange SFT vs its controls) + the Experiment 2 table.

    .venv/bin/python kaushik/run.py orange <stage> [args]

  pool      regenerate the orange-MCQ pool via OpenRouter -> data/        (build_pool.py)
  data      pool -> SFT rows for orange / orange_nopersona -> data/       (build_sft_data.py)
  configs   emit the run configs -> configs/                              (make_configs.py)
  smoke     tiny end-to-end run: 2 steps, 8 questions -> runs/smoke/
  train     launch training for every config in the background (train_run.py, one process each)
  eval      start the eval scheduler in the background (eval_scheduler.py; --fg to block)
  status    training / eval progress of every run + scheduler state
  registry  (re)write kaushik/registry/<run>.json for runs whose final adapter exists
  analyze   Experiment 2 table + plots -> results/                        (analyze.py)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs"
PY = sys.executable
MODEL_ORDER = {"Qwen/Qwen3.6-27B": 0, "openai/gpt-oss-20b": 1}  # critical path first


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def free_memory_pct() -> int | None:
    """macOS memory_pressure's system-wide free percentage (None if unavailable)."""
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else None


def _memory_ok(min_free_pct: int = 25) -> bool:
    free = free_memory_pct()
    if free is not None and free < min_free_pct:
        print(f"laptop free memory {free}% < {min_free_pct}%: not starting more local processes")
        return False
    return True


def _spawn(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(f"\n=== {datetime.now().isoformat()} {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                             env=os.environ.copy(), start_new_session=True)
    return p.pid


def pool(argv):
    import build_pool
    build_pool.main(argv)


def data(argv):
    import build_sft_data
    build_sft_data.main(argv)


def configs(argv):
    import make_configs
    make_configs.main(argv)


def train(argv):
    """Launch train_run.py for each config as a detached background process."""
    ap = argparse.ArgumentParser(prog="orange train")
    ap.add_argument("--configs", default=str(HERE / "configs" / "*.yaml"), help="glob")
    ap.add_argument("--force", action="store_true", help="relaunch finished/running runs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.configs))
    cfgs = [(p, yaml.safe_load(Path(p).read_text())) for p in paths]
    cfgs.sort(key=lambda pc: (MODEL_ORDER.get(pc[1]["base_model"], 9), pc[0]))
    launched = 0
    for path, raw in cfgs:
        run_dir = ROOT / raw["run_dir"]
        launcher = run_dir / "launcher.json"
        if not args.force:
            if (run_dir / "train_result.json").exists():
                print(f"done     {raw['run_name']}")
                continue
            if launcher.exists() and _pid_alive(json.loads(launcher.read_text())["pid"]):
                print(f"running  {raw['run_name']}")
                continue
        if args.dry_run:
            print(f"would launch {raw['run_name']} ({raw['base_model']}, gpu {raw['gpu']})")
            continue
        if not _memory_ok():
            break
        pid = _spawn([PY, str(HERE / "train_run.py"), path], run_dir / "launcher.log")
        gpus = int(str(raw["gpu"]).rsplit(":", 1)[1]) if ":" in str(raw["gpu"]) else 1
        launcher.write_text(json.dumps({"pid": pid, "config": path, "gpus": gpus,
                                        "started": datetime.now().isoformat()}))
        marker = run_dir.parent / ".first_launch"
        if not raw.get("meta", {}).get("smoke") and not marker.exists():
            marker.write_text(datetime.now().isoformat())
        print(f"launched {raw['run_name']} pid {pid}")
        launched += 1
    print(f"{launched} launched")


def eval(argv):
    """Start eval_scheduler.py (background unless --fg); extra args pass through."""
    fg = "--fg" in argv
    argv = [a for a in argv if a != "--fg"]
    runs_dir = RUNS
    if "--runs-dir" in argv:
        runs_dir = Path(argv[argv.index("--runs-dir") + 1])
    cmd = [PY, str(HERE / "eval_scheduler.py"), *argv]
    if fg:
        subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), check=True)
        return
    pid_file = runs_dir / "eval_scheduler.pid"
    if pid_file.exists() and _pid_alive(int(pid_file.read_text())):
        print(f"scheduler already running (pid {pid_file.read_text()}); it picks up new runs itself")
        return
    if not _memory_ok():
        return
    pid = _spawn(cmd, runs_dir / "eval_scheduler.launcher.log")
    print(f"scheduler launched pid {pid}; log {runs_dir / 'eval_scheduler.log'}")


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.2f}"  # baseline mode has no compliance


def _total_jobs(run_dir: Path) -> int | None:
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        tr = json.loads((run_dir / "train_result.json").read_text())
    except FileNotFoundError:
        return None
    steps = {int(c["step"]) for c in tr["checkpoints"]}
    n = 0
    for blk in cfg["eval"].values():
        if not isinstance(blk, dict) or not blk.get("run", True):
            continue
        at = blk.get("eval_at_steps")
        n += len(steps & {int(s) for s in at}) if at is not None else len(steps)
    return n


def my_gpu_usage(runs_dir: Path) -> tuple[int, Counter]:
    """Containers of Modal apps this folder started (app ids scraped from our logs;
    every container here is one GPU: H200 x tp=1). Other agents share the app names."""
    ids = set()
    for log in [*runs_dir.glob("*/train.log"), *runs_dir.glob("smoke/*/train.log"),
                runs_dir / "eval_scheduler.log", runs_dir / "smoke" / "eval_scheduler.log"]:
        if log.exists():
            ids |= set(re.findall(r"ap-[A-Za-z0-9]{20,}", log.read_text(errors="replace")))
    out = subprocess.run([str(Path(PY).parent / "modal"), "container", "list", "--json"],
                         capture_output=True, text=True, check=False)
    try:  # rich colours the JSON even when piped
        containers = json.loads(re.sub(r"\x1b\[[0-9;]*m", "", out.stdout))
    except json.JSONDecodeError:
        return -1, Counter()
    # Deployed mode: every kaushik/ process shares the owner-named deployed apps,
    # so those containers are "ours" collectively (all sprint agents of this owner).
    deployed = {os.environ.get("COTCONTROL_MODAL_INFERENCE_APP"), os.environ.get("COTCONTROL_MODAL_TRAIN_APP")} - {None}
    mine = [c for c in containers if c.get("app_id") in ids or c.get("app_name") in deployed]
    return len(mine), Counter(c.get("app_name") for c in mine)


def status(argv):
    ap = argparse.ArgumentParser(prog="orange status")
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--gpus", action="store_true", help="also count our live Modal containers")
    args = ap.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    if args.gpus:
        n, by_app = my_gpu_usage(runs_dir)
        print(f"live Modal containers (= GPUs) of our apps: {n} {dict(by_app)}\n"
              "  (deployed apps are shared by every kaushik/ process; this folder's own share is "
              "the scheduler's 'eval GPUs' + 'training' below)\n")
    print(f"{'run':<36} {'train':<10} {'eval':>9}  last")
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists()):
        if (run_dir / "train_result.json").exists():
            tr = json.loads((run_dir / "train_result.json").read_text())
            train_state = f"done {tr['final_loss']:.3f}" if tr.get("final_loss") is not None else "done"
        elif (run_dir / "launcher.json").exists():
            pid = json.loads((run_dir / "launcher.json").read_text())["pid"]
            train_state = "running" if _pid_alive(pid) else "FAILED"
        else:
            train_state = "-"
        total = _total_jobs(run_dir)
        n_done = 0
        last = ""
        prog = run_dir / "eval_progress.jsonl"
        if prog.exists():
            lines = [ln for ln in prog.read_text().splitlines() if ln.strip()]
            n_done = len({(json.loads(ln)["block"], str(json.loads(ln)["step"])) for ln in lines})
            if lines:
                d = json.loads(lines[-1])
                last = (f"{d['block']}@{d['step']} compl={_fmt(d['compliance_rate'])} "
                        f"honest={_fmt(d['honest_rate'])} acc={_fmt(d['accuracy'])}")
        ev = f"{n_done}/{total}" if total is not None else "-"
        print(f"{run_dir.name:<36} {train_state:<10} {ev:>9}  {last}")
    state = runs_dir / "eval_scheduler.json"
    pid_file = runs_dir / "eval_scheduler.pid"
    alive = pid_file.exists() and _pid_alive(int(pid_file.read_text()))
    if state.exists():
        s = json.loads(state.read_text())
        print(f"\nscheduler {'alive' if alive else 'NOT running'} @ {s['time']}: cap {s['cap']}, "
              f"workspace containers {s.get('workspace_containers')}, "
              f"training {s['training_gpus']} + eval {s['eval_gpus_in_use']} GPUs, "
              f"{len(s['in_flight'])} jobs in flight, {s['pending']} pending, {s['done']} done, "
              f"{len(s['failed'])} failed")
    else:
        print(f"\nscheduler {'alive' if alive else 'not started'}")


def registry(argv):
    from train_run import write_registry_entry
    for tr in sorted(RUNS.glob("*/train_result.json")):
        out = write_registry_entry(tr.parent)
        print(f"{tr.parent.name}: {out}")


def smoke(argv):
    """2-step training + 8-question evals for one (arm, model), end to end."""
    ap = argparse.ArgumentParser(prog="orange smoke")
    ap.add_argument("--model", default="qwen36_27b")
    ap.add_argument("--arm", default="orange")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--skip-train", action="store_true", help="reuse runs/smoke/<run>/train_result.json")
    args = ap.parse_args(argv)
    import make_configs
    cfg = make_configs.make_config(args.arm, args.model, args.seed, smoke=True)
    path = make_configs.write_config(cfg, HERE / "configs" / "smoke")
    run_dir = ROOT / cfg["run_dir"]
    print(f"smoke config {path.relative_to(ROOT)}; run dir {run_dir.relative_to(ROOT)}")
    env = os.environ.copy()
    if not args.skip_train:
        subprocess.run([PY, str(HERE / "train_run.py"), str(path)], cwd=ROOT, env=env, check=True)
    subprocess.run([PY, str(HERE / "eval_scheduler.py"), "--runs-dir", str(run_dir.parent),
                    "--once", "--cap", str(args.cap), "--scan-interval", "5"], cwd=ROOT, env=env, check=True)
    import analyze
    analyze.main(["--runs-root", str(run_dir.parent), "--out", str(HERE / "results" / "smoke"), "--no-extra"])


def analyze(argv):
    import analyze as mod
    mod.main(argv)


STAGES = {
    "pool": pool, "data": data, "configs": configs, "smoke": smoke, "train": train,
    "eval": eval, "status": status, "registry": registry, "analyze": analyze,
}
