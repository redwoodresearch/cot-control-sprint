"""kaushik/orange_expand/sft — orange expansion, SFT half (PLAN.md stage 1 + analysis).

    .venv/bin/python kaushik/run.py orange_expand/sft <stage> [args]

  data      2A pool -> held-out 200 + SFT rows per training prompt -> data/   (build_sft_data.py)
  configs   12 run configs -> configs/                                        (make_configs.py)
  smoke     2-step / 8-question end-to-end run (default qwen8b eugene) -> runs/smoke/
  train     launch training for every config in the background (train_run.py, one process each)
  eval      start kaushik/orange's eval scheduler on runs/ (--fg to block; args pass through)
  status    training / eval progress + scheduler state
  analyze   summary tables -> results/                                        (analyze.py)

Modal apps are forced to the deployed sub-orange-expand pair by common.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

import common
from common import HERE, MODEL_ORDER, ROOT, RUNS_DIR

ocli = common.orange_mod("cli")   # kaushik/orange/cli.py: _spawn, _memory_ok, _pid_alive, eval
PY = sys.executable


def data(argv):
    common.load_py(HERE / "build_sft_data.py", "oex_build_sft_data").main(argv)


def configs(argv):
    common.load_py(HERE / "make_configs.py", "oex_make_configs").main(argv)


def _launch(path: str, raw: dict) -> int:
    run_dir = ROOT / raw["run_dir"]
    pid = ocli._spawn([PY, str(HERE / "train_run.py"), path], run_dir / "launcher.log")
    gpus = int(str(raw["gpu"]).rsplit(":", 1)[1]) if ":" in str(raw["gpu"]) else 1
    (run_dir / "launcher.json").write_text(json.dumps(
        {"pid": pid, "config": path, "gpus": gpus, "started": datetime.now().isoformat()}))
    return pid


def train(argv):
    """One detached train_run.py per config, cheapest model first."""
    ap = argparse.ArgumentParser(prog="sft train")
    ap.add_argument("--configs", default=str(HERE / "configs" / "*.yaml"), help="glob")
    ap.add_argument("--force", action="store_true", help="relaunch finished/running runs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cfgs = [(p, yaml.safe_load(Path(p).read_text())) for p in sorted(glob.glob(args.configs))]
    order = {k: i for i, k in enumerate(MODEL_ORDER)}
    cfgs.sort(key=lambda pc: (order.get(pc[1]["meta"]["model_key"], 9), pc[0]))
    launched = 0
    for path, raw in cfgs:
        run_dir = ROOT / raw["run_dir"]
        launcher = run_dir / "launcher.json"
        if not args.force:
            if (run_dir / "train_result.json").exists():
                print(f"done     {raw['run_name']}")
                continue
            if launcher.exists() and ocli._pid_alive(json.loads(launcher.read_text())["pid"]):
                print(f"running  {raw['run_name']}")
                continue
        if args.dry_run:
            print(f"would launch {raw['run_name']} ({raw['base_model']}, gpu {raw['gpu']})")
            continue
        if not ocli._memory_ok():
            break
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"launched {raw['run_name']} pid {_launch(path, raw)}")
        launched += 1
    print(f"{launched} launched")


def eval(argv):
    """kaushik/orange's eval_scheduler.py on this folder's runs/ (deployed-app mode)."""
    if "--runs-dir" not in argv:
        argv = ["--runs-dir", str(RUNS_DIR), *argv]
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ocli.eval(argv)


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
        n += (len(steps & {int(s) for s in at}) if at is not None else len(steps)) + int(bool(blk.get("eval_base_model")))
    return n


def status(argv):
    ap = argparse.ArgumentParser(prog="sft status")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = ap.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    print(f"{'run':<40} {'train':<12} {'eval':>7}  last")
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists()):
        if (run_dir / "train_result.json").exists():
            tr = json.loads((run_dir / "train_result.json").read_text())
            train_state = f"done {tr['final_loss']:.4f}" if tr.get("final_loss") is not None else "done"
        elif (run_dir / "launcher.json").exists():
            pid = json.loads((run_dir / "launcher.json").read_text())["pid"]
            train_state = "running" if ocli._pid_alive(pid) else "FAILED"
        else:
            train_state = "-"
        total, n_done, last = _total_jobs(run_dir), 0, ""
        prog = run_dir / "eval_progress.jsonl"
        if prog.exists():
            lines = [json.loads(ln) for ln in prog.read_text().splitlines() if ln.strip()]
            n_done = len({(d["block"], str(d["step"])) for d in lines})
            if lines:
                d = lines[-1]
                last = (f"{d['block']}@{d['step']} honest={ocli._fmt(d['honest_rate'])} "
                        f"acc={ocli._fmt(d['accuracy'])}")
        print(f"{run_dir.name:<40} {train_state:<12} {f'{n_done}/{total}' if total is not None else '-':>7}  {last}")
    state, pid_file = runs_dir / "eval_scheduler.json", runs_dir / "eval_scheduler.pid"
    alive = pid_file.exists() and ocli._pid_alive(int(pid_file.read_text()))
    if state.exists():
        s = json.loads(state.read_text())
        print(f"\nscheduler {'alive' if alive else 'NOT running'} @ {s['time']}: cap {s['cap']}, workspace "
              f"{s.get('workspace_containers')}, training {s['training_gpus']} + eval {s['eval_gpus_in_use']} GPUs, "
              f"{len(s['in_flight'])} in flight, {s['pending']} pending, {s['done']} done, {len(s['failed'])} failed")
    else:
        print(f"\nscheduler {'alive' if alive else 'not started'}")


def smoke(argv):
    ap = argparse.ArgumentParser(prog="sft smoke")
    ap.add_argument("--model", default="qwen8b")
    ap.add_argument("--variant", default="eugene")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args(argv)
    mc = common.load_py(HERE / "make_configs.py", "oex_make_configs")
    cfg = mc.sft_config(args.variant, args.model, smoke=True)
    path = mc.write_config(cfg, HERE / "configs" / "smoke")
    run_dir = ROOT / cfg["run_dir"]
    print(f"smoke config {path.relative_to(ROOT)}; run dir {run_dir.relative_to(ROOT)}")
    env = os.environ.copy()
    if not args.skip_train:
        subprocess.run([PY, str(HERE / "train_run.py"), str(path)], cwd=ROOT, env=env, check=True)
    subprocess.run([PY, str(common.ORANGE / "eval_scheduler.py"), "--runs-dir", str(run_dir.parent),
                    "--once", "--cap", "4", "--scan-interval", "5"], cwd=ROOT, env=env, check=True)
    for f in sorted((run_dir / "eval").glob("*/*.json")):
        d = json.loads(f.read_text())
        print(f"{f.parent.name}: {d['summary']}")


def analyze(argv):
    common.load_py(HERE / "analyze.py", "oex_analyze").main(argv)


STAGES = {"data": data, "configs": configs, "smoke": smoke, "train": train, "eval": eval,
          "status": status, "analyze": analyze}
