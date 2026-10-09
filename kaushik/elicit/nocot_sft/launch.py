"""Launch scripts/train_eval.py per config as a detached process (pattern:
kaushik/fragility/launch.py), report status, and set up the 1A reference eval.

    .venv/bin/python kaushik/run.py elicit/nocot_sft launch [--arms ...] [--seeds ...] [--configs ...] [--smoke]
    .venv/bin/python kaushik/run.py elicit/nocot_sft status [--smoke]
    .venv/bin/python kaushik/run.py elicit/nocot_sft ref_1a          # eval-only: 1A step 50 + checkpoint-0 on val

Each run dir (runs/<run>/) holds train_eval.py's artifacts plus manifest.json
(dataset + config sha256, git commit, pid, command) and launcher.log. Only pids
recorded in our own manifests are ever signalled (we never signal any here).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml
from common import CONFIGS, PY, REF_1A_RUN, ROOT, RUNS, atomic_write_json, git_commit, sha256_file


def _parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    return ap


def select(args) -> list[tuple[Path, dict]]:
    paths = ([Path(p) for p in args.configs] if args.configs
             else sorted((CONFIGS / ("smoke" if args.smoke else "")).glob("*.yaml")))
    out = []
    for p in paths:
        raw = yaml.safe_load(p.read_text())
        if args.arms and raw["arm"] not in args.arms:
            continue
        if args.seeds and raw["train"]["seed"] not in args.seeds:
            continue
        out.append((p, raw))
    return out


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


def launch_one(cfg_path: Path, raw: dict, eval_only: bool = False) -> dict:
    run_dir = ROOT / raw["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if eval_only or (run_dir / "train_result.json").exists():  # never retrain a finished adapter
        cmd = [PY, "scripts/train_eval.py", "--eval-only", str(run_dir.relative_to(ROOT))]
    else:
        cmd = [PY, "scripts/train_eval.py", str(cfg_path.resolve().relative_to(ROOT))]
    data = ROOT / raw["train"]["data_path"]
    manifest = {
        "run_name": raw["run_name"], "arm": raw["arm"], "base_model": raw["base_model"], "seed": raw["train"]["seed"],
        "config": str(cfg_path.resolve().relative_to(ROOT)), "config_sha256": sha256_file(cfg_path),
        "data_path": raw["train"]["data_path"], "data_sha256": sha256_file(data) if data.exists() else None,
        "git_commit": git_commit(), "command": " ".join(cmd),
        "modal_apps": {k: os.environ.get(k) for k in ("COTCONTROL_MODAL_INFERENCE_APP", "COTCONTROL_MODAL_TRAIN_APP")},
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
    with open(RUNS / "launches.jsonl", "a") as f:
        f.write(json.dumps({k: manifest[k] for k in ("run_name", "pid", "launched_at", "command")}) + "\n")
    print(f"launched {raw['run_name']} (pid {proc.pid})", flush=True)
    return manifest


def run_state(run_dir: Path, raw: dict) -> dict:
    mpath = run_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    alive = pid_alive(manifest.get("pid"))
    blocks = {}
    for name, blk in raw["eval"].items():
        if not blk.get("run", True):
            continue
        want = len(blk["eval_at_steps"]) if blk.get("eval_at_steps") else len(raw["train"]["save_at_steps"]) + 1
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
        lines = [ln.strip() for ln in (run_dir / "launcher.log").read_text(errors="replace").splitlines() if ln.strip()]
        last = lines[-1][:140] if lines else ""
    return {"run": raw["run_name"], "phase": phase, "pid": manifest.get("pid"), "alive": alive,
            "launched_at": manifest.get("launched_at"), "eval_blocks": blocks, "last_log": last}


def status(argv: list[str] | None = None) -> None:
    args = _parser("status of runs").parse_args(argv)
    states = [run_state(ROOT / raw["run_dir"], raw) for _, raw in select(args)]
    for s in states:
        done = sum(int(v.split("/")[0]) for v in s["eval_blocks"].values())
        want = sum(int(v.split("/")[1]) for v in s["eval_blocks"].values())
        print(f"{s['run']:<34} {s['phase']:<13} eval {done:>2}/{want:<2} | {s['last_log']}")
    counts = {}
    for s in states:
        counts[s["phase"]] = counts.get(s["phase"], 0) + 1
    print(f"-- {counts}", flush=True)
    atomic_write_json({"at": datetime.now().astimezone().isoformat(), "runs": states},
                      RUNS / ("smoke/status.json" if args.smoke else "status.json"))


def launch(argv: list[str] | None = None) -> None:
    ap = _parser("launch runs")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    RUNS.mkdir(parents=True, exist_ok=True)
    for cfg_path, raw in select(args):
        s = run_state(ROOT / raw["run_dir"], raw)
        if s["phase"] == "done" and not args.force:
            print(f"skip {raw['run_name']}: finished")
        elif s["alive"] and not args.force:
            print(f"skip {raw['run_name']}: running (pid {s['pid']})")
        else:
            launch_one(cfg_path, raw)
            time.sleep(2)


def ref_1a(argv: list[str] | None = None) -> None:
    """Eval-only run of the existing 1A adapter (fragility seed 0, step 50) and its
    checkpoint-0 (= untrained) under the new val protocol, in our own run dir."""
    ap = argparse.ArgumentParser(description=ref_1a.__doc__)
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 50])
    args = ap.parse_args(argv)
    from make_configs import eval_section
    src_cfg = yaml.safe_load((REF_1A_RUN / "config.yaml").read_text())
    src_tr = json.loads((REF_1A_RUN / "train_result.json").read_text())
    run_name = "ref1a-mcq_noreason-gptoss20b-seed0"
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ev = eval_section([args.steps[0], args.steps[-1]], smoke=False)
    for k, blk in ev.items():  # test numbers for 1A exist already (fragility); val only here
        blk["eval_at_steps"] = list(args.steps)
        if k.startswith("final"):
            blk["run"] = False
    cfg = {**src_cfg, "run_name": run_name, "arm": "ref_1a", "run_dir": str(run_dir.relative_to(ROOT)), "eval": ev,
           "source_run": str(REF_1A_RUN.relative_to(ROOT))}
    (run_dir / "config.yaml").write_text(yaml.dump(cfg, sort_keys=False, width=1000))
    tr = {**src_tr, "checkpoints": [c for c in src_tr["checkpoints"] if c["step"] in args.steps],
          "source_run": str(REF_1A_RUN.relative_to(ROOT))}
    (run_dir / "train_result.json").write_text(json.dumps(tr, indent=2))
    launch_one(run_dir / "config.yaml", cfg, eval_only=True)


def test_step(argv: list[str] | None = None) -> None:
    """Eval-only TEST blocks for a non-final checkpoint of an arm (3 seeds), in
    runs/<run>-step<N>/ (config + train_result copied from the source run; the
    val blocks are switched off). Used when selection picks the mid checkpoint."""
    ap = argparse.ArgumentParser(description=test_step.__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--peak", type=int, default=2)
    args = ap.parse_args(argv)
    for seed in args.seeds:
        src = RUNS / f"nocot-{args.arm}-gptoss20b-seed{seed}"
        src_cfg = yaml.safe_load((src / "config.yaml").read_text())
        src_tr = json.loads((src / "train_result.json").read_text())
        run_name = f"{src_cfg['run_name']}-step{args.step}"
        run_dir = RUNS / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        ev = {}
        for k, blk in src_cfg["eval"].items():
            blk = dict(blk)
            blk["run"] = k.startswith("final")
            blk["eval_at_steps"] = [args.step]
            blk["chunk_parallel"] = blk["parallel_checkpoints"] = args.peak
            ev[k] = blk
        cfg = {**src_cfg, "run_name": run_name, "run_dir": str(run_dir.relative_to(ROOT)), "eval": ev,
               "source_run": str(src.relative_to(ROOT))}
        (run_dir / "config.yaml").write_text(yaml.dump(cfg, sort_keys=False, width=1000))
        tr = {**src_tr, "checkpoints": [c for c in src_tr["checkpoints"] if c["step"] == args.step],
              "source_run": str(src.relative_to(ROOT))}
        (run_dir / "train_result.json").write_text(json.dumps(tr, indent=2))
        launch_one(run_dir / "config.yaml", cfg, eval_only=True)
        time.sleep(2)


if __name__ == "__main__":
    status()
