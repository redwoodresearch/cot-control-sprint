"""Stages for `kaushik/run.py elicit/gepa_lora <stage>` (Workstream B2: no-CoT context-distilled GEPA LoRA).

    build_data   train-split prompts x 9 modes -> data/prompts.jsonl (+ holdout flag, drift prompts)
    deploy       deploy the custom distillation app (distill_app.py)
    train        --run NAME [--lr --rank --alpha --steps --bs --seed --save-at ...]: spawn one training job
                 (detached; call id + config in runs/<run>/train.json)
    status       [--run NAME]: job state + tail of the trainer's log.jsonl
    probe        --lora PATH: first-reasoning-token check in the vLLM eval path (base vs student vs teacher)
    eval         evaluate one candidate on val/test under the shared protocol (evalrun.py)
    launch       background-launch several evals: launch <run> -- <eval args> [;; <run> -- ...]
    noise        --lora PATH...: KL(base||lora) + 3 KL-matched random adapters (noise.py)
    analyze      metrics table with bootstrap CIs -> results/
    spend        Modal spend for this workstream
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime

from gl_common import (DATA, DISTILL_APP, HERE, INFERENCE_APP, PY, ROOT, RUNS, SUB, TRAIN_APP, atomic_write_json,
                       gepa_prompt, gepa_prompt_sha256, git_commit, load_json, read_jsonl, set_apps)


def _stage(module: str, fn: str = "main"):
    def run(argv: list[str]) -> None:
        set_apps()
        getattr(importlib.import_module(module), fn)(argv)
    return run


def deploy(argv: list[str]) -> None:
    set_apps()
    cmd = [str(ROOT / ".venv/bin/modal"), "deploy", str(HERE / "distill_app.py"), "--name", DISTILL_APP]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "COTCONTROL_MODAL_OWNER": "kaushik-reddy"})


def train(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-first", type=float, default=2.0)
    ap.add_argument("--w-turn", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=0.0, help="KL(base||student) anchor at header positions")
    ap.add_argument("--save-at", default="50,100,200,300,400")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    args = ap.parse_args(argv)
    set_apps()
    import modal

    rows, drift = read_jsonl(DATA / "prompts.jsonl"), read_jsonl(DATA / "drift.jsonl")
    cfg = dict(run_name=args.run, seed=args.seed, lora_rank=args.rank, lora_alpha=args.alpha, lr=args.lr,
               max_steps=args.steps, batch_size=args.bs, micro_bs=args.micro_bs, w_first=args.w_first,
               w_turn=args.w_turn, w_anchor=args.w_anchor,
               save_at=[int(s) for s in args.save_at.split(",") if s], eval_every=args.eval_every,
               target_modules=args.targets.split(","))
    fn = modal.Function.from_name(DISTILL_APP, "train_distill")
    call = fn.spawn(cfg, rows, drift, gepa_prompt())
    rec = {"run": args.run, "call_id": call.object_id, "config": cfg, "n_rows": len(rows), "n_drift": len(drift),
           "gepa_prompt_sha256": gepa_prompt_sha256(), "data_meta": load_json(DATA / "meta.json"),
           "git_commit": git_commit(), "started": datetime.now().astimezone().isoformat()}
    atomic_write_json(rec, RUNS / args.run / "train.json")
    print(f"spawned {args.run}: call {call.object_id} -> {RUNS / args.run / 'train.json'}")


def status(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--tail", type=int, default=3000)
    args = ap.parse_args(argv)
    set_apps()
    import modal

    runs = [args.run] if args.run else sorted(p.parent.name for p in RUNS.glob("*/train.json"))
    read_log = modal.Function.from_name(DISTILL_APP, "read_log")
    for run in runs:
        rec = load_json(RUNS / run / "train.json")
        call = modal.FunctionCall.from_id(rec["call_id"])
        try:
            res = call.get(timeout=0)
            state = "done"
            if not (RUNS / run / "result.json").exists():
                atomic_write_json(res, RUNS / run / "result.json")
        except TimeoutError:
            state, res = "running", None
        except Exception as e:  # noqa: BLE001
            state, res = f"FAILED: {type(e).__name__}: {str(e)[:500]}", None
        print(f"== {run}: {state}")
        tail = read_log.remote(run, args.tail)
        (RUNS / run / "log.jsonl").write_text(tail) if tail else None
        print(tail[-args.tail:])


def launch(argv: list[str]) -> None:
    """launch <run> -- <eval args> [;; <run> -- <eval args> ...]: one detached eval process per run."""
    env = {**os.environ, "COTCONTROL_MODAL_INFERENCE_APP": INFERENCE_APP, "COTCONTROL_MODAL_TRAIN_APP": TRAIN_APP,
           "COTCONTROL_MODAL_OWNER": "kaushik-reddy", "PYTHONUNBUFFERED": "1"}
    groups, cur = [], []
    for a in argv:
        if a == ";;":
            groups.append(cur)
            cur = []
        else:
            cur.append(a)
    groups.append(cur)
    for g in groups:
        run = g[0]
        (RUNS / run).mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(ROOT / "kaushik" / "run.py"), "elicit/gepa_lora", "eval", "--run", run] + [a for a in g[1:] if a != "--"]
        log = open(RUNS / run / "eval.log", "a")
        log.write(f"\n[{datetime.now().isoformat()}] {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        (RUNS / run / "pid").write_text(str(p.pid))
        print(f"launched {run} pid {p.pid} -> {RUNS / run / 'eval.log'}")


def spend(argv: list[str]) -> None:
    subprocess.run([PY, str(HERE.parents[0] / "spend.py"), SUB], cwd=ROOT, check=False)


STAGES = {
    "build_data": _stage("build_data"),
    "deploy": deploy,
    "train": train,
    "status": status,
    "probe": _stage("probe"),
    "eval": _stage("evalrun"),
    "launch": launch,
    "noise": _stage("noise"),
    "analyze": _stage("analyze"),
    "spend": spend,
}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    STAGES[sys.argv[1]](sys.argv[2:])
