"""Stages for `kaushik/run.py elicit/lora_arith <stage>` (Workstream C3: LoRA arithmetic — GEPA LoRA minus brevity).

    deploy       deploy the distill (copy of B2's), arith (CPU combiner) and random-lora apps under ELICIT_SUB
    train        --run NAME --teacher brief|verbose|gepa [--seed k ...]: spawn one B2-style distillation job
    status       [--run NAME]: job state + tail of the trainer's log.jsonl
    combine      --name N --terms PATH:COEFF,PATH:COEFF[,...]: build a servable combined adapter on the volume
    eval         evaluate one candidate on val/test under the shared protocol (evalrun.py)
    launch       background-launch several evals: launch <run> -- <eval args> [;; <run> -- ...]
    noise        --group G --adapters A,B,C: KL(base||lora) + KL-matched random adapters (noise.py)
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

from gl_common import (ARITH_APP, DATA, DISTILL_APP, HERE, INFERENCE_APP, PY, RANDOM_LORA_APP, ROOT, RUNS, SUB,
                       TEACHER_PROMPTS, TRAIN_APP, VOL_DIR, atomic_write_json, gepa_prompt, git_commit, load_json,
                       read_jsonl, set_apps)


def _stage(module: str, fn: str = "main"):
    def run(argv: list[str]) -> None:
        set_apps()
        getattr(importlib.import_module(module), fn)(argv)
    return run


def deploy(argv: list[str]) -> None:
    set_apps()
    env = {**os.environ, "COTCONTROL_MODAL_OWNER": "kaushik-reddy", "ELICIT_SUB": SUB}
    env.pop("COTCONTROL_MODAL_RANDOM_LORA_APP", None)
    modal = str(ROOT / ".venv/bin/modal")
    cmds = {
        "distill": [modal, "deploy", str(HERE / "distill_app.py"), "--name", DISTILL_APP],
        "arith": [modal, "deploy", str(HERE / "arith_app.py"), "--name", ARITH_APP],
        "random-lora": [modal, "deploy", "-m", "cotcontrol.training.random_lora", "--name", RANDOM_LORA_APP],
    }
    which = argv or list(cmds)
    procs = {k: subprocess.Popen(cmds[k], cwd=ROOT, env=env) for k in which}
    for k, p in procs.items():
        print(f"{k}: exit {p.wait()}")


def train(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--teacher", required=True, choices=[*TEACHER_PROMPTS, "gepa"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-first", type=float, default=2.0)
    ap.add_argument("--w-turn", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=0.0)
    ap.add_argument("--save-at", default="100,200,300,400")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    args = ap.parse_args(argv)
    set_apps()
    import modal

    rows, drift = read_jsonl(DATA / "prompts.jsonl"), read_jsonl(DATA / "drift.jsonl")
    prompt = gepa_prompt() if args.teacher == "gepa" else TEACHER_PROMPTS[args.teacher]
    cfg = dict(run_name=args.run, seed=args.seed, lora_rank=args.rank, lora_alpha=args.alpha, lr=args.lr,
               max_steps=args.steps, batch_size=args.bs, micro_bs=args.micro_bs, w_first=args.w_first,
               w_turn=args.w_turn, w_anchor=args.w_anchor,
               save_at=[int(s) for s in args.save_at.split(",") if s], eval_every=args.eval_every,
               target_modules=args.targets.split(","))
    fn = modal.Function.from_name(DISTILL_APP, "train_distill")
    call = fn.spawn(cfg, rows, drift, prompt)
    rec = {"run": args.run, "call_id": call.object_id, "teacher": args.teacher, "teacher_prompt": prompt,
           "config": cfg, "n_rows": len(rows), "n_drift": len(drift), "data_meta": load_json(DATA / "meta.json"),
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


def combine(argv: list[str]) -> None:
    """combine --name gepa_m1.0brief --terms /checkpoints/...:1,/checkpoints/...:-1 -> /checkpoints/lora_arith/combos/<name>"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--terms", required=True, help="comma-separated PATH:COEFF")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    set_apps()
    import modal

    terms = [(t.rsplit(":", 1)[0], float(t.rsplit(":", 1)[1])) for t in args.terms.split(",")]
    dst = f"{VOL_DIR}/combos/{args.name}"
    res = modal.Function.from_name(ARITH_APP, "combine").remote(terms, dst, args.overwrite)
    atomic_write_json(res, RUNS / "_combos" / f"{args.name}.json")
    print(json.dumps({k: v for k, v in res.items() if k != "terms"}, indent=1))
    print(f"-> {dst}  (max_rel_err {res['max_rel_err']:.2e})")


def launch(argv: list[str]) -> None:
    """launch <run> -- <eval args> [;; <run> -- <eval args> ...]: one detached eval process per run."""
    env = {**os.environ, "COTCONTROL_MODAL_INFERENCE_APP": INFERENCE_APP, "COTCONTROL_MODAL_TRAIN_APP": TRAIN_APP,
           "COTCONTROL_MODAL_OWNER": "kaushik-reddy", "PYTHONUNBUFFERED": "1", "ELICIT_SUB": SUB}
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
        cmd = [PY, str(ROOT / "kaushik" / "run.py"), "elicit/lora_arith", "eval", "--run", run] + [a for a in g[1:] if a != "--"]
        log = open(RUNS / run / "eval.log", "a")
        log.write(f"\n[{datetime.now().isoformat()}] {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        (RUNS / run / "pid").write_text(str(p.pid))
        print(f"launched {run} pid {p.pid} -> {RUNS / run / 'eval.log'}")


def spend(argv: list[str]) -> None:
    subprocess.run([PY, str(HERE.parents[0] / "spend.py"), SUB], cwd=ROOT, check=False)


STAGES = {
    "deploy": deploy,
    "train": train,
    "status": status,
    "combine": combine,
    "eval": _stage("evalrun"),
    "launch": launch,
    "noise": _stage("noise"),
    "analyze": _stage("analyze"),
    "spend": spend,
}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    STAGES[sys.argv[1]](sys.argv[2:])
