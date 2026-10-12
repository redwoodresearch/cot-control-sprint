"""The no-CoT distillation training job (local side of common/distill_app.py).

    run.py elicit/gepa_lora distill run_name=gl_r16_lr1e-4_s0 distill.seed=0 [--dry-run]   # spawn + wait
    run.py elicit/gepa_lora distill --status [--run NAME] [--tail 3000]                    # call state + log tail
    run.py sweep elicit/lora_arith/sweeps/train.yaml --dry-run                             # one job per run

The `distill:` section of the resolved config is the trainer's whole configuration (data, teacher prompt +
pinned sha256, LoRA, optimizer, loss weights, packing budgets); `run_name` and `base_model` are added. The
Modal call is spawned (call id + config in runs/<run>/train.json), the scheduler job then polls it and writes
runs/<run>/result.json (the trainer's return value: checkpoints, eval history, full config) and log.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import time
from datetime import datetime
from pathlib import Path

from kaushik.core.config import run_dir_of
from kaushik.core.io import (
    ROOT,
    atomic_write_json,
    git_commit,
    read_json,
    read_jsonl,
    rel,
)
from kaushik.core.scheduler import Job, run_scheduler
from kaushik.elicit.common import (
    app_name,
    check_modal,
    resolve_prompt,
    runs_dir,
    setup_modal,
    sha256_text,
)

META_KEYS = ("data", "teacher", "gpu", "timeout_s")   # distill keys that are not trainer hyperparameters


def trainer_config(cfg: dict) -> dict:
    """The dict handed to train_distill: distill hyperparameters + run_name, base_model, teacher template kwargs."""
    d = cfg["distill"]
    tc = {k: copy.deepcopy(v) for k, v in d.items() if k not in META_KEYS}
    tc["run_name"], tc["base_model"] = cfg["run_name"], cfg["base_model"]
    tc["teacher_template_kwargs"] = (d.get("teacher") or {}).get("chat_template_kwargs")
    return tc


def teacher_prompt(cfg: dict) -> str:
    t = cfg["distill"]["teacher"]
    return resolve_prompt(t["system_prompt"], t.get("sha256"))


def spawn(cfg: dict) -> dict:
    import modal

    d = cfg["distill"]
    run_dir = run_dir_of(cfg)
    rows, drift = read_jsonl(ROOT / d["data"]["prompts"]), read_jsonl(ROOT / d["data"]["drift"])
    prompt = teacher_prompt(cfg)
    tc = trainer_config(cfg)
    fn = modal.Function.from_name(app_name(cfg, "distill"), "train_distill")
    call = fn.spawn(tc, rows, drift, prompt)
    rec = {"run": cfg["run_name"], "call_id": call.object_id, "config": tc,
           "teacher": {"system_prompt": d["teacher"]["system_prompt"], "sha256": sha256_text(prompt),
                       "chat_template_kwargs": d["teacher"].get("chat_template_kwargs")},
           "n_rows": len(rows), "n_drift": len(drift), "data_meta": read_json(ROOT / d["data"]["meta"]),
           "git_commit": git_commit(), "started": datetime.now().astimezone().isoformat()}
    atomic_write_json(rec, run_dir / "train.json", indent=1)
    print(f"spawned {cfg['run_name']}: call {call.object_id} -> {rel(run_dir / 'train.json')}", flush=True)
    return rec


def poll(cfg: dict, run_dir: Path, tail: int = 3000) -> tuple[str, dict | None]:
    """('done' | 'running' | 'FAILED: ...', result); writes result.json and log.jsonl."""
    import modal

    rec = read_json(run_dir / "train.json")
    call = modal.FunctionCall.from_id(rec["call_id"])
    try:
        res = call.get(timeout=0)
        state = "done"
        if not (run_dir / "result.json").exists():
            atomic_write_json(res, run_dir / "result.json", indent=1)
    except TimeoutError:
        state, res = "running", None
    except Exception as e:  # noqa: BLE001 - the call failed remotely; report, do not crash the poller
        state, res = f"FAILED: {type(e).__name__}: {str(e)[:500]}", None
    try:
        log = modal.Function.from_name(app_name(cfg, "distill"), "read_log").remote(
            cfg["run_name"], tail, cfg["distill"]["out_root"])
        if log:
            (run_dir / "log.jsonl").write_text(log)
    except Exception as e:  # noqa: BLE001 - log tail is best effort
        print(f"[distill] read_log failed: {type(e).__name__}: {e}", flush=True)
    return state, res


async def run(cfg: dict, poll_s: float = 60.0) -> dict:
    run_dir = run_dir_of(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "train.json").exists():
        spawn(cfg)
    while True:
        state, res = poll(cfg, run_dir)
        if state == "done":
            print(f"[distill] {cfg['run_name']} done: {res.get('runtime_s')} s, {len(res.get('checkpoints', []))} checkpoints",
                  flush=True)
            return res
        if state.startswith("FAILED"):
            raise RuntimeError(f"{cfg['run_name']} {state}")
        await asyncio.sleep(poll_s)


def finished(run_dir: Path) -> bool:
    """result.json fetched, or the trainer's log (committed with the last checkpoint) ends with its done event;
    a run with only train.json is polled (never respawned) by the job."""
    if (run_dir / "result.json").exists():
        return True
    log = run_dir / "log.jsonl"
    return log.exists() and '"event": "done"' in log.read_text()[-2000:]


def jobs(cfg: dict) -> list[Job]:
    setup_modal(cfg)
    run_dir = run_dir_of(cfg)
    from kaushik.core.jobs import gpu_count

    return [Job(name=f"distill:{cfg['run_name']}", weight=gpu_count(cfg["distill"].get("gpu")),
                run=lambda: run(cfg), done=lambda: finished(run_dir),
                priority=(0, cfg["run_name"]), group=str(run_dir))]


def status(cfg: dict, only: str | None, tail: int) -> None:
    setup_modal(cfg)
    root = runs_dir(cfg)
    runs = [only] if only else sorted(p.parent.name for p in root.glob("*/train.json"))
    for name in runs:
        run_dir = root / name
        if (run_dir / "result.json").exists():
            res = read_json(run_dir / "result.json")
            print(f"== {name}: done ({res.get('runtime_s')} s, steps {res.get('steps')})")
            continue
        state, _ = poll({**cfg, "run_name": name}, run_dir, tail)
        print(f"== {name}: {state}")
        if (run_dir / "log.jsonl").exists():
            print((run_dir / "log.jsonl").read_text()[-tail:])


def main(cfg: dict, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="distill", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--run", default=None, help="with --status: one run")
    ap.add_argument("--tail", type=int, default=3000)
    a = ap.parse_args(argv)
    if a.status:
        status(cfg, a.run, a.tail)
        return
    if a.dry_run:
        tc = trainer_config(cfg)
        print(f"distill {cfg['run_name']} on {app_name(cfg, 'distill')}: teacher sha256 {sha256_text(teacher_prompt(cfg))[:12]}..., "
              f"{len(read_jsonl(ROOT / cfg['distill']['data']['prompts']))} rows; config {tc}")
    else:
        from kaushik.core.jobs import prepare_run

        check_modal(cfg)
        prepare_run(cfg)
    t0 = time.time()
    run_scheduler(jobs(cfg), runs_dir(cfg), cap=a.cap, name="distill_scheduler", dry_run=a.dry_run)
    if not a.dry_run:
        print(f"[distill] {time.time() - t0:.0f} s")
