"""Evaluate every finished run's checkpoints under ONE Modal app and ONE
global GPU budget.

scripts/train_eval.py evaluates blocks one after another, each in its own
Modal app (an engine cold start per block; 27 blocks per run here), and has no
notion of a cross-run GPU cap. This scheduler scans a runs/ folder for runs
with a train_result.json, turns (eval block x checkpoint) into jobs, and
dispatches them in priority order whenever the budget allows: Qwen3.6-27B
before gpt-oss-20b; within a model the lowercase_thinking val blocks of every
run first (seeds and arms interleaved: orange-s0, nopersona-s0, orange-s1, ...),
then the other val modes, then the final_* test blocks — so the headline
(orange - orange_nopersona) comparison has all seeds early. The budget counts
live training processes (launcher.json pids, one GPU each) against --cap (the
sprint's per-agent GPU cap minus a margin), can be lowered after
--cap-after-hours (--cap-later; off by default) and overridden live through
runs/.gpu_cap; nothing is dispatched while the workspace as a whole has more
than --workspace-max active containers. A multi-mode block with `chunk` set is
one job weighing ceil(n_prompts / chunk) GPUs, fanned out as concurrent
engine calls. Generation and grading are cotcontrol's (eval_cotcontrolqa +
modal_vllm) through scripts/train_eval.py's helpers; nothing here grades.

    .venv/bin/python kaushik/orange/eval_scheduler.py [--runs-dir kaushik/orange/runs] [--cap 30] [--once]

Per run (same layout as train_eval.py, so analysis is shared):
    eval/<block>/checkpoint-<step>.json   full per-rollout eval JSON (one per job)
    eval_progress.jsonl                   one line per finished job (compliance, honest, ...)
    eval_summary.json, summary.png        written once every job of the run is done
Scheduler state: <runs-dir>/eval_scheduler.log, eval_scheduler.json, eval_scheduler.pid.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("COTCONTROL_MODAL_OWNER", "kaushik-reddy")
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from train_run import load_train_eval

te = load_train_eval()
from cotcontrol.eval.data import assign_tasks, load_dataset
from cotcontrol.eval.eval import eval_cotcontrolqa
from cotcontrol.eval.grading import is_honest
from cotcontrol.inference import modal_vllm
from cotcontrol.inference.modal_vllm import make_generate_fn
from cotcontrol.run_logging import tee_stdio

MODEL_PRIORITY = {"Qwen/Qwen3.6-27B": 0, "openai/gpt-oss-20b": 1}
ARM_PRIORITY = {"orange": 0, "orange_nopersona": 1}
COND_PRIORITY = {"persona": 0, "direct": 1, "stated": 2}
MAX_ATTEMPTS = 3


def block_group(name: str) -> int:
    """0 = lowercase_thinking val blocks, 1 = other val blocks, 2 = final_* test blocks."""
    if name.startswith("final"):
        return 2
    return 0 if name.endswith("lowercase_thinking") else 1


@dataclass(order=True)
class Job:
    priority: tuple
    run_dir: Path = field(compare=False)
    block: str = field(compare=False)
    blk: dict = field(compare=False)
    step: object = field(compare=False)
    lora_path: str | None = field(compare=False)
    weight: int = field(compare=False)
    tc: object = field(compare=False)

    @property
    def key(self) -> tuple:
        return (str(self.run_dir), self.block, str(self.step))

    @property
    def out_path(self) -> Path:
        stem = f"checkpoint-{self.step}" if self.lora_path else "base"
        return self.run_dir / "eval" / self.block / f"{stem}.json"


# ----------------------------------------------------------------------------- helpers

def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def training_gpus(runs_dir: Path) -> int:
    """GPUs held by live training processes launched by cli.py (launcher.json)."""
    n = 0
    for lj in runs_dir.glob("*/launcher.json"):
        if (lj.parent / "train_result.json").exists():
            continue
        try:
            info = json.loads(lj.read_text())
        except json.JSONDecodeError:
            continue
        if pid_alive(int(info["pid"])):
            n += int(info.get("gpus", 1))
    return n


class Budget:
    def __init__(self, runs_dir: Path, cap: int, cap_later: int | None, after_hours: float):
        self.runs_dir, self.cap, self.cap_later, self.after_hours = runs_dir, cap, cap_later, after_hours

    def cap_now(self) -> int:
        override = self.runs_dir / ".gpu_cap"
        if override.exists():
            try:
                return int(override.read_text().strip())
            except ValueError:
                pass
        marker = self.runs_dir / ".first_launch"
        if self.cap_later is not None and marker.exists():
            launched = datetime.fromisoformat(marker.read_text().strip())
            if (datetime.now() - launched).total_seconds() > self.after_hours * 3600:
                return self.cap_later
        return self.cap


_WORKSPACE: dict = {"t": 0.0, "n": None}


def workspace_containers(max_age: float = 60.0) -> int | None:
    """Active containers across the whole Modal workspace (all users), cached."""
    if time.time() - _WORKSPACE["t"] > max_age:
        out = subprocess.run([sys.executable, "-m", "modal", "container", "list", "--json"],
                             capture_output=True, text=True, check=False)
        try:
            _WORKSPACE["n"] = len(json.loads(re.sub(r"\x1b\[[0-9;]*m", "", out.stdout)))
        except json.JSONDecodeError:
            _WORKSPACE["n"] = None
        _WORKSPACE["t"] = time.time()
    return _WORKSPACE["n"]


_N_PROMPTS: dict = {}


def n_prompts(blk: dict) -> int:
    key = (blk["dataset"], blk["mode"], blk["split"], blk["max_samples"], blk["subsample_seed"],
           tuple(blk["allowed_modes"] or ()), int(blk["seed"]))
    if key not in _N_PROMPTS:
        samples = load_dataset(blk["dataset"], blk["mode"], None, blk["max_samples"],
                               blk["subsample_seed"], blk["split"])
        _N_PROMPTS[key] = len(assign_tasks(samples, blk["mode"], int(blk["seed"]),
                                           allowed_modes=blk["allowed_modes"]))
    return _N_PROMPTS[key]


def chunked_generate_fn(base_model: str, gen_cfg, lora_path: str | None, chunk: int):
    """generate_fn splitting prompts into `chunk`-sized concurrent engine calls
    (the job's weight already reserved that many GPUs)."""
    inner = make_generate_fn(base_model, gen_cfg, lora_path)
    if not chunk:
        return inner

    async def fn(messages_list):
        parts = [messages_list[i:i + chunk] for i in range(0, len(messages_list), chunk)]
        outs = await asyncio.gather(*[inner(p) for p in parts])
        return [r for o in outs for r in o]

    return fn


def honest_rate(result: dict) -> float | None:
    samples = [s for rec in result["results"] for s in rec["samples"] if not s["error"]]
    return sum(is_honest(s) for s in samples) / len(samples) if samples else None


def progress_line(job: Job, result: dict, seconds: float | None) -> dict:
    s = result["summary"]
    return {"block": job.block, "step": job.step, "lora_path": job.lora_path,
            "accuracy": s["accuracy"], "compliance_rate": s["compliance_rate"],
            "honest_rate": honest_rate(result), "meta_discussion_rate": s["meta_discussion_rate"],
            "n_errors": s["n_errors"], "n_rollouts": s["n_rollouts"], "per_mode": s["per_mode"],
            "seconds": seconds, "time": datetime.now().isoformat()}


def read_progress(run_dir: Path) -> dict[tuple, dict]:
    out = {}
    path = run_dir / "eval_progress.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out[(d["block"], str(d["step"]))] = d  # last wins
    return out


def append_progress(run_dir: Path, line: dict) -> None:
    with (run_dir / "eval_progress.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")


def write_eval_summary(run_dir: Path, jobs: list[Job]) -> None:
    """train_eval.py's eval_summary.json schema (+ honest_rate) from the progress lines."""
    progress = read_progress(run_dir)
    tr = json.loads((run_dir / "train_result.json").read_text())
    evals, blocks = {}, []
    for job in jobs:
        if job.block not in evals:
            evals[job.block] = {"config": job.blk, "checkpoints": []}
            blocks.append(job.block)
        line = progress.get((job.block, str(job.step)))
        if line:
            evals[job.block]["checkpoints"].append({k: v for k, v in line.items() if k != "block"})
    for block in evals.values():
        block["checkpoints"].sort(key=lambda c: (c["step"] == "base", c["step"]))
    summary = {"run_name": tr["run_name"], "base_model": tr.get("base_model") or jobs[0].tc.base_model,
               "evals": evals, "checkpoints": evals[blocks[0]]["checkpoints"] if blocks else []}
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    try:
        te.plot_summary(summary, run_dir)
    except Exception as e:  # noqa: BLE001 - plotting is cosmetic
        print(f"[summary] plot failed for {run_dir.name}: {e}")
    print(f"[summary] wrote {run_dir / 'eval_summary.json'}")


# ----------------------------------------------------------------------------- discovery

_RUN_CACHE: dict = {}


def load_run(run_dir: Path):
    """(train_result, TrainConfig, [(block, blk)], meta) for a finished run, cached by mtime."""
    tr_path = run_dir / "train_result.json"
    mtime = tr_path.stat().st_mtime
    if _RUN_CACHE.get(run_dir, (None,))[0] != mtime:
        tr = json.loads(tr_path.read_text())
        tc, ec, raw = te.load_config(run_dir / "config.yaml")
        tc.run_name = tr["run_name"]
        _RUN_CACHE[run_dir] = (mtime, tr, tc, te.eval_blocks(ec), raw.get("meta", {}))
    return _RUN_CACHE[run_dir][1:]


def discover_jobs(runs_dir: Path, only_blocks: list[str] | None) -> list[Job]:
    jobs = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not ((run_dir / "train_result.json").exists() and (run_dir / "config.yaml").exists()):
            continue
        try:
            tr, tc, blocks, meta = load_run(run_dir)
        except (OSError, ValueError, KeyError) as e:
            print(f"[discover] skipping {run_dir.name}: {e}")
            continue
        mp = MODEL_PRIORITY.get(tc.base_model, 5)
        arm_rank = ARM_PRIORITY.get(meta.get("arm"), 5)
        for name, blk in blocks:
            if only_blocks and not any(fnmatch.fnmatch(name, pat) for pat in only_blocks):
                continue
            targets = [(c["step"], c["path"]) for c in tr["checkpoints"]]
            if blk["eval_at_steps"] is not None:
                wanted = {int(s) for s in blk["eval_at_steps"]}
                targets = [(s, p) for s, p in targets if int(s) in wanted]
            if blk["eval_base_model"]:
                targets = [("base", None)] + targets
            chunk = int(blk.get("chunk") or 0)
            weight = (max(1, math.ceil(n_prompts(blk) / chunk)) if chunk else 1) * int(blk["tensor_parallel_size"])
            for step, path in targets:
                prio = (mp, block_group(name), int(tc.seed), arm_rank,
                        int(step) if str(step).isdigit() else -1,
                        COND_PRIORITY.get(name.replace("final_", "").split("_")[0], 5), name)
                jobs.append(Job(prio, run_dir, name, blk, step, path, weight, tc))
    return jobs


# ----------------------------------------------------------------------------- scheduler

async def run_job(job: Job) -> None:
    blk, tc = job.blk, job.tc
    gen_cfg = te.build_generate_config(tc, blk)
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = await eval_cotcontrolqa(
        model=tc.base_model,
        system_prompt=te.resolve_system_prompt(blk["system_prompt"]),
        generate_fn=chunked_generate_fn(tc.base_model, gen_cfg, job.lora_path, int(blk.get("chunk") or 0)),
        save_dir=job.out_path.parent,
        save_name=job.out_path.stem,
        dataset=blk["dataset"],
        mode=blk["mode"],
        allowed_modes=blk["allowed_modes"],
        seed=int(blk["seed"]),
        state_requirement=bool(blk["state_requirement"]),
        max_samples=blk["max_samples"],
        subsample_seed=blk["subsample_seed"],
        split=blk["split"],
        grade_meta_discussion=blk["grade_meta_discussion"],
        meta_discussion_scope=blk.get("meta_discussion_scope", "compliant"),
        judge_model=blk["judge_model"],
        judge_concurrency=int(blk["judge_concurrency"]),
        backend_info={"base_model": tc.base_model, "lora_path": job.lora_path,
                      "generate_config": asdict(gen_cfg)},
    )
    line = progress_line(job, result, round(time.time() - t0))
    append_progress(job.run_dir, line)
    print(f"[done] {job.run_dir.name} {job.block} step {job.step}: "
          f"compliance={line['compliance_rate']} honest={line['honest_rate']} "
          f"accuracy={line['accuracy']} errors={line['n_errors']} ({line['seconds']}s)", flush=True)


async def scheduler(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    budget = Budget(runs_dir, args.cap, args.cap_later, args.cap_after_hours)
    state_path = runs_dir / "eval_scheduler.json"
    pending: list[Job] = []
    in_flight: dict[tuple, tuple[Job, asyncio.Task]] = {}
    done: set[tuple] = set()
    failed: set[tuple] = set()
    attempts: Counter = Counter()
    summarized: set[Path] = set()
    in_use, last_scan, idle_since = 0, 0.0, None
    jobs: list[Job] = []

    def summarize_complete_runs() -> None:
        by_run: dict[Path, list[Job]] = {}
        for job in jobs:
            by_run.setdefault(job.run_dir, []).append(job)
        for run_dir, run_jobs in by_run.items():
            if run_dir not in summarized and all(j.key in done for j in run_jobs):
                write_eval_summary(run_dir, run_jobs)
                summarized.add(run_dir)

    async with modal_vllm.app.run():
        # Logged so cli.status --gpus can attribute this app's containers to us. In
        # deployed mode (COTCONTROL_MODAL_INFERENCE_APP) app.run() is a no-op and the
        # engines live in the named deployed app instead.
        try:
            app_ref = modal_vllm.DEPLOYED_APP or modal_vllm.app.app_id
        except Exception:  # noqa: BLE001 - informational only
            app_ref = "unknown"
        print(f"[scheduler] modal app {app_ref}", flush=True)
        while True:
            now = time.time()
            if now - last_scan >= args.scan_interval:
                last_scan = now
                jobs = discover_jobs(runs_dir, args.blocks)
                queued = {j.key for j in pending}
                for job in jobs:
                    if job.key in in_flight or job.key in done or job.key in failed or job.key in queued:
                        continue
                    if job.out_path.exists():
                        # Finished by an earlier scheduler; make sure it has a progress line.
                        if (job.block, str(job.step)) not in read_progress(job.run_dir):
                            try:
                                append_progress(job.run_dir, progress_line(
                                    job, json.loads(job.out_path.read_text()), None))
                            except (json.JSONDecodeError, KeyError):
                                job.out_path.unlink()  # truncated by a crash: redo
                                pending.append(job)
                                continue
                        done.add(job.key)
                        continue
                    pending.append(job)
                pending.sort()
                summarize_complete_runs()  # per-run summaries once every job of a run is done

            for key, (job, task) in list(in_flight.items()):
                if not task.done():
                    continue
                in_use -= job.weight
                del in_flight[key]
                exc = task.exception()
                if exc is None:
                    done.add(key)
                    continue
                attempts[key] += 1
                print(f"[error] {job.run_dir.name} {job.block} step {job.step} "
                      f"(attempt {attempts[key]}/{MAX_ATTEMPTS}): {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exception(exc)
                if attempts[key] < MAX_ATTEMPTS:
                    pending.append(job)
                    pending.sort()
                else:
                    failed.add(key)
                last_scan = 0.0  # re-check summaries promptly

            train_gpus = training_gpus(runs_dir)
            limit = budget.cap_now() - train_gpus
            ws = workspace_containers()
            if pending and ws is not None and ws > args.workspace_max:
                limit = 0  # guard: the workspace is saturated; launch nothing new
                if int(now) % 60 < args.tick:
                    print(f"[hold] workspace has {ws} active containers > {args.workspace_max}", flush=True)
            while pending and in_use + pending[0].weight <= limit:
                job = pending.pop(0)
                in_use += job.weight
                in_flight[job.key] = (job, asyncio.create_task(run_job(job)))
                print(f"[start] {job.run_dir.name} {job.block} step {job.step} "
                      f"(weight {job.weight}; in use {in_use}/{limit}, training {train_gpus})", flush=True)

            state_path.write_text(json.dumps({
                "time": datetime.now().isoformat(), "cap": budget.cap_now(),
                "workspace_containers": ws,
                "training_gpus": train_gpus, "eval_gpus_in_use": in_use,
                "in_flight": [{"run": j.run_dir.name, "block": j.block, "step": j.step, "weight": j.weight}
                              for j, _ in in_flight.values()],
                "pending": len(pending), "done": len(done), "failed": sorted(failed),
            }, indent=1))

            idle = not pending and not in_flight and train_gpus == 0
            if idle and args.once:
                summarize_complete_runs()
                break
            if idle:
                idle_since = idle_since or now
                if now - idle_since > args.idle_exit_minutes * 60 or (runs_dir / ".stop_scheduler").exists():
                    summarize_complete_runs()
                    break
            else:
                idle_since = None
            await asyncio.sleep(args.tick)
    print(f"[scheduler] exit: {len(done)} done, {len(failed)} failed")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--cap", type=int, default=65, help="concurrent GPUs (eval + training); sprint cap 70")
    ap.add_argument("--cap-later", type=int, default=None, help="optional lower cap after --cap-after-hours")
    ap.add_argument("--cap-after-hours", type=float, default=6.0)
    ap.add_argument("--workspace-max", type=int, default=800,
                    help="dispatch nothing while the whole workspace has more active containers")
    ap.add_argument("--blocks", nargs="+", default=None, help="glob patterns of block names")
    ap.add_argument("--once", action="store_true", help="exit when nothing is pending or training")
    ap.add_argument("--idle-exit-minutes", type=float, default=45.0)
    ap.add_argument("--scan-interval", type=float, default=60.0)
    ap.add_argument("--tick", type=float, default=5.0)
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "eval_scheduler.pid").write_text(str(os.getpid()))
    with tee_stdio(runs_dir / "eval_scheduler.log", mode="a"):
        print(f"[scheduler] start {datetime.now().isoformat()} pid {os.getpid()} args {vars(args)}")
        te.with_slot_retry(lambda: asyncio.run(scheduler(args)), "eval-scheduler")


if __name__ == "__main__":
    main()
