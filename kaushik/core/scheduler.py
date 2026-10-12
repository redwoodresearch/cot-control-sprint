"""One GPU scheduler for every kaushik/ experiment (replaces the per-folder copies).

A Job is a unit of GPU work with a weight (GPUs it holds while running), a
`done()` predicate (its outputs exist), an optional `ready()` gate (inputs exist,
e.g. train_result.json before an eval) and either a coroutine factory or a
subprocess command. The scheduler dispatches pending jobs in priority order while
`in use + weight <= cap`, where

* cap = `<runs_dir>/.gpu_cap` (live override) else the cap given (default
  defaults.yaml scheduler.max_gpus), minus GPUs held outside the scheduler
  (`external_gpus()`, e.g. training processes launched by hand);
* nothing starts while the Modal workspace has more than
  `workspace_max_containers` active containers, while `<runs_dir>/.stop_launches`
  exists, or (subprocess jobs only) while laptop free memory is below
  `min_free_memory_pct`;
* a failed job is retried up to `max_attempts` times; a subprocess job whose
  pid file names a live process (an earlier scheduler) is adopted, not restarted.

State: `<runs_dir>/<name>.json` (rewritten every tick), `<name>.pid`, and the
caller's log (`run_scheduler` tees stdout to `<name>.log`). `--dry-run` prints
the plan and dispatches nothing.

    jobs = [Job("eval:run/block", weight=2, run=coro_factory, done=lambda: out.exists()), ...]
    run_scheduler(jobs, runs_dir, cap=40)                    # blocks until every job is done or failed
"""

# ruff: noqa: DTZ005 — naive local timestamps, like the rest of the repo
from __future__ import annotations

import asyncio
import os
import subprocess
import time
import traceback
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kaushik.core.config import defaults
from kaushik.core.io import ROOT, atomic_write_json, free_memory_pct, pid_alive
from kaushik.core.modal_apps import workspace_containers


@dataclass
class Job:
    name: str
    weight: int
    run: Callable[[], Awaitable] | list[str] | None = None   # coroutine factory | subprocess argv
    done: Callable[[], bool] = lambda: False
    ready: Callable[[], bool] = lambda: True
    priority: tuple = ()
    log: Path | None = None            # subprocess stdout/stderr (appended)
    pid_file: Path | None = None       # subprocess pid; a live one from an earlier scheduler is adopted
    group: str | None = None           # e.g. the run dir; `on_group_done` fires when a group completes
    expand: Callable[[], list[Job]] | None = None   # placeholder: replaced by expand() once ready()
    cwd: Path = ROOT

    @property
    def is_process(self) -> bool:
        return isinstance(self.run, list)

    def __lt__(self, other: Job) -> bool:
        return (self.priority, self.name) < (other.priority, other.name)


@dataclass
class _Running:
    job: Job
    started: float
    task: asyncio.Task | None = None
    proc: subprocess.Popen | None = None
    pid: int | None = None


@dataclass
class Scheduler:
    jobs: list[Job]
    runs_dir: Path
    cap: int | None = None
    name: str = "scheduler"
    external_gpus: Callable[[], int] | None = None
    on_group_done: Callable[[str, list[Job]], None] | None = None
    dry_run: bool = False
    settings: dict = field(default_factory=lambda: dict(defaults()["scheduler"]))

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir)
        self.cap = self.settings["max_gpus"] if self.cap is None else int(self.cap)
        self.in_flight: dict[str, _Running] = {}
        self.done: set[str] = set()
        self.failed: set[str] = set()
        self.attempts: Counter = Counter()
        self.in_use = 0
        self.groups_done: set[str] = set()

    # ----------------------------------------------------------------- guards

    def cap_now(self) -> int:
        f = self.runs_dir / self.settings["cap_file"]
        if f.exists():
            try:
                return int(f.read_text().strip())
            except ValueError:
                pass
        return self.cap

    def limit(self, need_process: bool) -> tuple[int, str]:
        """(GPUs available for new launches, reason when held)."""
        if (self.runs_dir / self.settings["stop_file"]).exists():
            return 0, f"{self.settings['stop_file']} present"
        ws = workspace_containers()
        if ws is not None and ws > self.settings["workspace_max_containers"]:
            return 0, f"workspace has {ws} active containers > {self.settings['workspace_max_containers']}"
        if need_process:
            free = free_memory_pct()
            if free is not None and free < self.settings["min_free_memory_pct"]:
                return 0, f"laptop free memory {free}% < {self.settings['min_free_memory_pct']}%"
        ext = int(self.external_gpus()) if self.external_gpus else 0
        return self.cap_now() - ext, ""

    # ----------------------------------------------------------------- dispatch

    def _start(self, job: Job) -> None:
        now = time.time()
        if job.is_process:
            if job.pid_file and job.pid_file.exists() and pid_alive(int(job.pid_file.read_text().strip() or 0)):
                pid = int(job.pid_file.read_text().strip())
                self.in_flight[job.name] = _Running(job, job.pid_file.stat().st_mtime, pid=pid)
                print(f"[adopt] {job.name} (pid {pid}, weight {job.weight})", flush=True)
            else:
                log = job.log or (self.runs_dir / f"{job.name.replace('/', '_').replace(':', '_')}.log")
                log.parent.mkdir(parents=True, exist_ok=True)
                fh = log.open("a")
                fh.write(f"\n===== {datetime.now().isoformat()} {' '.join(map(str, job.run))}\n")
                fh.flush()
                proc = subprocess.Popen([str(c) for c in job.run], stdout=fh, stderr=subprocess.STDOUT,
                                        cwd=job.cwd, env=os.environ.copy(), start_new_session=True)
                if job.pid_file:
                    job.pid_file.parent.mkdir(parents=True, exist_ok=True)
                    job.pid_file.write_text(str(proc.pid))
                self.in_flight[job.name] = _Running(job, now, proc=proc, pid=proc.pid)
                print(f"[start] {job.name} pid {proc.pid} (weight {job.weight}; in use {self.in_use + job.weight})",
                      flush=True)
        else:
            task = asyncio.create_task(job.run())
            self.in_flight[job.name] = _Running(job, now, task=task)
            print(f"[start] {job.name} (weight {job.weight}; in use {self.in_use + job.weight})", flush=True)
        self.in_use += job.weight

    def _reap(self, pending: list[Job]) -> None:
        for name, r in list(self.in_flight.items()):
            job, err = r.job, None
            if r.task is not None:
                if not r.task.done():
                    continue
                err = r.task.exception()
            elif r.proc is not None:
                rc = r.proc.poll()
                if rc is None:
                    continue
                err = None if rc == 0 and job.done() else RuntimeError(f"exit code {rc}, done={job.done()}")
            else:  # adopted
                if pid_alive(r.pid):
                    continue
                err = None if job.done() else RuntimeError("adopted process ended without outputs")
            del self.in_flight[name]
            self.in_use -= job.weight
            minutes = (time.time() - r.started) / 60
            if err is None:
                self.done.add(name)
                print(f"[done] {name} ({minutes:.0f} min)", flush=True)
                continue
            self.attempts[name] += 1
            print(f"[error] {name} (attempt {self.attempts[name]}/{self.settings['max_attempts']}): "
                  f"{type(err).__name__}: {err}", flush=True)
            if r.task is not None:
                traceback.print_exception(err)
            if self.attempts[name] < self.settings["max_attempts"]:
                pending.append(job)
                pending.sort()
            else:
                self.failed.add(name)

    def _expand(self, pending: list[Job]) -> None:
        """Replace ready placeholders by the jobs they expand to (done ones are recorded)."""
        for j in [p for p in pending if p.expand is not None and p.ready()]:
            pending.remove(j)
            new = j.expand()
            self.jobs.remove(j)
            self.jobs.extend(new)
            for n in new:
                (self.done.add(n.name) if n.done() else pending.append(n))
            print(f"[expand] {j.name} -> {len(new)} jobs", flush=True)
        pending.sort()

    def _group_hooks(self) -> None:
        if not self.on_group_done:
            return
        by_group: dict[str, list[Job]] = {}
        for j in self.jobs:
            if j.group:
                by_group.setdefault(j.group, []).append(j)
        for g, js in by_group.items():
            if g not in self.groups_done and all(j.name in self.done for j in js):
                self.groups_done.add(g)
                try:
                    self.on_group_done(g, js)
                except Exception as e:  # noqa: BLE001 - hooks are reporting only
                    print(f"[group] hook failed for {g}: {type(e).__name__}: {e}", flush=True)

    def _state(self, pending: list[Job], limit: int, hold: str) -> None:
        atomic_write_json({
            "time": datetime.now().isoformat(), "pid": os.getpid(), "cap": self.cap_now(), "limit": limit,
            "hold": hold, "workspace_containers": workspace_containers(),
            "gpus_in_use": self.in_use,
            "in_flight": [{"job": r.job.name, "weight": r.job.weight, "pid": r.pid,
                           "minutes": round((time.time() - r.started) / 60, 1)} for r in self.in_flight.values()],
            "pending": [j.name for j in pending], "done": len(self.done), "failed": sorted(self.failed),
        }, self.runs_dir / f"{self.name}.json", indent=1)

    async def run(self) -> dict:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for j in sorted(self.jobs):
            if j.done():
                self.done.add(j.name)
            else:
                pending.append(j)
        print(f"[{self.name}] {len(pending)} pending, {len(self.done)} already done, cap {self.cap_now()}", flush=True)
        if self.dry_run:
            for j in pending:
                kind = "placeholder" if j.expand else "process" if j.is_process else "coroutine"
                print(f"  would run {j.name} (weight {j.weight}, {kind}, ready={j.ready()})")
            return self.summary()   # no group hooks either: a dry run writes nothing
        (self.runs_dir / f"{self.name}.pid").write_text(str(os.getpid()))
        idle_since = None
        tick = float(self.settings["tick_s"])
        while pending or self.in_flight:
            self._reap(pending)
            self._group_hooks()
            self._expand(pending)
            ready = [j for j in pending if j.ready()]
            limit, hold = self.limit(any(j.is_process for j in ready))
            if hold and ready and int(time.time()) % 60 < tick:
                print(f"[hold] {hold}", flush=True)
            for j in list(ready):
                if self.in_use + j.weight <= limit:
                    pending.remove(j)
                    self._start(j)
            self._state(pending, limit, hold)
            if not self.in_flight and not ready:
                idle_since = idle_since or time.time()
                if time.time() - idle_since > float(self.settings["idle_exit_minutes"]) * 60:
                    print(f"[{self.name}] nothing ready for {self.settings['idle_exit_minutes']} min; exiting "
                          f"with {len(pending)} unready jobs", flush=True)
                    break
            else:
                idle_since = None
            await asyncio.sleep(tick)
        self._group_hooks()
        print(f"[{self.name}] exit: {len(self.done)} done, {len(self.failed)} failed", flush=True)
        return self.summary()

    def summary(self) -> dict:
        return {"done": sorted(self.done), "failed": sorted(self.failed),
                "pending": [j.name for j in self.jobs if j.name not in self.done | self.failed]}


def run_scheduler(jobs: list[Job], runs_dir: Path, cap: int | None = None, name: str = "scheduler",
                  dry_run: bool = False, **kw) -> dict:
    """Blocking entry point: tee output to <runs_dir>/<name>.log and run the loop."""
    from cotcontrol.run_logging import tee_stdio

    sched = Scheduler(jobs, Path(runs_dir), cap=cap, name=name, dry_run=dry_run, **kw)
    if dry_run:
        return asyncio.run(sched.run())
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    with tee_stdio(Path(runs_dir) / f"{name}.log", mode="a"):
        print(f"[{name}] start {datetime.now().isoformat()} pid {os.getpid()} cap {sched.cap} "
              f"apps {os.environ.get('COTCONTROL_MODAL_INFERENCE_APP')}", flush=True)
        return asyncio.run(sched.run())
