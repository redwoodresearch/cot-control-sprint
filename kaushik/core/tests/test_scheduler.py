from __future__ import annotations

import asyncio
import sys

import pytest

from kaushik.core import scheduler as sch

FAST = {"max_gpus": 4, "cap_file": ".gpu_cap", "stop_file": ".stop_launches", "workspace_max_containers": 800,
        "min_free_memory_pct": 25, "max_attempts": 3, "tick_s": 0.01, "scan_interval_s": 1, "idle_exit_minutes": 0.001}


@pytest.fixture(autouse=True)
def no_guards(monkeypatch):
    monkeypatch.setattr(sch, "workspace_containers", lambda max_age=60.0: 10)
    monkeypatch.setattr(sch, "free_memory_pct", lambda: 90)


def _coro_job(name, weight, log, outputs, fail_times=0):
    attempts = {"n": 0}

    async def run():
        log.append(("start", name))
        await asyncio.sleep(0.02)
        attempts["n"] += 1
        if attempts["n"] <= fail_times:
            raise RuntimeError("boom")
        outputs.add(name)
        log.append(("end", name))

    return sch.Job(name=name, weight=weight, run=run, done=lambda: name in outputs)


def test_cap_limits_concurrency_and_runs_all(tmp_path):
    log, outputs = [], set()
    jobs = [_coro_job(f"j{i}", 2, log, outputs) for i in range(4)]
    s = sch.Scheduler(jobs, tmp_path, cap=4, settings=dict(FAST))
    summary = asyncio.run(s.run())
    assert sorted(summary["done"]) == ["j0", "j1", "j2", "j3"] and not summary["failed"]
    # never more than 2 jobs (weight 2 each) in flight under cap 4
    running, peak = 0, 0
    for ev, _ in log:
        running += 1 if ev == "start" else -1
        peak = max(peak, running)
    assert peak == 2
    assert (tmp_path / "scheduler.json").exists()


def test_retry_then_fail(tmp_path):
    log, outputs = [], set()
    jobs = [_coro_job("flaky", 1, log, outputs, fail_times=1), _coro_job("dead", 1, log, outputs, fail_times=99)]
    s = sch.Scheduler(jobs, tmp_path, cap=4, settings=dict(FAST))
    summary = asyncio.run(s.run())
    assert summary["done"] == ["flaky"] and summary["failed"] == ["dead"]
    assert s.attempts["dead"] == 3


def test_done_jobs_skipped_and_placeholder_expands(tmp_path):
    log, outputs = [], {"already"}
    gate = tmp_path / "train_result.json"
    expanded = [_coro_job("e1", 1, log, outputs), _coro_job("e2", 1, log, outputs)]
    placeholder = sch.Job(name="eval:*", weight=1, ready=gate.exists, expand=lambda: expanded)
    done_job = _coro_job("already", 1, log, outputs)

    async def opener():
        await asyncio.sleep(0.05)
        gate.write_text("{}")

    async def both():
        s = sch.Scheduler([placeholder, done_job], tmp_path, cap=4, settings=dict(FAST))
        t = asyncio.create_task(opener())
        out = await s.run()
        await t
        return out

    summary = asyncio.run(both())
    assert set(summary["done"]) == {"already", "e1", "e2"}
    assert ("start", "already") not in log


def test_guards_hold_launches(tmp_path, monkeypatch):
    s = sch.Scheduler([], tmp_path, cap=4, settings=dict(FAST))
    assert s.limit(need_process=True) == (4, "")
    monkeypatch.setattr(sch, "workspace_containers", lambda max_age=60.0: 801)
    assert s.limit(need_process=False)[0] == 0
    monkeypatch.setattr(sch, "workspace_containers", lambda max_age=60.0: None)   # unknown does not block
    monkeypatch.setattr(sch, "free_memory_pct", lambda: 10)
    assert s.limit(need_process=False)[0] == 4 and s.limit(need_process=True)[0] == 0
    (tmp_path / ".gpu_cap").write_text("7")
    monkeypatch.setattr(sch, "free_memory_pct", lambda: 90)
    assert s.limit(need_process=True)[0] == 7
    (tmp_path / ".stop_launches").write_text("")
    assert s.limit(need_process=False)[0] == 0


def test_process_job_and_adoption(tmp_path):
    out = tmp_path / "out.txt"
    job = sch.Job(name="proc", weight=1, run=[sys.executable, "-c", f"open({str(out)!r}, 'w').write('ok')"],
                  done=out.exists, log=tmp_path / "proc.log", pid_file=tmp_path / "proc.pid")
    summary = asyncio.run(sch.Scheduler([job], tmp_path, cap=1, settings=dict(FAST)).run())
    assert summary["done"] == ["proc"] and out.read_text() == "ok"
    assert (tmp_path / "proc.pid").exists() and "===== " in (tmp_path / "proc.log").read_text()


def test_dry_run_dispatches_nothing(tmp_path):
    log, outputs = [], set()
    s = sch.Scheduler([_coro_job("j", 1, log, outputs)], tmp_path, cap=1, settings=dict(FAST), dry_run=True)
    summary = asyncio.run(s.run())
    assert summary["pending"] == ["j"] and not log
