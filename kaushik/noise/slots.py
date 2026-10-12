"""LEGACY (kept for kaushik/elicit/*/evalrun.py, which load this file by path): a
cross-process cap on concurrent Modal GPUs. The noise pipeline itself no longer uses it; its
GPU accounting is kaushik/core/scheduler.py (cap file <runs>/.gpu_cap). Delete once elicit/*
has moved to the core scheduler.

Original description: cross-process cap on concurrent Modal GPUs for the noise pipeline.

The sprint budget is a hard cap on GPUs in flight (15, dropping to 7 six hours
after the first full launch). Every GPU-backed engine call in this folder
acquires `tensor_parallel_size` slots first. Slots are lock files under
runs/_gpu_slots/ holding the owner pid, so several `process`/`kl` subprocesses
share one pool; a slot whose pid is dead is stale and reclaimed. The cap is
read from runs/gpu_cap.json on every acquire so `watch` can lower it live.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

DEFAULT_CAP = 15
MAX_SLOTS = 256  # upper bound on slot files, not on the cap


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GpuSlots:
    def __init__(self, runs_dir: Path):
        self.dir = Path(runs_dir) / "_gpu_slots"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cap_file = Path(runs_dir) / "gpu_cap.json"

    def state(self) -> dict:
        try:
            return json.loads(self.cap_file.read_text())
        except (OSError, ValueError):
            return {}

    def cap(self) -> int:
        try:
            return int(self.state()["cap"])
        except (ValueError, KeyError):
            return DEFAULT_CAP

    def set_cap(self, cap: int, **extra) -> None:
        state = {}
        try:
            state = json.loads(self.cap_file.read_text())
        except (OSError, ValueError):
            pass
        state.update({"cap": int(cap), "updated": time.time(), **extra})
        tmp = self.cap_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.cap_file)

    def _try_take(self, i: int) -> bool:
        p = self.dir / f"slot_{i:02d}"
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(p.read_text().strip() or -1)
            except (OSError, ValueError):
                return False
            if pid == os.getpid() or (pid > 0 and _alive(pid)):
                return False
            try:  # stale: owner is gone
                p.unlink()
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except (FileExistsError, FileNotFoundError):
                return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True

    def held(self) -> int:
        """Slots currently held by live processes (approx. GPUs in flight)."""
        n = 0
        for p in self.dir.glob("slot_*"):
            try:
                pid = int(p.read_text().strip() or -1)
            except (OSError, ValueError):
                continue
            n += pid > 0 and _alive(pid)
        return n

    def release(self, taken: list[int]) -> None:
        for i in taken:
            try:
                (self.dir / f"slot_{i:02d}").unlink()
            except FileNotFoundError:
                pass

    async def acquire(self, n: int = 1, poll: float = 5.0) -> list[int]:
        while True:
            cap = min(self.cap(), MAX_SLOTS)
            taken: list[int] = []
            for i in range(cap):
                if self._try_take(i):
                    taken.append(i)
                    if len(taken) == n:
                        return taken
            self.release(taken)
            await asyncio.sleep(poll)

    @asynccontextmanager
    async def slot(self, n: int = 1):
        taken = await self.acquire(n)
        try:
            yield
        finally:
            self.release(taken)
