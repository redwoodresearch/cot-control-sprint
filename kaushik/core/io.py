"""Small I/O helpers shared by every kaushik/ experiment: atomic JSON, JSONL,
sha256, git commit, module-by-path loading, process spawning and the laptop
memory guard. Nothing here imports Modal or cotcontrol.

    from kaushik.core.io import atomic_write_json, read_jsonl, write_jsonl, git_commit
"""

# ruff: noqa: DTZ005 — naive local timestamps, like the rest of the repo
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
KAUSHIK = ROOT / "kaushik"


def atomic_write_json(obj, path: Path, indent: int = 2) -> Path:
    """Write <path>.tmp then rename, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent))
    os.replace(tmp, path)
    return path


def read_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def append_jsonl(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def tail_jsonl(path: Path) -> dict | None:
    rows = read_jsonl(path)
    return rows[-1] if rows else None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(cwd: Path = ROOT) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_py(path: Path, name: str | None = None) -> ModuleType:
    """Import a .py file by path as module `name` (default: a name derived from the
    path under the repo root), cached in sys.modules. Use this for scripts that are
    not packages (scripts/train_eval.py, experiment job files)."""
    path = Path(path).resolve()
    if name is None:
        inside = path.is_relative_to(ROOT)
        name = "kpath_" + re.sub(r"\W", "_", str((path.relative_to(ROOT) if inside else path).with_suffix("")))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_callable(ref: str):
    """'kaushik/orange_expand/prompts.py:eval_block' or 'pkg.module:fn' -> callable."""
    target, _, attr = ref.rpartition(":")
    if not attr:
        raise ValueError(f"callable reference needs '<module-or-path>:<name>': {ref!r}")
    if target.endswith(".py"):
        mod = load_py(ROOT / target if not Path(target).is_absolute() else Path(target))
    else:
        mod = importlib.import_module(target)
    return getattr(mod, attr)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def spawn(cmd: list[str], log: Path, cwd: Path = ROOT) -> int:
    """Detached subprocess (own session) appending stdout+stderr to `log`; -> pid."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(f"\n===== {datetime.now().isoformat()} {' '.join(map(str, cmd))}\n")
        p = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=cwd,
                             env=os.environ.copy(), start_new_session=True)
    return p.pid


def free_memory_pct() -> int | None:
    """macOS `memory_pressure` system-wide free percentage; None where unavailable."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"free percentage:\s*(\d+)%", out.stdout)
    return int(m.group(1)) if m else None


def rel(path: Path) -> str:
    """Repo-root-relative string for paths inside the repo (as-is otherwise)."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
