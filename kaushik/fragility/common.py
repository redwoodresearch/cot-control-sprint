"""Shared paths/helpers for kaushik/fragility (Experiment 1 arms 1A, 1D, 1E)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA, CONFIGS, RUNS, RESULTS = HERE / "data", HERE / "configs", HERE / "runs", HERE / "results"
ORANGE = ROOT / "orange-training"
PY = str(ROOT / ".venv" / "bin" / "python")

VAL_MODES = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "end_of_sentence"]
ARMS = ["mcq_noreason", "mcq_reason", "style_sft"]  # 1A, 1D, 1E


def orange(name: str):
    """Import a module from orange-training/ read-only (the folder name has a
    hyphen, so it is not a package; its modules import each other as siblings)."""
    for p in (str(ROOT), str(ORANGE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    key = f"orange_training_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ORANGE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[key] = mod
    # orange's modules put their folder at sys.path[0] for sibling imports;
    # leaving it there would shadow our own make_configs/build_sft_data/analyze.
    sys.path[:] = [p for p in sys.path if p != str(ORANGE)]
    return sys.modules[key]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def atomic_write_json(obj, path: Path) -> None:
    """Write <path>.tmp then rename, so pollers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 — not a git checkout
        return None


def answer_letter(item: dict) -> str:
    return "ABCD"[item["answer_index"]]
