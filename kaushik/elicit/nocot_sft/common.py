"""Shared paths/helpers for kaushik/elicit/nocot_sft (Workstream A: no-CoT SFT)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA, CONFIGS, RUNS, RESULTS = HERE / "data", HERE / "configs", HERE / "runs", HERE / "results"
FRAG = ROOT / "kaushik" / "fragility"
FRAG_DATA = FRAG / "data"
REGISTRY = ROOT / "kaushik" / "registry"
PY = str(ROOT / ".venv" / "bin" / "python")
EXPERIMENT = "elicit/nocot_sft"
SUB = "sub-nocot-sft"
APPS = {
    "COTCONTROL_MODAL_INFERENCE_APP": f"cot-control-sprint_kaushik-reddy_{SUB}_inference",
    "COTCONTROL_MODAL_TRAIN_APP": f"cot-control-sprint_kaushik-reddy_{SUB}_train",
}

BASE_MODEL = "openai/gpt-oss-20b"
MODEL_KEY = "gptoss20b"
PREFILL = "<|channel|>analysis<|message|><|end|><|start|>assistant"   # masked empty analysis channel
TRAIN_KWARGS = {"reasoning_effort": "low"}      # 1A's convention for reasoning-off rows
EVAL_KWARGS = {"reasoning_effort": "medium"}    # the shared eval protocol; anchor rows were sampled here
ATTN = "kernels-community/vllm-flash-attn3"

# 1A reference adapter (fragility, seed 0) re-evaluated on the new val protocol.
REF_1A_RUN = FRAG / "runs" / "fragility-mcq_noreason-gptoss20b-seed0"


def set_apps() -> None:
    """Point every Modal call at this workstream's deployed app pair (cost tracking)."""
    for k, v in APPS.items():
        os.environ[k] = v


def frag(name: str):
    """Import a kaushik/fragility module read-only (its modules import `common` as a
    sibling, which would collide with ours, so load under a private name)."""
    key = f"fragility_{name}"
    if key in sys.modules:
        return sys.modules[key]
    saved = sys.modules.get("common")
    sys.path.insert(0, str(FRAG))
    try:
        sys.modules.pop("common", None)
        spec = importlib.util.spec_from_file_location(key, FRAG / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[key] = mod
    finally:
        sys.path.remove(str(FRAG))
        if saved is not None:
            sys.modules["common"] = saved
    return mod


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


def write_dataset(rows: list[dict], path: Path, manifest: dict) -> None:
    write_jsonl(rows, path)
    manifest = {"n_rows": len(rows), "sha256": sha256_file(path), **manifest, "example_row": rows[0]}
    path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows)} rows -> {path.relative_to(ROOT)} (sha256 {manifest['sha256'][:12]}...)")


def atomic_write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None
