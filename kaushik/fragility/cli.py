"""Stages for `kaushik/run.py fragility <stage>` (Experiment 1 arms 1A/1D/1E).

    build_pool    public-QA train splits -> data/mcq_pool.json (+ eval-overlap check)
    build_data    pool -> data/mcq_noreason.jsonl (1A rows)
    distill       base-model rollouts -> data/mcq_reason_<model>.jsonl (1D rows)
    style         pirate rewrites -> data/style_sft.jsonl (1E rows)
    make_configs  configs/*.yaml (train_eval.py); --smoke for the tiny end-to-end check
    schedule      launch train_eval.py per config in the background under the GPU cap
    launch        unconditional launch of selected configs
    status        progress of launched runs
    registry      kaushik/registry/<run>.json for finished final adapters
    analyze       Experiment 1 table + plots -> results/

Modules are imported lazily so listing stages never touches Modal/OpenRouter.
"""

from __future__ import annotations

import importlib
import subprocess

from common import HERE, ROOT


def _build_pool(argv: list[str]) -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:  # the repo venv has no parquet reader; use an ephemeral overlay
        cmd = ["uv", "run", "--with", "pyarrow", "python", str(HERE / "build_pool.py"), *argv]
        print("pyarrow not in .venv ->", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    importlib.import_module("build_pool").main(argv)


def _stage(module: str, fn: str = "main"):
    def run(argv: list[str]) -> None:
        getattr(importlib.import_module(module), fn)(argv)
    return run


STAGES = {
    "build_pool": _build_pool,
    "build_data": _stage("build_sft_data"),
    "distill": _stage("distill"),
    "style": _stage("style"),
    "make_configs": _stage("make_configs"),
    "schedule": _stage("launch", "schedule"),
    "launch": _stage("launch", "launch"),
    "status": _stage("launch", "status"),
    "migrate": _stage("launch", "migrate"),
    "registry": _stage("registry"),
    "split_eval": _stage("split_eval"),
    "analyze": _stage("analyze"),
}
