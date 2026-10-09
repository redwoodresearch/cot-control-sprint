"""Stages for `kaushik/run.py elicit/nocot_sft <stage>` (Workstream A: no-CoT SFT).

    gen_answers   gpt-oss-20b's own explained MCQ answers -> data/answers/ (arm c/d source)
    build_data    data/*.jsonl for every arm (+ manifests with sha256)
    make_configs  configs/*.yaml for train_eval.py (--wave 1|2|test)
    launch        launch selected configs in the background (--configs / --arms / --seeds)
    status        progress of launched runs
    ref_1a        eval-only run of the existing 1A adapter (+ checkpoint-0) on the val protocol
    registry      kaushik/registry/<run>.json for final adapters
    analyze       val / test tables with bootstrap CIs -> results/

Every stage pins the Modal apps to this workstream's deployed pair.
"""

from __future__ import annotations

import importlib

from common import set_apps

set_apps()


def _stage(module: str, fn: str = "main"):
    def run(argv: list[str]) -> None:
        getattr(importlib.import_module(module), fn)(argv)
    return run


STAGES = {
    "gen_answers": _stage("gen_answers"),
    "build_data": _stage("build_data"),
    "make_configs": _stage("make_configs"),
    "launch": _stage("launch", "launch"),
    "status": _stage("launch", "status"),
    "ref_1a": _stage("launch", "ref_1a"),
    "test_step": _stage("launch", "test_step"),
    "registry": _stage("registry"),
    "analyze": _stage("analyze"),
}
