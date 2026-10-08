#!/usr/bin/env python
"""Runner for the kaushik/ sprint experiments (spec: sprint_2026_10_08.md).

    .venv/bin/python kaushik/run.py --list
    .venv/bin/python kaushik/run.py <experiment> <stage> [stage args...]
    .venv/bin/python kaushik/run.py fragility/cpt train --dry-run

<experiment> is a path under kaushik/ (e.g. fragility, fragility/cpt, retry).
Its cli.py defines STAGES = {name: fn(argv: list[str]) -> None}. This file
sets the shared environment *before* anything imports cotcontrol's Modal apps
(volume names are resolved at import time), then dispatches.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def setup_env() -> None:
    # Checkpoints on our own Modal volume (Redwood naming); HF cache stays shared.
    os.environ.setdefault("COTCONTROL_MODAL_OWNER", "kaushik-reddy")
    # Call the deployed apps (kaushik/deploy_modal.sh) instead of opening one
    # ephemeral app per run/eval block: the workspace caps ephemeral apps at 100.
    # Set either to "" to fall back to ephemeral apps.
    os.environ.setdefault("COTCONTROL_MODAL_INFERENCE_APP", "cot-control-sprint_kaushik-reddy_inference")
    os.environ.setdefault("COTCONTROL_MODAL_TRAIN_APP", "cot-control-sprint_kaushik-reddy_train")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")  # OPENROUTER_API_KEY etc.; never printed
    except ImportError:
        pass
    sys.path.insert(0, str(ROOT))


def experiments() -> list[str]:
    return sorted(str(p.parent.relative_to(HERE)) for p in HERE.rglob("cli.py"))


def load_stages(experiment: str) -> dict:
    path = HERE / experiment / "cli.py"
    if not path.exists():
        raise SystemExit(f"no {path.relative_to(ROOT)}; known: {experiments()}")
    spec = importlib.util.spec_from_file_location(f"kaushik_{experiment.replace('/', '_')}_cli", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod.STAGES


def main() -> None:
    setup_env()
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "--list"):
        for exp in experiments():
            print(f"{exp}: {', '.join(load_stages(exp))}")
        return
    stages = load_stages(args[0])
    if len(args) < 2 or args[1] not in stages:
        raise SystemExit(f"{args[0]} stages: {', '.join(stages)}")
    stages[args[1]](args[2:])


if __name__ == "__main__":
    main()
