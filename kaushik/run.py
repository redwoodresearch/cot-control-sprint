#!/usr/bin/env python
"""Runner for the kaushik/ experiments (design: ARCHITECTURE.md; library: core/README.md).

    .venv/bin/python kaushik/run.py --list
    .venv/bin/python kaushik/run.py <experiment> <job> [key=value ...] [job args]
    .venv/bin/python kaushik/run.py sweep <sweep.yaml> [--dry-run | --check GLOB... | --write]
    .venv/bin/python kaushik/run.py orange_expand/sft train model=qwen8b run_name=x --dry-run

<experiment> is a folder under kaushik/ with a config.yaml (migrated) or a cli.py
with STAGES (legacy; its stages keep working unchanged). This file sets the shared
Modal environment *before* anything imports cotcontrol's Modal apps (volume and
app names resolve at import time), then dispatches through kaushik/core/runner.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def setup_env() -> None:
    # Checkpoints on our own Modal volume (Redwood naming); HF cache stays shared.
    # Call the deployed apps (kaushik/deploy_modal.sh) instead of opening one
    # ephemeral app per run/eval block: the workspace caps ephemeral apps at 100.
    # Set either app variable to "" to fall back to ephemeral apps; export
    # COTCONTROL_MODAL_*_APP (or use modal_apps.setup_env(sub=...)) for a sub pair.
    sys.path.insert(0, str(ROOT))
    from kaushik.core.modal_apps import setup_env as _setup

    _setup(sub=os.environ.get("KAUSHIK_MODAL_SUB") or None)


def main() -> None:
    setup_env()
    from kaushik.core.runner import main as run

    run(sys.argv[1:])


if __name__ == "__main__":
    main()
