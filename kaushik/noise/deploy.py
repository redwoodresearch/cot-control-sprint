"""(Re)deploy the CPU adapter-build Modal app (config.yaml: modal.random_lora_app). Run after
changing the remote-side code of cotcontrol/training/random_lora.py.

    .venv/bin/python kaushik/run.py noise deploy [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from kaushik.core.io import KAUSHIK, ROOT, load_py

nz = load_py(KAUSHIK / "noise" / "common.py")


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="noise deploy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    a = ap.parse_args(argv)
    cmd = [sys.executable, "-m", "modal", "deploy", "-m", "cotcontrol.training.random_lora",
           "--name", cfg["modal"]["random_lora_app"]]
    print(" ".join(cmd))
    if a.dry_run:
        return
    # The module must build its own app, not look the deployed one up: unset the selector.
    env = {k: v for k, v in os.environ.items() if k != nz.ENV_RANDOM_LORA_APP}
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
