"""Experiment 1 publication figures (figures/make_figures.py; reads results/ only).

    .venv/bin/python kaushik/run.py fragility figures
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import resolve
from kaushik.core.io import load_py

EXP = "fragility"


def main(cfg: dict, argv: list[str] | None = None) -> None:
    load_py(Path(__file__).resolve().parent / "figures" / "make_figures.py").main()


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
