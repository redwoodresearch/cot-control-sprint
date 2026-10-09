"""Metrics table for len_teacher runs (Workstream B's definitions: honest, honest_long vs the UNTRAINED median,
heldout, accuracy, chars, 95% bootstrap CIs over questions; seeds `_s<k>` pooled).

    kaushik/run.py elicit/len_teacher analyze [--runs a,b,...] [--boot 1000] [--out results/<name>.md] [--per-mode]

Workstream B's teacher_* / untrained_* runs are symlinked into runs/ as reference rows (gain recovered is
relative to the GEPA teacher).
"""

from __future__ import annotations

import importlib.util
import os
import sys

from lt_common import RESULTS, ROOT, RUNS

GD = ROOT / "kaushik/elicit/gepa_distill"


def _gd_analyze():
    sys.path.insert(0, str(GD))
    try:
        spec = importlib.util.spec_from_file_location("gd_analyze_ro", GD / "analyze.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(GD))
    mod.RUNS, mod.RESULTS = RUNS, RESULTS
    return mod


def link_references() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    for name in ("teacher_val", "untrained_val", "teacher_test", "untrained_test"):
        src, dst = GD / "runs" / name, RUNS / name
        if src.is_dir() and not dst.exists():
            os.symlink(src, dst)


def main(argv: list[str] | None = None) -> None:
    link_references()
    _gd_analyze().main(argv)


if __name__ == "__main__":
    main()
