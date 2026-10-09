"""Metrics table for lora_arith runs — honest, honest_long, heldout, accuracy, chars, 95% bootstrap CIs over questions.

    kaushik/run.py elicit/lora_arith analyze [--runs a,b,...] [--boot 1000] [--out results/<name>.md]

Thin wrapper over Workstream B's analyze.py (same definitions, so numbers are directly comparable): honest =
cotcontrol.eval.grading.is_honest; honest_long = honest among traces at least the UNTRAINED median length for that
(split, block); gain recovered = (candidate - untrained) / (teacher - untrained). Workstream B's teacher_* and
untrained_* runs are symlinked into runs/ so they appear as reference rows. Runs sharing a stem with _s<k>
suffixes (seeds) are pooled.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from gl_common import GD_RUNS, RESULTS, ROOT, RUNS


def _gd_analyze():
    gd = ROOT / "kaushik/elicit/gepa_distill"
    sys.path.insert(0, str(gd))
    try:
        spec = importlib.util.spec_from_file_location("gd_analyze_ro", gd / "analyze.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(gd))
    mod.RUNS, mod.RESULTS = RUNS, RESULTS
    return mod


def link_references() -> None:
    """Expose Workstream B's reference evals (read-only symlinks) under our runs/."""
    RUNS.mkdir(parents=True, exist_ok=True)
    for name in ("teacher_val", "untrained_val", "teacher_test", "untrained_test"):
        src, dst = GD_RUNS / name, RUNS / name
        if src.is_dir() and not dst.exists():
            os.symlink(src, dst)


def main(argv: list[str] | None = None) -> None:
    link_references()
    mod = _gd_analyze()
    mod.main(argv)


if __name__ == "__main__":
    main()
