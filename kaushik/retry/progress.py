"""Progress of a run group from its attempt logs (experiment job).

    run.py retry progress run=main        # items started / finished / honest-compliant per (model, arm)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.retry import retry_common as rc


def main(cfg: dict, argv: list[str] | None = None) -> None:
    from cotcontrol.eval.retry import load_attempts

    argparse.ArgumentParser(prog="retry progress", description=__doc__).parse_args(argv)
    run_dir = rc.RUNS / str(cfg["run"])
    for path in sorted(run_dir.glob("**/attempts.jsonl")):
        eps = load_attempts(path)
        man = json.loads((path.parent / "manifest.json").read_text()) if (path.parent / "manifest.json").exists() else {}
        n_items, n_att = man.get("n_items", "?"), man.get("n_attempts", cfg["n_attempts"])
        succ = sum(any(r["honest"] for r in recs) for recs in eps.values())
        finished = sum(any(r["honest"] for r in recs) and man.get("stop_on_success", True) or len(recs) >= n_att
                       for recs in eps.values())
        errs = sum(1 for _ in (path.parent / "errors.jsonl").open()) if (path.parent / "errors.jsonl").exists() else 0
        print(f"{path.parent.relative_to(rc.RUNS)}: {len(eps)}/{n_items} items started, {finished} finished, "
              f"{succ} with an honest-compliant attempt, {sum(map(len, eps.values()))} attempts, {errs} errored generations")
