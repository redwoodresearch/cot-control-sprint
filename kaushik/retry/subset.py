"""Item sets: the seeded, mode-stratified main subset and the power statement (experiment job).

    run.py retry subset --power                      # item counts and SEs of every item spec
    run.py retry subset [--write]                    # write subset.out (configs/main_subset.json) from item_specs.main

The subset is data (its ids are what runs/main holds), so it is never rewritten without --write.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.retry import retry_common as rc


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="retry subset", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", action="store_true", help="sample-size statement per item spec")
    ap.add_argument("--write", action="store_true", help="(re)write the subset file")
    a = ap.parse_args(argv)
    if a.power:
        for name, spec in cfg["item_specs"].items():
            n = len(rc.load_items("openai/gpt-oss-20b", **spec))
            print(f"{name}: {spec} -> {rc.power_statement(n)}; {rc.power_statement(n, 0.3)}")
        return
    s = cfg["subset"]
    spec = {k: v for k, v in cfg["item_specs"]["main"].items() if k != "subset"}
    d = rc.make_subset(spec, int(s["n"]), int(s["seed"]), rc.HERE / s["out"] if a.write else None)
    print(f"{d['n']} items (per mode {d['per_mode']}) from {d['n_source_items']}; sha256 {d['sha256']}"
          + (f" -> {s['out']}" if a.write else " (not written; --write)"))
