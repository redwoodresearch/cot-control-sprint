"""Print one item's files per scenario (eyeball check of the wording; experiment job).

    run.py hidden_constraint preview [slice=dev] [--scenario S] [--idx 0] [--control]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.hidden_constraint import runner


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hidden_constraint preview", description=__doc__)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args(argv)
    sc = runner.scenarios_mod()
    for name in ([a.scenario] if a.scenario else list(cfg["scenarios"])):
        item = sc.build_items(name, cfg["slice"], a.idx + 1)[a.idx]
        files = item.control_files if a.control else item.files
        print(f"\n{'=' * 100}\n{name} | {item.item_id} | {'control' if a.control else 'constrained'} | "
              f"reveal={item.reveal_path} | terms={item.forbidden_terms or item.forbidden_pattern} | answer={item.answer}")
        for path, content in sorted(files.items()):
            print(f"\n--- {path} ---\n{content}")
