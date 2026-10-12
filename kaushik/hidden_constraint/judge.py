"""Fill in missing constraint-mention judge outputs (experiment job).

    run.py hidden_constraint judge [--runs heldout_x,heldout_y]      # default: every run with episodes.jsonl

Needed only when OPENROUTER_API_KEY was absent during a run (the run logs a warning). Rewrites
episodes.jsonl and summary.json of each run; re-run `analyze` afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import runs_root
from kaushik.hidden_constraint import runner


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hidden_constraint judge", description=__doc__)
    ap.add_argument("--runs", default=None, help="comma-separated run names (default: all)")
    a = ap.parse_args(argv)
    if not runner.openrouter_available():
        raise SystemExit("OPENROUTER_API_KEY missing (.env)")
    runs = runs_root(cfg)
    dirs = [runs / r for r in a.runs.split(",")] if a.runs else \
        sorted(p for p in runs.iterdir() if p.is_dir() and (p / "episodes.jsonl").exists())

    async def go():
        for d in dirs:
            n = await runner.fill_judge(d, cfg)
            print(f"{d.name}: judged {n} episodes")
    asyncio.run(go())
