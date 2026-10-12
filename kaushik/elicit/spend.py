"""Modal spend so far for the elicitation workstreams (apps are new on 2026-10-09, so all their cost counts).

    .venv/bin/python kaushik/elicit/spend.py                 # every *_sub-* app of ours, plus total
    .venv/bin/python kaushik/elicit/spend.py sub-nocot-sft   # one workstream
    .venv/bin/python kaushik/elicit/spend.py --start 2026-10-09

Hourly resolution: only completed hours are reported (a daily report shows $0 until the day ends), so
the number lags by up to ~1 h. Treat it as a lower bound and keep headroom. The app prefix comes from
kaushik/config/defaults.yaml (modal.app_prefix).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kaushik.core.config import defaults
from kaushik.core.modal_apps import modal_bin


def spend(substring: str = "", start: str = "2026-10-09") -> dict[str, float]:
    prefix = defaults()["modal"]["app_prefix"] + "_sub-"
    out = subprocess.run([*modal_bin(), "billing", "report", "--start", start, "--resolution", "h", "--json"],
                         capture_output=True, text=True, check=True).stdout
    rows = json.loads(re.sub(r"\x1b\[[0-9;]*m", "", out))
    by_app: dict[str, float] = defaultdict(float)
    for r in rows:
        d = r["description"]
        if d.startswith(prefix) and substring in d:
            by_app[d] += float(r["cost"])
    return dict(by_app)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sub", nargs="?", default="", help="substring of the app name, e.g. sub-gepa-lora")
    ap.add_argument("--start", default="2026-10-09", help="first day of the report (the sub-apps were created then)")
    a = ap.parse_args(argv)
    s = spend(a.sub, a.start)
    for k, v in sorted(s.items()):
        print(f"{v:9.2f}  {k}")
    print(f"{sum(s.values()):9.2f}  TOTAL")


if __name__ == "__main__":
    main()
