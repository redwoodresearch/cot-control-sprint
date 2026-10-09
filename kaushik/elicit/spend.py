#!/usr/bin/env python
"""Modal spend so far for the elicitation workstreams (apps are new on 2026-10-09, so all their cost counts).

    .venv/bin/python kaushik/elicit/spend.py                 # every *_sub-* app of ours, plus total
    .venv/bin/python kaushik/elicit/spend.py sub-nocot-sft   # one workstream

Hourly resolution: only completed hours are reported (a daily report shows $0 until the day ends), so
the number lags by up to ~1 h. Treat it as a lower bound and keep headroom.
"""
import json
import re
import subprocess
import sys
from collections import defaultdict

START = "2026-10-09"
PREFIX = "cot-control-sprint_kaushik-reddy_"


def spend(substring: str = "") -> dict[str, float]:
    out = subprocess.run([".venv/bin/modal", "billing", "report", "--start", START, "--resolution", "h", "--json"],
                         capture_output=True, text=True, check=True).stdout
    rows = json.loads(re.sub(r"\x1b\[[0-9;]*m", "", out))
    by_app: dict[str, float] = defaultdict(float)
    for r in rows:
        d = r["description"]
        if d.startswith(PREFIX + "sub-") and substring in d:
            by_app[d] += float(r["cost"])
    return dict(by_app)


if __name__ == "__main__":
    s = spend(sys.argv[1] if len(sys.argv) > 1 else "")
    for k, v in sorted(s.items()):
        print(f"{v:9.2f}  {k}")
    print(f"{sum(s.values()):9.2f}  TOTAL")
