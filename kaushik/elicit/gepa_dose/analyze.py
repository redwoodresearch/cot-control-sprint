"""Metrics table for gepa_dose runs (Workstream B's definitions: honest = cotcontrol.eval.grading.is_honest;
honest_long = honest among traces >= the UNTRAINED median reasoning length for that split/block, 9 training modes;
95% bootstrap CIs over questions; runs sharing a stem with _s<k> suffixes are pooled).

    kaushik/run.py elicit/gepa_dose analyze --runs a,b,... [--subset RUN] [--out results/x.md] [--boot 1000]

--subset RUN restricts every run to the questions present in RUN's modes.json (the 40-question sweep subset), so
the full-val reference runs (untrained_val, teacher_val, x1_val = B2's lr1e-4_c400_val) are comparable; the
honest_long bar always comes from the FULL untrained reference. Reference runs are symlinked into runs/.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dose_common import GD, GL, RESULTS, RUNS, atomic_write_json, gd_module, load_json

REFS = {"teacher_val": GD / "runs/teacher_val", "untrained_val": GD / "runs/untrained_val",
        "teacher_test": GD / "runs/teacher_test", "x1_val": GL / "runs/lr1e-4_c400_val",
        **{f"x1_test_s{k}": GL / f"runs/lr1e-4_c400_test_s{k}" for k in range(3)}}
COLS = ["honest", "honest_long", "heldout", "accuracy", "strict"]


def link_references() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    for name, src in REFS.items():
        if src.is_dir() and not (RUNS / name).exists():
            os.symlink(src, RUNS / name)


def qids_of(run: str) -> set[str]:
    d = load_json(RUNS / run / "modes.json")
    return {f"{r['dataset']}:{r['id']}" for r in d["results"]}


def metrics(gd, group: list[str], split: str, n_boot: int, qids: set[str] | None) -> dict:
    ref = gd.untrained_blocks(split)
    bars = {b: float(np.median([r["chars"] for r in gd.rollouts(p)])) for b, p in ref.items() if b != "baseline"}
    pooled = defaultdict(list)
    for run in group:
        blocks = gd.run_blocks(run) if run != "untrained_test" else ref
        for b, p in blocks.items():
            pooled[b].extend(r for r in gd.rollouts(p) if qids is None or r["qid"] in qids)
    out = {"runs": group, "split": split, "bars": bars, "n": {b: len(v) for b, v in pooled.items()}}
    if pooled.get("modes"):
        rows = pooled["modes"]
        bar = bars.get("modes", 0)
        out["honest"] = gd.boot_ci(rows, "honest", n_boot)
        out["honest_long"] = gd.boot_ci(rows, "honest", n_boot, subset=lambda r: r["chars"] >= bar)
        out["strict"] = gd.boot_ci(rows, "compliance", n_boot)
        out["acc_constrained"] = gd.boot_ci(rows, "correct", n_boot)
        out["chars_median"] = float(np.median([r["chars"] for r in rows]))
        out["frac_long"] = float(np.mean([r["chars"] >= bar for r in rows]))
        out["per_mode_honest"] = {m: float(np.nanmean([r["honest"] for r in rows if r["mode"] == m]))
                                  for m in sorted({r["mode"] for r in rows})}
    if pooled.get("heldout"):
        rows = pooled["heldout"]
        out["heldout"] = gd.boot_ci(rows, "honest", n_boot)
        out["heldout_chars_median"] = float(np.median([r["chars"] for r in rows]))
    if pooled.get("baseline"):
        rows = pooled["baseline"]
        out["accuracy"] = gd.boot_ci(rows, "correct", n_boot)
        out["baseline_chars_median"] = float(np.median([r["chars"] for r in rows]))
    return out


def split_of(run: str) -> str:
    return "test" if run == "untrained_test" else load_json(RUNS / run / "manifest.json")["split"]


def table(runs: list[str], n_boot: int, qids: set[str] | None) -> tuple[str, dict]:
    gd = gd_module("analyze")
    gd.RUNS, gd.RESULTS = RUNS, RESULTS
    groups = defaultdict(list)
    for r in runs:
        groups[re.sub(r"_s\d+$", "", r)].append(r)
    res = {name: metrics(gd, g, split_of(g[0]), n_boot, qids) for name, g in groups.items()}
    lines = ["| run | split | n_modes | " + " | ".join(COLS) + " | chars med (bar) | frac ≥ bar | heldout chars | baseline chars |",
             "|" + "---|" * (len(COLS) + 7)]
    for name, m in sorted(res.items(), key=lambda kv: (kv[1]["split"], kv[0])):
        lines.append(f"| {name} | {m['split']} | {m['n'].get('modes', 0)} | " + " | ".join(gd.fmt(m.get(c)) for c in COLS) +
                     f" | {gd.fmt(m.get('chars_median'))} ({gd.fmt(m['bars'].get('modes'))}) | {m.get('frac_long', float('nan')):.2f}"
                     f" | {gd.fmt(m.get('heldout_chars_median'))} | {gd.fmt(m.get('baseline_chars_median'))} |")
    return "\n".join(lines), res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gepa_dose analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="comma-separated (default: every run with a modes.json)")
    ap.add_argument("--subset", default=None, help="run whose modes.json defines the question subset")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-mode", action="store_true")
    args = ap.parse_args(argv)
    link_references()
    runs = args.runs.split(",") if args.runs else sorted(p.parent.name for p in RUNS.glob("*/modes.json"))
    qids = qids_of(args.subset) if args.subset else None
    md, res = table(runs, args.boot, qids)
    head = f"# gepa_dose ({'subset of ' + str(len(qids)) + ' questions from ' + args.subset if qids else 'full split'}; honest metric, 95% bootstrap CIs over questions; %)\n\n"
    print(md)
    if args.per_mode:
        for name, m in res.items():
            if "per_mode_honest" in m:
                print(name, {k: round(100 * v, 1) for k, v in m["per_mode_honest"].items()})
    out = Path(args.out) if args.out else RESULTS / "table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(head + md + "\n")
    atomic_write_json(res, out.with_suffix(".json"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
