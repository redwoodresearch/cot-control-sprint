"""Metrics tables for candidate evals (Workstream B's definitions, shared by every elicit workstream so the
numbers are directly comparable): honest, honest_long, held-out, accuracy, chars with 95% bootstrap CIs over
questions.

    run.py elicit/<ws> analyze [--runs a,b,...] [--boot 1000] [--out results/<name>.md] [--per-mode] [--subset RUN]

Per rollout (cotcontrol eval JSON): honest = cotcontrol.eval.grading.is_honest; chars = len(reasoning).
honest_long = honest among traces at least the UNTRAINED median length for that (split, block): the untrained
reference is runs/untrained_val for val and `analysis.untrained_test` (the fragility checkpoint-0 test evals)
for test. Metrics: honest / honest_long / strict / chars on the 9-training-mode block, heldout (/ heldout_long)
on the 3 held-out modes, accuracy on the unconstrained block. Runs sharing a stem with _s<k> suffixes (seeds)
are pooled. Gain recovered = (candidate - untrained) / (teacher - untrained) on the same split.

`--subset RUN` restricts every run to the questions present in RUN's modes.json (the 40-question sweeps), so
full-split reference runs are comparable; the honest_long bar always comes from the FULL untrained reference.
`analysis.style`: `distill` (gain-recovered column, untrained rows added per split) or `dose` (heldout / baseline
chars columns, no automatic untrained rows). `analysis.refs` are symlinked into runs/ as reference rows.

Bootstrap: one `numpy.random.default_rng(analysis.bootstrap_seed)` per metric, resampling questions (clusters)
with replacement; this draw order is what the committed tables were made with, so it is kept here rather than
mapped onto kaushik/core/stats (whose matrix bootstrap consumes the generator differently).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from kaushik.core.io import ROOT, read_json
from kaushik.elicit.common import results_dir, runs_dir

BLOCK_FILES = {"modes": "modes.json", "heldout": "heldout.json", "baseline": "baseline.json"}
COLS = {"distill": ["honest", "honest_long", "heldout", "heldout_long", "accuracy", "strict"],
        "dose": ["honest", "honest_long", "heldout", "accuracy", "strict"]}


class Analysis:
    def __init__(self, cfg: dict):
        from cotcontrol.eval.grading import is_honest

        self.is_honest = is_honest
        self.cfg = cfg
        an = cfg["analysis"]
        self.runs = runs_dir(cfg)
        self.results = results_dir(cfg)
        self.style = an.get("style", "distill")
        self.seed = int(an.get("bootstrap_seed", 0))
        self.untrained_test = {b: ROOT / p for b, p in (an.get("untrained_test") or {}).items()}
        self.refs = {k: ROOT / v for k, v in (an.get("refs") or {}).items()}

    # ------------------------------------------------------------------ rows

    def link_references(self) -> None:
        """Expose the reference evals (read-only symlinks) under our runs/."""
        self.runs.mkdir(parents=True, exist_ok=True)
        for name, src in self.refs.items():
            dst = self.runs / name
            if src.is_dir() and not dst.exists():
                os.symlink(src, dst)

    def rollouts(self, path: Path) -> list[dict]:
        """[{qid, mode, honest, compliance, correct, chars}] for every non-error rollout of one eval JSON."""
        data = read_json(path)
        rows = []
        for rec in data["results"]:
            for s in rec["samples"]:
                if s["error"]:
                    continue
                graded = s["compliance"] is not None
                rows.append({"qid": f"{rec['dataset']}:{rec['id']}", "mode": rec["mode"],
                             "honest": float(self.is_honest(s)) if graded else np.nan,
                             "compliance": float(s["compliance"]) if graded else np.nan,
                             "correct": float(bool(s["correct"])), "chars": len(s["reasoning"] or "")})
        return rows

    def run_blocks(self, run: str) -> dict[str, Path]:
        d = self.runs / run
        return {b: d / f for b, f in BLOCK_FILES.items() if (d / f).exists()}

    def untrained_blocks(self, split: str) -> dict[str, Path]:
        if split == "val":
            return self.run_blocks("untrained_val")
        return {b: p for b, p in self.untrained_test.items() if p.exists()}

    def split_of(self, run: str) -> str:
        if run == "untrained_test":
            return "test"
        return read_json(self.runs / run / "manifest.json")["split"]

    def qids_of(self, run: str) -> set[str]:
        return {f"{r['dataset']}:{r['id']}" for r in read_json(self.runs / run / "modes.json")["results"]}

    # ------------------------------------------------------------------ bootstrap

    def boot_ci(self, rows: list[dict], metric: str, n_boot: int, subset=None) -> tuple[float, float, float]:
        """Mean and 95% percentile CI, resampling questions (clusters) with replacement; NaNs excluded."""
        by_q = defaultdict(list)
        for r in rows:
            if subset is not None and not subset(r):
                continue
            v = r[metric]
            if not np.isnan(v):
                by_q[r["qid"]].append(v)
        if not by_q:
            return np.nan, np.nan, np.nan
        sums = np.array([sum(v) for v in by_q.values()])
        cnts = np.array([len(v) for v in by_q.values()])
        mean = sums.sum() / cnts.sum()
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
        draws = sums[idx].sum(1) / cnts[idx].sum(1)
        return mean, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    def metrics(self, group: list[str], split: str, n_boot: int, qids: set[str] | None = None,
                label: list[str] | None = None) -> dict:
        """Pooled over the runs in `group` (seeds): {metric: (mean, lo, hi)} plus n, bars, medians."""
        ref = self.untrained_blocks(split)
        bars = {b: float(np.median([r["chars"] for r in self.rollouts(p)])) for b, p in ref.items() if b != "baseline"}
        pooled = defaultdict(list)
        for run in group:
            blocks = ref if run == "untrained_test" else self.run_blocks(run)
            for b, p in blocks.items():
                pooled[b].extend(r for r in self.rollouts(p) if qids is None or r["qid"] in qids)
        out = {"runs": label or group, "split": split, "bars": bars, "n": {b: len(v) for b, v in pooled.items()}}
        if pooled.get("modes"):
            rows = pooled["modes"]
            bar = bars.get("modes", 0)
            out["honest"] = self.boot_ci(rows, "honest", n_boot)
            out["honest_long"] = self.boot_ci(rows, "honest", n_boot, subset=lambda r: r["chars"] >= bar)
            out["strict"] = self.boot_ci(rows, "compliance", n_boot)
            out["acc_constrained"] = self.boot_ci(rows, "correct", n_boot)
            out["chars_median"] = float(np.median([r["chars"] for r in rows]))
            out["frac_long"] = float(np.mean([r["chars"] >= bar for r in rows]))
            out["per_mode_honest"] = {m: float(np.nanmean([r["honest"] for r in rows if r["mode"] == m]))
                                      for m in sorted({r["mode"] for r in rows})}
        if pooled.get("heldout"):
            rows = pooled["heldout"]
            out["heldout"] = self.boot_ci(rows, "honest", n_boot)
            if self.style == "distill":
                hbar = bars.get("heldout", 0)
                out["heldout_long"] = self.boot_ci(rows, "honest", n_boot, subset=lambda r: r["chars"] >= hbar)
            out["heldout_chars_median"] = float(np.median([r["chars"] for r in rows]))
        if pooled.get("baseline"):
            rows = pooled["baseline"]
            out["accuracy"] = self.boot_ci(rows, "correct", n_boot)
            out["baseline_chars_median"] = float(np.median([r["chars"] for r in rows]))
        return out

    # ------------------------------------------------------------------ table

    def table(self, groups: dict[str, list[str]], n_boot: int, qids: set[str] | None = None) -> tuple[str, dict]:
        res = {name: self.metrics(g, self.split_of(g[0]), n_boot, qids) for name, g in groups.items()}
        if self.style == "distill":
            for split in {m["split"] for m in res.values()}:
                key = f"untrained_{split}"
                if key not in res:
                    res[key] = (self.metrics(["untrained_val"], "val", n_boot) if split == "val"
                                else self.metrics(["untrained_test"], "test", n_boot, label=["fragility ckpt-0 (test)"]))
        cols = COLS[self.style]
        if self.style == "distill":
            lines = ["| run | split | n_modes | " + " | ".join(cols) + " | chars med (bar) | frac ≥ bar | gain rec. honest / long |",
                     "|" + "---|" * (len(cols) + 6)]
        else:
            lines = ["| run | split | n_modes | " + " | ".join(cols) + " | chars med (bar) | frac ≥ bar | heldout chars | baseline chars |",
                     "|" + "---|" * (len(cols) + 7)]
        for name, m in sorted(res.items(), key=lambda kv: (kv[1]["split"], kv[0])):
            split = m["split"]
            cells = f"| {name} | {split} | {m['n'].get('modes', 0)} | " + " | ".join(fmt(m.get(c)) for c in cols) + \
                f" | {fmt(m.get('chars_median'))} ({fmt(m['bars'].get('modes'))}) | {m.get('frac_long', float('nan')):.2f}"
            if self.style == "distill":
                u, t = res.get(f"untrained_{split}"), res.get(f"teacher_{split}")
                rec = "-"
                if u and t and "honest" in m and "honest" in t and name not in (f"untrained_{split}", f"teacher_{split}"):
                    rec = f"{gain(m, u, t, 'honest')} / {gain(m, u, t, 'honest_long')}"
                lines.append(cells + f" | {rec} |")
            else:
                lines.append(cells + f" | {fmt(m.get('heldout_chars_median'))} | {fmt(m.get('baseline_chars_median'))} |")
        return "\n".join(lines), res


def gain(m: dict, u: dict, t: dict, k: str) -> str:
    """Gain recovered: (candidate - untrained) / (teacher - untrained)."""
    den = t[k][0] - u[k][0]
    return f"{100 * (m[k][0] - u[k][0]) / den:.0f}%" if den > 0 else "n/a"


def fmt(t) -> str:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "-"
    if isinstance(t, tuple):
        if np.isnan(t[0]):
            return "-"
        return f"{100 * t[0]:.1f} [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"
    return f"{t:.0f}"


def group_runs(runs: list[str]) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for r in runs:
        groups[re.sub(r"_s\d+$", "", r)].append(r)
    return dict(groups)


def main(cfg: dict, argv: list[str]) -> None:
    an = Analysis(cfg)
    ap = argparse.ArgumentParser(prog="analyze", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="comma-separated run names (default: every run with a modes.json)")
    ap.add_argument("--boot", type=int, default=int(cfg["analysis"]["bootstrap_draws"]))
    ap.add_argument("--out", default=None, help="markdown path (default results/table.md)")
    ap.add_argument("--per-mode", action="store_true")
    ap.add_argument("--subset", default=None, help="run whose modes.json defines the question subset")
    args = ap.parse_args(argv)
    an.link_references()
    runs = args.runs.split(",") if args.runs else sorted(p.parent.name for p in an.runs.glob("*/modes.json"))
    qids = an.qids_of(args.subset) if args.subset else None
    md, res = an.table(group_runs(runs), args.boot, qids)
    print(md)
    if args.per_mode:
        for name, m in res.items():
            if "per_mode_honest" in m:
                scale = 100 if an.style == "dose" else 1
                print(name, {k: round(scale * v, 1 if an.style == "dose" else 3) for k, v in m["per_mode_honest"].items()})
    out = Path(args.out) if args.out else an.results / "table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if an.style == "distill":
        head = "# gepa_distill results (honest metric, 95% bootstrap CIs over questions; values in %)\n\n"
    else:
        head = (f"# gepa_dose ({'subset of ' + str(len(qids)) + ' questions from ' + args.subset if qids else 'full split'}; "
                "honest metric, 95% bootstrap CIs over questions; %)\n\n")
    out.write_text(head + md + "\n")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, default=str))
    os.replace(tmp, out.with_suffix(".json"))
    print(f"-> {out}")
