"""Metrics for gepa_distill runs: honest, honest_long, held-out, accuracy, chars with 95% bootstrap CIs over questions.

    kaushik/run.py elicit/gepa_distill analyze [--runs a,b,...] [--boot 1000] [--out results/<name>.md]

Per rollout (cotcontrol eval JSON): honest = cotcontrol.eval.grading.is_honest; chars = len(reasoning).
honest_long = honest among traces at least the UNTRAINED median length for that (split, block): the untrained
reference is runs/untrained_val for val and the fragility checkpoint-0 test evals for test (gd_common.UNTRAINED_TEST).
Metrics: honest / honest_long / chars on the 9-training-mode block, heldout / heldout_long on the 3 held-out modes,
accuracy on the unconstrained block. Runs sharing a stem with _s<k> suffixes (seeds) are pooled.
Gain recovered = (candidate - untrained) / (teacher - untrained) on the same split.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from gd_common import RESULTS, ROOT, RUNS, UNTRAINED_TEST, load_json

sys.path.insert(0, str(ROOT))
from cotcontrol.eval.grading import is_honest  # noqa: E402

BLOCK_FILES = {"modes": "modes.json", "heldout": "heldout.json", "baseline": "baseline.json"}


def rollouts(path: Path) -> list[dict]:
    """[{qid, mode, honest, compliance, correct, chars, error}] for every rollout of one eval JSON."""
    data = load_json(path)
    rows = []
    for rec in data["results"]:
        for s in rec["samples"]:
            if s["error"]:
                continue
            rows.append({"qid": f"{rec['dataset']}:{rec['id']}", "mode": rec["mode"],
                         "honest": float(is_honest(s)) if s["compliance"] is not None else np.nan,
                         "compliance": float(s["compliance"]) if s["compliance"] is not None else np.nan,
                         "correct": float(bool(s["correct"])), "chars": len(s["reasoning"] or "")})
    return rows


def run_blocks(run: str) -> dict[str, Path]:
    d = RUNS / run
    return {b: d / f for b, f in BLOCK_FILES.items() if (d / f).exists()}


def untrained_blocks(split: str) -> dict[str, Path]:
    return run_blocks("untrained_val") if split == "val" else {b: p for b, p in UNTRAINED_TEST.items() if p.exists()}


def split_of(run: str) -> str:
    m = load_json(RUNS / run / "manifest.json")
    return m["split"]


def boot_ci(rows: list[dict], metric: str, n_boot: int, seed: int = 0, subset=None) -> tuple[float, float, float]:
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
    sums = np.array([sum(v) for v in by_q.values()]); cnts = np.array([len(v) for v in by_q.values()])
    mean = sums.sum() / cnts.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    draws = sums[idx].sum(1) / cnts[idx].sum(1)
    return mean, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def metrics(run_group: list[str], split: str, n_boot: int) -> dict:
    """Pooled over the runs in run_group (seeds). Returns {metric: (mean, lo, hi)} plus n."""
    ref = untrained_blocks(split)
    bars = {b: float(np.median([r["chars"] for r in rollouts(p)])) for b, p in ref.items() if b != "baseline"}
    pooled = defaultdict(list)
    for run in run_group:
        for b, p in run_blocks(run).items():
            pooled[b].extend(rollouts(p))
    out = {"runs": run_group, "split": split, "bars": bars, "n": {b: len(v) for b, v in pooled.items()}}
    if pooled.get("modes"):
        rows = pooled["modes"]
        out["honest"] = boot_ci(rows, "honest", n_boot)
        out["honest_long"] = boot_ci(rows, "honest", n_boot, subset=lambda r: r["chars"] >= bars.get("modes", 0))
        out["strict"] = boot_ci(rows, "compliance", n_boot)
        out["acc_constrained"] = boot_ci(rows, "correct", n_boot)
        out["chars_median"] = float(np.median([r["chars"] for r in rows]))
        out["frac_long"] = float(np.mean([r["chars"] >= bars.get("modes", 0) for r in rows]))
        out["per_mode_honest"] = {m: float(np.nanmean([r["honest"] for r in rows if r["mode"] == m]))
                                  for m in sorted({r["mode"] for r in rows})}
    if pooled.get("heldout"):
        rows = pooled["heldout"]
        out["heldout"] = boot_ci(rows, "honest", n_boot)
        out["heldout_long"] = boot_ci(rows, "honest", n_boot, subset=lambda r: r["chars"] >= bars.get("heldout", 0))
        out["heldout_chars_median"] = float(np.median([r["chars"] for r in rows]))
    if pooled.get("baseline"):
        rows = pooled["baseline"]
        out["accuracy"] = boot_ci(rows, "correct", n_boot)
        out["baseline_chars_median"] = float(np.median([r["chars"] for r in rows]))
    return out


def untrained_metrics(split: str, n_boot: int) -> dict:
    if split == "val":
        return metrics(["untrained_val"], "val", n_boot)
    ref = untrained_blocks("test")
    bars = {b: float(np.median([r["chars"] for r in rollouts(p)])) for b, p in ref.items() if b != "baseline"}
    pooled = {b: rollouts(p) for b, p in ref.items()}
    out = {"runs": ["fragility ckpt-0 (test)"], "split": "test", "bars": bars, "n": {b: len(v) for b, v in pooled.items()}}
    rows = pooled["modes"]
    out["honest"] = boot_ci(rows, "honest", n_boot)
    out["honest_long"] = boot_ci(rows, "honest", n_boot, subset=lambda r: r["chars"] >= bars["modes"])
    out["strict"] = boot_ci(rows, "compliance", n_boot)
    out["acc_constrained"] = boot_ci(rows, "correct", n_boot)
    out["chars_median"] = float(np.median([r["chars"] for r in rows]))
    out["frac_long"] = float(np.mean([r["chars"] >= bars["modes"] for r in rows]))
    out["per_mode_honest"] = {m: float(np.nanmean([r["honest"] for r in rows if r["mode"] == m])) for m in sorted({r["mode"] for r in rows})}
    rows = pooled["heldout"]
    out["heldout"] = boot_ci(rows, "honest", n_boot)
    out["heldout_long"] = boot_ci(rows, "honest", n_boot, subset=lambda r: r["chars"] >= bars["heldout"])
    out["heldout_chars_median"] = float(np.median([r["chars"] for r in rows]))
    rows = pooled["baseline"]
    out["accuracy"] = boot_ci(rows, "correct", n_boot)
    out["baseline_chars_median"] = float(np.median([r["chars"] for r in rows]))
    return out


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


def table(groups: dict[str, list[str]], n_boot: int) -> tuple[str, dict]:
    res = {}
    for name, g in groups.items():
        split = split_of(g[0])
        res[name] = metrics(g, split, n_boot)
    for split in {m["split"] for m in res.values()}:
        key = f"untrained_{split}"
        if key not in res:
            res[key] = untrained_metrics(split, n_boot)
    cols = ["honest", "honest_long", "heldout", "heldout_long", "accuracy", "strict"]
    lines = ["| run | split | n_modes | " + " | ".join(cols) + " | chars med (bar) | frac ≥ bar | gain rec. honest / long |",
             "|" + "---|" * (len(cols) + 6)]
    for name, m in sorted(res.items(), key=lambda kv: (kv[1]["split"], kv[0])):
        split = m["split"]
        u = res.get(f"untrained_{split}"); t = res.get(f"teacher_{split}")
        rec = "-"
        if u and t and "honest" in m and "honest" in t and name not in (f"untrained_{split}", f"teacher_{split}"):
            def r(k):
                den = t[k][0] - u[k][0]
                return f"{100 * (m[k][0] - u[k][0]) / den:.0f}%" if den > 0 else "n/a"
            rec = f"{r('honest')} / {r('honest_long')}"
        lines.append(f"| {name} | {split} | {m['n'].get('modes', 0)} | " + " | ".join(fmt(m.get(c)) for c in cols) +
                     f" | {fmt(m.get('chars_median'))} ({fmt(m['bars'].get('modes'))}) | {m.get('frac_long', float('nan')):.2f} | {rec} |")
    return "\n".join(lines), res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gepa_distill analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="comma-separated run names (default: every run with a modes.json)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=None, help="markdown path (default results/table.md)")
    ap.add_argument("--per-mode", action="store_true")
    args = ap.parse_args(argv)
    runs = args.runs.split(",") if args.runs else sorted(p.parent.name for p in RUNS.glob("*/modes.json"))
    md, res = table(group_runs(runs), args.boot)
    print(md)
    if args.per_mode:
        for name, m in res.items():
            if "per_mode_honest" in m:
                print(name, {k: round(v, 3) for k, v in m["per_mode_honest"].items()})
    out = Path(args.out) if args.out else RESULTS / "table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# gepa_distill results (honest metric, 95% bootstrap CIs over questions; values in %)\n\n" + md + "\n")
    from gd_common import atomic_write_json
    atomic_write_json(res, out.with_suffix(".json"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
