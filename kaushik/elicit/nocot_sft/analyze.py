"""Val (selection) and test (reporting) tables for the no-CoT SFT arms.

Per rollout: honest compliance (cotcontrol.eval.grading.is_honest), raw strict
compliance, accuracy, reasoning chars, degen (< 200 chars), and `long` = chars >= the
UNTRAINED median for that (split, block, mode). Endpoints (PLAN.md):
  honest       plain honest compliance
  honest_long  honest compliance among long traces (primary; among-long = conditional)
  correct      accuracy on the unconstrained baseline block (guardrail: within 3 pts)
  chars        median reasoning chars (guardrail: >= 80% of untrained)
Intervals: 95% two-way cluster bootstrap over questions x seeds (1000 draws).

References pulled in: untrained (val: ref1a checkpoint-0; test: fragility 1A seed-0
checkpoint-0), 1A (val: ref1a step 50; test: fragility seeds 0-2 step 50), noise_1A
(kaushik/noise fragility-mcq_noreason-gptoss20b, test) and our own noise groups
(kaushik/noise elicit-nocot_sft-<arm>-gptoss20b) when present.

    .venv/bin/python kaushik/run.py elicit/nocot_sft analyze [--n-boot 1000] [--split val|test|both]
-> results/val_table.md, results/test_table.md, results/rollouts.csv.gz, results/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from common import FRAG, RESULTS, ROOT, RUNS

from cotcontrol.eval.grading import is_honest

NOISE_RUNS = ROOT / "kaushik" / "noise" / "runs"
DEGEN_CHARS = 200
CI_METRICS = ["honest", "honest_long", "raw", "correct"]


def kind_of(block: str) -> str:
    return "heldout" if "heldout" in block else "baseline" if "baseline" in block else "modes"


def load_eval_json(path: Path, arm: str, seed: int, step, source: str) -> list[dict]:
    data = json.loads(path.read_text())
    split = data["config"].get("split") or "all"
    block = path.parent.name
    rows = []
    for rec in data["results"]:
        for s in rec["samples"]:
            if s.get("error"):
                continue
            chars = len(s.get("reasoning_text_graded") or s.get("reasoning") or "")
            graded = s.get("compliance") is not None
            rows.append({"arm": arm, "seed": seed, "step": step, "source": source, "split": split,
                         "block": block, "kind": kind_of(block), "mode": rec["mode"],
                         "qid": f"{rec['dataset']}:{rec['id']}",
                         "honest": float(is_honest(s)) if graded else np.nan,
                         "raw": float(s["compliance"] == 1) if graded else np.nan,
                         "correct": float(bool(s.get("correct"))), "chars": chars,
                         "degen": float(chars < DEGEN_CHARS)})
    return rows


def load_run_dir(run_dir: Path, arm: str, seed: int, source: str, steps=None) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "eval").glob("*/checkpoint-*.json")):
        step = int(path.stem.split("-")[1])
        if steps is not None and step not in steps:
            continue
        rows += load_eval_json(path, arm, seed, step, source)
    return rows


def load_noise_group(group_dir: Path, arm: str) -> list[dict]:
    rows = []
    for path in sorted((group_dir / "eval").glob("*/noise*.json")):
        k = int(re.search(r"noise(\d+)", path.stem)[1])
        rows += load_eval_json(path, arm, k, -1, str(group_dir.relative_to(ROOT)))
    return rows


def load_all() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "eval").is_dir() and p.name != "smoke"):
        cfg = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
        m = re.match(r"(?:nocot|ref1a)-(.+?)-gptoss20b-seed(\d+)", run_dir.name)
        if not m:
            continue
        arm, seed = m[1], int(m[2])
        if run_dir.name.startswith("ref1a"):
            for step, name in ((0, "untrained"), (50, "1A")):
                rows += load_run_dir(run_dir, name, seed, run_dir.name, steps=[step])
        else:
            rows += load_run_dir(run_dir, arm, seed, run_dir.name)
        print(f"loaded {run_dir.name} ({cfg.get('arm', arm)})")
    # test references from fragility (same protocol: T=1, 16k, medium, final_* blocks)
    for seed in (0, 1, 2):
        rd = FRAG / "runs" / f"fragility-mcq_noreason-gptoss20b-seed{seed}"
        if (rd / "eval").is_dir():
            rows += [r for r in load_run_dir(rd, "1A", seed, rd.name, steps=[50]) if r["block"].startswith("final")]
            if seed == 0:
                rows += [r for r in load_run_dir(rd, "untrained", 0, rd.name, steps=[0]) if r["block"].startswith("final")]
    g = NOISE_RUNS / "fragility-mcq_noreason-gptoss20b"
    if g.is_dir():
        rows += load_noise_group(g, "noise_1A")
    for g in sorted(NOISE_RUNS.glob("elicit-nocot_sft-*-gptoss20b")):
        rows += load_noise_group(g, "noise_" + g.name.split("elicit-nocot_sft-")[1].rsplit("-gptoss20b", 1)[0])
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no eval JSONs yet")
    add_long(df)
    return df


def add_long(df: pd.DataFrame) -> None:
    """long = chars >= untrained median per (split, kind, mode); fallback (split, kind), then (split)."""
    base = df[df.arm == "untrained"]
    keys = [["split", "kind", "mode"], ["split", "kind"], ["split"]]
    bars = [base.groupby(k)["chars"].median() for k in keys]

    def bar_for(row):
        for b, k in zip(bars, keys):
            idx = tuple(row[c] for c in k) if len(k) > 1 else row[k[0]]
            if idx in b.index:
                return b.loc[idx]
        return np.nan
    df["long_bar"] = df.apply(bar_for, axis=1)
    df["long"] = (df.chars >= df.long_bar).astype(float)
    df["honest_long"] = df.honest.where(df.long == 1)   # conditional on long
    df["honest_x_long"] = df.honest * df.long            # unconditional (honest AND long)


def boot_draws(df: pd.DataFrame, metric: str, n_boot: int, seed: int = 0) -> np.ndarray:
    sub = df.dropna(subset=[metric])
    if sub.empty:
        return np.full(n_boot, np.nan)
    sums = sub.pivot_table(index="seed", columns="qid", values=metric, aggfunc="sum", fill_value=0.0)
    cnts = sub.pivot_table(index="seed", columns="qid", values=metric, aggfunc="count", fill_value=0)
    a, c = sums.to_numpy(float), cnts.to_numpy(float)
    S, Q = a.shape
    rng = np.random.default_rng(seed)
    rows, cols = rng.integers(0, S, (n_boot, S)), rng.integers(0, Q, (n_boot, Q))

    def draw(x):
        return np.take_along_axis(x[rows], cols[:, None, :], axis=2).sum(axis=(1, 2))
    return draw(a) / np.maximum(draw(c), 1)


def summarize(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    out = []
    for (split, arm, step, kind), sub in df.groupby(["split", "arm", "step", "kind"], sort=True):
        row = {"split": split, "arm": arm, "step": step, "kind": kind, "n": len(sub),
               "n_seeds": sub.seed.nunique(), "chars": float(sub.chars.median()), "degen": float(sub.degen.mean()),
               "long_frac": float(sub.long.mean())}
        for m in CI_METRICS + ["honest_x_long"]:
            vals = sub[m].dropna()
            row[m] = float(vals.mean()) if len(vals) else np.nan
            if m in CI_METRICS and len(vals):
                lo, hi = np.percentile(boot_draws(sub, m, n_boot), [2.5, 97.5])
                row[f"{m}_lo"], row[f"{m}_hi"] = float(lo), float(hi)
        row["honest_by_seed"] = " ".join(f"{v:.3f}" for v in sub.groupby("seed")["honest"].mean().sort_index()
                                         if not np.isnan(v))
        out.append(row)
    return pd.DataFrame(out)


def cell(r, m):
    if np.isnan(r.get(m, np.nan)):
        return "–"
    s = f"{100 * r[m]:.1f}"
    if f"{m}_lo" in r and not np.isnan(r[f"{m}_lo"]):
        s += f" [{100 * r[f'{m}_lo']:.1f}, {100 * r[f'{m}_hi']:.1f}]"
    return s


def table(summary: pd.DataFrame, split: str) -> str:
    s = summary[summary.split == split]
    if s.empty:
        return f"(no {split} results yet)\n"
    arms = {a: i for i, a in enumerate(["untrained", "1A", "noise_1A", "gentle", "mix", "fmt", "fmt_mix"])}
    lines = [f"### {split} split (percent; 95% bootstrap CI over questions x seeds)", "",
             "| arm | step | seeds | modes honest | modes honest_long | held-out honest | held-out honest_long "
             "| baseline acc | median chars (modes) | long frac | degen | honest by seed |",
             "|" + "---|" * 12]
    for (arm, step), g in sorted(s.groupby(["arm", "step"]), key=lambda x: (arms.get(x[0][0], 99), x[0][1])):
        rows = {r["kind"]: r for r in g.to_dict("records")}
        md, ho, ba = rows.get("modes", {}), rows.get("heldout", {}), rows.get("baseline", {})
        if not md:
            continue
        lines.append(f"| {arm} | {'' if step < 0 else step} | {md['n_seeds']} | {cell(md, 'honest')} | {cell(md, 'honest_long')} "
                     f"| {cell(ho, 'honest') if ho else '–'} | {cell(ho, 'honest_long') if ho else '–'} "
                     f"| {cell(ba, 'correct') if ba else '–'} | {md['chars']:.0f} | {md['long_frac']:.2f} | {md['degen']:.2f} "
                     f"| {md['honest_by_seed']} |")
    return "\n".join(lines) + "\n"


def selection(summary: pd.DataFrame) -> dict:
    """Pre-registered rule (STATE.md): val, among our arms, highest honest_long whose
    guardrails pass (baseline acc within 3 pts of untrained; median chars >= 80%)."""
    v = summary[summary.split == "val"]
    u_md = v[(v.arm == "untrained") & (v.kind == "modes")]
    u_ba = v[(v.arm == "untrained") & (v.kind == "baseline")]
    if u_md.empty or u_ba.empty:
        return {"error": "untrained val reference missing"}
    u_chars, u_acc = float(u_md.chars.iloc[0]), float(u_ba.correct.iloc[0])
    cands = []
    for (arm, step), g in v[~v.arm.isin(["untrained", "1A"]) & ~v.arm.str.startswith("noise")].groupby(["arm", "step"]):
        rows = {r["kind"]: r for r in g.to_dict("records")}
        md, ba, ho = rows.get("modes"), rows.get("baseline"), rows.get("heldout", {})
        if not md or not ba:
            continue
        ok_acc, ok_len = ba["correct"] >= u_acc - 0.03, md["chars"] >= 0.8 * u_chars
        cands.append({"arm": arm, "step": int(step), "honest_long": md["honest_long"], "honest": md["honest"],
                      "heldout_honest_long": ho.get("honest_long", np.nan), "acc": ba["correct"], "chars": md["chars"],
                      "pass_acc": bool(ok_acc), "pass_len": bool(ok_len), "pass": bool(ok_acc and ok_len)})
    passing = [c for c in cands if c["pass"]]
    key = lambda c: (np.nan_to_num(c["honest_long"]), c["honest"], np.nan_to_num(c["heldout_honest_long"]))  # noqa: E731
    best = max(passing, key=key) if passing else None
    return {"untrained_val": {"chars": u_chars, "acc": u_acc}, "candidates": sorted(cands, key=key, reverse=True),
            "selected": best}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--split", default="both", choices=["val", "test", "both"])
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_all()
    df.to_csv(RESULTS / "rollouts.csv.gz", index=False)
    summary = summarize(df, args.n_boot)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    sel = selection(summary)
    (RESULTS / "selection.json").write_text(json.dumps(sel, indent=2, default=float))
    for split in (["val", "test"] if args.split == "both" else [args.split]):
        md = table(summary, split)
        (RESULTS / f"{split}_table.md").write_text(md)
        print(md)
    print("selection:", json.dumps(sel.get("selected"), default=float))
    for c in sel.get("candidates", []):
        print(f"  {c['arm']:<8} step {c['step']:>3}: honest_long {100 * np.nan_to_num(c['honest_long']):.1f} "
              f"honest {100 * c['honest']:.1f} acc {100 * c['acc']:.1f} chars {c['chars']:.0f} "
              f"pass={c['pass']} (acc {c['pass_acc']}, len {c['pass_len']})")


if __name__ == "__main__":
    main()
