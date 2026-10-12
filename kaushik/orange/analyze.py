"""Experiment 2 table: orange SFT against its matched controls (config `analysis`).

Reads every per-rollout eval JSON (runs/<run>/eval/<block>/checkpoint-<step>.json)
from the folders in `analysis.sources` —

    kaushik/orange/runs       orange, orange_nopersona           (this folder)
    kaushik/fragility/runs    mcq_noreason                       (arm 1A, trained there)
    kaushik/orange/rl/runs    rl_orange_*, rl_mcq                (Experiment 2B)
    kaushik/noise/runs        noise: KL-matched random adapters  (per registry entry)

— and derives, per (model, split, condition, mode, arm, step): honest compliance
(cotcontrol.eval.grading.is_honest), raw strict compliance, accuracy, chars (mean
reasoning length), degen (traces < analysis.degen_chars) and compl_long (compliance
among traces at least as long as the untrained median for that model/condition/mode),
each with a 95% bootstrap interval over questions AND seeds (kaushik/core/stats.py:
matrix / boot_means, the same draw order as the committed tables), plus the seed
spread. `untrained` = checkpoint-0 of every SFT run (one resample per run).

The quantities of interest are differences with paired-question, independent-seed
bootstrap intervals (`analysis.differences`). Section "length-matched" repeats
orange-training/plotting.ipynb section 7: compliance at matched reasoning length
(>= analysis.matched_min_words words and by length band).

Conditions are inferred from each JSON's own config (EUGENE system prompt ->
persona; other system prompt -> direct; requirement in the user turn -> stated),
and the run name is parsed as <arm>-<model>-seed<k>[-...], so other agents'
layouts need not match block names exactly. Questions are keyed (dataset, id).

    .venv/bin/python kaushik/run.py orange analyze [--plot] [--boot 2000]

Outputs (results/): rollouts.csv.gz (every rollout, one row), cells.csv (every
table cell with intervals), differences.csv, length_matched.csv, table.md (the
Experiment 2 table), compliance_vs_step_<model>.png.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cotcontrol.eval.prompts import HELDOUT_MODES
from kaushik.core import stats as st
from kaushik.core.config import model_key_of, resolve, results_dir
from kaushik.core.results import load_rollouts
from kaushik.core.stats import METRICS, POOLED

EXP = "orange"
ROLLOUT_COLUMNS = ["experiment", "arm", "matched_arm", "model", "seed", "run", "block", "condition", "kind", "split",
                   "step", "mode", "qid", "compliance", "honest", "correct", "chars", "words", "truncated", "task"]
CELL_KEYS = ["model", "split", "condition", "kind", "mode", "arm", "step"]
DIFF_KEYS = ["model", "split", "condition", "kind", "mode"]
RUN_RE = re.compile(r"^(?P<arm>.+?)-(?P<model>[A-Za-z0-9_.]+)-seed(?P<seed>\d+)(?P<rest>.*)$")
NOISE_GROUP_RE = re.compile(r"^(?P<exp>[a-z]+(?:-[a-z]+)?)-(?P<arm>.+)-(?P<slug>[a-z0-9]+)$")


# ----------------------------------------------------------------------------- loading


def _known(arm: str, known_arms: list[str]) -> str:
    return next((k for k in sorted(known_arms, key=len, reverse=True) if arm == k or arm.endswith("-" + k)), arm)


def parse_run_name(name: str, known_arms: list[str]) -> tuple[str, int | None, str | None]:
    """'orange-qwen36_27b-seed3' -> ('orange', 3, None). Launcher prefixes such as
    'fragility-' or 'rl2b-' are stripped by matching the tail against known_arms;
    a '-noise<k>' suffix makes the arm 'noise[<base arm>]' (seed = k)."""
    m = RUN_RE.match(name)
    if not m:
        return name, None, None
    arm, seed, rest = _known(m["arm"], known_arms), int(m["seed"]), m["rest"]
    if "noise" in rest:
        k = re.search(r"noise[-_]?(\d+)", rest)
        return f"noise[{arm}]", (int(k.group(1)) if k else seed), arm
    return arm, seed, None


def parse_noise_group(run: str, stem: str, known_arms: list[str]) -> tuple[str, int | None, str]:
    """kaushik/noise layout: runs/<experiment>-<arm>-<modelslug>/eval/<block>/noise<k>.json
    -> ('noise[<arm>]', k, <arm>)."""
    m = NOISE_GROUP_RE.match(run)
    arm = _known(m["arm"], known_arms) if m else run
    k = re.search(r"noise[-_]?(\d+)", stem)
    return f"noise[{arm}]", (int(k.group(1)) if k else None), arm


def block_kind(cfg: dict) -> str:
    mode, allowed = cfg.get("mode"), cfg.get("allowed_modes")
    if mode == "baseline":
        return "baseline"
    if mode in ("all", "random"):
        return "heldout" if allowed and set(allowed) <= set(HELDOUT_MODES) else "standard"
    return "heldout" if mode in HELDOUT_MODES else "fixed"


def condition_of(cfg: dict) -> str:
    sp = cfg.get("system_prompt") or ""
    return "persona" if "EUGENE" in sp else "direct" if sp else "stated" if cfg.get("state_requirement", True) else "bare"


def make_extra(known_arms: list[str]):
    """Row hook for core.results.load_rollouts: experiment / arm / seed / model / condition / kind / split / step."""
    per_file: dict[Path, dict] = {}

    def extra(row: dict, cfg: dict, path: Path) -> dict:
        if path not in per_file:   # the same for every row of a file
            source, run = row["source"], row["run"]
            if source == "noise":
                arm, seed, matched = parse_noise_group(run, path.stem, known_arms)
                step = "noise"
            else:
                arm, seed, matched = parse_run_name(run, known_arms)
                step = int(row["step"]) if row["step"].isdigit() else row["step"]
            per_file[path] = {"experiment": source, "arm": arm, "matched_arm": matched,
                              "model": model_key_of(row["base_model"]) or row["base_model"].split("/")[-1].lower(),
                              "seed": seed if seed is not None else run, "condition": condition_of(cfg),
                              "kind": block_kind(cfg), "split": cfg.get("split") or "all", "step": step}
        return per_file[path]
    return extra


def load(sources: dict[str, Path], cache_dir: Path, known_arms: list[str]) -> pd.DataFrame:
    df = load_rollouts(sources, cache_dir, extra=make_extra(known_arms), patterns=("*/eval/*/*.json",))
    return df[ROLLOUT_COLUMNS] if len(df) else df


# ----------------------------------------------------------------------------- statistics


def untrained_bars(df: pd.DataFrame, sft_arms: list[str]) -> dict:
    """Median reasoning chars of the untrained model per (model, condition, mode), from
    checkpoint-0 of the SFT runs (val); test blocks fall back to the (model, condition) bar."""
    return st.untrained_bars(df[(df.step == 0) & df.arm.isin(sft_arms)])


def with_pooled_and_untrained(df: pd.DataFrame, sft_arms: list[str]) -> pd.DataFrame:
    """Add the 'untrained' arm (step-0 rollouts of every SFT run, seed := run) and
    mode='(pooled)' copies of every non-baseline row."""
    untrained = df[(df.step == 0) & df.arm.isin(sft_arms)].assign(arm="untrained", seed=lambda d: d.run)
    full = pd.concat([df, untrained], ignore_index=True)
    pooled = full[full.kind != "baseline"].assign(mode=POOLED)
    return pd.concat([full, pooled], ignore_index=True)


def cell_stats(g: pd.DataFrame, rng: np.random.Generator, B: int, degen_chars: int) -> dict:
    out = {"n": len(g), "n_seeds": g.seed.nunique(), "accuracy": g.correct.mean(),
           "chars": g.chars.mean(), "degen": (g.chars < degen_chars).mean(),
           "truncated": g.truncated.mean(), "n_long": int(g.long.sum())}
    for metric in METRICS:
        M = st.matrix(g, metric)
        out[metric] = st.nanmean(M)
        out[f"{metric}_lo"], out[f"{metric}_hi"] = st.ci(st.boot_means(M, rng, B))
        per_seed = g.groupby("seed")[metric].mean()
        out[f"{metric}_seed_min"], out[f"{metric}_seed_max"] = per_seed.min(), per_seed.max()
    return out


def summarize_cells(df: pd.DataFrame, rng: np.random.Generator, B: int, degen_chars: int) -> pd.DataFrame:
    return pd.DataFrame([{**dict(zip(CELL_KEYS, key)), **cell_stats(g, rng, B, degen_chars)}
                         for key, g in df.groupby(CELL_KEYS, sort=False, dropna=False)])


def differences(df: pd.DataFrame, rng: np.random.Generator, B: int, pairs: list[tuple[str, str]],
                final_only: bool = True) -> pd.DataFrame:
    """A - B per (model, split, condition, kind, mode) at the last checkpoint of each arm,
    with paired-question, independent-seed bootstrap intervals (core.stats.paired_boot)."""
    rows = []
    for a, b in pairs:
        da, db = df[df.arm == a], df[df.arm == b]
        if da.empty or db.empty:
            continue
        for key, ga in da.groupby(DIFF_KEYS, sort=False):
            gb = db[(db[DIFF_KEYS] == pd.Series(key, index=DIFF_KEYS)).all(axis=1)]
            if gb.empty:
                continue
            steps_a, steps_b = sorted(ga.step.unique(), key=str), sorted(gb.step.unique(), key=str)
            if final_only:
                steps_a, steps_b = [steps_a[-1]], [steps_b[-1]]
            for sa in steps_a:
                sb = sa if sa in steps_b else steps_b[-1]
                xa, xb = ga[ga.step == sa], gb[gb.step == sb]
                rows.append({**dict(zip(DIFF_KEYS, key)), "a": a, "b": b, "step_a": sa, "step_b": sb,
                             "n_a": len(xa), "n_b": len(xb), "seeds_a": xa.seed.nunique(), "seeds_b": xb.seed.nunique(),
                             **st.paired_boot(xa, xb, rng, B)})
    return pd.DataFrame(rows)


def length_matched(df: pd.DataFrame, rng: np.random.Generator, B: int, pairs: list[tuple[str, str]],
                   bands: list[tuple[int, int | None]], min_words: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """orange-training section 7: compliance by reasoning-length band and the >= min_words
    'length-matched' compliance, pooled over trained checkpoints (steps > 0) and the fixed val
    modes; plus matched differences for the pairs of interest."""
    t = df[(df.kind == "fixed") & (df.split == "val") & (df["mode"] != POOLED)
           & (df.step.map(lambda s: isinstance(s, (int, np.integer)) and s > 0))]
    rows, diffs = [], []
    for (model, cond, arm), g in t.groupby(["model", "condition", "arm"], sort=False):
        w = g.words.clip(lower=1)
        row = {"model": model, "condition": cond, "arm": arm, "n": len(g), "raw": g.compliance.mean(),
               f"matched_{min_words}w": g[w >= min_words].compliance.mean(), f"frac_under_{min_words}w": (w < min_words).mean(),
               f"compliance_under_{min_words}w": g[w < min_words].compliance.mean(), "honest": g.honest.mean()}
        for lo, hi in bands:
            sel = (w >= lo) & ((w < hi) if hi is not None else True)
            label = f"{lo}-{hi}w" if hi is not None else f"{lo}+w"
            row[label] = g[sel].compliance.mean() if sel.any() else np.nan
            row[f"n_{label}"] = int(sel.sum())
        rows.append(row)
    for a, b in pairs:
        for (model, cond), g in t.groupby(["model", "condition"], sort=False):
            xa, xb = g[(g.arm == a) & (g.words >= min_words)], g[(g.arm == b) & (g.words >= min_words)]
            if xa.empty or xb.empty:
                continue
            tasks = sorted(set(xa.task) | set(xb.task))
            cols = rng.integers(0, len(tasks), (B, len(tasks)))
            Ma, Mb = st.matrix(xa, "compliance", tasks), st.matrix(xb, "compliance", tasks)
            lo, hi = st.ci(st.boot_means(Ma, rng, B, cols) - st.boot_means(Mb, rng, B, cols))
            diffs.append({"model": model, "condition": cond, "a": a, "b": b,
                          f"matched_{min_words}w_diff": float(np.nanmean(Ma) - np.nanmean(Mb)), "lo": lo, "hi": hi,
                          "n_a": len(xa), "n_b": len(xb)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


# ----------------------------------------------------------------------------- reporting


def arm_sorter(arm_order: list[str]):
    def key(arm: str) -> tuple:
        k = "noise" if arm.startswith("noise") else arm
        return (arm_order.index(k) if k in arm_order else len(arm_order), arm)
    return key


def final_step_of(cells: pd.DataFrame, arm: str, model: str, split: str) -> object:
    s = cells[(cells.arm == arm) & (cells.model == model) & (cells.split == split)].step
    ints = [x for x in s.unique() if isinstance(x, (int, np.integer))]
    return max(ints) if ints else (s.iloc[0] if len(s) else None)


def _kind_cell(r: pd.DataFrame, kind: str, m: str) -> str:
    x = r[r.kind == kind]
    return st.pct_ci(x.iloc[0], m) if len(x) else "-"


def write_table_md(cells: pd.DataFrame, diffs: pd.DataFrame, lm: pd.DataFrame, lm_diffs: pd.DataFrame, out: Path,
                   val_modes: list[str], arm_order: list[str], degen_chars: int, min_words: int) -> str:
    pct, pct_ci, md_table, arm_sort = st.pct, st.pct_ci, st.md_table, arm_sorter(arm_order)
    conds = ("persona", "direct", "stated")
    parts = ["# Experiment 2 table\n",
             ("Honest = honest compliance (`is_honest`), raw = strict compliance, compl_long = compliance "
              "among traces at least as long as the untrained median, acc = accuracy, chars = mean reasoning "
              f"length, degen = traces < {degen_chars} chars. Brackets: 95% bootstrap over questions and seeds; "
              "seed range = min–max of per-seed honest compliance. Each arm at its final checkpoint; "
              "`untrained` = checkpoint-0 of every SFT run.\n")]
    for model in sorted(cells.model.unique()):
        for split in ("val", "test"):
            sub = cells[(cells.model == model) & (cells.split == split) & (cells.kind == "fixed")]
            if sub.empty:
                continue
            parts.append(f"\n## {model} — {split} split, fixed-trait modes pooled ({', '.join(val_modes)})\n")
            rows = []
            for cond in conds:
                for arm in sorted(sub.arm.unique(), key=arm_sort):
                    step = final_step_of(sub, arm, model, split)
                    r = sub[(sub.condition == cond) & (sub.arm == arm) & (sub["mode"] == POOLED) & (sub.step == step)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    rows.append([cond, arm, str(step), f"{r.n_seeds}", pct_ci(r, "honest"),
                                 f"{pct(r.honest_seed_min)}–{pct(r.honest_seed_max)}", pct_ci(r, "compliance"),
                                 pct_ci(r, "compl_long"), pct_ci(r, "correct"), f"{r.chars:.0f}", pct(r.degen)])
            parts.append(md_table(["condition", "arm", "step", "seeds", "honest", "seed range", "raw",
                                   "compl_long", "acc", "chars", "degen"], rows))
            rows = []
            for cond in conds:
                for arm in sorted(sub.arm.unique(), key=arm_sort):
                    step = final_step_of(sub, arm, model, split)
                    r = sub[(sub.condition == cond) & (sub.arm == arm) & (sub.step == step)]
                    if r.empty:
                        continue
                    by_mode = {m: pct_ci(r[r["mode"] == m].iloc[0], "honest") if (r["mode"] == m).any() else "-"
                               for m in val_modes}
                    rows.append([cond, arm, *by_mode.values()])
            parts.append(f"\nPer-mode honest compliance ({split}, final step):\n")
            parts.append(md_table(["condition", "arm", *val_modes], rows))
        std = cells[(cells.model == model) & ((cells.kind.isin(["standard", "heldout"]) & (cells["mode"] == POOLED))
                                              | (cells.kind == "baseline"))]
        if not std.empty:
            parts.append(f"\n## {model} — standard stated eval (test, final checkpoint)\n")
            rows = []
            for arm in sorted(std.arm.unique(), key=arm_sort):
                step = final_step_of(std, arm, model, "test")
                r = std[(std.arm == arm) & (std.step == step)]
                rows.append([arm, str(step), *[_kind_cell(r, k, m) for k, m in (
                    ("standard", "honest"), ("standard", "compliance"), ("standard", "correct"),
                    ("heldout", "honest"), ("heldout", "compliance"), ("heldout", "correct"),
                    ("baseline", "correct"))]])
            parts.append(md_table(["arm", "step", "9-mode honest", "9-mode raw", "9-mode acc", "held-out honest",
                                   "held-out raw", "held-out acc", "baseline acc"], rows))
    if not diffs.empty:
        parts.append("\n## Quantities of interest (A − B, final checkpoints, modes pooled)\n")
        d = diffs[diffs["mode"] == POOLED]
        rows = [[r.model, r.split, r.kind, r.condition, f"{r.a} − {r.b}", f"{r.step_a}/{r.step_b}",
                 f"{r.seeds_a}/{r.seeds_b}", pct_ci(r, "honest"), pct_ci(r, "compliance"),
                 pct_ci(r, "compl_long"), pct_ci(r, "correct")]
                for _, r in d.sort_values(["model", "split", "kind", "condition", "a", "b"]).iterrows()]
        parts.append(md_table(["model", "split", "kind", "condition", "difference", "steps", "seeds",
                               "Δ honest", "Δ raw", "Δ compl_long", "Δ acc"], rows))
    if not lm.empty:
        parts.append("\n## Length-matched comparison (val, steps 10–50 pooled, fixed modes)\n")
        parts.append(f"Raw vs compliance among traces ≥ {min_words} words (orange-training section 7), and "
                     "compliance by reasoning-length band (words).\n")
        band_cols = [c for c in lm.columns if re.fullmatch(r"\d+(-\d+|\+)w", c)]
        rows = [[r.model, r.condition, r.arm, pct(r.raw), pct(r[f"matched_{min_words}w"]), pct(r[f"frac_under_{min_words}w"]),
                 pct(r[f"compliance_under_{min_words}w"]), *[f"{pct(r[c])} (n={r['n_' + c]})" for c in band_cols]]
                for _, r in lm.sort_values(["model", "condition", "arm"],
                                           key=lambda s: s.map(arm_sort) if s.name == "arm" else s).iterrows()]
        parts.append(md_table(["model", "condition", "arm", "raw", f"matched ≥{min_words}w", f"% <{min_words}w",
                               f"compl <{min_words}w", *band_cols], rows))
        if not lm_diffs.empty:
            rows = [[r.model, r.condition, f"{r.a} − {r.b}", f"{pct(r[f'matched_{min_words}w_diff'])} [{pct(r.lo)}, {pct(r.hi)}]"]
                    for _, r in lm_diffs.iterrows()]
            parts.append(f"\nLength-matched (≥ {min_words} words) differences:\n")
            parts.append(md_table(["model", "condition", "difference", "Δ matched compliance"], rows))
    text = "\n".join(parts) + "\n"
    out.write_text(text)
    return text


def plot_curves(cells: pd.DataFrame, out_dir: Path, val_modes: list[str], arm_order: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils import PALETTE, set_matplotlib_style

    set_matplotlib_style()
    arm_sort = arm_sorter(arm_order)
    for model in sorted(cells.model.unique()):
        sub = cells[(cells.model == model) & (cells.split == "val") & (cells.kind == "fixed")
                    & cells.step.map(lambda s: isinstance(s, (int, np.integer)))]
        if sub.empty:
            continue
        modes = [POOLED] + list(val_modes)
        conds = ["persona", "direct", "stated"]
        fig, axes = plt.subplots(len(conds), len(modes), figsize=(3.2 * len(modes), 2.6 * len(conds)),
                                 sharex=True, sharey=True, squeeze=False)
        arms = [a for a in sorted(sub.arm.unique(), key=arm_sort) if a != "untrained"]
        for i, cond in enumerate(conds):
            for j, mode in enumerate(modes):
                ax = axes[i][j]
                for k, arm in enumerate(arms):
                    s = sub[(sub.condition == cond) & (sub["mode"] == mode) & (sub.arm == arm)].sort_values("step")
                    if s.empty:
                        continue
                    color = PALETTE[k % len(PALETTE)]
                    x = s.step.astype(int).to_numpy()
                    h, c, lo, hi = (100 * s[k].to_numpy(dtype=float)
                                    for k in ("honest", "compliance", "honest_seed_min", "honest_seed_max"))
                    ax.plot(x, h, marker="o", color=color, label=f"{arm} honest")
                    ax.plot(x, c, ls=":", lw=1, color=color, label=f"{arm} raw")
                    ax.fill_between(x, lo, hi, color=color, alpha=0.12)
                ax.set_title(f"{cond} / {mode}", fontsize=8)
                if i == len(conds) - 1:
                    ax.set_xlabel("SFT step")
                if j == 0:
                    ax.set_ylabel("compliance (%)")
        axes[0][0].legend(fontsize=6)
        fig.suptitle(f"{model}: compliance vs step (val; shading = seed range of honest compliance)",
                     x=0.02, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout()
        path = out_dir / f"compliance_vs_step_{model}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path}")


# ----------------------------------------------------------------------------- main


def main(cfg: dict, argv: list[str] | None = None) -> None:
    an = cfg["analysis"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=None, help="override analysis.sources.orange (e.g. runs/smoke)")
    ap.add_argument("--no-extra", action="store_true", help="only this folder's runs")
    ap.add_argument("--out", default=str(results_dir(cfg)))
    ap.add_argument("--boot", type=int, default=int(an["bootstrap_draws"]), help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=int(an["bootstrap_seed"]))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    sources = {k: ROOT / v for k, v in an["sources"].items()}
    if args.runs_root:
        sources["orange"] = Path(args.runs_root)
    if args.no_extra:
        sources = {"orange": sources["orange"]}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sft_arms, pairs = list(an["sft_arms"]), [tuple(p) for p in an["differences"]]
    degen, min_words = int(an["degen_chars"]), int(an["matched_min_words"])
    bands = [(int(lo), None if hi is None else int(hi)) for lo, hi in an["length_bands_words"]]

    df = load(sources, out_dir / ".cache", list(an["known_arms"]))
    if df.empty:
        raise SystemExit(f"no eval JSONs under {[str(p) for p in sources.values()]}")
    print(f"{len(df):,} rollouts from {df.run.nunique()} runs; arms {sorted(df.arm.unique())}; "
          f"models {sorted(df.model.unique())}")
    df.to_csv(out_dir / "rollouts.csv.gz", index=False)

    df = st.add_long(df, untrained_bars(df, sft_arms))
    full = with_pooled_and_untrained(df, sft_arms)
    cells = summarize_cells(full, rng, args.boot, degen)
    cells.to_csv(out_dir / "cells.csv", index=False)
    diffs = differences(full, rng, args.boot, pairs)
    diffs.to_csv(out_dir / "differences.csv", index=False)
    lm, lm_diffs = length_matched(full, rng, args.boot, pairs, bands, min_words)
    lm.to_csv(out_dir / "length_matched.csv", index=False)
    lm_diffs.to_csv(out_dir / "length_matched_differences.csv", index=False)
    text = write_table_md(cells, diffs, lm, lm_diffs, out_dir / "table.md", list(an["val_modes"]), list(an["arm_order"]),
                          degen, min_words)
    print(text)
    if args.plot:
        plot_curves(cells, out_dir, list(an["val_modes"]), list(an["arm_order"]))


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
