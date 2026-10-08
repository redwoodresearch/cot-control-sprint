"""Experiment 2 table: orange SFT against its matched controls.

Reads every per-rollout eval JSON (runs/<run>/eval/<block>/checkpoint-<step>.json)
from the sprint folders that hold Experiment 2 arms —

    kaushik/orange/runs       orange, orange_nopersona           (this folder)
    kaushik/fragility/runs    mcq_noreason                       (arm 1A, trained there)
    kaushik/orange/rl/runs    rl_orange_*, rl_mcq                (Experiment 2B)
    kaushik/noise/runs        noise: KL-matched random adapters  (per registry entry)

— and derives, per (model, split, condition, mode, arm, step): honest compliance
(cotcontrol.eval.grading.is_honest), raw strict compliance, accuracy, chars (mean
reasoning length), degen (traces < 200 chars) and compl_long (compliance among
traces at least as long as the untrained median for that model/condition/mode),
each with a 95% bootstrap interval over questions AND seeds, plus the seed
spread. `untrained` = checkpoint-0 of every SFT run (one resample per run).

The quantities of interest are differences with paired-question, independent-
seed bootstrap intervals: (orange - orange_nopersona), (orange - mcq_noreason),
(rl_orange_* - rl_mcq). (orange - untrained) is deliberately not a headline.
Section "length-matched" repeats orange-training/plotting.ipynb section 7:
compliance at matched reasoning length (>= 50 words and by length band).

Conditions are inferred from each JSON's own config (EUGENE system prompt ->
persona; other system prompt -> direct; requirement in the user turn -> stated),
and the run name is parsed as <arm>-<model>-seed<k>[-...], so other agents'
layouts need not match block names exactly.

    .venv/bin/python kaushik/run.py orange analyze [--plot] [--boot 2000]

Outputs (results/): rollouts.csv.gz (every rollout, one row), cells.csv (every
table cell with intervals), differences.csv, length_matched.csv, table.md (the
Experiment 2 table), compliance_vs_step_<model>.png.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from cotcontrol.eval.grading import is_honest
from cotcontrol.eval.prompts import HELDOUT_MODES

SOURCES = {
    "orange": HERE / "runs",
    "fragility": ROOT / "kaushik" / "fragility" / "runs",
    "rl": HERE / "rl" / "runs",
    "noise": ROOT / "kaushik" / "noise" / "runs",
}
MODEL_KEYS = {"Qwen/Qwen3.6-27B": "qwen36_27b", "openai/gpt-oss-20b": "gptoss20b",
              "Qwen/Qwen3-8B": "qwen8b", "Qwen/Qwen3-32B": "qwen32b"}
SFT_ARMS = ("orange", "orange_nopersona", "mcq_noreason")
KNOWN_ARMS = ["orange", "orange_nopersona", "mcq_noreason", "mcq_reason", "style_sft",
              "rl_orange_noreason", "rl_orange_reason", "rl_mcq_noreason", "rl_mcq_reason"]
# (A, B) pairs reported as A - B. noise[<arm>] = KL-matched random adapters of that arm.
DIFFERENCES = [("orange", "orange_nopersona"), ("orange", "mcq_noreason"),
               ("rl_orange_noreason", "rl_mcq_noreason"), ("rl_orange_reason", "rl_mcq_reason"),
               ("orange", "noise[orange]"), ("orange_nopersona", "noise[orange_nopersona]"),
               ("mcq_noreason", "noise[mcq_noreason]")]
ARM_ORDER = ["untrained", "noise", "orange", "orange_nopersona", "mcq_noreason",
             "rl_orange_noreason", "rl_orange_reason", "rl_mcq_noreason", "rl_mcq_reason"]
VAL_MODES = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "end_of_sentence"]
DEGEN_CHARS = 200
POOLED = "(pooled)"
BANDS = [(0, 50), (50, 200), (200, 800), (800, 3200), (3200, 10 ** 9)]


# ----------------------------------------------------------------------------- loading

def parse_run_name(name: str, experiment: str) -> tuple[str, int | None, str | None]:
    """'orange-qwen36_27b-seed3' -> ('orange', 3, None). Launcher prefixes such as
    'fragility-' or 'rl2b-' are stripped by matching the tail against KNOWN_ARMS;
    a '-noise<k>' suffix makes the arm 'noise[<base arm>]' (seed = k)."""
    m = re.match(r"^(?P<arm>.+?)-(?P<model>[A-Za-z0-9_.]+)-seed(?P<seed>\d+)(?P<rest>.*)$", name)
    if not m:
        return name, None, None
    arm, seed, rest = m["arm"], int(m["seed"]), m["rest"]
    arm = next((k for k in sorted(KNOWN_ARMS, key=len, reverse=True)
                if arm == k or arm.endswith("-" + k)), arm)
    if "noise" in rest:
        k = re.search(r"noise[-_]?(\d+)", rest)
        return f"noise[{arm}]", (int(k.group(1)) if k else seed), arm
    return arm, seed, None


def parse_noise_group(run: str, stem: str) -> tuple[str, int | None, str]:
    """kaushik/noise layout: runs/<experiment>-<arm>-<modelslug>/eval/<block>/noise<k>.json
    -> ('noise[<arm>]', k, <arm>)."""
    m = re.match(r"^(?P<exp>[a-z]+(?:-[a-z]+)?)-(?P<arm>.+)-(?P<slug>[a-z0-9]+)$", run)
    arm = next((k for k in sorted(KNOWN_ARMS, key=len, reverse=True)
                if m and (m["arm"] == k or m["arm"].endswith("-" + k))), m["arm"] if m else run)
    k = re.search(r"noise[-_]?(\d+)", stem)
    return f"noise[{arm}]", (int(k.group(1)) if k else None), arm


def block_kind(cfg: dict) -> str:
    mode, allowed = cfg.get("mode"), cfg.get("allowed_modes")
    if mode == "baseline":
        return "baseline"
    if mode in ("all", "random"):
        return "heldout" if allowed and set(allowed) <= set(HELDOUT_MODES) else "standard"
    return "heldout" if mode in HELDOUT_MODES else "fixed"


def parse_file(path: Path, experiment: str) -> list[dict]:
    run, block = path.parents[2].name, path.parent.name
    data = json.loads(path.read_text())
    cfg = data["config"]
    base_model = cfg["model"]
    sp = cfg.get("system_prompt") or ""
    condition = ("persona" if "EUGENE" in sp else "direct" if sp
                 else "stated" if cfg.get("state_requirement", True) else "bare")
    step = path.stem.replace("checkpoint-", "")
    if experiment == "noise":
        arm, seed, matched = parse_noise_group(run, path.stem)
        step = "noise"
    else:
        arm, seed, matched = parse_run_name(run, experiment)
    base = {
        "experiment": experiment, "arm": arm, "matched_arm": matched,
        "model": MODEL_KEYS.get(base_model, base_model.split("/")[-1].lower()),
        "seed": seed if seed is not None else run, "run": run, "block": block,
        "condition": condition, "kind": block_kind(cfg), "split": cfg.get("split") or "all",
        "step": int(step) if step.isdigit() else step,
    }
    rows = []
    for rec in data["results"]:
        for s in rec["samples"]:
            if s["error"]:
                continue
            text = s.get("reasoning_text_graded") or ""
            rows.append({**base, "mode": rec["mode"], "qid": str(rec["id"]),
                         "compliance": s["compliance"] == 1, "honest": is_honest(s),
                         "correct": s["correct"] is True, "chars": len(text),
                         "words": len(text.split()), "truncated": s.get("finish_reason") == "length"})
    return rows


def load_rollouts(sources: dict[str, Path], cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment, root in sources.items():
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/eval/*/*.json")):
            if "smoke" in path.parents[2].name:
                continue
            cache = cache_dir / f"{experiment}__{path.parents[2].name}__{path.parent.name}__{path.stem}.pkl"
            mtime = path.stat().st_mtime
            if cache.exists():
                c = pickle.loads(cache.read_bytes())
                if c["mtime"] == mtime:
                    rows += c["rows"]
                    continue
            try:
                parsed = parse_file(path, experiment)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"skipping {path}: {e}")
                continue
            cache.write_bytes(pickle.dumps({"mtime": mtime, "rows": parsed}))
            rows += parsed
    df = pd.DataFrame(rows)
    if len(df):
        df["task"] = df["qid"] + "|" + df["mode"]
    return df


# ----------------------------------------------------------------------------- statistics

def _matrix(g: pd.DataFrame, value: str, tasks: list[str] | None = None) -> np.ndarray:
    """seeds x tasks matrix of a metric (NaN where missing), seed = one training run."""
    m = g.pivot_table(index="seed", columns="task", values=value, aggfunc="mean")
    if tasks is not None:
        m = m.reindex(columns=tasks)
    return m.to_numpy(dtype=float)


def boot_means(M: np.ndarray, rng: np.random.Generator, B: int, cols: np.ndarray | None = None,
               batch: int = 200) -> np.ndarray:
    """B bootstrap means resampling seeds (rows) and questions (cols) with replacement.
    `cols` (B x Q) lets two arms share the question resample (paired differences)."""
    S, Q = M.shape
    if cols is None:
        cols = rng.integers(0, Q, (B, Q))
    out = np.empty(B)
    for i in range(0, B, batch):
        rows = rng.integers(0, S, (min(batch, B - i), S))
        samp = M[rows[:, :, None], cols[i:i + len(rows), None, :]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN resample -> NaN
            out[i:i + len(rows)] = np.nanmean(samp.reshape(len(rows), -1), axis=1)
    return out


def ci(values: np.ndarray) -> tuple[float, float]:
    v = values[~np.isnan(values)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def untrained_bars(df: pd.DataFrame) -> dict:
    """Median reasoning chars of the untrained model per (model, condition, mode),
    from checkpoint-0 of the SFT runs (val); test blocks fall back to the val bar."""
    u = df[(df.step == 0) & df.arm.isin(SFT_ARMS)]
    bars = u.groupby(["model", "condition", "mode"]).chars.median().to_dict()
    bars.update({k[:2]: v for k, v in u.groupby(["model", "condition"]).chars.median().to_dict().items()})
    return bars


def add_long(df: pd.DataFrame, bars: dict) -> pd.DataFrame:
    bar = [bars.get((m, c, mo), bars.get((m, c), np.nan)) for m, c, mo in zip(df.model, df.condition, df["mode"])]
    df = df.assign(bar=bar)
    df["long"] = df.chars >= df.bar
    df["compl_long"] = np.where(df.long, df.compliance.astype(float), np.nan)
    return df


def with_pooled_and_untrained(df: pd.DataFrame) -> pd.DataFrame:
    """Add mode='(pooled)' copies (non-baseline) and the 'untrained' arm (step-0 rollouts
    of every SFT run, seed := run)."""
    untrained = df[(df.step == 0) & df.arm.isin(SFT_ARMS)].assign(arm="untrained", seed=lambda d: d.run)
    full = pd.concat([df, untrained], ignore_index=True)
    pooled = full[full.kind != "baseline"].assign(mode=POOLED)
    return pd.concat([full, pooled], ignore_index=True)


CELL_KEYS = ["model", "split", "condition", "kind", "mode", "arm", "step"]


def cell_stats(g: pd.DataFrame, rng: np.random.Generator, B: int) -> dict:
    out = {"n": len(g), "n_seeds": g.seed.nunique(), "accuracy": g.correct.mean(),
           "chars": g.chars.mean(), "degen": (g.chars < DEGEN_CHARS).mean(),
           "truncated": g.truncated.mean(), "n_long": int(g.long.sum())}
    for metric in ("honest", "compliance", "compl_long", "correct"):
        M = _matrix(g, metric)
        with np.errstate(all="ignore"):
            out[metric] = float(np.nanmean(M)) if np.isfinite(M).any() else np.nan
        lo, hi = ci(boot_means(M, rng, B))
        out[f"{metric}_lo"], out[f"{metric}_hi"] = lo, hi
        per_seed = g.groupby("seed")[metric].mean()
        out[f"{metric}_seed_min"], out[f"{metric}_seed_max"] = per_seed.min(), per_seed.max()
    return out


def summarize_cells(df: pd.DataFrame, rng: np.random.Generator, B: int) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(CELL_KEYS, sort=False, dropna=False):
        rows.append({**dict(zip(CELL_KEYS, key)), **cell_stats(g, rng, B)})
    return pd.DataFrame(rows)


def differences(df: pd.DataFrame, rng: np.random.Generator, B: int,
                pairs=DIFFERENCES, final_only: bool = True) -> pd.DataFrame:
    """A - B per (model, split, condition, kind, mode, step) with paired-question,
    independent-seed bootstrap intervals, for honest / compliance / compl_long."""
    rows = []
    keys = ["model", "split", "condition", "kind", "mode"]
    for a, b in pairs:
        da, db = df[df.arm == a], df[df.arm == b]
        if da.empty or db.empty:
            continue
        for key, ga in da.groupby(keys, sort=False):
            gb = db[(db[keys] == pd.Series(key, index=keys)).all(axis=1)]
            if gb.empty:
                continue
            steps_a = sorted(ga.step.unique(), key=str)
            steps_b = sorted(gb.step.unique(), key=str)
            if final_only:  # compare the last checkpoint of each arm
                steps_a, steps_b = [steps_a[-1]], [steps_b[-1]]
            for sa in steps_a:
                sb = sa if sa in steps_b else steps_b[-1]
                xa, xb = ga[ga.step == sa], gb[gb.step == sb]
                tasks = sorted(set(xa.task) | set(xb.task))
                cols = rng.integers(0, len(tasks), (B, len(tasks)))
                row = {**dict(zip(keys, key)), "a": a, "b": b, "step_a": sa, "step_b": sb,
                       "n_a": len(xa), "n_b": len(xb), "seeds_a": xa.seed.nunique(), "seeds_b": xb.seed.nunique()}
                for metric in ("honest", "compliance", "compl_long", "correct"):
                    Ma, Mb = _matrix(xa, metric, tasks), _matrix(xb, metric, tasks)
                    with np.errstate(all="ignore"):
                        row[metric] = float(np.nanmean(Ma) - np.nanmean(Mb))
                    row[f"{metric}_lo"], row[f"{metric}_hi"] = ci(
                        boot_means(Ma, rng, B, cols) - boot_means(Mb, rng, B, cols))
                rows.append(row)
    return pd.DataFrame(rows)


def length_matched(df: pd.DataFrame, rng: np.random.Generator, B: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """orange-training section 7: compliance by reasoning-length band and the >=50-word
    'length-matched' compliance, pooled over trained checkpoints (steps 10-50) and the
    fixed val modes; plus matched differences for the pairs of interest."""
    t = df[(df.kind == "fixed") & (df.split == "val") & (df["mode"] != POOLED)
           & (df.step.map(lambda s: isinstance(s, (int, np.integer)) and s > 0))]
    rows, diffs = [], []
    for (model, cond, arm), g in t.groupby(["model", "condition", "arm"], sort=False):
        w = g.words.clip(lower=1)
        row = {"model": model, "condition": cond, "arm": arm, "n": len(g), "raw": g.compliance.mean(),
               "matched_50w": g[w >= 50].compliance.mean(), "frac_under_50w": (w < 50).mean(),
               "compliance_under_50w": g[w < 50].compliance.mean(), "honest": g.honest.mean()}
        for lo, hi in BANDS:
            sel = (w >= lo) & (w < hi)
            label = f"{lo}-{hi}w" if hi < 10 ** 8 else f"{lo}+w"
            row[label] = g[sel].compliance.mean() if sel.any() else np.nan
            row[f"n_{label}"] = int(sel.sum())
        rows.append(row)
    for a, b in DIFFERENCES:
        for (model, cond), g in t.groupby(["model", "condition"], sort=False):
            xa, xb = g[(g.arm == a) & (g.words >= 50)], g[(g.arm == b) & (g.words >= 50)]
            if xa.empty or xb.empty:
                continue
            tasks = sorted(set(xa.task) | set(xb.task))
            cols = rng.integers(0, len(tasks), (B, len(tasks)))
            Ma, Mb = _matrix(xa, "compliance", tasks), _matrix(xb, "compliance", tasks)
            d = boot_means(Ma, rng, B, cols) - boot_means(Mb, rng, B, cols)
            lo, hi = ci(d)
            diffs.append({"model": model, "condition": cond, "a": a, "b": b,
                          "matched_50w_diff": float(np.nanmean(Ma) - np.nanmean(Mb)), "lo": lo, "hi": hi,
                          "n_a": len(xa), "n_b": len(xb)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


# ----------------------------------------------------------------------------- reporting

def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def pct_ci(row, m: str) -> str:
    if np.isnan(row[m]):
        return "-"
    return f"{pct(row[m])} [{pct(row[f'{m}_lo'])}, {pct(row[f'{m}_hi'])}]"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def arm_sort(arm: str) -> tuple:
    key = "noise" if arm.startswith("noise") else arm
    return (ARM_ORDER.index(key) if key in ARM_ORDER else len(ARM_ORDER), arm)


def final_step_of(cells: pd.DataFrame, arm: str, model: str, split: str) -> object:
    s = cells[(cells.arm == arm) & (cells.model == model) & (cells.split == split)].step
    ints = [x for x in s.unique() if isinstance(x, (int, np.integer))]
    return max(ints) if ints else (s.iloc[0] if len(s) else None)


def _kind_cell(r: pd.DataFrame, kind: str, m: str) -> str:
    x = r[r.kind == kind]
    return pct_ci(x.iloc[0], m) if len(x) else "-"


def write_table_md(cells: pd.DataFrame, diffs: pd.DataFrame, lm: pd.DataFrame, lm_diffs: pd.DataFrame,
                   out: Path) -> str:
    parts = ["# Experiment 2 table\n",
             ("Honest = honest compliance (`is_honest`), raw = strict compliance, compl_long = compliance "
              "among traces at least as long as the untrained median, acc = accuracy, chars = mean reasoning "
              "length, degen = traces < 200 chars. Brackets: 95% bootstrap over questions and seeds; "
              "seed range = min–max of per-seed honest compliance. Each arm at its final checkpoint; "
              "`untrained` = checkpoint-0 of every SFT run.\n")]
    for model in sorted(cells.model.unique()):
        for split in ("val", "test"):
            sub = cells[(cells.model == model) & (cells.split == split) & (cells.kind == "fixed")]
            if sub.empty:
                continue
            parts.append(f"\n## {model} — {split} split, fixed-trait modes pooled "
                         f"({', '.join(VAL_MODES)})\n")
            rows = []
            for cond in ("persona", "direct", "stated"):
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
            # per-mode honest compliance at the final step
            rows = []
            for cond in ("persona", "direct", "stated"):
                for arm in sorted(sub.arm.unique(), key=arm_sort):
                    step = final_step_of(sub, arm, model, split)
                    r = sub[(sub.condition == cond) & (sub.arm == arm) & (sub.step == step)]
                    if r.empty:
                        continue
                    by_mode = {m: pct_ci(r[r["mode"] == m].iloc[0], "honest") if (r["mode"] == m).any() else "-"
                               for m in VAL_MODES}
                    rows.append([cond, arm, *by_mode.values()])
            parts.append(f"\nPer-mode honest compliance ({split}, final step):\n")
            parts.append(md_table(["condition", "arm", *VAL_MODES], rows))
        # standard / heldout / baseline (test, final)
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
        rows = []
        d = diffs[diffs["mode"] == POOLED]
        for _, r in d.sort_values(["model", "split", "kind", "condition", "a", "b"]).iterrows():
            rows.append([r.model, r.split, r.kind, r.condition, f"{r.a} − {r.b}", f"{r.step_a}/{r.step_b}",
                         f"{r.seeds_a}/{r.seeds_b}", pct_ci(r, "honest"), pct_ci(r, "compliance"),
                         pct_ci(r, "compl_long"), pct_ci(r, "correct")])
        parts.append(md_table(["model", "split", "kind", "condition", "difference", "steps", "seeds",
                               "Δ honest", "Δ raw", "Δ compl_long", "Δ acc"], rows))
    if not lm.empty:
        parts.append("\n## Length-matched comparison (val, steps 10–50 pooled, fixed modes)\n")
        parts.append("Raw vs compliance among traces ≥ 50 words (orange-training section 7), and "
                     "compliance by reasoning-length band (words).\n")
        band_cols = [c for c in lm.columns if re.fullmatch(r"\d+(-\d+|\+)w", c)]
        rows = [[r.model, r.condition, r.arm, pct(r.raw), pct(r.matched_50w), pct(r.frac_under_50w),
                 pct(r.compliance_under_50w), *[f"{pct(r[c])} (n={r['n_' + c]})" for c in band_cols]]
                for _, r in lm.sort_values(["model", "condition", "arm"], key=lambda s: s.map(arm_sort) if s.name == "arm" else s).iterrows()]
        parts.append(md_table(["model", "condition", "arm", "raw", "matched ≥50w", "% <50w", "compl <50w", *band_cols], rows))
        if not lm_diffs.empty:
            rows = [[r.model, r.condition, f"{r.a} − {r.b}", f"{pct(r.matched_50w_diff)} [{pct(r.lo)}, {pct(r.hi)}]"]
                    for _, r in lm_diffs.iterrows()]
            parts.append("\nLength-matched (≥ 50 words) differences:\n")
            parts.append(md_table(["model", "condition", "difference", "Δ matched compliance"], rows))
    text = "\n".join(parts) + "\n"
    out.write_text(text)
    return text


def plot_curves(cells: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils import PALETTE, set_matplotlib_style

    set_matplotlib_style()
    for model in sorted(cells.model.unique()):
        sub = cells[(cells.model == model) & (cells.split == "val") & (cells.kind == "fixed")
                    & cells.step.map(lambda s: isinstance(s, (int, np.integer)))]
        if sub.empty:
            continue
        modes = [POOLED] + VAL_MODES
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
                    x = s.step.astype(int).to_numpy()  # object column (ints + "noise") -> int
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

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=None, help="override kaushik/orange/runs (e.g. runs/smoke)")
    ap.add_argument("--no-extra", action="store_true", help="only this folder's runs")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args(argv)

    sources = dict(SOURCES)
    if args.runs_root:
        sources["orange"] = Path(args.runs_root)
    if args.no_extra:
        sources = {"orange": sources["orange"]}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_rollouts(sources, out_dir / ".cache")
    if df.empty:
        raise SystemExit(f"no eval JSONs under {[str(p) for p in sources.values()]}")
    print(f"{len(df):,} rollouts from {df.run.nunique()} runs; arms {sorted(df.arm.unique())}; "
          f"models {sorted(df.model.unique())}")
    df.to_csv(out_dir / "rollouts.csv.gz", index=False)

    df = add_long(df, untrained_bars(df))
    full = with_pooled_and_untrained(df)
    cells = summarize_cells(full, rng, args.boot)
    cells.to_csv(out_dir / "cells.csv", index=False)
    diffs = differences(full, rng, args.boot)
    diffs.to_csv(out_dir / "differences.csv", index=False)
    lm, lm_diffs = length_matched(full, rng, args.boot)
    lm.to_csv(out_dir / "length_matched.csv", index=False)
    lm_diffs.to_csv(out_dir / "length_matched_differences.csv", index=False)
    text = write_table_md(cells, diffs, lm, lm_diffs, out_dir / "table.md")
    print(text)
    if args.plot:
        plot_curves(cells, out_dir)


if __name__ == "__main__":
    main()
