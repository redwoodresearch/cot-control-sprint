"""Publication figures for Experiment 2B (orange RL vs rl_mcq), FINAL checkpoints on the TEST split.

Regenerate (about 3 minutes; run from the repo root):

  1. Fresh analysis into a scratch dir (never touches ../results/, which is the owner's):
       .venv/bin/python kaushik/run.py orange/rl analyze \
           --out kaushik/orange/rl/figures/_analysis --n-boot 2000 --no-plots
  2. Figures (re-extracts the per-trace test cache `_analysis/test_items.csv.gz` whenever it is older
     than `_analysis/summary.json`, or with --rebuild-cache; the cache is read with analyze.py's own
     loaders, so it holds exactly the rollouts analyze.py scored):
       .venv/bin/python kaushik/orange/rl/figures/make_figures.py

Inputs: _analysis/{summary,training}.json (KL table, realized dose, training curves, cross-check),
_analysis/test_items.csv.gz (one row per test rollout: honest/strict compliance, accuracy, trace length),
and, read-only, ../../results/{cells,differences}.csv (Experiment 2A SFT reference numbers).
Outputs, all inside this folder: fig*.png (200 dpi) + fig*.pdf, data/*.csv (the numbers behind each
figure), data/consistency_checks.txt, data/palette_check.txt.

Statistics: percentile bootstrap, 2,000 draws, resampling training seeds and test questions
(analyze.bootstrap, seed 0); differences share the question draw and resample seeds per arm. Reference
rows: `untrained` = checkpoint 0 of the five rl_mcq_noreason runs (the only test checkpoint-0 runs);
`noise` = the noise agent's KL-matched random adapters (3 per RL arm).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ANALYSIS = HERE / "_analysis"
DATA = HERE / "data"
SFT_RES = ROOT / "kaushik" / "orange" / "results"
CACHE = ANALYSIS / "test_items.csv.gz"
sys.path.insert(0, str(ROOT))

N_BOOT = 2000
FIXED = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "end_of_sentence"]
CONDS = ["persona", "direct", "stated", "standard"]
COND_BLOCKS = {"persona": [f"persona_{m}" for m in FIXED], "direct": [f"direct_{m}" for m in FIXED],
               "stated": [f"stated_{m}" for m in FIXED], "standard": ["stated_standard"]}
COND_TITLE = {
    "persona": "Persona prompt asks for\nthe behaviour (4 modes)",
    "direct": "Behaviour asked for\nplainly: direct (4 modes)",
    "stated": "Constraint stated in the\nquestion (4 modes)",
    "standard": "Standard CoT-Control eval\n(stated, 9 modes)",
}
COND_SHORT = {"persona": "Persona prompt (4 modes)", "direct": "Direct instruction (4 modes)",
              "stated": "Stated in question (4 modes)", "standard": "Standard eval (9 modes)"}
VARIANTS = ["noreason", "reason"]
VARIANT_LABEL = {"noreason": "Reasoning OFF during RL", "reason": "Reasoning ON during RL"}
CHECKS: list[str] = []
CHECKS_HEAD: list[str] = []
KL_TICKS, KL_TICK_LABELS = [0.003, 0.01, 0.03, 0.1], ["0.003", "0.01", "0.03", "0.1"]

# ------------------------------------------------------------------ palette and style
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
ORANGE, GREEN, LIGHT_ORANGE, LIGHT_GREEN = "#eb6834", "#1baf7a", "#f2b79f", "#9fdcc3"
# Colour follows the arm in every figure and matches 2A (orange = orange arm, green = generic MCQ arm,
# greys = reference rows told apart by marker shape and label). Hues = reference-palette slots 2 and 3.
KEYS = {
    "rl_orange": {"label": "RL orange (persona prompt)", "color": ORANGE, "light": LIGHT_ORANGE, "marker": "o", "ms": 8},
    "rl_mcq": {"label": "RL generic MCQ (no persona)", "color": GREEN, "light": LIGHT_GREEN, "marker": "^", "ms": 8},
    "noise_orange": {"label": "Random adapter, KL-matched to RL orange", "color": MUTED, "light": GRID, "marker": "v", "ms": 8},
    "noise_mcq": {"label": "Random adapter, KL-matched to RL generic MCQ", "color": MUTED, "light": GRID, "marker": "v", "ms": 8},
    "untrained": {"label": "Untrained model", "color": INK2, "light": AXIS, "marker": "D", "ms": 6},
    "sft": {"label": "Orange SFT (2A, reference)", "color": ORANGE, "light": LIGHT_ORANGE, "marker": "o", "ms": 8},
}
ROW_ORDER = ["rl_orange", "rl_mcq", "noise_orange", "noise_mcq", "untrained", "sft"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
        "xtick.major.size": 3, "ytick.major.size": 0, "lines.solid_capstyle": "round",
    })


def check_palette(colors: list[str], surface: str = SURFACE) -> None:
    """Python port of the dataviz skill's validate_palette.js (light mode, all pairs), as in 2A."""
    machado = {
        "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
        "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    }
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])

    def lin(h):
        v = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
        return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

    def lab(v):
        return m2 @ np.cbrt(m1 @ v)

    def de(a, b, kind=None):
        va, vb = lin(a), lin(b)
        if kind:
            va, vb = (np.clip(np.array(machado[kind]) @ v, 0, 1) for v in (va, vb))
        return 100 * np.linalg.norm(lab(va) - lab(vb))

    def contrast(a, b):
        def lum(h):
            return lin(h) @ np.array([0.2126, 0.7152, 0.0722])
        hi, lo = sorted((lum(a), lum(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    pairs = [(a, b) for i, a in enumerate(colors) for b in colors[i + 1:]]
    lch = [(lab(lin(c))[0], np.hypot(*lab(lin(c))[1:])) for c in colors]
    cvd = min(de(a, b, k) for a, b in pairs for k in machado)
    nor = min(de(a, b) for a, b in pairs)
    con = {c: round(float(contrast(c, surface)), 2) for c in colors}
    lines = [f"palette {colors} on surface {surface} (light mode, all pairs; hues = reference slots 2 and 3)",
             (f"lightness band 0.43-0.77: {'PASS' if all(0.43 <= L <= 0.77 for L, _ in lch) else 'FAIL'} "
              f"(L = {[round(float(L), 3) for L, _ in lch]})"),
             (f"chroma floor 0.10: {'PASS' if all(c >= 0.10 for _, c in lch) else 'FAIL'} "
              f"(C = {[round(float(c), 3) for _, c in lch]})"),
             f"CVD separation (min protan/deutan dE x100, target >= 8): {'PASS' if cvd >= 8 else 'FAIL'} ({cvd:.1f})",
             f"normal-vision floor (dE x100 >= 15): {'PASS' if nor >= 15 else 'FAIL'} ({nor:.1f})",
             (f"contrast vs surface (>= 3:1 or relief): {con} -> "
              f"{'PASS' if min(con.values()) >= 3 else 'WARN: relief via direct labels, legends, marker shapes and data labels'}"),
             "neutral reference rows (random adapter, untrained) are told apart by marker shape and label, not hue"]
    (DATA / "palette_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if cvd < 8 or nor < 15:
        raise SystemExit("palette fails the dataviz gates")


# ------------------------------------------------------------------ per-trace test cache
def build_cache() -> None:
    """Test-split per-rollout table through analyze.py's own loaders (no ad-hoc parsing)."""
    from kaushik.orange.rl import analyze

    rows, nfiles = [], 0
    run_names = []
    for run_dir in sorted(p for p in analyze.RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")
                          and "smoke" not in p.name and (p / "config.yaml").exists()):
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        arm, seed = cfg["task"]["arm"], str(cfg["train"]["seed"])
        run_names.append(run_dir.name)
        found: dict[tuple, Path] = {}
        for g in ("final/eval/*/checkpoint-*.json", "final/part*/eval/*/checkpoint-*.json"):
            for path in run_dir.glob(g):
                key = (path.parent.name, path.stem)
                if key not in found or path.stat().st_mtime > found[key].stat().st_mtime:
                    found[key] = path
        for (block, stem), path in sorted(found.items()):
            step = stem.replace("checkpoint-", "")
            label = "untrained" if step == "0" else arm
            for qid, it in analyze.load_items(path).items():
                rows.append((label, run_dir.name, seed, step, block, qid, it["honest"], it["raw"], it["correct"], it["chars"]))
            nfiles += 1
    for r in analyze.load_noise_rows(set(run_names)):
        if r["split"] != "test":
            continue
        nfiles += 1
        for qid, it in r["items"].items():
            rows.append((r["arm"], r["run"], str(r["seed"]), "noise", r["block"], qid, it["honest"], it["raw"],
                         it["correct"], it["chars"]))
    df = pd.DataFrame(rows, columns=["arm", "run", "seed", "step", "block", "qid", "honest", "raw", "correct", "chars"])
    ANALYSIS.mkdir(exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"cache: {len(df):,} test rollouts from {nfiles} eval files -> {CACHE.name}")


def load_items(rebuild: bool) -> tuple[pd.DataFrame, pd.Series]:
    summary = ANALYSIS / "summary.json"
    if not summary.exists():
        raise SystemExit("run the analysis first (see the docstring)")
    if rebuild or not CACHE.exists() or CACHE.stat().st_mtime < summary.stat().st_mtime:
        build_cache()
    df = pd.read_csv(CACHE, dtype={"seed": str, "step": str})
    # untrained long-trace bar per test block: median reasoning length of the untrained model's traces
    bars = df[df.arm == "untrained"].groupby("block")["chars"].median()
    df["long"] = (df["chars"] >= df["block"].map(bars)).astype(float)
    df["raw_long"] = df["raw"] * df["long"]
    df["degen"] = (df["chars"] < 200).astype(float)
    return df, bars


# ------------------------------------------------------------------ statistics
def _boot(MA: np.ndarray, MB: np.ndarray | None = None) -> dict:
    from kaushik.orange.rl import analyze

    return analyze.bootstrap(MA, MB, n_boot=N_BOOT)


def _blocks(cond: str, modes: list[str] | None) -> list[str]:
    b = COND_BLOCKS[cond]
    return b if modes is None or cond == "standard" else [x for x in b if x.partition("_")[2] in modes]


def arm_name(key: str, variant: str) -> str:
    return {"rl_orange": f"rl_orange_{variant}", "rl_mcq": f"rl_mcq_{variant}", "noise_orange": f"noise:rl_orange_{variant}",
            "noise_mcq": f"noise:rl_mcq_{variant}", "untrained": "untrained"}[key]


def pivot(df: pd.DataFrame, arm: str, cond: str, modes: list[str] | None, metric: str) -> pd.DataFrame:
    """[seed x question] matrix: per seed, each question's mean over the selected modes."""
    d = df[(df.arm == arm) & df.block.isin(_blocks(cond, modes))]
    p = d.groupby(["seed", "qid"])[metric].mean().unstack("qid")
    return p.dropna(axis=1)


def common_modes(df: pd.DataFrame, arm: str, cond: str) -> list[str]:
    """Modes for which every adapter/seed of `arm` has the test block (noise evals may be unfinished)."""
    d = df[df.arm == arm]
    seeds = d.seed.unique()
    if cond == "standard":
        return ["standard"] if len(seeds) and all(((d.seed == s) & (d.block == "stated_standard")).any() for s in seeds) else []
    return [m for m in FIXED if len(seeds) and all(((d.seed == s) & (d.block == f"{cond}_{m}")).any() for s in seeds)]


def ratio_matrix(df: pd.DataFrame, arm: str, cond: str, modes) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pivot(df, arm, cond, modes, "raw_long"), pivot(df, arm, cond, modes, "long")


def boot_ratio(NUM: np.ndarray, DEN: np.ndarray, seed: int = 0) -> dict:
    """Ratio-of-means bootstrap (strict compliance among long traces), same resampling as analyze.bootstrap."""
    rng = np.random.default_rng(seed)
    S, Q = NUM.shape
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        q = rng.integers(0, Q, Q)
        s = rng.integers(0, S, S)
        den = DEN[s][:, q].sum()
        out[i] = NUM[s][:, q].sum() / den if den > 0 else np.nan
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return {"mean": float(NUM.sum() / max(DEN.sum(), 1e-12)), "lo": float(lo), "hi": float(hi)}


_CACHE_LEVEL: dict = {}


def level(df: pd.DataFrame, key: str, variant: str, cond: str, modes: list[str] | None = None) -> dict | None:
    """Everything one (arm, variant, condition) cell needs. `modes` None = all four fixed modes (or the
    standard block); noise rows pass their common modes. None if the cell has no data."""
    arm = arm_name(key, variant)
    tag = (arm, cond, tuple(modes) if modes else None)
    if tag in _CACHE_LEVEL:
        return _CACHE_LEVEL[tag]
    out = None
    if modes is None or len(modes):
        H = pivot(df, arm, cond, modes, "honest")
        if H.shape[0] and H.shape[1]:
            out = {"arm": arm, "key": key, "variant": variant, "condition": cond, "n_seeds": H.shape[0],
                   "n_questions": H.shape[1], "modes": ",".join(m for m in (modes or (FIXED if cond != "standard" else ["standard"]))),
                   "n_modes": len(modes) if modes else (len(FIXED) if cond != "standard" else 1)}
            out["n_rollouts"] = out["n_seeds"] * out["n_questions"] * out["n_modes"]
            for metric in ("honest", "raw", "correct", "chars", "degen", "long"):
                M = pivot(df, arm, cond, modes, metric)
                b = _boot(M.to_numpy(float))
                out[metric] = {k: b[k] for k in ("mean", "lo", "hi")}
                out[f"{metric}_seed_values"] = M.mean(axis=1).to_dict()
                out[f"{metric}_seed_min"], out[f"{metric}_seed_max"] = float(M.mean(axis=1).min()), float(M.mean(axis=1).max())
            NUM, DEN = ratio_matrix(df, arm, cond, modes)
            out["compl_long"] = boot_ratio(NUM.to_numpy(float), DEN.to_numpy(float))
            sv = (NUM.sum(axis=1) / DEN.sum(axis=1).replace(0, np.nan)).to_dict()
            out["compl_long_seed_values"] = sv
            vals = [v for v in sv.values() if not np.isnan(v)]
            out["compl_long_seed_min"], out["compl_long_seed_max"] = (min(vals), max(vals)) if vals else (np.nan, np.nan)
    _CACHE_LEVEL[tag] = out
    return out


def diff(df: pd.DataFrame, a: tuple, b: tuple, cond: str, modes: list[str] | None = None, metric: str = "honest") -> dict | None:
    """Bootstrap of arm A minus arm B on shared questions; a, b = (key, variant)."""
    MA = pivot(df, arm_name(*a), cond, modes, metric)
    MB = pivot(df, arm_name(*b), cond, modes, metric)
    if MA.empty or MB.empty:
        return None
    common = MA.columns.intersection(MB.columns)
    r = _boot(MA[common].to_numpy(float), MB[common].to_numpy(float))
    return {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "n_seeds_a": MA.shape[0], "n_seeds_b": MB.shape[0],
            "n_questions": len(common)}


# ------------------------------------------------------------------ 2A reference (read only)
def load_sft_reference() -> dict:
    out: dict = {"cells": {}, "diffs": {}}
    cells_p, diffs_p = SFT_RES / "cells.csv", SFT_RES / "differences.csv"
    if not (cells_p.exists() and diffs_p.exists()):
        return out
    cells = pd.read_csv(cells_p, dtype={"step": str})
    diffs = pd.read_csv(diffs_p, dtype={"step_a": str, "step_b": str})
    kind = {"persona": "fixed", "direct": "fixed", "stated": "fixed", "standard": "standard"}
    for cond in CONDS:
        cond_name = "stated" if cond == "standard" else cond
        s = cells[(cells.model == "qwen36_27b") & (cells.split == "test") & (cells["mode"] == "(pooled)") & (cells.arm == "orange")
                  & (cells.step == "50") & (cells.condition == cond_name) & (cells.kind == kind[cond])]
        if len(s) == 1:
            out["cells"][cond] = s.iloc[0].to_dict()
        for other in ("mcq_noreason", "noise[orange]"):
            d = diffs[(diffs.model == "qwen36_27b") & (diffs.split == "test") & (diffs["mode"] == "(pooled)") & (diffs.a == "orange")
                      & (diffs.b == other) & (diffs.condition == cond_name) & (diffs.kind == kind[cond])]
            if len(d) == 1:
                out["diffs"][(cond, other)] = d.iloc[0].to_dict()
    return out


# ------------------------------------------------------------------ drawing helpers
def save(fig, stem: str, df: pd.DataFrame | None = None) -> None:
    fig.savefig(HERE / f"{stem}.png", dpi=200)
    fig.savefig(HERE / f"{stem}.pdf")
    plt.close(fig)
    if df is not None:
        df.to_csv(DATA / f"{stem}.csv", index=False)
    print(f"saved {stem}")


def finish(fig, note: str, handles=None, ncol: int = 4) -> None:
    """Legend row(s) above a wrapped footnote, both below the plot area (the layout engine gets the rest)."""
    w, h = fig.get_size_inches()
    note = "\n".join(textwrap.fill(par, width=int(w * 15.5)) for par in note.split("\n"))
    note_in = 0.165 * (note.count("\n") + 1) + 0.14
    leg_in = (0.25 * (-(-len(handles) // ncol)) + 0.12) if handles else 0.0
    bottom = (note_in + leg_in) / h
    fig.get_layout_engine().set(rect=(0, bottom, 1, 1 - bottom))
    fig.text(0.005, 0.09 / h, note, fontsize=8, color=INK2, ha="left", va="bottom", linespacing=1.4)
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, note_in / h), ncol=ncol, fontsize=8.5,
                   columnspacing=1.6, handletextpad=0.5)


def pct(x: float) -> float:
    return 100 * x


def sgn(x: float) -> str:
    """Signed one-decimal percentage-point string without a negative zero."""
    r = round(x, 1)
    return "0.0" if r == 0 else f"{r:+.1f}"


def ci_label(m: float, lo: float, hi: float) -> str:
    return f"{sgn(m)} [{sgn(lo)}, {sgn(hi)}]"


def draw_interval(ax, y, mean, lo, hi, smin, smax, key, lw=3.2, filled=True, ms=None, dy=0.0, alpha=1.0):
    st = KEYS[key]
    y = y + dy
    if smin is not None and not (np.isnan(smin) or np.isnan(smax)):
        ax.plot([smin, smax], [y, y], color=st["light"], lw=1.2, zorder=2, solid_capstyle="butt")
        for x in (smin, smax):
            ax.plot([x, x], [y - 0.11, y + 0.11], color=st["light"], lw=1.2, zorder=2)
    if lo is not None and not (np.isnan(lo) or np.isnan(hi)):
        ax.plot([lo, hi], [y, y], color=st["color"], lw=lw, zorder=3, solid_capstyle="butt", alpha=alpha)
    ax.plot([mean], [y], marker=st["marker"], ms=ms or st["ms"], color=st["color"] if filled else SURFACE,
            mec=SURFACE if filled else st["color"], mew=1.4 if filled else 1.8, zorder=4, ls="", alpha=alpha)


def legend_handles(keys, extra=True):
    hs = []
    for k in keys:
        st = KEYS[k]
        hollow = k == "sft"
        hs.append(Line2D([], [], marker=st["marker"], color=SURFACE if hollow else st["color"], ls="", ms=st["ms"],
                         mec=st["color"] if hollow else SURFACE, mew=1.8 if hollow else 1.4, label=st["label"]))
    if extra:
        hs += [Line2D([], [], color=INK2, lw=3.2, label="95% CI (bootstrap over questions x seeds)"),
               Line2D([], [], color=AXIS, lw=1.2, label="lowest to highest training seed / random adapter")]
    return hs


# ------------------------------------------------------------------ figure 1: levels
def fig1(df, sft, cells_out: list) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.9), sharex=True, sharey="row", layout="constrained")
    partial = {v: any(len(common_modes(df, arm_name(k, v), c)) < (1 if c == "standard" else 4) for k in ("noise_orange", "noise_mcq") for c in CONDS)
               for v in VARIANTS}
    partial_note = any(partial.values())
    for r, variant in enumerate(VARIANTS):
        for c, cond in enumerate(CONDS):
            ax = axes[r, c]
            ax.grid(axis="y", visible=False)
            for i, key in enumerate(ROW_ORDER):
                y = i + (0.45 if key == "sft" else 0)
                if key == "sft":
                    s = sft["cells"].get(cond)
                    if s is None:
                        ax.text(0.3, y, "not available", va="center", fontsize=8.5, color=MUTED, style="italic")
                        continue
                    v, lo, hi, smin, smax = (pct(s[k]) for k in ("honest", "honest_lo", "honest_hi", "honest_seed_min", "honest_seed_max"))
                    cells_out.append({"variant_row": variant, "condition": cond, "row": key, "source": "2A SFT cells.csv", "n_seeds": int(s["n_seeds"]),
                                      "honest_pct": v, "ci_lo": lo, "ci_hi": hi, "seed_min": smin, "seed_max": smax})
                else:
                    modes = None
                    if key.startswith("noise"):
                        modes = common_modes(df, arm_name(key, variant), cond)
                    lv = level(df, key, variant, cond, modes if modes is not None else None) if (modes is None or modes) else None
                    if lv is None:
                        ax.text(0.3, y, "random adapter not yet evaluated" if key.startswith("noise") else "no data",
                                va="center", fontsize=8.5, color=MUTED, style="italic")
                        cells_out.append({"variant_row": variant, "condition": cond, "row": key, "source": "missing"})
                        continue
                    v, lo, hi = (pct(lv["honest"][k]) for k in ("mean", "lo", "hi"))
                    smin, smax = pct(lv["honest_seed_min"]), pct(lv["honest_seed_max"])
                    cells_out.append({"variant_row": variant, "condition": cond, "row": key, "source": "test items",
                                      "arm": lv["arm"], "modes": lv["modes"], "n_seeds": lv["n_seeds"], "n_questions": lv["n_questions"],
                                      "n_rollouts": lv["n_rollouts"], "honest_pct": v, "ci_lo": lo, "ci_hi": hi,
                                      "seed_min": smin, "seed_max": smax,
                                      "seed_values_pct": ";".join(f"{k}={pct(x):.2f}" for k, x in lv["honest_seed_values"].items())})
                part = key.startswith("noise") and len(modes) < (1 if cond == "standard" else 4)
                draw_interval(ax, y, v, lo, hi, smin, smax, key, filled=key != "sft", alpha=0.5 if part else 1.0)
                bold = key in ("rl_orange", "sft")
                ax.text(max(hi, smax) + 0.4, y, f"{v:.1f}" + (f"  ({len(modes)} of 4 modes)" if part else ""), va="center", fontsize=9 if not part else 8.2,
                        color=INK if not part else INK2, fontweight="bold" if bold else "normal")
            ax.axhline(4.72, color=AXIS, lw=0.8, ls=(0, (3, 3)))
            ax.set_xlim(-0.8, 13.0)
            ax.set_xticks([0, 4, 8, 12])
            ax.set_ylim(5.75, -0.55)
            ax.set_yticks([0, 1, 2, 3, 4, 5.45])
            labels = [KEYS[k]["label"] for k in ROW_ORDER]
            if partial[variant]:
                labels[2] += " (partial)"
                labels[3] += " (partial)"
            ax.set_yticklabels(labels, fontsize=8.8)
            if r == 0:
                ax.set_title(COND_TITLE[cond], loc="left", fontsize=9.5, color=INK, pad=6)
            if r == 1:
                ax.set_xlabel("Honest compliance (%)")
    for r, variant in enumerate(VARIANTS):
        axes[r, 0].set_ylabel(VARIANT_LABEL[variant], fontsize=11, fontweight="bold", color=INK, labelpad=8)
    fig.suptitle("Orange RL leaves honest compliance at the level of the untrained model; only the 2A SFT reference moves it",
                 x=0.005, ha="left", fontsize=12.5, fontweight="bold")
    note = ("Qwen3.6-27B, test split (500 questions), final checkpoint (step 50), 5 training seeds per RL arm; untrained = checkpoint 0 of the 5 rl_mcq_noreason runs; "
            "random adapters: 3 per RL arm, test split.\nHonest compliance = followed the stated rule, gave an answer, trace not hollow. 'Standard eval' pools the 9 training modes. "
            "Orange SFT (2A) is reasoning-off SFT, 5 seeds, shown for scale only; its standard-eval cell used all 9 modes per question, the RL arms one random mode per question (500 questions per seed).")
    if partial_note:
        note += ("\nPartial (faded rows): some random adapters are still being evaluated, so they pool only the modes finished for all 3 adapters (data/fig1_main_result.csv, column `modes`) "
                 "and are NOT like-for-like with the 4-mode rows next to them (modes differ in difficulty); fig2 panel B compares on identical modes. 'not yet evaluated' = no complete block yet.")
    finish(fig, note, legend_handles(["rl_orange", "rl_mcq", "noise_orange", "untrained", "sft"]), ncol=4)
    save(fig, "fig1_main_result", pd.DataFrame(cells_out))


# ------------------------------------------------------------------ figure 2: paired differences
def fig2(df, sft, dose: dict) -> pd.DataFrame:
    rows_out = []
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.4), sharex=True, sharey=True, layout="constrained")
    ypos, headers = {}, {}
    y = 0.0
    for variant in VARIANTS:
        headers[variant] = y
        y += 1
        for cond in CONDS:
            ypos[(variant, cond)] = y
            y += 1
        y += 0.6
    for variant in VARIANTS:
        for cond in CONDS:
            yy = ypos[(variant, cond)]
            # ---- panel A: orange minus generic MCQ (both RL)
            d = diff(df, ("rl_orange", variant), ("rl_mcq", variant), cond)
            a_ = axes[0]
            if d:
                draw_interval(a_, yy, pct(d["mean"]), pct(d["lo"]), pct(d["hi"]), None, None, "rl_orange", dy=-0.16)
                a_.text(max(pct(d["hi"]), pct(d["mean"])) + 0.3, yy - 0.16, ci_label(pct(d["mean"]), pct(d["lo"]), pct(d["hi"])), va="center",
                        fontsize=8.5, fontweight="bold", bbox={"fc": SURFACE, "ec": "none", "pad": 0.8, "alpha": 0.85}, zorder=5)
                rows_out.append({"panel": "A", "variant": variant, "condition": cond, "comparison": "rl_orange - rl_mcq", "diff_pp": pct(d["mean"]),
                                 "ci_lo": pct(d["lo"]), "ci_hi": pct(d["hi"]), "n_seeds_a": d["n_seeds_a"], "n_seeds_b": d["n_seeds_b"], "n_questions": d["n_questions"],
                                 "modes": "standard" if cond == "standard" else ",".join(FIXED)})
            s = sft["diffs"].get((cond, "mcq_noreason"))
            if s is not None:
                draw_interval(a_, yy, pct(s["honest"]), pct(s["honest_lo"]), pct(s["honest_hi"]), None, None, "sft", filled=False, dy=0.16, lw=2.2)
                a_.text(max(pct(s["honest_hi"]), pct(s["honest"])) + 0.3, yy + 0.16, ci_label(pct(s["honest"]), pct(s["honest_lo"]), pct(s["honest_hi"])),
                        va="center", fontsize=8.5, color=INK2, bbox={"fc": SURFACE, "ec": "none", "pad": 0.8, "alpha": 0.85}, zorder=5)
                rows_out.append({"panel": "A", "variant": variant, "condition": cond, "comparison": "SFT 2A: orange - mcq_noreason", "diff_pp": pct(s["honest"]),
                                 "ci_lo": pct(s["honest_lo"]), "ci_hi": pct(s["honest_hi"]), "source": "2A differences.csv"})
            # ---- panel B: each arm minus its own KL-matched random adapter
            b_ = axes[1]
            for key, nkey, dy in (("rl_orange", "noise_orange", -0.24), ("rl_mcq", "noise_mcq", 0.0)):
                modes = common_modes(df, arm_name(nkey, variant), cond)
                dd = diff(df, (key, variant), (nkey, variant), cond, modes) if modes else None
                if dd:
                    draw_interval(b_, yy, pct(dd["mean"]), pct(dd["lo"]), pct(dd["hi"]), None, None, key, dy=dy, lw=2.8, ms=7)
                    rows_out.append({"panel": "B", "variant": variant, "condition": cond, "comparison": f"{key} - noise matched to {key}",
                                     "diff_pp": pct(dd["mean"]), "ci_lo": pct(dd["lo"]), "ci_hi": pct(dd["hi"]), "n_seeds_a": dd["n_seeds_a"],
                                     "n_seeds_b": dd["n_seeds_b"], "n_questions": dd["n_questions"], "modes": ",".join(modes)})
                else:
                    b_.text(-6.6, yy + dy, f"{'orange' if key == 'rl_orange' else 'MCQ'}: random adapter not yet evaluated", va="center", fontsize=7.5,
                            color=MUTED, style="italic")
                    rows_out.append({"panel": "B", "variant": variant, "condition": cond, "comparison": f"{key} - noise matched to {key}", "diff_pp": np.nan})
            s = sft["diffs"].get((cond, "noise[orange]"))
            if s is not None:
                draw_interval(b_, yy, pct(s["honest"]), pct(s["honest_lo"]), pct(s["honest_hi"]), None, None, "sft", filled=False, dy=0.24, lw=2.2, ms=7)
                rows_out.append({"panel": "B", "variant": variant, "condition": cond, "comparison": "SFT 2A: orange - noise[orange]", "diff_pp": pct(s["honest"]),
                                 "ci_lo": pct(s["honest_lo"]), "ci_hi": pct(s["honest_hi"]), "source": "2A differences.csv"})
    for ax in axes:
        ax.axvline(0, color=INK2, lw=1.2, zorder=1)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-7.2, 10.5)
        ax.set_xticks([-6, -3, 0, 3, 6, 9])
        ax.set_ylim(y - 0.3, -0.6)
        ax.set_xlabel("Difference in honest compliance (percentage points)")
        for variant in VARIANTS:
            ax.text(-7.0, headers[variant], VARIANT_LABEL[variant], fontsize=10.5, fontweight="bold", va="center", color=INK)
    axes[0].set_yticks([ypos[(v, c)] for v in VARIANTS for c in CONDS])
    axes[0].set_yticklabels([COND_SHORT[c] for v in VARIANTS for c in CONDS], fontsize=9)
    axes[0].set_title("A. RL orange minus RL generic MCQ (the matched baseline)", loc="left", fontsize=10.5, fontweight="bold", pad=8)
    axes[1].set_title("B. Each RL arm minus a random adapter of the same size (matched KL)", loc="left", fontsize=10.5, fontweight="bold", pad=8)
    hs = [Line2D([], [], marker="o", color=ORANGE, ls="", ms=8, mec=SURFACE, mew=1.4, label="RL orange (A: minus RL generic MCQ; B: minus its matched random adapter)"),
          Line2D([], [], marker="^", color=GREEN, ls="", ms=8, mec=SURFACE, mew=1.4, label="RL generic MCQ minus its matched random adapter (B only)"),
          Line2D([], [], marker="o", color=SURFACE, mec=ORANGE, mew=1.8, ls="", ms=8,
                 label="2A orange SFT minus generic MCQ SFT (A) / minus its random adapter (B), reference"),
          Line2D([], [], color=INK2, lw=3, label="95% CI (bootstrap over questions x seeds)")]
    mc = {d["arm"]: d for d in dose}
    cave = ("CAVEAT: the generic-MCQ reward is already about 0.95 at step 0 and saturates, so RL generic MCQ is an under-dosed baseline: prompt groups that carried gradient (of 2,400) "
            f"were {mc['rl_mcq_noreason']['groups_used']:.0f} (reasoning off) and {mc['rl_mcq_reason']['groups_used']:.0f} (on) for generic MCQ versus "
            f"{mc['rl_orange_noreason']['groups_used']:.0f} and {mc['rl_orange_reason']['groups_used']:.0f} for orange (fig3, fig5). By KL(base || adapter) the two RL arms moved the model by "
            "comparable amounts (fig3; KL is measured without the persona prompt), so panel B, equal-size random adapters, is the dose-matched reference.")
    fig.suptitle("Orange RL does no better than the RL generic-MCQ baseline, or than random adapters of equal size", x=0.005, ha="left", fontsize=12.5, fontweight="bold")
    note = (cave + "\nQwen3.6-27B, test split (500 questions), final checkpoint (step 50), 5 seeds per RL arm vs 3 random adapters; 95% CI by bootstrap over questions and seeds. "
            "Panel B pools only the modes finished for all 3 random adapters (the RL arm is restricted to the same modes); 'not yet evaluated' = no complete block yet. "
            "2A SFT reference: reasoning-off SFT; 2A has no generic-MCQ SFT run on the 4-mode stated block, and its standard block used all 9 modes per question (RL: one random mode per question).")
    finish(fig, note, hs, ncol=2)
    save(fig, "fig2_paired_differences", pd.DataFrame(rows_out))
    return pd.DataFrame(rows_out)


# ------------------------------------------------------------------ figure 3: matched KL and dose
def fig3(df, summary: dict, training: list[dict]) -> None:
    kl = {k["arm"]: k for k in summary["kl"]}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), layout="constrained")
    shape = {"noreason": "o", "reason": "s"}
    rows_out = []
    # --- panel A: KL per adapter
    axA = axes[0, 0]
    axA.grid(axis="y", visible=False)
    order = [("rl_orange", "noreason"), ("rl_mcq", "noreason"), ("rl_orange", "reason"), ("rl_mcq", "reason")]
    for i, (key, variant) in enumerate(order):
        k = kl.get(f"{key}_{variant}")
        if not k:
            continue
        col = ORANGE if key == "rl_orange" else GREEN
        xs = [p["kl"] for p in k["per_seed"]]
        axA.plot(xs, [i - 0.12] * len(xs), ls="", marker=shape[variant], ms=8, color=col, mec=SURFACE, mew=1.2, zorder=4)
        nx = k["noise_achieved"]
        axA.plot(nx, [i + 0.14] * len(nx), ls="", marker="v", ms=7, color=SURFACE, mec=MUTED, mew=1.6, zorder=4)
        axA.plot([k["median"]], [i - 0.12], marker="|", ms=18, color=INK, mew=1.6, zorder=5)
        for p in k["per_seed"]:
            rows_out.append({"panel": "A", "arm": f"{key}_{variant}", "kind": "trained final adapter", "seed": p["seed"], "kl_per_token": p["kl"], "kl_se": p.get("se")})
        for j, x in enumerate(nx):
            rows_out.append({"panel": "A", "arm": f"{key}_{variant}", "kind": "random adapter", "seed": f"noise{j}", "kl_per_token": x})
    axA.set_xscale("log")
    axA.set_xticks(KL_TICKS)
    axA.set_xticklabels(KL_TICK_LABELS)
    axA.minorticks_off()
    axA.set_yticks(range(4))
    axA.set_yticklabels([f"{KEYS[k]['label'].split(' (')[0]}\n{VARIANT_LABEL[v].replace(' during RL', '').lower()}" for k, v in order], fontsize=8.8)
    axA.set_ylim(3.6, -0.6)
    axA.set_xlabel("KL(base || adapter) per token, nats (log scale)")
    axA.set_title("A. How far each final adapter moved the model", loc="left", fontsize=10.5, fontweight="bold")
    axA.text(0.99, 0.02, "black tick = median of the 5 trained seeds (the random adapters' KL target)", transform=axA.transAxes, ha="right", fontsize=7.8, color=INK2)
    # --- panel B: realized dose
    axB = axes[0, 1]
    axB.grid(axis="y", visible=False)
    for i, (key, variant) in enumerate(order):
        col = ORANGE if key == "rl_orange" else GREEN
        runs = [t for t in training if t["arm"] == f"{key}_{variant}"]
        vals = []
        for t in runs:
            used = sum(e.get("n_groups_used", 0) for e in t["metrics"] if not e.get("skipped"))
            vals.append(used)
            rows_out.append({"panel": "B", "arm": f"{key}_{variant}", "kind": "groups with gradient", "seed": t["seed"], "groups_with_gradient": used})
        axB.plot(vals, [i] * len(vals), ls="", marker=shape[variant], ms=8, color=col, mec=SURFACE, mew=1.2, zorder=4)
        axB.plot([np.mean(vals)], [i], marker="|", ms=18, color=INK, mew=1.6, zorder=5)
        axB.text(max(vals) + 8, i, f"mean {np.mean(vals):.0f}", va="center", fontsize=8.5)
    axB.text(0.99, 0.02, "black tick = mean of the 5 seeds", transform=axB.transAxes, ha="right", fontsize=7.8, color=INK2)
    axB.set_xlim(0, 400)
    axB.set_yticks(range(4))
    axB.set_yticklabels([""] * 4)
    axB.set_ylim(3.6, -0.6)
    axB.set_xlabel("Prompt groups that carried gradient over 50 iterations (maximum 2,400)")
    axB.set_title("B. Realized training dose (groups with all-equal rewards are dropped)", loc="left", fontsize=10.5, fontweight="bold")
    # --- panels C, D: compliance vs KL
    for ax, cond, title in ((axes[1, 0], "persona", "C. Persona prompt (4 modes)"), (axes[1, 1], "direct", "D. Direct instruction (4 modes)")):
        base = level(df, "untrained", "noreason", cond)
        ax.axhspan(pct(base["honest"]["lo"]), pct(base["honest"]["hi"]), color=GRID, zorder=0, alpha=0.8)
        ax.axhline(pct(base["honest"]["mean"]), color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(0.99, 0.97, "dashed line and band: untrained model (KL 0), mean and 95% CI", transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=INK2)
        for key, variant in order:
            col = ORANGE if key == "rl_orange" else GREEN
            k = kl.get(f"{key}_{variant}")
            if not k:
                continue
            lv = level(df, key, variant, cond)
            for p in k["per_seed"]:
                v = lv["honest_seed_values"].get(str(p["seed"]))
                if v is None:
                    continue
                ax.plot([p["kl"]], [pct(v)], marker=shape[variant], ms=8, color=col, mec=SURFACE, mew=1.2, ls="", zorder=4)
                rows_out.append({"panel": "C" if cond == "persona" else "D", "arm": f"{key}_{variant}", "kind": "trained final adapter", "seed": p["seed"],
                                 "kl_per_token": p["kl"], "honest_pct": pct(v)})
            nkey = "noise_orange" if key == "rl_orange" else "noise_mcq"
            modes = common_modes(df, arm_name(nkey, variant), cond)
            nl = level(df, nkey, variant, cond, modes) if modes else None
            if nl:
                for j, x in enumerate(k["noise_achieved"]):
                    v = nl["honest_seed_values"].get(f"noise{j}")
                    if v is None:
                        continue
                    ax.plot([x], [pct(v)], marker="v", ms=7, color=SURFACE, mec=MUTED, mew=1.6, ls="", zorder=3)
                    rows_out.append({"panel": "C" if cond == "persona" else "D", "arm": f"{key}_{variant}", "kind": "random adapter", "seed": f"noise{j}",
                                     "kl_per_token": x, "honest_pct": pct(v), "modes": ",".join(modes)})
        ax.set_xscale("log")
        ax.set_xticks(KL_TICKS)
        ax.set_xticklabels(KL_TICK_LABELS)
        ax.minorticks_off()
        ax.set_xlim(0.0015, 0.2)
        ax.set_ylim(-0.6, 12 if cond == "direct" else 3)
        ax.set_xlabel("KL(base || adapter) per token, nats (log scale)")
        ax.set_ylabel("Honest compliance (%)")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    hs = [Line2D([], [], marker="o", color=ORANGE, ls="", ms=8, mec=SURFACE, label="RL orange, reasoning off"),
          Line2D([], [], marker="s", color=ORANGE, ls="", ms=8, mec=SURFACE, label="RL orange, reasoning on"),
          Line2D([], [], marker="o", color=GREEN, ls="", ms=8, mec=SURFACE, label="RL generic MCQ, reasoning off"),
          Line2D([], [], marker="s", color=GREEN, ls="", ms=8, mec=SURFACE, label="RL generic MCQ, reasoning on"),
          Line2D([], [], marker="v", color=SURFACE, mec=MUTED, mew=1.6, ls="", ms=7, label="random adapter (hollow; matched to the arm's median KL)")]
    fig.suptitle("Generic-MCQ RL is not smaller than orange RL on the KL axis, but fewer of its prompt groups carried gradient",
                 x=0.005, ha="left", fontsize=12.5, fontweight="bold")
    finish(fig, "Qwen3.6-27B final adapters (step 50), 5 seeds per RL arm. KL(base || adapter) is the noise agent's measurement on 200 val prompts with T=1 samples from the base model, "
           "with no persona prompt, so it may understate how far orange adapters moved the model in their trained context. C and D: one dot per adapter (seed mean), test split, 500 questions x 4 modes; "
           "random adapters use only the modes finished for all 3 adapters (fewer than 4 for some reasoning-on arms), so they are not exactly like-for-like with the 4-mode trained dots.", hs, ncol=3)
    save(fig, "fig3_matched_kl_and_dose", pd.DataFrame(rows_out))


# ------------------------------------------------------------------ figure 4: length / degeneration
def fig4(df, sft) -> None:
    rows_out = []
    CONDS3 = ["persona", "direct", "stated"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.8), sharey=True, layout="constrained")
    ords = [("untrained", None), ("rl_mcq", "noreason"), ("rl_orange", "noreason"), ("rl_mcq", "reason"), ("rl_orange", "reason"), ("sft", None)]
    ylab = ["Untrained model", "RL generic MCQ, reasoning off", "RL orange, reasoning off", "RL generic MCQ, reasoning on",
            "RL orange, reasoning on", "Orange SFT (2A, reference)"]
    ypos = [0, 1.3, 2.3, 3.6, 4.6, 6.0]
    for r, cond in enumerate(CONDS3):
        for c, (metric, title) in enumerate([("chars", "Mean reasoning length (thousand characters)"), ("degen", "Traces under 200 characters (%)"),
                                             ("compl", "Compliance: all traces (filled) vs traces as long as\nthe untrained median (hollow), %"),
                                             ("correct", "Accuracy (%)")]):
            ax = axes[r, c]
            ax.grid(axis="y", visible=False)
            for (key, variant), y in zip(ords, ypos):
                if key == "sft":
                    s = sft["cells"].get(cond)
                    if s is None:
                        continue
                    if metric == "chars":
                        ax.plot([s["chars"] / 1000], [y], marker="o", ms=8, color=SURFACE, mec=ORANGE, mew=1.8, ls="", zorder=4)
                        ax.text(s["chars"] / 1000 + 1.0, y, f"{s['chars'] / 1000:.1f}", va="center", fontsize=8.5)
                    elif metric == "degen":
                        ax.plot([pct(s["degen"])], [y], marker="o", ms=8, color=SURFACE, mec=ORANGE, mew=1.8, ls="", zorder=4)
                        ax.text(pct(s["degen"]) + 0.8, y, f"{pct(s['degen']):.1f}", va="center", fontsize=8.5)
                    elif metric == "compl":
                        draw_interval(ax, y - 0.17, pct(s["honest"]), pct(s["honest_lo"]), pct(s["honest_hi"]), None, None, "sft", filled=False, lw=2.2, ms=7)
                        draw_interval(ax, y + 0.17, pct(s["compl_long"]), pct(s["compl_long_lo"]), pct(s["compl_long_hi"]), None, None, "sft", filled=False, lw=2.2, ms=7)
                        ax.text(max(pct(s["honest_hi"]), pct(s["honest"])) + 0.4, y - 0.17, f"{pct(s['honest']):.1f}", va="center", fontsize=8)
                        ax.text(max(pct(s["compl_long_hi"]), pct(s["compl_long"])) + 0.4, y + 0.17, f"{pct(s['compl_long']):.1f}", va="center", fontsize=8)
                    else:
                        ax.plot([pct(s["correct"])], [y], marker="o", ms=8, color=SURFACE, mec=ORANGE, mew=1.8, ls="", zorder=4)
                        ax.text(pct(s["correct"]) + 1.2, y, f"{pct(s['correct']):.1f}", va="center", fontsize=8.5)
                    rows_out.append({"condition": cond, "row": "sft", "metric": metric, "source": "2A cells.csv"})
                    continue
                lv = level(df, key, variant or "noreason", cond)
                color_key = key
                if metric in ("chars", "degen", "correct"):
                    m = lv[metric]
                    sc = 1 / 1000 if metric == "chars" else 100
                    draw_interval(ax, y, m["mean"] * sc, m["lo"] * sc, m["hi"] * sc, lv[f"{metric}_seed_min"] * sc, lv[f"{metric}_seed_max"] * sc, color_key,
                                  filled=True)
                    ax.text(max(m["hi"] * sc, lv[f"{metric}_seed_max"] * sc) + (0.5 if metric != "correct" else 1), y, f"{m['mean'] * sc:.1f}", va="center",
                            fontsize=8.5, fontweight="bold" if key == "rl_orange" else "normal")
                    rows_out.append({"condition": cond, "row": f"{key}_{variant}" if variant else key, "metric": metric, "mean": m["mean"] * sc, "ci_lo": m["lo"] * sc,
                                     "ci_hi": m["hi"] * sc, "seed_min": lv[f"{metric}_seed_min"] * sc, "seed_max": lv[f"{metric}_seed_max"] * sc,
                                     "n_seeds": lv["n_seeds"], "n_questions": lv["n_questions"]})
                else:
                    h, cl = lv["honest"], lv["compl_long"]
                    draw_interval(ax, y - 0.17, pct(h["mean"]), pct(h["lo"]), pct(h["hi"]), pct(lv["honest_seed_min"]), pct(lv["honest_seed_max"]), color_key, filled=True, lw=2.6, ms=7)
                    draw_interval(ax, y + 0.17, pct(cl["mean"]), pct(cl["lo"]), pct(cl["hi"]), pct(lv["compl_long_seed_min"]) if not np.isnan(lv["compl_long_seed_min"]) else None,
                                  pct(lv["compl_long_seed_max"]) if not np.isnan(lv["compl_long_seed_max"]) else None, color_key, filled=False, lw=2.6, ms=7)
                    ax.text(max(pct(h["hi"]), pct(lv["honest_seed_max"])) + 0.4, y - 0.17, f"{pct(h['mean']):.1f}", va="center", fontsize=8)
                    ax.text(max(pct(cl["hi"]), pct(lv["compl_long_seed_max"]) if not np.isnan(lv["compl_long_seed_max"]) else 0) + 0.4, y + 0.17, f"{pct(cl['mean']):.1f}",
                            va="center", fontsize=8)
                    rows_out.append({"condition": cond, "row": f"{key}_{variant}" if variant else key, "metric": "honest_all_traces", "mean": pct(h["mean"]), "ci_lo": pct(h["lo"]),
                                     "ci_hi": pct(h["hi"]), "seed_min": pct(lv["honest_seed_min"]), "seed_max": pct(lv["honest_seed_max"]), "n_seeds": lv["n_seeds"],
                                     "n_questions": lv["n_questions"]})
                    rows_out.append({"condition": cond, "row": f"{key}_{variant}" if variant else key, "metric": "strict_compliance_among_long_traces", "mean": pct(cl["mean"]),
                                     "ci_lo": pct(cl["lo"]), "ci_hi": pct(cl["hi"]), "seed_min": pct(lv["compl_long_seed_min"]), "seed_max": pct(lv["compl_long_seed_max"]),
                                     "n_seeds": lv["n_seeds"], "n_questions": lv["n_questions"]})
                    fl = lv["long"]
                    rows_out.append({"condition": cond, "row": f"{key}_{variant}" if variant else key, "metric": "share_of_traces_as_long_as_untrained_median", "mean": pct(fl["mean"]),
                                     "ci_lo": pct(fl["lo"]), "ci_hi": pct(fl["hi"]), "seed_min": pct(lv["long_seed_min"]), "seed_max": pct(lv["long_seed_max"]),
                                     "n_seeds": lv["n_seeds"], "n_questions": lv["n_questions"]})
            ax.set_ylim(6.8, -0.7)
            ax.set_yticks(ypos)
            ax.set_yticklabels(ylab, fontsize=8.8)
            if metric == "chars":
                ax.set_xlim(0, 32)
            elif metric == "degen":
                ax.set_xlim(0, 21)
            elif metric == "compl":
                ax.set_xlim(-0.6, 11)
            else:
                ax.set_xlim(38, 62)
            if r == 0:
                ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)
        axes[r, 0].annotate({"persona": "Persona prompt", "direct": "Direct instruction", "stated": "Stated in question"}[cond], xy=(-0.62, 0.5),
                            xycoords="axes fraction", rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    hs = legend_handles(["rl_orange", "rl_mcq", "untrained", "sft"], extra=False) + [
        Line2D([], [], color=INK2, lw=3, label="95% CI (bootstrap over questions x seeds)"),
        Line2D([], [], color=AXIS, lw=1.2, label="lowest to highest seed")]
    rl_rows = [(k, v, c) for k in ("rl_orange", "rl_mcq") for v in VARIANTS for c in CONDS3]
    shorter = sum(level(df, k, v, c)["chars"]["mean"] < level(df, "untrained", "noreason", c)["chars"]["mean"] for k, v, c in rl_rows)
    cl_max = max(pct(level(df, k, v, c)["compl_long"]["mean"]) for k, v, c in rl_rows)
    CHECKS.append(f"fig4: mean reasoning length below untrained in {shorter} of {len(rl_rows)} RL arm x condition cells; max strict compliance among long traces = {cl_max:.2f}%")
    fig.suptitle(f"RL shortens reasoning in {'all' if shorter == len(rl_rows) else shorter} {len(rl_rows)} arm-by-condition cells; compliance among traces "
                 f"as long as the untrained model's stays at or below {max(cl_max, 0.1):.1f}%", x=0.005, ha="left", fontsize=12.5, fontweight="bold")
    finish(fig, "Qwen3.6-27B final checkpoints, test split, 4 fixed modes pooled, 5 seeds per arm, 500 questions. 'As long as the untrained median' = at least the untrained model's median trace length "
           "in the same test block; compliance there is strict (raw) compliance, as in 2A. Orange SFT (2A) is reasoning-off SFT, shown for reference; the 2A tables give no interval on length, degenerate share or accuracy.",
           hs, ncol=4)
    save(fig, "fig4_length_degeneration", pd.DataFrame(rows_out))


# ------------------------------------------------------------------ figure 5: training curves
def fig5(training: list[dict]) -> None:
    rows_out = []
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.2), sharex=True, layout="constrained")
    for c, variant in enumerate(VARIANTS):
        for key, color in (("rl_orange", ORANGE), ("rl_mcq", GREEN)):
            runs = [t for t in training if t["arm"] == f"{key}_{variant}"]
            if not runs:
                continue
            lab = KEYS[key]["label"].split(" (")[0]
            # row 0: training reward per iteration
            ax = axes[0, c]
            series = {}
            for t in runs:
                series[t["seed"]] = {e["iteration"]: e["reward_mean"] for e in t["progress"] if e.get("reward_mean") is not None}
            its = sorted(set.intersection(*(set(s) for s in series.values())))
            M = np.array([[series[s][i] for i in its] for s in sorted(series)])
            ax.fill_between(its, M.min(0), M.max(0), color=color, alpha=0.14, lw=0)
            for row in M:
                ax.plot(its, row, color=color, lw=0.6, alpha=0.5)
            ax.plot(its, M.mean(0), color=color, lw=2.2, label=lab)
            for i, it in enumerate(its):
                rows_out.append({"panel": "train_reward", "variant": variant, "arm": f"{key}_{variant}", "iteration": it, "mean": M[:, i].mean(), "min": M[:, i].min(), "max": M[:, i].max(),
                                 "n_seeds": M.shape[0]})
            # row 1: held-out reward at checkpoints
            ax = axes[1, c]
            held = {t["seed"]: {e["iteration"]: e["heldout"]["reward_mean"] for e in t["progress"]
                                if e.get("heldout") and e["heldout"].get("reward_mean") is not None} for t in runs}
            hi_ = sorted(set.intersection(*(set(s) for s in held.values())))
            H = np.array([[held[s][i] for i in hi_] for s in sorted(held)])
            ax.fill_between(hi_, H.min(0), H.max(0), color=color, alpha=0.14, lw=0)
            ax.plot(hi_, H.mean(0), "o-", color=color, lw=2.0, ms=5, label=lab)
            for i, it in enumerate(hi_):
                rows_out.append({"panel": "heldout_reward", "variant": variant, "arm": f"{key}_{variant}", "iteration": it, "mean": H[:, i].mean(), "min": H[:, i].min(), "max": H[:, i].max(),
                                 "n_seeds": H.shape[0]})
            # row 2: cumulative groups with gradient
            ax = axes[2, c]
            cum = []
            for t in runs:
                used = {e["iteration"]: (0 if e.get("skipped") else e.get("n_groups_used", 0)) for e in t["metrics"]}
                cum.append(np.cumsum([used.get(i, 0) for i in range(50)]))
            C = np.array(cum)
            x = np.arange(1, 51)
            ax.fill_between(x, C.min(0), C.max(0), color=color, alpha=0.14, lw=0)
            for row in C:
                ax.plot(x, row, color=color, lw=0.6, alpha=0.5)
            ax.plot(x, C.mean(0), color=color, lw=2.2, label=lab)
            for i in range(50):
                rows_out.append({"panel": "cumulative_groups_with_gradient", "variant": variant, "arm": f"{key}_{variant}", "iteration": i + 1, "mean": C[:, i].mean(),
                                 "min": C[:, i].min(), "max": C[:, i].max(), "n_seeds": C.shape[0]})
    for c, variant in enumerate(VARIANTS):
        axes[0, c].set_title(VARIANT_LABEL[variant], loc="left", fontsize=11, fontweight="bold")
        axes[0, c].set_ylim(0.25, 1.02)
        axes[1, c].set_ylim(0.25, 1.02)
        axes[2, c].set_ylim(0, 420)
        axes[2, c].set_xlabel("RL iteration")
        for r in range(3):
            axes[r, c].legend(loc="center right" if r < 2 else "upper left", fontsize=9)
    axes[0, 0].set_ylabel("Training reward\n(share of rollouts rewarded)")
    axes[1, 0].set_ylabel("Held-out reward\n(unseen prompts, every 10 its)")
    axes[2, 0].set_ylabel("Cumulative prompt groups\nthat carried gradient")
    for ax in axes[2]:
        ax.text(0.99, 0.04, "maximum possible over 50 iterations: 2,400 (axis cut)", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    t90 = {}
    for arm in ("rl_orange_noreason", "rl_orange_reason"):
        firsts = []
        for t in (t for t in training if t["arm"] == arm):
            rew = {e["iteration"]: e["reward_mean"] for e in t["progress"] if e.get("reward_mean") is not None}
            its = sorted(rew)
            ma = [np.mean([rew[j] for j in its[max(0, i - 1):i + 2]]) for i in range(len(its))]
            firsts.append(next((its[i] for i, v in enumerate(ma) if v >= 0.9), np.nan))
        t90[arm] = firsts
        CHECKS.append(f"fig5: first iteration with 3-iteration mean training reward >= 0.9, {arm}, per seed: {firsts}")
    mcq0 = {a: float(np.mean([next(e["reward_mean"] for e in t["progress"] if e.get("reward_mean") is not None) for t in training if t["arm"] == a]))
            for a in ("rl_mcq_noreason", "rl_mcq_reason")}
    CHECKS.append(f"fig5: iteration-0 training reward of generic MCQ arms: {mcq0}")
    fig.suptitle(f"Generic-MCQ reward is saturated from step 0 (about {np.mean(list(mcq0.values())):.2f}), so few of its prompt groups carry gradient;\n"
                 f"the orange reward reaches 0.9 within {int(np.nanmax(t90['rl_orange_noreason']))} (reasoning off) to {int(np.nanmax(t90['rl_orange_reason']))} (on) iterations",
                 x=0.005, ha="left", fontsize=12.5, fontweight="bold")
    finish(fig, "Training rollouts, T=1, 48 prompts x 8 rollouts per iteration, 5 seeds per arm (thick line = mean over seeds, band = lowest to highest seed, thin lines = individual seeds). "
           "Reward = 1 iff the final answer is the target option (the orange option, or the correct answer for generic MCQ); it never reads the reasoning. A prompt group is dropped (no gradient) when its "
           "8 rewards are identical. Held-out reward: 96 unseen prompts x 4 rollouts at iterations 0, 10, ..., 50. Bottom row: axis cut at 420; the maximum possible is 2,400.")
    save(fig, "fig5_training_reward", pd.DataFrame(rows_out))


# ------------------------------------------------------------------ consistency checks
def consistency(df, summary: dict) -> None:
    """Re-derive analyze.py's headline test numbers from the per-trace cache and compare."""
    bad, n = 0, 0
    lines = ["Cross-check of figure numbers against _analysis/summary.json (analyze.py's own bootstrap, same seed):"]
    for e in summary["headline"]:
        if e["split"] != "test":
            continue
        for a in e["arms"]:
            key = {"untrained": "untrained"}.get(a["arm"])
            if a["arm"].startswith("noise:"):
                continue
            key = key or a["arm"].rsplit("_", 1)[0]
            variant = e["variant"]
            lv = level(df, key, variant, e["condition"])
            for metric in ("honest", "chars", "degen", "correct"):
                n += 1
                for part in ("mean", "lo", "hi"):
                    x, y = lv[metric][part], a[metric][part]
                    if abs(x - y) > 1e-9 * max(1.0, abs(y)):
                        bad += 1
                        lines.append(f"MISMATCH {e['variant']}/{e['condition']}/{a['arm']}/{metric}/{part}: figure {x} vs analysis {y}")
        # orange - mcq difference
        if "diff" in e:
            d = diff(df, ("rl_orange", e["variant"]), ("rl_mcq", e["variant"]), e["condition"])
            n += 1
            for part in ("mean", "lo", "hi"):
                if abs(d[part] - e["diff"]["honest"][part]) > 1e-9:
                    bad += 1
                    lines.append(f"MISMATCH diff {e['variant']}/{e['condition']}/{part}: figure {d[part]} vs analysis {e['diff']['honest'][part]}")
    for s in summary["standard"]:
        if s["block"] != "stated_standard" or s["arm"].startswith("noise:"):
            continue
        key = "untrained" if s["arm"] == "untrained" else s["arm"].rsplit("_", 1)[0]
        variant = "noreason" if s["arm"] == "untrained" else s["arm"].rsplit("_", 1)[1]
        lv = level(df, key, variant, "standard")
        n += 1
        for part in ("mean", "lo", "hi"):
            if abs(lv["honest"][part] - s["honest"][part]) > 1e-9:
                bad += 1
                lines.append(f"MISMATCH standard/{s['arm']}/{part}: figure {lv['honest'][part]} vs analysis {s['honest'][part]}")
    lines.insert(1, f"{n} comparisons (arms x conditions x metrics, orange-mcq differences, 9-mode standard block); mismatches: {bad}")
    print("\n".join(lines[:2]))
    CHECKS_HEAD[:] = lines
    if bad:
        (DATA / "consistency_checks.txt").write_text("\n".join(lines) + "\n")
        raise SystemExit("figure numbers disagree with analyze.py output")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild-cache", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    style()
    check_palette([ORANGE, GREEN])
    summary = json.loads((ANALYSIS / "summary.json").read_text())
    training = json.loads((ANALYSIS / "training.json").read_text())
    df, bars = load_items(a.rebuild_cache)
    CHECKS.append("untrained long-trace bar (median chars of untrained test traces, per block): "
                  + "; ".join(f"{b}={v:.0f}" for b, v in bars.items()))
    consistency(df, summary)
    sft = load_sft_reference()
    fig1(df, sft, [])
    fig2(df, sft, summary["dose"])
    fig3(df, summary, training)
    fig4(df, sft)
    fig5(training)
    (DATA / "consistency_checks.txt").write_text("\n".join(CHECKS_HEAD + CHECKS) + "\n")


if __name__ == "__main__":
    main()
