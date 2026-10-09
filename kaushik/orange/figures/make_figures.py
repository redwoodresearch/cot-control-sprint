"""Publication figures for Experiment 2A (orange persona SFT vs its controls).

Re-run:  .venv/bin/python kaushik/orange/figures/make_figures.py
Reads only kaushik/orange/results/{cells,differences}.csv, rollouts.csv.gz and the
earlier run's committed notebook output; writes PNG (200 dpi) + PDF per figure and the
CSV behind each figure into kaushik/orange/figures/ (data/ for the CSVs).
RL arms (rl_*) are provisional and left out of every figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
DATA = HERE / "data"
NOTEBOOK = HERE.parents[2] / "orange-training" / "plotting.ipynb"
POOLED = "(pooled)"
FIXED_MODES = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "end_of_sentence"]
SFT_ARMS = ("orange", "orange_nopersona", "mcq_noreason")
CHECKS: list[str] = []

# ------------------------------------------------------------------ palette and style
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
# colour follows the arm in every figure; hues = reference-palette slots 2, 1, 3 (validated below);
# the two reference arms are neutrals, told apart by marker shape and direct labels.
ARMS = {
    "orange": {"label": "Orange SFT (persona prompt)", "color": "#eb6834", "marker": "o", "ms": 8},
    "orange_nopersona": {"label": "Orange SFT, no persona prompt", "color": "#2a78d6", "marker": "s", "ms": 7},
    "mcq_noreason": {"label": "Generic MCQ SFT", "color": "#1baf7a", "marker": "^", "ms": 8},
    "noise[orange]": {"label": "Random adapter (KL-matched)", "color": MUTED, "marker": "v", "ms": 8},
    "untrained": {"label": "Untrained model", "color": INK2, "marker": "D", "ms": 6},
}
ARM_STEP = {"orange": "50", "orange_nopersona": "50", "mcq_noreason": "50", "noise[orange]": "noise", "untrained": "0"}
MODELS = [("qwen36_27b", "Qwen3.6-27B"), ("gptoss20b", "gpt-oss-20b")]
CONDS = [  # (condition, kind of test block, column title)
    ("persona", "fixed", "Persona prompt asks for the behaviour\n(4 fixed-trait modes)"),
    ("direct", "fixed", "Behaviour asked for plainly (direct)\n(4 fixed-trait modes)"),
    ("stated", "standard", "Standard eval: constraint stated in the question\n(9 standard modes)"),
]
COND_SHORT = {"persona": "Persona prompt", "direct": "Direct instruction", "stated": "Standard eval (stated)"}


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
    """Python port of the dataviz skill's validate_palette.js (light mode, all pairs)."""
    machado = {
        "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
        "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
        "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
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
        lum = lambda h: lin(h) @ np.array([0.2126, 0.7152, 0.0722])
        hi, lo = sorted((lum(a), lum(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    pairs = [(a, b) for i, a in enumerate(colors) for b in colors[i + 1:]]
    lch = [(lab(lin(c))[0], np.hypot(*lab(lin(c))[1:])) for c in colors]
    cvd = min(de(a, b, k) for a, b in pairs for k in ("protan", "deutan"))
    nor = min(de(a, b) for a, b in pairs)
    con = {c: round(float(contrast(c, surface)), 2) for c in colors}
    lines = [f"palette {colors} on surface {surface} (light mode, all pairs; hues = reference slots 2,1,3)",
             (f"lightness band 0.43-0.77: {'PASS' if all(0.43 <= L <= 0.77 for L, _ in lch) else 'FAIL'} "
             f"(L = {[round(float(L), 3) for L, _ in lch]})"),
             (f"chroma floor 0.10: {'PASS' if all(c >= 0.10 for _, c in lch) else 'FAIL'} "
             f"(C = {[round(float(c), 3) for _, c in lch]})"),
             f"CVD separation (min protan/deutan dE x100, target >= 8): {'PASS' if cvd >= 8 else 'FAIL'} ({cvd:.1f})",
             f"normal-vision floor (dE x100 >= 15): {'PASS' if nor >= 15 else 'FAIL'} ({nor:.1f})",
             (f"contrast vs surface (>= 3:1 or relief): {con} -> "
             f"{'PASS' if min(con.values()) >= 3 else 'WARN: relief via direct labels, legend and marker shapes'}"),
             "neutral reference arms (random adapter, untrained) are told apart by marker shape and labels, not hue"]
    (DATA / "palette_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if cvd < 8 or nor < 15:
        raise SystemExit("palette fails the dataviz gates")


# ------------------------------------------------------------------ helpers
def save(fig, stem: str, df: pd.DataFrame | None = None) -> None:
    fig.savefig(HERE / f"{stem}.png", dpi=200)
    fig.savefig(HERE / f"{stem}.pdf")
    plt.close(fig)
    if df is not None:
        df.to_csv(DATA / f"{stem}.csv", index=False)
    print(f"saved {stem}")


def footer(fig, text: str) -> None:
    fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.925))
    fig.text(0.005, 0.006, text, fontsize=8, color=INK2, ha="left", va="bottom", linespacing=1.4)


def cell(cells, model, split, kind, cond, arm, step, mode=POOLED):
    s = cells[(cells.model == model) & (cells.split == split) & (cells.kind == kind) & (cells["mode"] == mode)
              & (cells.condition == cond) & (cells.arm == arm) & (cells.step == step)]
    assert len(s) <= 1, (model, split, kind, cond, arm, step, len(s))
    return s.iloc[0] if len(s) else None


def load_rollouts() -> pd.DataFrame:
    cols = ["experiment", "arm", "model", "seed", "condition", "kind", "split", "step", "mode", "qid",
            "compliance", "honest", "chars", "words"]
    r = pd.read_csv(RES / "rollouts.csv.gz", usecols=cols, low_memory=False, dtype={"step": str})
    return r


def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


# ------------------------------------------------------------------ figure 1: levels
def fig1(cells: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.9), sharex=True, sharey=True, layout="constrained")
    out = []
    for r, (model, mlabel) in enumerate(MODELS):
        for c, (cond, kind, ctitle) in enumerate(CONDS):
            ax = axes[r, c]
            ax.grid(axis="y", visible=False)
            for i, arm in enumerate(ARMS):
                s = cell(cells, model, "test", kind, cond, arm, ARM_STEP[arm])
                st = ARMS[arm]
                if s is None:
                    ax.text(0.25, i, "not run", va="center", fontsize=8.5, color=MUTED, style="italic")
                    continue
                v, lo, hi, smin, smax = (100 * s[k] for k in ("honest", "honest_lo", "honest_hi",
                                                              "honest_seed_min", "honest_seed_max"))
                out.append({"model": model, "condition": cond, "eval_set": kind, "split": "test", "arm": arm,
                                "step": s.step, "n_rollouts": int(s.n), "n_seeds": int(s.n_seeds), "honest_pct": v,
                                "ci95_lo": lo, "ci95_hi": hi, "seed_min": smin, "seed_max": smax})
                ax.plot([smin, smax], [i, i], color=AXIS if arm != "orange" else "#f2b79f", lw=1.2, zorder=2,
                        solid_capstyle="butt")
                for x in (smin, smax):
                    ax.plot([x, x], [i - 0.12, i + 0.12], color=AXIS if arm != "orange" else "#f2b79f", lw=1.2, zorder=2)
                ax.plot([lo, hi], [i, i], color=st["color"], lw=3.2, zorder=3, solid_capstyle="butt")
                ax.plot([v], [i], marker=st["marker"], ms=st["ms"], color=st["color"], mec=SURFACE, mew=1.4,
                        zorder=4, ls="")
                if arm == "orange" or v >= 1.0:
                    ax.text(max(hi, smax) + 0.45, i, fmt_pct(v), va="center", fontsize=9, color=INK,
                            fontweight="bold" if arm == "orange" else "normal")
            ax.set_xlim(-0.8, 12.8)
            ax.set_xticks([0, 4, 8, 12])
            ax.set_ylim(len(ARMS) - 0.45, -0.55)
            ax.set_yticks(range(len(ARMS)))
            ax.set_yticklabels([a["label"] for a in ARMS.values()])
            if r == 0:
                ax.set_title(ctitle, loc="left", fontsize=9.5, color=INK, pad=6)
            if r == 1:
                ax.set_xlabel("Honest compliance (%)")
    for r, (_, mlabel) in enumerate(MODELS):
        axes[r, 0].set_ylabel(mlabel, fontsize=11, fontweight="bold", color=INK, labelpad=8)
    handles = [Line2D([], [], marker=a["marker"], color=a["color"], ls="", ms=a["ms"], mec=SURFACE, label=a["label"])
               for a in ARMS.values()]
    handles += [Line2D([], [], ls="", label=" "), Line2D([], [], color=INK2, lw=3.2, label="95% CI (bootstrap over questions x seeds)"),
                Line2D([], [], color=AXIS, lw=1.2, label="lowest to highest training seed")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8.5, columnspacing=1.6, handletextpad=0.5)
    fig.suptitle("Orange SFT raises honest compliance only with the persona prompt, and only on Qwen3.6-27B",
                 x=0.005, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, "fig1_main_result", pd.DataFrame(out))


# ------------------------------------------------------------------ figure 2: paired differences
def fig2(diffs: pd.DataFrame) -> None:
    comps = [("orange_nopersona", "vs same data, no persona prompt"), ("mcq_noreason", "vs generic MCQ SFT"),
             ("noise[orange]", "vs random adapter (KL-matched)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharex=True, sharey=True, layout="constrained")
    out = []
    ypos = lambda g, j: g * 4.0 + j
    for ax, (model, mlabel) in zip(axes, MODELS):
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK2, lw=1.0, zorder=1)
        for g, (cond, kind, ctitle) in enumerate(CONDS):
            ax.text(-6.7, ypos(g, 0) - 0.95, COND_SHORT[cond] + (" (4 modes)" if kind == "fixed" else " (9 modes)"),
                    fontsize=9.5, fontweight="bold", va="center", color=INK,
                    bbox={"fc": SURFACE, "ec": "none", "pad": 1.5}, zorder=5)
            for j, (b, _) in enumerate(comps):
                y, st = ypos(g, j), ARMS[b]
                for split, k, dy in (("test", kind, -0.13), ("val", "fixed", 0.15)):
                    s = diffs[(diffs.model == model) & (diffs.split == split) & (diffs.kind == k) & (diffs.condition == cond)
                              & (diffs.a == "orange") & (diffs.b == b) & (diffs["mode"] == POOLED)]
                    if s.empty:
                        if split == "test":
                            ax.text(-6.3, y, "not run", va="center", fontsize=8.5, color=MUTED, style="italic")
                        continue
                    s = s.iloc[0]
                    v, lo, hi = (100 * s[x] for x in ("honest", "honest_lo", "honest_hi"))
                    out.append({"model": model, "condition": cond, "eval_set": k, "split": split, "comparison": f"orange - {b}",
                                    "delta_pp": v, "ci95_lo": lo, "ci95_hi": hi, "seeds_a": int(s.seeds_a), "seeds_b": int(s.seeds_b)})
                    test = split == "test"
                    ax.plot([lo, hi], [y + dy] * 2, color=st["color"], lw=2.8 if test else 1.4, zorder=3,
                            solid_capstyle="butt", alpha=1 if test else 0.8)
                    ax.plot([v], [y + dy], marker=st["marker"], ms=8 if test else 6, ls="", zorder=4,
                            mfc=st["color"] if test else SURFACE, mec=st["color"],
                            mew=1.4 if test else 1.5)
                    sig = (lo > 0 or hi < 0) and test
                    if test and (sig or abs(v) >= 1):
                        ax.text(max(hi, 0.0) + 0.35, y + dy, f"{v:+.1f}", va="center", fontsize=8.5, color=INK,
                                fontweight="bold" if sig else "normal")
        ax.set_title(mlabel, loc="left", fontsize=11, fontweight="bold", pad=6)
        ax.set_xlim(-7, 11.5)
        ax.set_xticks([-6, -3, 0, 3, 6, 9])
        ax.set_xlabel("Orange SFT minus comparison (percentage points of honest compliance)")
    ys = [ypos(g, j) for g in range(3) for j in range(3)]
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([c[1] for _ in range(3) for c in comps])
    axes[0].set_ylim(ypos(2, 2) + 0.7, -1.7)
    handles = [Line2D([], [], marker="o", color=INK2, ls="", ms=8, label="test split, final checkpoint (primary)"),
               Line2D([], [], marker="o", mfc=SURFACE, mec=INK2, color=INK2, ls="", ms=6,
                      label="val split, 4 fixed modes (agreement check)")]
    handles += [Line2D([], [], marker=ARMS[b]["marker"], color=ARMS[b]["color"], ls="", ms=7, label=f"orange SFT {lab}")
                for b, lab in comps]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8.5, columnspacing=1.6)
    fig.suptitle("The persona-prompt gain beats every control; elsewhere the gaps are small, borderline or negative",
                 x=0.005, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, "fig2_paired_differences", pd.DataFrame(out))


# ------------------------------------------------------------------ figure 3: length
def band_ci(d: pd.DataFrame, band: pd.Series, draws, n_boot: int = 2000) -> tuple[float, float, float, int]:
    """Honest compliance among traces in a length band; 95% CI resamples seeds and questions
    independently (as the results tables do), pooling traces within each (seed, question)."""
    g = d.assign(k=(d.honest & band).astype(int), n=band.astype(int)).groupby(["seed", "qid"])[["k", "n"]].sum()
    k, n = g.k.unstack(fill_value=0).to_numpy(), g.n.unstack(fill_value=0).to_numpy()
    if n.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    rows, cols = draws(k.shape)
    ratio = np.array([k[np.ix_(rows[b], cols[b])].sum() / max(n[np.ix_(rows[b], cols[b])].sum(), 1) for b in range(n_boot)])
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    return k.sum() / n.sum(), lo, hi, int(n.sum())


def fig3(cells: pd.DataFrame, r: pd.DataFrame) -> None:
    q = r[(r.model == "qwen36_27b") & (r.kind == "fixed") & (r.condition == "persona") & r["mode"].isin(FIXED_MODES)]
    # untrained median reasoning length per mode (val, checkpoint 0 of the SFT arms) = the table's compl_long bar
    bars = q[(q.split == "val") & (q.step == "0") & q.arm.isin(SFT_ARMS)].groupby("mode").chars.median()
    bar_mean = float(bars.mean())
    test = q[q.split == "test"].assign(bar=lambda d: d["mode"].map(bars))
    groups = {
        "untrained": test[(test.step == "0") & test.arm.isin(SFT_ARMS)],
        "orange": test[(test.arm == "orange") & (test.step == "50")],
        "orange_nopersona": test[(test.arm == "orange_nopersona") & (test.step == "50")],
        "mcq_noreason": test[(test.arm == "mcq_noreason") & (test.step == "50")],
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1.35, 1.0]})
    # (a) distribution of reasoning length
    ax = axes[0]
    ax.grid(axis="x", visible=False)
    qs, qrows, short = np.linspace(0, 1, 1001), [], {}
    short_label = {"untrained": "Untrained", "orange": "Orange SFT", "orange_nopersona": "Orange, no persona prompt",
                   "mcq_noreason": "Generic MCQ SFT"}
    for arm, d in groups.items():
        xs = np.maximum(np.quantile(d.chars, qs), 10)
        short[arm] = float((d.chars < d.bar).mean())
        mean_k = cell(cells, "qwen36_27b", "test", "fixed", "persona", arm, ARM_STEP[arm])["chars"] / 1000
        st = ARMS[arm]
        ax.plot(xs, qs * 100, color=st["color"], lw=2.4 if arm == "orange" else 1.6, zorder=3 if arm == "orange" else 2,
                label=f"{short_label[arm]}: {mean_k:.1f}k")
        qrows += [{"arm": arm, "quantile": float(a), "chars": float(b), "mean_chars_k": mean_k,
                       "share_shorter_than_untrained_median": short[arm]} for a, b in zip(qs, xs)]
    ax.axvline(bar_mean, color=INK2, lw=1.0, ls=(0, (1, 2)))
    ax.text(bar_mean * 1.07, 3, "untrained median\n(about 9.3k characters)", fontsize=8.5, color=INK2, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(60, 6e4)
    ax.set_xticks([100, 1000, 10000])
    ax.set_xticklabels(["100", "1k", "10k"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Reasoning length per trace (characters, log scale)")
    ax.set_ylabel("Share of traces at most this long (%)")
    ax.set_title("(a) Orange SFT shortens persona traces", loc="left", fontsize=10.5, fontweight="bold", pad=8)
    for arm in ("orange", "untrained"):
        ax.plot([bar_mean], [100 * short[arm]], marker=ARMS[arm]["marker"], ms=7, color=ARMS[arm]["color"], mec=SURFACE, mew=1.2,
                ls="", zorder=5)
    ax.text(1.3e4, 40, f"Shorter than the untrained\nmedian: {100 * short['orange']:.0f}% of orange\ntraces vs {100 * short['untrained']:.0f}% of untrained",
            fontsize=8.5, color=INK, va="top")
    ax.legend(loc="upper left", fontsize=8, handlelength=1.6, borderaxespad=0.2, title="mean length (characters)",
              title_fontsize=8, alignment="left")
    # (b) honest compliance by length band
    ax = axes[1]
    ax.grid(axis="x", visible=False)
    bands = [(0, 50, "< 50"), (50, 200, "50-200"), (200, 800, "200-800"), (800, 3200, "800-3,200"), (3200, 10**9, "> 3,200")]
    v = q[(q.split == "val") & q.step.isin(["10", "20", "30", "40", "50"])]
    rng, brows = np.random.default_rng(0), []
    pre = {}

    def draws(shape):  # same seed/question resamples for every band and arm with this shape
        if shape not in pre:
            pre[shape] = (rng.integers(0, shape[0], (2000, shape[0])), rng.integers(0, shape[1], (2000, shape[1])))
        return pre[shape]

    for arm, dx in (("orange", -0.2), ("orange_nopersona", 0.0), ("mcq_noreason", 0.2)):
        d, st = v[v.arm == arm], ARMS[arm]
        w = d.words.clip(lower=1)
        for bi, (lo_w, hi_w, lab) in enumerate(bands):
            m, lo, hi, n = band_ci(d, (w >= lo_w) & (w < hi_w), draws)
            raw = d.compliance[(w >= lo_w) & (w < hi_w)].mean()
            brows.append({"arm": arm, "band_words": lab, "n_traces": n, "honest_pct": 100 * m, "ci95_lo": 100 * lo, "ci95_hi": 100 * hi,
                              "raw_compliance_pct": 100 * raw})
            ax.plot([bi + dx] * 2, [100 * lo, 100 * hi], color=st["color"], lw=2.4, solid_capstyle="butt", zorder=3)
            ax.plot([bi + dx], [100 * m], marker=st["marker"], ms=st["ms"] - (1 if arm == "orange" else 0), ls="",
                    mfc=SURFACE if n < 300 else st["color"], mec=st["color"], mew=1.5, zorder=4)
            if arm == "orange":
                ax.text(bi + dx, 100 * hi + 0.6, fmt_pct(100 * m), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b[2] for b in bands])
    ax.set_xlim(-0.6, len(bands) - 0.4)
    ax.set_ylim(-0.8, 22)
    ax.set_xlabel("Reasoning length of the trace (words)")
    ax.set_ylabel("Honest compliance (%)")
    ax.set_title("(b) Compliance sits in short traces", loc="left", fontsize=10.5, fontweight="bold", pad=8)
    hs = [Line2D([], [], marker=ARMS[a]["marker"], color=ARMS[a]["color"], ls="", ms=7, label=ARMS[a]["label"])
          for a in ("orange", "orange_nopersona", "mcq_noreason")]
    hs.append(Line2D([], [], marker="o", mfc=SURFACE, mec=INK2, color=INK2, ls="", ms=6, label="hollow: fewer than 300 traces in band"))
    ax.legend(handles=hs, loc="upper right", fontsize=8, borderaxespad=0.3)
    # (c) all traces vs traces at least as long as the untrained median
    ax = axes[2]
    ax.grid(axis="y", visible=False)
    crows = []
    c_arms = ("orange", "orange_nopersona", "mcq_noreason")
    for i, arm in enumerate(c_arms):
        s = cell(cells, "qwen36_27b", "test", "fixed", "persona", arm, "50")
        st = ARMS[arm]
        a_v, a_lo, a_hi = (100 * s[k] for k in ("honest", "honest_lo", "honest_hi"))
        l_v, l_lo, l_hi = (100 * s[k] for k in ("compl_long", "compl_long_lo", "compl_long_hi"))
        frac_long = s.n_long / s.n
        crows += [{"arm": arm, "subset": "all traces", "pct": a_v, "ci95_lo": a_lo, "ci95_hi": a_hi, "n_rollouts": int(s.n),
                       "metric": "honest compliance"},
                  {"arm": arm, "subset": "traces at least as long as untrained median", "pct": l_v, "ci95_lo": l_lo, "ci95_hi": l_hi,
                       "n_rollouts": int(s.n_long), "metric": "compliance (equals honest compliance at this length)"}]
        ax.plot([l_v, a_v], [i, i], color=AXIS, lw=1.2, zorder=1)
        ax.plot([a_lo, a_hi], [i - 0.1, i - 0.1], color=st["color"], lw=2.8, solid_capstyle="butt", zorder=3)
        ax.plot([a_v], [i - 0.1], marker=st["marker"], ms=st["ms"], color=st["color"], mec=SURFACE, mew=1.2, ls="", zorder=4)
        ax.plot([l_lo, l_hi], [i + 0.1, i + 0.1], color=st["color"], lw=2.8, solid_capstyle="butt", zorder=3)
        ax.plot([l_v], [i + 0.1], marker=st["marker"], ms=st["ms"] - 1, mfc=SURFACE, mec=st["color"], mew=1.6, ls="", zorder=4)
        if arm == "orange":
            ax.text(a_hi + 0.35, i - 0.1, fmt_pct(a_v), va="center", fontsize=9, fontweight="bold")
            ax.text(l_hi + 0.35, i + 0.1, fmt_pct(l_v), va="center", fontsize=9, fontweight="bold")
        ax.text(3.2, i + 0.1, f"{100 * frac_long:.0f}% of its traces are that long", fontsize=8, color=INK2, va="center")
    ax.set_xlim(-0.6, 10.5)
    ax.set_ylim(3.5, -0.6)
    ax.set_yticks(range(3))
    ax.set_yticklabels([ARMS[a]["label"].replace(" (persona prompt)", "\n(persona prompt)").replace(", no persona prompt", ",\nno persona prompt")
                        for a in c_arms])
    ax.set_xlabel("Honest compliance (%)")
    ax.set_title("(c) At the untrained model's length: ~0", loc="left", fontsize=10.5, fontweight="bold", pad=8)
    hs = [Line2D([], [], marker="o", color=INK2, ls="", ms=7, label="all traces"),
          Line2D([], [], marker="o", mfc=SURFACE, mec=INK2, mew=1.6, color=INK2, ls="", ms=6,
                 label="only traces at least as long as the untrained median")]
    ax.legend(handles=hs, loc="lower left", fontsize=8, borderaxespad=0.3, bbox_to_anchor=(0.0, 0.0))
    fig.suptitle("Qwen3.6-27B, persona condition: the gain lives in shortened traces", x=0.005, ha="left", fontsize=12.5, fontweight="bold")
    footer(fig, "(a) and (c): test split, final checkpoint, 5 seeds x 500 questions x 4 modes; (c) intervals are the tables' 95% bootstrap over questions x seeds.\n"
                "(b): val split, checkpoints 10-50 pooled (5 seeds x 200 questions x 4 modes x 5 checkpoints); 95% CI = bootstrap resampling seeds and questions (2,000 draws).")
    save(fig, "fig3_length", None)
    pd.DataFrame(qrows).to_csv(DATA / "fig3a_length_quantiles.csv", index=False)
    bdf = pd.DataFrame(brows)
    bdf.to_csv(DATA / "fig3b_compliance_by_length_band.csv", index=False)
    pd.DataFrame(crows).to_csv(DATA / "fig3c_all_vs_long.csv", index=False)
    # consistency checks against the tables
    lm = pd.read_csv(RES / "length_matched.csv")
    names = {"0-50w": "< 50", "50-200w": "50-200", "200-800w": "200-800", "800-3200w": "800-3,200", "3200+w": "> 3,200"}
    worst = 0.0
    for arm in c_arms:
        row = lm[(lm.model == "qwen36_27b") & (lm.condition == "persona") & (lm.arm == arm)].iloc[0]
        for col, lab in names.items():
            mine = bdf[(bdf.arm == arm) & (bdf.band_words == lab)].raw_compliance_pct.iloc[0] / 100
            worst = max(worst, abs(mine - row[col]))
    CHECKS.append(f"fig3b: raw compliance per length band recomputed from rollouts vs length_matched.csv, max abs diff {worst:.1e}")
    s = cell(cells, "qwen36_27b", "test", "fixed", "persona", "orange", "50")
    o = groups["orange"]
    CHECKS.append(f"fig3c: compl_long over all traces {100 * o[o.chars >= o.bar].compliance.mean():.3f}% vs cells.csv {100 * s.compl_long:.3f}% "
                  "(cells.csv averages per-(seed, task) means)")
    share = (o.honest & (o.chars >= o.bar)).sum() / o.honest.sum()
    CHECKS.append(f"fig3: {100 * (1 - share):.1f}% of orange honest-compliant persona traces (test) are shorter than the untrained median")


# ------------------------------------------------------------------ figure 4: curves
def fig4(cells: pd.DataFrame, r: pd.DataFrame, model: str, mlabel: str, stem: str) -> None:
    arms = [a for a in ("orange", "orange_nopersona", "mcq_noreason") if not cells[(cells.model == model) & (cells.arm == a)].empty]
    sub = cells[(cells.model == model) & (cells.split == "val") & (cells.kind == "fixed") & (cells["mode"] == POOLED)
                & cells.step.isin([str(s) for s in range(0, 51, 10)])]
    rr = r[(r.model == model) & (r.split == "val") & (r.kind == "fixed") & (r.arm == "orange") & (r.experiment == "orange")
           & r["mode"].isin(FIXED_MODES) & r.step.isin([str(s) for s in range(0, 51, 10)])].assign(step=lambda d: d.step.astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True, sharey="row", layout="constrained")
    out = []
    for c, (cond, _, ctitle) in enumerate(CONDS):
        ctitle = ctitle.split("\n")[0].replace("(direct)", "").strip() if cond != "stated" else "Standard-style eval: constraint stated in the question"
        for row, (metric, ylab) in enumerate((("honest", "Honest compliance (%)"), ("chars", "Mean reasoning length (thousand characters)"))):
            ax = axes[row, c]
            ax.grid(axis="x", visible=False)
            for arm in arms:
                s = sub[(sub.condition == cond) & (sub.arm == arm)].assign(step=lambda d: d.step.astype(int)).sort_values("step")
                if s.empty:
                    continue
                st = ARMS[arm]
                scale = 100 if metric == "honest" else 1 / 1000
                x, y = s.step.to_numpy(), scale * s[metric].to_numpy(dtype=float)
                if metric == "honest":
                    ax.fill_between(x, 100 * s.honest_lo, 100 * s.honest_hi, color=st["color"], alpha=0.11, lw=0)
                for _, rw in s.iterrows():
                    out.append({"model": model, "condition": cond, "arm": arm, "step": int(rw.step), "n_rollouts": int(rw.n),
                                    "n_seeds": int(rw.n_seeds), "honest_pct": 100 * rw.honest, "ci95_lo": 100 * rw.honest_lo,
                                    "ci95_hi": 100 * rw.honest_hi, "seed_min": 100 * rw.honest_seed_min,
                                    "seed_max": 100 * rw.honest_seed_max, "mean_chars": rw.chars})
                if arm == "orange":
                    per = rr[rr.condition == cond].groupby(["seed", "step"])[("honest" if metric == "honest" else "chars")].mean().unstack(0)
                    for sd in per.columns:
                        ax.plot(per.index, scale * per[sd], color=st["color"], lw=0.8, alpha=0.4, zorder=2)
                    if metric == "honest":
                        chk = np.abs(100 * per.min(axis=1).to_numpy() - 100 * s.honest_seed_min.to_numpy()).max()
                        CHECKS.append(f"fig4 {model} {cond}: per-seed min recomputed from rollouts vs cells.csv, max abs diff {chk:.1e} pp")
                ax.plot(x, y, color=st["color"], lw=1.9, marker=st["marker"], ms=st["ms"] - 2.5, mec=SURFACE, mew=1.0,
                        zorder=4 if arm == "orange" else 3)
            ax.set_xticks(range(0, 51, 10))
            if row == 0:
                ax.set_title(ctitle, loc="left", fontsize=9.5, color=INK, pad=6)
                ax.set_ylim(-0.4, 14)
            else:
                ax.set_ylim(0, 27)
                ax.set_xlabel("SFT step (0 = untrained)")
            if c == 0:
                ax.set_ylabel(ylab)
    hs = [Line2D([], [], marker=ARMS[a]["marker"], color=ARMS[a]["color"], lw=1.9, ms=6, label=ARMS[a]["label"]) for a in arms]
    hs += [Line2D([], [], color=INK2, lw=6, alpha=0.2, label="95% CI (bootstrap over questions x seeds)"),
           Line2D([], [], color=ARMS["orange"]["color"], lw=0.8, alpha=0.5, label="orange SFT, one line per seed")]
    fig.legend(handles=hs, loc="outside lower center", ncol=3, fontsize=8.5, columnspacing=1.6)
    fig.suptitle(f"{mlabel}: honest compliance and trace length across SFT steps (val split, 4 fixed-trait modes pooled)", x=0.005, ha="left",
                 fontsize=12.5, fontweight="bold")
    save(fig, stem, pd.DataFrame(out))


# ------------------------------------------------------------------ figure 5: earlier run
def earlier_run() -> pd.DataFrame:
    """Per-seed raw persona compliance (%) by step, from the printed table in the earlier notebook."""
    for c in json.loads(NOTEBOOK.read_text())["cells"]:
        for o in c.get("outputs", []):
            lines = "".join(o.get("text", "")).splitlines()
            if lines and lines[0].split()[:2] == ["step", "0"] and len(lines) > 2 and lines[1].strip() == "seed":
                steps = [int(x) for x in lines[0].split()[1:]]
                rows = {int(ln.split()[0]): [float(x) for x in ln.split()[1:]] for ln in lines[2:] if ln.strip()}
                return pd.DataFrame(rows, index=steps).T
    raise SystemExit("per-seed table not found in orange-training/plotting.ipynb")


def fig5(cells: pd.DataFrame, r: pd.DataFrame) -> None:
    old = earlier_run()
    new = r[(r.model == "qwen36_27b") & (r.split == "val") & (r.kind == "fixed") & (r.arm == "orange") & (r.experiment == "orange")
            & (r.condition == "persona") & (r["mode"] == "lowercase_thinking")]
    new = (100 * new.assign(step=new.step.astype(int)).groupby(["seed", "step"]).compliance.mean().unstack(0)).T
    ref = cells[(cells.model == "qwen36_27b") & (cells.split == "val") & (cells.kind == "fixed") & (cells.condition == "persona")
                & (cells["mode"] == "lowercase_thinking") & (cells.arm == "orange") & cells.step.isin(["0", "10", "20", "30", "40", "50"])]
    CHECKS.append("fig5: this-run mean raw compliance (lowercase, persona, val) over all traces vs cells.csv, max abs diff "
                  f"{np.abs(new.mean().to_numpy() - 100 * ref.sort_values('step', key=lambda s: s.astype(int)).compliance.to_numpy()).max():.2f} pp "
                  "(cells.csv merges questions whose ids collide across source datasets, see below)")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, sharey=True, layout="constrained")
    out = []
    panels = [(old, "Earlier run (orange-training/)", "3 seeds, temperature 0 (harness default),\n12,000 max tokens, 2,086-item training pool"),
              (new, "This rerun (kaushik/orange/)", "5 seeds, temperature 1, 16,000 max tokens,\nregenerated 2,136-item training pool")]
    st, peaks = ARMS["orange"], []
    for ax, (df, head, sub), dy in zip(axes, panels, (-16, 30)):
        ax.grid(axis="x", visible=False)
        for sd, row in df.iterrows():
            ax.plot(row.index, row.values, color=st["color"], lw=0.9, alpha=0.45, marker="o", ms=3, zorder=2)
            for stp, val in row.items():
                out.append({"run": head, "seed": sd, "step": int(stp), "raw_compliance_pct": val})
        m = df.mean()
        ax.plot(m.index, m.values, color=st["color"], lw=2.4, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=4)
        pk = m.idxmax()
        ax.annotate(f"mean peaks at {m[pk]:.0f}%\n(step {pk})", xy=(pk, m[pk]), xytext=(pk + 1.5, m[pk] + dy), fontsize=9, ha="left",
                    va="top", arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
        peaks.append(m[pk])
        ax.text(0.02, 0.97, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="top")
        ax.set_title(head, loc="left", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xticks(range(0, 51, 10))
        ax.set_xlabel("SFT step (0 = untrained)")
        ax.set_ylim(-1, 62)
    axes[0].set_ylabel("Raw compliance (%)\nfollows the rule; hollow traces not excluded")
    fig.suptitle(f"Context only, not like-for-like: mean raw compliance peaked at {peaks[0]:.0f}% in the earlier run and {peaks[1]:.0f}% in this rerun",
                 x=0.005, ha="left", fontsize=12.5, fontweight="bold")
    footer(fig, "Both panels: Qwen3.6-27B, persona condition, lowercase_thinking mode only, val split; thin lines = individual seeds, thick line = mean over seeds.\n"
                "Earlier-run numbers are the per-seed table printed in orange-training/plotting.ipynb; this rerun is computed from results/rollouts.csv.gz.")
    save(fig, "fig5_earlier_run_context", pd.DataFrame(out))


def main() -> None:
    DATA.mkdir(exist_ok=True)
    style()
    check_palette([ARMS[a]["color"] for a in ("orange_nopersona", "orange", "mcq_noreason")])
    cells = pd.read_csv(RES / "cells.csv", dtype={"step": str})
    diffs = pd.read_csv(RES / "differences.csv", dtype={"step_a": str, "step_b": str})
    r = load_rollouts()
    for split in ("val", "test"):
        d = r[(r.split == split) & (r.kind == "fixed") & (r.arm == "orange") & (r.model == "qwen36_27b") & (r.step == "50")
              & (r.condition == "persona") & (r["mode"] == "lowercase_thinking") & (r.seed == 0)]
        CHECKS.append(f"{split}: {len(d)} questions per (seed, mode) but only {d.qid.nunique()} distinct qid; analyze.py keys tasks as qid|mode, "
                      "so repeated ids (likely per-source-dataset indices) are merged in cells.csv/differences.csv (means shift by <= 0.4 pp)")
    fig1(cells)
    fig2(diffs)
    fig3(cells, r)
    fig4(cells, r, "qwen36_27b", "Qwen3.6-27B", "fig4_training_curves_qwen36_27b")
    fig4(cells, r, "gptoss20b", "gpt-oss-20b", "fig4b_training_curves_gptoss20b")
    fig5(cells, r)
    (DATA / "consistency_checks.txt").write_text("\n".join(CHECKS) + "\n")
    print("\n".join(CHECKS))


if __name__ == "__main__":
    main()
