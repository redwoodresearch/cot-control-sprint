"""Simple slide figures for the 2026-10-09 elicitation write-up (gpt-oss-20b). Reads data/interventions.csv and
data/curves.csv (from collect.py); writes figures/slides/fig{1,2,3}_*.{png,pdf}. The post versions (fig1-5, with
subtitles, CIs and the orange expansion) are built by post_figures.py, which imports the data helpers from here.

    .venv/bin/python kaushik/elicit/writeup/figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "slides"
FIG.mkdir(parents=True, exist_ok=True)


BLUE, ORANGE, GREY, INK, MUTED, GRID, GUARD = "#2a78d6", "#eb6834", "#8a8a84", "#0b0b0b", "#52514e", "#e6e6e3", "#f1f5fb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#bdbdb8", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})

df = pd.read_csv(HERE / "data/interventions.csv")
test = df[(df.split == "test") & (df.label != "steering random control (broken)")].copy()
U = test[test.label == "untrained"].iloc[0]


def pareto(d: pd.DataFrame, x: str) -> pd.DataFrame:
    """Points not dominated in (x, honest): no other point has >= x and > honest (or > x and >= honest)."""
    keep = []
    for _, r in d.iterrows():
        dom = d[((d[x] >= r[x]) & (d.honest > r.honest)) | ((d[x] > r[x]) & (d.honest >= r.honest))]
        if dom.empty:
            keep.append(r)
    return pd.DataFrame(keep).sort_values(x)


def draw_frontier(ax, d, x):
    f = pareto(d, x)
    xs, ys = list(f[x]), list(f.honest)
    px, py = [], []
    for i in range(len(xs)):  # step: best compliance achievable at this x or beyond
        px += [xs[i]]; py += [ys[i]]
        if i + 1 < len(xs):
            px += [xs[i + 1]]; py += [ys[i]]
    ax.plot(px, py, color=INK, lw=1, ls=(0, (3, 3)), alpha=0.6, zorder=1)
    return f


KEYS = [("untrained", "untrained"), ("RANDOM", "random LoRA (control)"), ("GEPA prompt (teacher)", "GEPA system prompt"),
        ("GEPA LoRA x1", "GEPA LoRA"), ("GEPA LoRA x1.5", "GEPA LoRA ×1.5"), ("GEPA LoRA x0.75 + GEPA prompt", "GEPA LoRA + GEPA prompt"),
        ("GEPA LoRA - 2 brevity LoRA", "GEPA LoRA − brevity LoRA"),
        ("1A answer-only SFT", "answer-only SFT"), ("no-CoT SFT: fmt @25", "no-CoT format SFT"),
        ("steering L12 a1", "best steering vector")]


def simple_points() -> pd.DataFrame:
    cols = ["honest", "honest_lo", "honest_hi", "chars_pct", "accuracy"]
    rnd = test[test.label.str.startswith("noise for")][cols].mean()  # mean of the 4 KL-matched random-LoRA controls
    rows = []
    for k, name in KEYS:
        r = rnd if k == "RANDOM" else test[test.label == k].iloc[0]
        rows.append({**{c: r[c] for c in cols}, "name": name})
    return pd.DataFrame(rows)



def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def base(title, xlabel, ylabel, xlim, ylim):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes([0.08, 0.14, 0.9, 0.72])
    fig.text(0.08, 0.95, title, fontsize=24, fontweight="bold", ha="left", va="top", color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel(xlabel, fontsize=18, labelpad=8); ax.set_ylabel(ylabel, fontsize=18)
    return fig, ax


def scatter_fig(d, x, title, xlabel, xlim, shade, offsets, name, pct_x=False):
    fig, ax = base(title, xlabel, "honest compliance", xlim, (0, 12))
    ax.axvspan(*shade, color=GUARD, zorder=0)
    if pct_x:
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    on = set(pareto(d, x).name)
    draw_frontier(ax, d, x)
    for _, r in d.iterrows():
        c = BLUE if r["name"] in on else GREY
        ax.errorbar(r[x], r.honest, yerr=[[r.honest - r.honest_lo], [r.honest_hi - r.honest]], fmt="none", ecolor=c,
                    elinewidth=1.4, alpha=0.6, zorder=2)
        ax.scatter(r[x], r.honest, s=140, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy, ha = offsets[r["name"]]
        ax.annotate(r["name"], (r[x], r.honest), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=16, color=INK, fontweight="bold" if r["name"] in on else "normal")
    save(fig, name)


def fig1():
    off = {"untrained": (0, -24, "center"), "random LoRA (control)": (14, -14, "left"),
           "GEPA system prompt": (14, 18, "left"), "GEPA LoRA": (0, 34, "center"), "GEPA LoRA ×1.5": (-14, 0, "right"),
           "GEPA LoRA + GEPA prompt": (-8, -26, "left"),
           "GEPA LoRA − brevity LoRA": (12, 12, "left"), "answer-only SFT": (12, 12, "left"),
           "no-CoT format SFT": (-12, 0, "right"), "best steering vector": (-12, 14, "right")}
    scatter_fig(simple_points(), "chars_pct", "More control, shorter reasoning", "reasoning length (% of untrained)", (0, 150), (80, 150),
                off, "fig1_control_vs_length", pct_x=True)


def fig2():
    off = {"untrained": (0, -24, "center"), "random LoRA (control)": (-12, -14, "right"),
           "GEPA system prompt": (12, 8, "left"), "GEPA LoRA": (0, 34, "center"), "GEPA LoRA ×1.5": (-14, 0, "right"),
           "GEPA LoRA + GEPA prompt": (-12, -6, "right"),
           "GEPA LoRA − brevity LoRA": (-12, 12, "right"), "answer-only SFT": (-12, 12, "right"),
           "no-CoT format SFT": (-12, 0, "right"), "best steering vector": (-10, 14, "right")}
    scatter_fig(simple_points(), "accuracy", "More control, lower accuracy", "accuracy", (0, 56),
                (U.accuracy - 3, U.accuracy + 3), off, "fig2_control_vs_accuracy", pct_x=True)


def fig3():
    cv = pd.read_csv(HERE / "data/curves.csv")
    arms = [("untrained", GREY, "untrained", 0), ("GEPA prompt", ORANGE, "GEPA prompt", 0), ("GEPA LoRA", BLUE, "GEPA LoRA", 0),
            ("GEPA LoRA - 2 brevity LoRA", "#1baf7a", "GEPA LoRA − brevity LoRA", 0)]
    bins = ["<1k", "1-2k", "2-2.6k", "2.6-4k", "4-8k", ">8k"]
    fig, ax = base("Control fades as reasoning gets longer", "reasoning length (characters)", "honest compliance",
                   (-2.3, 5.3), (0, 20))
    ax.set_yticks(range(0, 21, 5))
    x = np.arange(len(bins))
    ax.set_xticks(x, ["<1k", "1–2k", "2–2.6k", "2.6–4k", "4–8k", ">8k"])
    ends = []
    for key, color, name, _ in arms:
        g = cv[cv.arm == key].set_index("bin").loc[bins]
        ax.plot(x, g.honest, color=color, lw=3, marker="o", ms=7, zorder=3)
        ends.append([g.honest.iloc[0], name, color])
    ends.sort(key=lambda t: t[0])
    prev = -99
    for y0, name, color in ends:  # direct labels at the left end, kept apart
        y = max(y0, prev + 1.5); prev = y
        ax.annotate(name, (-0.12, y0), xytext=(-0.12 - 0.15, y), textcoords="data", ha="right", va="center", fontsize=16,
                    color=color if color != GREY else MUTED, fontweight="bold")
    save(fig, "fig3_compliance_by_length")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("wrote", sorted(p.name for p in FIG.glob("fig*")))
