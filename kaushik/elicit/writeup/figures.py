"""Slide figures for the 2026-10-09 elicitation write-up (gpt-oss-20b). Reads data/interventions.csv and
data/curves.csv (from collect.py); writes figures/fig{1,2,3}_*.{png,pdf}.

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
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# dataviz reference palette in its fixed slot order (first three slots validated for all-pairs use in scatter;
# the five-slot order is the validated adjacent order for the line chart). Text wears ink, never series colour.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY, INK, MUTED, GRID, GUARD = "#8a8a84", "#0b0b0b", "#52514e", "#e6e6e3", "#f1f5fb"
FAMILY_COLOR = {"lora": BLUE, "prompt": ORANGE, "sft": AQUA, "control": GREY, "steer": GREY}
FAMILY_MARKER = {"lora": "o", "prompt": "s", "sft": "D", "control": "o", "steer": "^"}
FAMILY_LABEL = {"lora": "GEPA-prompt LoRA", "prompt": "system prompt", "sft": "no-CoT SFT",
                "control": "untrained / random control", "steer": "steering vector"}
DEF = "honest compliance = followed the instruction about its reasoning AND answered AND reasoning not empty/degenerate."

plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#bdbdb8", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK})

df = pd.read_csv(HERE / "data/interventions.csv")
df = df[df.label != "steering random control (broken)"]
test = df[(df.split == "test")].copy()
val_full = df[(df.split == "val") & (df.subset == "full")].copy()
val_40 = df[(df.split == "val") & (df.subset == "40q")].copy()
U = test[test.label == "untrained"].iloc[0]


def frame(title: str, subtitle: str, footnote: str):
    fig = plt.figure(figsize=(12, 6.75), facecolor="white")
    fig.text(0.015, 0.968, title, fontsize=18, fontweight="bold", ha="left", va="top", linespacing=1.25, color=INK)
    fig.text(0.015, 0.868, subtitle, fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.4)
    fig.text(0.015, 0.012, footnote, fontsize=9.5, color="#777770", ha="left", va="bottom", linespacing=1.4)
    return fig


def scatter(ax, d: pd.DataFrame, x: str, alpha=1.0, size=70, whiskers=True):
    for fam, g in d.groupby("family"):
        if whiskers:
            ax.errorbar(g[x], g.honest, yerr=[g.honest - g.honest_lo, g.honest_hi - g.honest],
                        fmt="none", ecolor=FAMILY_COLOR[fam], elinewidth=1, capsize=0, alpha=0.55 * alpha, zorder=2)
        ax.scatter(g[x], g.honest, s=size, marker=FAMILY_MARKER[fam], facecolor=FAMILY_COLOR[fam],
                   edgecolor="white", linewidth=1.2, alpha=alpha, zorder=3)


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


def label(ax, r, text, dx, dy, x="chars_pct", ha="left", fs=10.5, weight="normal", color=INK):
    far = abs(dx) > 16 or abs(dy) > 16
    ax.annotate(text, (r[x], r.honest), xytext=(dx, dy), textcoords="offset points", fontsize=fs, color=color,
                ha=ha, va="center", fontweight=weight,
                arrowprops={"arrowstyle": "-", "color": "#b0b0aa", "lw": 0.7, "shrinkA": 0, "shrinkB": 4} if far else None)


def row(d, lab):
    return d[d.label == lab].iloc[0]


def legend(ax, fams, loc, **kw):
    hs = [plt.Line2D([], [], ls="none", marker=FAMILY_MARKER[f], color=FAMILY_COLOR[f], markersize=8, markeredgecolor="white")
          for f in fams]
    ax.legend(hs, [FAMILY_LABEL[f] for f in fams], loc=loc, frameon=False, fontsize=10, handletextpad=0.4,
              labelspacing=0.35, **kw)


# ------------------------------------------------------------------ fig 1: control vs length
def fig1_detailed():
    fig = frame("Control gains come with shorter reasoning: inside the length guardrail\n"
                "the best arm is the GEPA LoRA minus a brevity LoRA, at 6.6%",
                "gpt-oss-20b · honest compliance on the 9 training instructions vs median reasoning length (% of the untrained "
                "model's) · 95% bootstrap CIs over questions\ndashed = Pareto frontier (best compliance at this length or longer) · "
                "shaded = length guardrail (≥ 80% of untrained)\nleft: held-out test split (500 questions; LoRAs pooled over 3 seeds) · "
                "right: validation only (200 questions; faint = 40-question screens, no CIs)",
                DEF + "\nOther no-CoT SFT checkpoints (unlabelled ◆) score 1.2–2.4%. GEPA-prompt LoRA = the GEPA system prompt distilled "
                "into weights with no loss on reasoning tokens.")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=0.06, right=0.985, top=0.735, bottom=0.145, wspace=0.07)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, bx):
        a.axvspan(80, 400, color=GUARD, zorder=0)
        a.axvline(100, color="#cfcfca", lw=0.8, zorder=0)
        a.yaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
        a.set_ylim(-0.4, 13.5)
        a.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlim(20, 190); bx.set_xlim(20, 275)
    ax.set_ylabel("honest compliance (9 training instructions)")
    ax.set_xlabel("median reasoning length, % of untrained (test: 2,612 chars)")
    bx.set_xlabel("median reasoning length, % of untrained (val)")
    ax.set_title("TEST (reported)", loc="left", fontsize=11.5, color=MUTED, pad=6)
    bx.set_title("VALIDATION-ONLY ARMS (not confirmed on test)", loc="left", fontsize=11.5, color=MUTED, pad=6)
    ax.text(81.5, 13.2, "length guardrail: ≥ 80%", fontsize=9.5, color="#5a7fae", va="top")

    scatter(ax, test, "chars_pct")
    draw_frontier(ax, test, "chars_pct")
    L = lambda lab, text, dx, dy, **k: label(ax, row(test, lab), text, dx, dy, **k)
    L("untrained", "untrained 1.5%", 10, 10, weight="bold")
    L("GEPA prompt (teacher)", "GEPA system prompt 9.4%", 12, 0, weight="bold")
    L("GEPA LoRA x1", "GEPA LoRA (distilled) 9.7%", 12, 28, weight="bold")
    L("GEPA LoRA x1.5", "LoRA ×1.5  9.8%", -14, 18)
    L("GEPA LoRA x0.75 + GEPA prompt", "LoRA ×0.75 + prompt 8.6%", 12, -12)
    L("GEPA LoRA - 2 brevity LoRA", "GEPA LoRA − 2·brevity LoRA 6.6%", 12, 10, weight="bold")
    L("1A answer-only SFT", "1A answer-only SFT 4.9%", 12, 6, weight="bold")
    L("no-CoT SFT: fmt @25", "no-CoT format SFT 2.8%", 0, 18, ha="center", weight="bold")
    L("steering L12 a1", "steering vectors (2)", -10, 20, ha="right", color=MUTED)
    L("noise for 1A", "random-LoRA\ncontrols (4)", 10, 0, color=MUTED)
    L("steering GEPA-brevity L12 a2", "steering: GEPA − brevity direction", 0, -18, ha="center", color=MUTED)
    legend(ax, ["lora", "prompt", "sft", "steer", "control"], "upper right", bbox_to_anchor=(1.0, 0.99))

    scatter(bx, val_40, "chars_pct", alpha=0.35, size=28, whiskers=False)
    scatter(bx, val_full, "chars_pct", alpha=0.95, size=55)
    bx.scatter([100], [1.0], s=55, marker="o", facecolor=GREY, edgecolor="white", linewidth=1.2, zorder=3)
    bx.annotate("untrained (val) 1.0%", (100, 1.0), xytext=(8, 8), textcoords="offset points", fontsize=9.5, color=INK)
    V = lambda lab, text, dx, dy, **k: label(bx, row(val_full, lab), text, dx, dy, fs=9.5, **k)
    V("GEPA LoRA lr 3e-4", "GEPA LoRA, lr 3e-4", 8, 6)
    V("GEPA LoRA + header anchor", "GEPA LoRA + length anchor", 8, 24)
    V("C2 prompt: steps_append", "GEPA prompt + numbered steps", 8, 8)
    V("C2 prompt: len_append", "GEPA prompt + length floor", 8, 14)
    V("C2 prompt: recheck_len", "re-check-the-rule prompt", 8, -2)
    V("C2 prompt: sustain", "sustain prompt", -8, 0, ha="right")
    V("C2 prompt: rewrite", "GEPA prompt rewritten\nwithout brevity cues", 8, 14)
    V("C2 prompt: GEPA @ effort high", "GEPA prompt,\nreasoning_effort high", -4, 22, ha="right")
    W = lambda lab, text, dx, dy, **k: label(bx, row(val_40, lab), text, dx, dy, fs=9, color="#7a7a74", **k)
    W("GEPA LoRA x1.3", "LoRA dose ×1.3 (40 q)", 8, 0)
    W("GEPA LoRA x0.5", "dose ×0.5", -6, 0, ha="right")
    W("GEPA LoRA - 2.5 brevity LoRA", "− 2.5·brevity", 6, 8)
    W("GEPA LoRA + 2.0 verbosity LoRA", "+ 2·verbosity", 6, -8)
    W("verbosity LoRA alone", "verbosity LoRA alone", 6, 0)
    W("brevity LoRA alone", "brevity LoRA alone", 6, -8)
    bx.text(0.98, 0.035, "off scale → length-floor prompt at effort high: 3.7% at 417%; untrained at effort high: 0.0% at 710%",
            transform=bx.transAxes, fontsize=8.5, color=MUTED, ha="right")
    bx.tick_params(axis="y", labelleft=False)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig1_control_vs_length_detailed.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2: control vs accuracy
def fig2_detailed():
    fig = frame("Control also costs accuracy: every arm above 3% breaks the ±3-point guardrail;\n"
                "the GEPA prompt (−3.6 pts) and the LoRA arithmetic (−4.0) come closest",
                "gpt-oss-20b, test split · honest compliance on the 9 training instructions vs accuracy on 500 unconstrained questions · "
                "95% bootstrap CIs over questions\ndashed = Pareto frontier (best compliance at this accuracy or better) · "
                "shaded = accuracy within 3 points of the untrained model",
                DEF + "\nAccuracy CIs are ±4–5 points at 500 questions; the pre-registered guardrail compares point estimates.")
    ax = fig.add_axes([0.07, 0.145, 0.915, 0.63])
    lo, hi = U.accuracy - 3, U.accuracy + 3
    ax.axvspan(lo, hi, color=GUARD, zorder=0)
    ax.axvline(U.accuracy, color="#cfcfca", lw=0.8, zorder=0)
    ax.text((lo + hi) / 2, 13.2, f"accuracy guardrail: {lo:.1f}–{hi:.1f}%", fontsize=9.5, color="#5a7fae", ha="center", va="top")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(35, 52); ax.set_ylim(-0.4, 13.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%"); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_ylabel("honest compliance (9 training instructions)")
    ax.set_xlabel("accuracy on unconstrained questions (untrained 48.6%)")
    d = test.dropna(subset=["accuracy"])
    for fam, g in d.groupby("family"):
        ax.errorbar(g.accuracy, g.honest, xerr=[g.accuracy - g.accuracy_lo, g.accuracy_hi - g.accuracy],
                    yerr=[g.honest - g.honest_lo, g.honest_hi - g.honest], fmt="none", ecolor=FAMILY_COLOR[fam],
                    elinewidth=0.8, alpha=0.35, zorder=2)
    scatter(ax, d, "accuracy", whiskers=False)
    draw_frontier(ax, d, "accuracy")
    L = lambda lab, text, dx, dy, **k: label(ax, row(d, lab), text, dx, dy, x="accuracy", **k)
    L("untrained", "untrained 1.5%", 10, 10, weight="bold")
    L("GEPA prompt (teacher)", "GEPA system prompt 9.4%", 10, 10, weight="bold")
    L("GEPA LoRA x1", "GEPA LoRA 9.7%", 10, 12, weight="bold")
    L("GEPA LoRA x1.5", "LoRA ×1.5  9.8%", -10, 10, ha="right")
    L("GEPA LoRA x0.75 + GEPA prompt", "LoRA ×0.75 + prompt 8.6%", 10, 0)
    L("GEPA LoRA - 2 brevity LoRA", "GEPA LoRA − 2·brevity LoRA 6.6%", 10, 10, weight="bold")
    L("1A answer-only SFT", "1A answer-only SFT 4.9%", 10, 8, weight="bold")
    L("no-CoT SFT: fmt @25", "no-CoT format SFT 2.8%", 10, 14, weight="bold")
    L("steering L15 a2", "steering vectors (3)", 0, -18, color=MUTED, ha="center")
    L("noise for 1A", "random-LoRA controls (4)", 0, -16, color=MUTED, ha="center")
    legend(ax, ["lora", "prompt", "sft", "steer", "control"], "upper left", bbox_to_anchor=(0.0, 0.99))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_control_vs_accuracy_detailed.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ simplified fig 1 / fig 2 (8 test points)
SIMPLE_DEF = ("honest compliance = followed the reasoning instruction AND answered AND reasoning not empty/degenerate")
KEYS = [("untrained", "untrained"), ("RANDOM", "random LoRA (control)"), ("GEPA prompt (teacher)", "GEPA system prompt"),
        ("GEPA LoRA x1", "GEPA LoRA"), ("GEPA LoRA x0.75 + GEPA prompt", "GEPA LoRA + GEPA prompt"),
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


def simple_frame(title, xlabel, xlim, ylim=(0, 11)):
    fig = plt.figure(figsize=(12, 6.75), facecolor="white")
    fig.text(0.04, 0.965, title, fontsize=24, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.04, 0.885, "gpt-oss-20b, test split · error bars = 95% CI", fontsize=15, color=MUTED, ha="left", va="top")
    ax = fig.add_axes([0.085, 0.17, 0.89, 0.64])
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.tick_params(labelsize=15)
    ax.set_xlabel(xlabel, fontsize=16, labelpad=8)
    ax.set_ylabel("honest compliance", fontsize=16)
    return fig, ax


def simple_draw(ax, d, x, offsets, foot_fig, foot):
    f = pareto(d, x)
    on = set(f.name)
    draw_frontier(ax, d, x)
    for _, r in d.iterrows():
        c = BLUE if r["name"] in on else GREY
        ax.errorbar(r[x], r.honest, yerr=[[r.honest - r.honest_lo], [r.honest_hi - r.honest]], fmt="none", ecolor=c,
                    elinewidth=1.4, capsize=0, alpha=0.6, zorder=2)
        ax.scatter(r[x], r.honest, s=130, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        dx, dy, ha = offsets[r["name"]]
        ax.annotate(r["name"], (r[x], r.honest), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=15, color=INK, fontweight="bold" if r["name"] in on else "normal")
    foot_fig.text(0.04, 0.045, foot, fontsize=12, color="#6b6b66", ha="left", va="bottom")
    foot_fig.text(0.04, 0.012, SIMPLE_DEF, fontsize=12, color="#6b6b66", ha="left", va="bottom")


def fig1():
    d = simple_points()
    fig, ax = simple_frame("More control has meant shorter reasoning", "median reasoning length (% of untrained)", (20, 150))
    ax.axvspan(80, 150, color=GUARD, zorder=0)
    ax.text(148, 10.6, "normal length", fontsize=12, color="#5a7fae", ha="right", va="top")
    off = {"untrained": (0, -22, "center"), "random LoRA (control)": (14, -14, "left"),
           "GEPA system prompt": (12, 14, "left"), "GEPA LoRA": (0, 22, "center"),
           "GEPA LoRA + GEPA prompt": (-8, -24, "left"),
           "GEPA LoRA − brevity LoRA": (12, 12, "left"), "answer-only SFT": (12, 12, "left"),
           "no-CoT format SFT": (-12, 0, "right"), "best steering vector": (-12, 14, "right")}
    simple_draw(ax, d, "chars_pct", off, fig, "")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig1_control_vs_length.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def fig2():
    d = simple_points()
    fig, ax = simple_frame("More control has also cost accuracy", "accuracy on unconstrained questions", (34, 52))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    lo, hi = U.accuracy - 3, U.accuracy + 3
    ax.axvspan(lo, hi, color=GUARD, zorder=0)
    ax.text((lo + hi) / 2 + 1.2, 10.8, "within 3 pts of untrained", fontsize=12, color="#5a7fae", ha="center", va="top")
    off = {"untrained": (0, -22, "center"), "random LoRA (control)": (-12, -14, "right"),
           "GEPA system prompt": (12, 8, "left"), "GEPA LoRA": (0, 22, "center"),
           "GEPA LoRA + GEPA prompt": (12, -6, "left"),
           "GEPA LoRA − brevity LoRA": (-12, 12, "right"), "answer-only SFT": (12, 12, "left"),
           "no-CoT format SFT": (-12, 0, "right"), "best steering vector": (-10, 14, "right")}
    simple_draw(ax, d, "accuracy", off, fig, "accuracy CIs are ±4–5 pts (not shown)")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_control_vs_accuracy.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3: P(honest | length bin)
def fig3():
    cv = pd.read_csv(HERE / "data/curves.csv")
    arms = [("untrained", GREY, "untrained"), ("GEPA prompt", ORANGE, "GEPA system prompt"), ("GEPA LoRA", BLUE, "GEPA LoRA"),
            ("1A answer-only SFT", AQUA, "1A answer-only SFT"), ("GEPA LoRA - 2 brevity LoRA", YELLOW, "GEPA LoRA − 2·brevity LoRA"),
            ("no-CoT SFT: fmt @25", MAGENTA, "no-CoT format SFT")]
    bins = ["<1k", "1-2k", "2-2.6k", "2.6-4k", "4-8k", ">8k"]
    fig = frame("Control is a function of reasoning length: interventions mostly move traces\n"
                "along one curve, and nothing exceeds ~2% once reasoning passes 2.6k characters",
                "gpt-oss-20b, test split, 9 training instructions · honest compliance within each reasoning-length bin · "
                "95% bootstrap CIs over questions\nmarker area ∝ share of the arm's traces in that bin: GEPA prompt and GEPA LoRA keep "
                "most traces under 1k chars; the LoRA arithmetic spreads them out",
                DEF + "\nThe three letter-case instructions are 0% honest above 2.6k chars for every arm. Untrained median = 2,612 chars.")
    ax = fig.add_axes([0.07, 0.145, 0.70, 0.63])
    x = np.arange(len(bins))
    ax.axvspan(2.5, 5.6, color=GUARD, zorder=0)
    ax.text(2.58, 19.6, "≥ untrained median length (the honest_long endpoint)", fontsize=9.5, color="#5a7fae", va="top")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(-2.5, 5.6); ax.set_ylim(-0.5, 20)
    ax.set_yticks(range(0, 21, 5)); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xticks(x, ["< 1k", "1–2k", "2–2.6k", "2.6–4k", "4–8k", "> 8k"])
    ax.set_xlabel("reasoning length (characters)")
    ax.set_ylabel("honest compliance within the length bin")
    ends = []
    for key, color, name in arms:
        g = cv[cv.arm == key].set_index("bin").loc[bins]
        ax.plot(x, g.honest, color=color, lw=2, zorder=3)
        ax.errorbar(x, g.honest, yerr=[g.honest - g.lo, g.hi - g.honest], fmt="none", ecolor=color, elinewidth=0.8, alpha=0.5, zorder=2)
        ax.scatter(x, g.honest, s=30 + 9 * g.frac, color=color, edgecolor="white", linewidth=1, zorder=4)
        ends.append((g.honest.iloc[0], name, color))
    ends.sort(key=lambda t: t[0])
    ys = []
    for y0, name, color in ends:  # direct labels at the left end, nudged apart
        y = y0
        while any(abs(y - z) < 1.2 for z in ys):
            y += 0.3
        ys.append(y)
        ax.annotate(name, (-0.65, y0), xytext=(0, (y - y0) * 10), textcoords="offset points", ha="right", va="center", fontsize=10.5,
                    color=INK, fontweight="bold" if name != "untrained" else "normal")
        ax.plot([-0.55, -0.1], [y0, y0], color=color, lw=2.5, solid_capstyle="round", zorder=3)
    keys = [("untrained", "untrained"), ("GEPA prompt (teacher)", "GEPA prompt"), ("GEPA LoRA x1", "GEPA LoRA"),
            ("GEPA LoRA - 2 brevity LoRA", "LoRA − 2·brevity"), ("1A answer-only SFT", "1A answer-only SFT"),
            ("no-CoT SFT: fmt @25", "no-CoT format SFT")]
    txt = "honest among traces\n≥ 2,612 chars (honest_long)\n\n" + "\n".join(
        f"{n}: {test[test.label == k].honest_long.iloc[0]:.1f}%" for k, n in keys)
    fig.text(0.795, 0.62, txt, fontsize=10, color=INK, va="top", ha="left", linespacing=1.5,
             bbox={"boxstyle": "round,pad=0.6", "fc": "#f7f7f5", "ec": "#dddcd6"})
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig3_compliance_by_length.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig1_detailed(); fig2_detailed(); fig1(); fig2(); fig3()
    print("wrote", sorted(p.name for p in FIG.glob("fig*")))
