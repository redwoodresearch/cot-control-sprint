"""Post figures for the CoT-controllability elicitation lab notes (gpt-oss-20b + the orange expansion).

Reads data/interventions.csv and data/curves.csv (via figures.py) and the orange-expansion results in
kaushik/orange_expand/{sft,baseline,rl}; writes figures/fig{1..5}_*.{png,pdf} (16:9, 200 dpi, white background).

    .venv/bin/python kaushik/elicit/writeup/post_figures.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figures import HERE, U, pareto, simple_points
from matplotlib.lines import Line2D

FIG = HERE / "figures"
OE = HERE.parents[1] / "orange_expand"

# dataviz reference palette (categorical slots 1, 2, 3, 7) + neutrals
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GREY, GRID, GUARD = "#0b0b0b", "#52514e", "#8a8a84", "#e6e6e3", "#eef3fa"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#bdbdb8", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "axes.titlesize": 15, "legend.fontsize": 13, "pdf.fonttype": 42})

FAMILY = {"untrained": "controls", "random LoRA (control)": "controls", "GEPA system prompt": "prompt",
          "GEPA LoRA": "GEPA LoRA family", "GEPA LoRA ×1.5": "GEPA LoRA family",
          "GEPA LoRA + GEPA prompt": "GEPA LoRA family", "GEPA LoRA − brevity LoRA": "GEPA LoRA family",
          "answer-only SFT": "no-CoT SFT", "no-CoT format SFT": "no-CoT SFT", "best steering vector": "steering"}
FAMILY_COLOR = {"prompt": ORANGE, "GEPA LoRA family": BLUE, "no-CoT SFT": AQUA, "steering": VIOLET, "controls": GREY}
COMPLIANCE = ("honest compliance = followed the reasoning instruction, answered, and the reasoning was not empty "
              "or degenerate")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def frame(title, subtitle, footnote, rect=(0.07, 0.15, 0.9, 0.68)):
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.04, 0.955, title, fontsize=22, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.04, 0.905, subtitle, fontsize=15, ha="left", va="top", color=MUTED)
    if footnote:
        fig.text(0.04, 0.025, footnote, fontsize=12, ha="left", va="bottom", color=MUTED)
    ax = fig.add_axes(rect)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def pct(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(lambda v, _: f"{v:.0f}%")


def family_legend(ax, loc, families=("prompt", "GEPA LoRA family", "no-CoT SFT", "steering", "controls")):
    handles = [Line2D([], [], marker="o", ls="", ms=10, color=FAMILY_COLOR[f], label=f) for f in families]
    ax.legend(handles=handles, loc=loc, frameon=False, title="method family", title_fontsize=13, alignment="left")


def step_frontier(ax, f, x):
    xs, ys = list(f[x]), list(f.honest)
    px, py = [], []
    for i in range(len(xs)):
        px += [xs[i]]; py += [ys[i]]
        if i + 1 < len(xs):
            px += [xs[i + 1]]; py += [ys[i]]
    ax.plot(px, py, color=INK, lw=1.2, ls=(0, (4, 3)), alpha=0.55, zorder=1)


def tradeoff(d, x, xlabel, xlim, shade, shade_label, offsets, xerr, name, title, subtitle, footnote, legend_loc):
    fig, ax = frame(title, subtitle, footnote)
    ax.set_xlim(*xlim); ax.set_ylim(0, 12.5)
    pct(ax); pct(ax, "x")
    ax.set_xlabel(xlabel, labelpad=8); ax.set_ylabel("honest compliance (9 training instructions)")
    ax.axvspan(*shade, color=GUARD, zorder=0)
    ax.text(shade[0] + 0.01 * (xlim[1] - xlim[0]), 12.2, shade_label, ha="left", va="top", fontsize=13, color=MUTED,
            style="italic")
    f = pareto(d, x)
    step_frontier(ax, f, x)
    on = set(f.name)
    for _, r in d.iterrows():
        c = FAMILY_COLOR[FAMILY[r["name"]]]
        ax.errorbar(r[x], r.honest, yerr=[[r.honest - r.honest_lo], [r.honest_hi - r.honest]],
                    xerr=None if xerr is None else [[r[x] - r[xerr[0]]], [r[xerr[1]] - r[x]]],
                    fmt="none", ecolor=c, elinewidth=1.3, alpha=0.5, zorder=2)
        ax.scatter(r[x], r.honest, s=150, color=c, edgecolor=INK if r["name"] in on else "white",
                   linewidth=1.6 if r["name"] in on else 1.2, zorder=3)
        dx, dy, ha = offsets[r["name"]]
        ax.annotate(r["name"], (r[x], r.honest), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=14, color=INK, fontweight="bold" if r["name"] in on else "normal", zorder=4)
    family_legend(ax, legend_loc)
    save(fig, name)


def fig1():
    d = simple_points()
    off = {"untrained": (0, -24, "center"), "random LoRA (control)": (14, -14, "left"),
           "GEPA system prompt": (14, 16, "left"), "GEPA LoRA": (0, 34, "center"), "GEPA LoRA ×1.5": (-14, 0, "right"),
           "GEPA LoRA + GEPA prompt": (-8, -26, "left"), "GEPA LoRA − brevity LoRA": (12, 12, "left"),
           "answer-only SFT": (12, 12, "left"), "no-CoT format SFT": (-12, 0, "right"),
           "best steering vector": (-12, 14, "right")}
    tradeoff(d, "chars_pct", "median reasoning length (% of untrained, characters)", (0, 150), (80, 150),
             "normal length (≥ 80% of untrained)", off, None, "fig1_tradeoff_length",
             "Every gain in CoT control came with shorter reasoning",
             "gpt-oss-20b, test split (9 instructions × 500 q; LoRAs pooled over 3 seeds); bars = 95% bootstrap CI; "
             "dashed = Pareto frontier",
             "GEPA LoRA ×1.3 (9.7%, 34% length) sits on top of GEPA LoRA ×1 and is not drawn; random LoRA = mean of "
             "4 KL-matched controls.",
             "upper right")


def fig2():
    d = simple_points()
    off = {"untrained": (0, -24, "center"), "random LoRA (control)": (-12, -16, "right"),
           "GEPA system prompt": (12, 12, "left"), "GEPA LoRA": (0, 34, "center"), "GEPA LoRA ×1.5": (-14, -6, "right"),
           "GEPA LoRA + GEPA prompt": (-12, -6, "right"), "GEPA LoRA − brevity LoRA": (-12, 12, "right"),
           "answer-only SFT": (-12, 12, "right"), "no-CoT format SFT": (14, 4, "left"),
           "best steering vector": (-12, 14, "right")}
    tradeoff(d, "accuracy", "accuracy on 500 unconstrained questions", (28, 56),
             (U.accuracy - 3, U.accuracy + 3), "within 3 pts of untrained", off, None,
             "fig2_tradeoff_accuracy", "The same gains also cost accuracy",
             "gpt-oss-20b, test split (same 10 arms as fig. 1); bars = 95% CI on compliance (accuracy CIs ±4–5 pts, "
             "not drawn)",
             "Guardrail compares point estimates; compliance = followed the instruction, answered, reasoning not "
             "degenerate.", "upper left")


def fig3():
    cv = pd.read_csv(HERE / "data/curves.csv")
    bins = ["<1k", "1-2k", "2-2.6k", "2.6-4k", "4-8k", ">8k"]
    arms = [("untrained", GREY, "-", "untrained"), ("GEPA prompt", ORANGE, "-", "GEPA prompt"),
            ("GEPA LoRA", BLUE, "-", "GEPA LoRA"), ("GEPA LoRA - 2 brevity LoRA", BLUE, (0, (5, 2)), "GEPA LoRA − brevity LoRA"),
            ("1A answer-only SFT", AQUA, "-", "answer-only SFT")]
    fig, ax = frame("CoT control is a function of reasoning length: above 2.6k characters no arm exceeds 2%",
                    "gpt-oss-20b, test split, per reasoning-length bin (9 instructions pooled; LoRAs pooled over "
                    "3 seeds); bands = 95% bootstrap CI over traces",
                    "Bins hold 2–55% of each arm's traces (fewest: GEPA LoRA >8k, 2%); compliance = followed the "
                    "instruction, answered, reasoning not degenerate.", rect=(0.25, 0.15, 0.72, 0.68))
    ax.yaxis.set_label_coords(-0.36, 0.5)
    x = np.arange(len(bins))
    ax.set_xlim(-0.4, 5.3); ax.set_ylim(0, 21)
    ax.set_xticks(x, ["<1k", "1–2k", "2–2.6k", "2.6–4k", "4–8k", ">8k"])
    ax.set_yticks(range(0, 21, 5)); pct(ax)
    ax.set_xlabel("reasoning length (characters)", labelpad=8); ax.set_ylabel("honest compliance")
    ax.axvline(2.5, color=MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
    ax.text(2.55, 20.5, "untrained median (2,612 chars)", ha="left", va="top", fontsize=13, color=MUTED, style="italic")
    ends = []
    for key, color, ls, name in arms:
        g = cv[cv.arm == key].set_index("bin").loc[bins]
        ax.fill_between(x, g.lo, g.hi, color=color, alpha=0.1, lw=0, zorder=2)
        ax.plot(x, g.honest, color=color, lw=2.4, ls=ls, marker="o", ms=7, zorder=3)
        ends.append([float(g.honest.iloc[0]), name, color])
    ends.sort(key=lambda t: t[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):  # push labels apart by at least one line height
        ys[i] = max(ys[i], ys[i - 1] + 1.6)
    for (y0, name, color), y in zip(ends, ys):
        ax.annotate(name, (0, y0), xytext=(-0.78, y), textcoords="data", ha="right", va="center", fontsize=14,
                    color=INK, fontweight="bold", arrowprops={"arrowstyle": "-", "color": color, "lw": 1, "alpha": 0.7,
                                                              "shrinkA": 0, "shrinkB": 6})
    save(fig, "fig3_compliance_by_length")


# ----------------------------------------------------------------------------------------------- orange expansion
OE_MODELS = [("qwen36_27b", "Qwen3.6-27B\n(2A reference)"), ("gptoss20b", "gpt-oss-20b"), ("gptoss120b", "gpt-oss-120b"),
             ("qwen8b", "Qwen3-8B"), ("qwen32b", "Qwen3-32B")]
OE_METHODS = [("untrained", "untrained", GREY), ("sft", "orange SFT", ORANGE), ("rl", "orange RL", BLUE)]
OE_ARM = {"untrained": ["untrained"], "sft": ["orange", "sft_eugene"], "rl": ["rl_eugene_lr5e-5", "rl_eugene"]}


def orange_cells() -> pd.DataFrame:
    """Pooled (lowercase + uppercase) honest compliance per model × method × eval condition, in %."""
    c = pd.read_csv(OE / "sft/results/cells.csv")
    m = pd.read_csv(OE / "sft/results/mechanism_cells.csv")
    d = pd.concat([c, m[m.model == "qwen36_27b"]])
    d = d[(d["mode"] == "(pooled)") & d.condition.isin(["persona", "nopersona"])]
    rows = []
    for model, _ in OE_MODELS:
        for method, _, _ in OE_METHODS:
            for arm in OE_ARM[method]:
                sub = d[(d.model == model) & (d.arm == arm)]
                if sub.empty:
                    continue
                for _, r in sub.iterrows():
                    rows.append({"model": model, "method": method, "arm": arm, "condition": r.condition,
                                 "honest": 100 * r.honest, "lo": 100 * r.honest_lo, "hi": 100 * r.honest_hi,
                                 "n_seeds": r.n_seeds, "step": r.step})
                break  # first available arm per method (gpt-oss-20b: the healthy LR 5e-5 RL run)
    return pd.DataFrame(rows)


def fig4():
    d = orange_cells()
    d.to_csv(HERE / "data/orange_transfer.csv", index=False)
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.04, 0.955, "Orange training transfers to CoT control only on Qwen3.6-27B; four new models: 0%",
             fontsize=22, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.04, 0.905, "Honest compliance when the system prompt forbids a reasoning format; lowercase/uppercase pooled, "
             "val 200 q, 1 seed (27B: 3), 95% CI",
             fontsize=15, ha="left", va="top", color=MUTED)
    fig.text(0.04, 0.025, "27B = 2A SFT adapters, step 50 (no 27B RL here); gpt-oss-20b RL = LR 5e-5 run, ckpt 58 "
             "(LR 1.5e-4 run, ckpt 10, also 0%); Qwen3-32B RL = ckpt 55; no RL for gpt-oss-120b / Qwen3-8B.",
             fontsize=12, ha="left", va="bottom", color=MUTED)
    panels = [("persona", "Prohibition phrased via the EUGENE persona"), ("nopersona", "Prohibition without a persona")]
    xs = np.arange(len(OE_MODELS))
    offs = {"untrained": -0.24, "sft": 0.0, "rl": 0.24}
    for j, (cond, ptitle) in enumerate(panels):
        ax = fig.add_axes([0.07 + j * 0.48, 0.17, 0.42, 0.62])
        ax.set_title(ptitle, loc="left", color=INK, fontweight="bold", pad=10)
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_xlim(-0.6, len(OE_MODELS) - 0.4); ax.set_ylim(-0.6, 14)
        ax.set_yticks(range(0, 15, 2)); pct(ax)
        ax.set_xticks(xs, [n for _, n in OE_MODELS], fontsize=13)
        ax.axhline(0, color="#bdbdb8", lw=0.8, zorder=1)
        if j == 0:
            ax.set_ylabel("honest CoT compliance")
        for i, (model, _) in enumerate(OE_MODELS):
            for method, _, color in OE_METHODS:
                r = d[(d.model == model) & (d.method == method) & (d.condition == cond)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = xs[i] + offs[method]
                ax.errorbar(x, r.honest, yerr=[[r.honest - r.lo], [r.hi - r.honest]], fmt="none", ecolor=color,
                            elinewidth=1.6, capsize=4, zorder=2)
                ax.scatter(x, r.honest, s=130, color=color, edgecolor="white", linewidth=1.2, zorder=3)
                if r.honest > 0:
                    ax.annotate(f"{r.honest:.1f}%", (x, r.hi), xytext=(0, 8), textcoords="offset points", ha="center",
                                va="bottom", fontsize=14, fontweight="bold", color=INK)
        ax.annotate("0% in every arm", (2.5, 0), xytext=(0, 22), textcoords="offset points", ha="center", va="bottom",
                    fontsize=13, color=MUTED, style="italic")
        ax.plot([0.7, 4.3], [1.0, 1.0], color=GRID, lw=1.2, zorder=1)
        if j == 1:
            handles = [Line2D([], [], marker="o", ls="", ms=10, color=c, label=lab) for _, lab, c in OE_METHODS]
            ax.legend(handles=handles, loc="upper right", frameon=False)
    save(fig, "fig4_orange_transfer")


def load_progress(run: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in (OE / "rl/runs" / run / "progress.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "reward_mean" in r and not r.get("final")]
    return pd.DataFrame(rows).drop_duplicates("iteration", keep="last").sort_values("iteration")


def fig5():
    spec = [("oex-rl-eugene-gptoss20b-seed0", "gpt-oss-20b, LR 1.5e-4", ORANGE, 10,
             "stopped at it. 20: reasoning blew up (21.6k chars, 30% truncated)"),
            ("oex-rl-eugene-gptoss20b-seed0-lr5e-5", "gpt-oss-20b, LR 5e-5", BLUE, 58,
             "stopped at it. 57 (rollouts stalled); ckpt 58 evaluated"),
            ("oex-rl-eugene-qwen3_32b-seed0", "Qwen3-32B, LR 1.5e-4", AQUA, 55,
             "stopped at it. 54 (reward saturated); ckpt 55 evaluated")]
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.04, 0.955, "Orange RL saturates on every run; high-LR gpt-oss-20b then blows up its reasoning",
             fontsize=22, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.04, 0.905, "GRPO on the EUGENE prohibition, reasoning on, 1 seed/run; one point per iteration (48 prompts × 8 "
             "rollouts, no CI); rings = evaluated ckpt",
             fontsize=15, ha="left", va="top", color=MUTED)
    fig.text(0.04, 0.025, "Iteration t's rollouts are sampled from checkpoint t, so checkpoints 55 and 58 are one update "
             "past the last logged point; x-axis stops where each run stopped (progress.jsonl).",
             fontsize=12, ha="left", va="bottom", color=MUTED)
    top = fig.add_axes([0.07, 0.52, 0.62, 0.33]); bot = fig.add_axes([0.07, 0.15, 0.62, 0.33], sharex=top)
    for ax in (top, bot):
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    runs = {name: load_progress(name) for name, *_ in spec}
    for name, label, color, ckpt, note in spec:
        df = runs[name]
        top.plot(df.iteration, df.reward_mean, color=color, lw=2.2, marker="o", ms=4, zorder=3)
        bot.plot(df.iteration, df.reasoning_chars_median, color=color, lw=2.2, marker="o", ms=4, zorder=3)
        last = df.iloc[-1]
        if ckpt in set(df.iteration):
            r = df[df.iteration == ckpt].iloc[0]
            for ax, y in ((top, r.reward_mean), (bot, r.reasoning_chars_median)):
                ax.scatter(ckpt, y, s=170, facecolor="none", edgecolor=color, linewidth=2.2, zorder=4)
        else:  # checkpoint saved after the last logged iteration: mark at the end of the line
            for ax, y in ((top, last.reward_mean), (bot, last.reasoning_chars_median)):
                ax.scatter(last.iteration, y, s=170, facecolor="none", edgecolor=color, linewidth=2.2, zorder=4)
        bot.annotate(f"{label}\n{note}", (last.iteration, last.reasoning_chars_median), xytext=(14, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=13, color=INK, annotation_clip=False,
                     arrowprops={"arrowstyle": "-", "color": color, "lw": 1, "shrinkA": 0, "shrinkB": 8})
    top.set_ylim(0.3, 1.18); top.set_xlim(-1, 60)
    top.set_yticks([0.4, 0.6, 0.8, 1.0])
    top.set_ylabel("mean training reward\n(orange option picked)")
    bot.set_yscale("log"); bot.set_ylim(350, 40000)
    bot.set_yticks([500, 1000, 2000, 5000, 10000, 20000], ["500", "1k", "2k", "5k", "10k", "20k"])
    bot.set_ylabel("median reasoning length\n(characters, log scale)")
    bot.set_xlabel("training iteration (= checkpoint number)", labelpad=8)
    plt.setp(top.get_xticklabels(), visible=False)
    # the orange run: checkpoint 10 ring + the reward collapse after it
    r10 = runs[spec[0][0]].set_index("iteration").loc[10]
    top.annotate("checkpoint 10 evaluated (reward 0.97)", (10, r10.reward_mean), xytext=(14, 26),
                 textcoords="offset points", ha="left", va="center", fontsize=13, color=INK,
                 arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1, "shrinkA": 0, "shrinkB": 8})
    last20 = runs[spec[0][0]].iloc[-1]
    top.annotate("reward falls as traces hit the 8k-token cap", (last20.iteration, last20.reward_mean), xytext=(14, 0),
                 textcoords="offset points", ha="left", va="center", fontsize=13, color=INK,
                 arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1, "shrinkA": 0, "shrinkB": 8})
    handles = [Line2D([], [], color=c, lw=2.4, label=lab) for _, lab, c, _, _ in spec]
    top.legend(handles=handles, loc="lower right", frameon=False)
    save(fig, "fig5_orange_rl_dynamics")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
