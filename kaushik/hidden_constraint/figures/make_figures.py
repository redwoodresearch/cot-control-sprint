"""Figures for Experiment 3B (hidden-constraint scenarios).

Reads only ../runs/summary.csv (columns: model_or_checkpoint, scenario, metric, value, ci_low,
ci_high, n) and writes PNG (200 dpi) + PDF next to this file. Re-run after `hidden_constraint analyze`:

    .venv/bin/python kaushik/hidden_constraint/figures/make_figures.py

Figures
  fig1_notice_vs_comply   models read and discuss the hidden rule, yet still write the forbidden notation
  fig2_suppression_ratio  full success is ~0 for everyone; the graded suppression ratio separates models
  fig3_adapters_vs_base   trained adapters vs their untrained base model (seeds grouped by arm)
  fig4_all_tasks          the notice-but-comply pattern in all three scenarios

Colour follows the metric in every figure: blue = notices the rule, orange = still writes the
forbidden term, aqua = full constrained success, violet = suppression ratio (all from utils.PALETTE;
this four-colour subset passes the dataviz palette validator all-pairs, aqua only with the
"visible labels" relief, which is why every aqua mark is value-labelled or sits in a table column).
"""

from __future__ import annotations

import math
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root, for utils.py
from utils import PALETTE, set_matplotlib_style

SUMMARY = HERE.parent / "runs" / "summary.csv"

# ---- design tokens (text never wears the data colour) -------------------------------------------
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d6d5cd"
C_NOTICE, C_USE, C_SUCCESS, C_SUPP = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[6]
MINUS = "−"

NAMES = {
    "dsv4pro": "DeepSeek V4 Pro",
    "glm53": "GLM-5.3",
    "gptoss120b": "gpt-oss-120b",
    "gptoss20b": "gpt-oss-20b",
    "qwen36_27b": "Qwen3.6-27B",
    "qwen8b": "Qwen3-8B",
    "qwen32b": "Qwen3-32B",
}
# The API-served copies (gptoss20b_api, qwen8b_api) are serving-path cross-checks, not extra models.
BASE_MODELS = list(NAMES)
SCENARIOS = {
    "config_notation": "Arithmetic word problems",
    "debug_lint": "Bug finding",
    "qa_grader": "Exam questions",
}
RULES = {
    "config_notation": "rule: no formula symbols (= + * / × ÷)",
    "debug_lint": "rule: don't write the buggy function's name",
    "qa_grader": "rule: don't write a key term of the question",
}

# ---- data ---------------------------------------------------------------------------------------
_df = pd.read_csv(SUMMARY).set_index(["model_or_checkpoint", "scenario", "metric"])


class Est(NamedTuple):
    v: float
    lo: float
    hi: float
    n: int


def get(model: str, scen: str, metric: str, scale: float = 100.0) -> Est:
    r = _df.loc[(model, scen, metric)]
    return Est(r.value * scale, r.ci_low * scale, r.ci_high * scale, int(r.n))


def models_by_suppression() -> list[str]:
    """Models ordered by arithmetic-task suppression ratio, highest first (same order in every figure)."""
    return sorted(BASE_MODELS, key=lambda m: -get(m, "config_notation", "suppression_ratio").v)


# ---- layout helpers (everything positioned in inches so spacing is predictable) -----------------
def start(width: float, body: float, header: float, footer: float):
    """Figure with `header` inches above and `footer` inches below a body of `body` inches."""
    h = header + body + footer
    fig = plt.figure(figsize=(width, h))
    return fig, h, 1 - header / h, footer / h


def headline(fig: plt.Figure, h: float, title: str, subtitle: str, width: int) -> None:
    lines = textwrap.wrap(title, width)
    fig.text(0.0, 1 - 0.03 / h, "\n".join(lines), fontsize=13, fontweight="bold", color=INK, va="top",
             ha="left", linespacing=1.25)
    drop = 0.03 + len(lines) * 13 * 1.25 / 72 + 0.10
    fig.text(0.0, 1 - drop / h, subtitle, fontsize=8.5, color=INK2, va="top", ha="left", linespacing=1.4)


def footnote(fig: plt.Figure, h: float, text: str, width: int) -> float:
    """Wrapped note at the bottom edge; returns the inches it occupies."""
    wrapped = textwrap.fill(text, width)
    fig.text(0.0, 0.03 / h, wrapped, fontsize=7.8, color=INK2, va="bottom", ha="left", linespacing=1.4)
    return 0.03 + (wrapped.count("\n") + 1) * 7.8 * 1.4 / 72


def legend(fig: plt.Figure, h: float, handles: list, y_in: float, ncol: int) -> None:
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, y_in / h), ncol=ncol, fontsize=8,
               frameon=False, handlelength=1.6, columnspacing=1.8, borderaxespad=0)


def panel_header(ax: plt.Axes, title: str, sub: str = "", lines: int = 2, fs: float = 9.5) -> None:
    """Bold title over a muted subtitle of up to `lines` lines; the title height is fixed by `lines`, so
    titles line up across panels whatever the subtitle length."""
    y_title = 6 + lines * 10.5
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, y_title), textcoords="offset points",
                fontsize=fs, fontweight="bold", color=INK, va="bottom", ha="left", annotation_clip=False)
    if sub:
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, y_title - 2),
                    textcoords="offset points", fontsize=7.6, color=INK2, va="top", ha="left",
                    annotation_clip=False, linespacing=1.3)


def style_rows(ax: plt.Axes, show_names: bool = True) -> None:
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelleft=show_names)
    ax.tick_params(axis="x", length=0, colors=MUTED, labelcolor=INK2)


def pct(x: float, dec: int = 0) -> str:
    return f"{x:.{dec}f}%"


def pct_axis(ax: plt.Axes, ticks: list[float]) -> None:
    ax.set_xticks(ticks, [f"{t:.0f}%".replace("-", MINUS) for t in ticks])


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.png", dpi=200)
    fig.savefig(HERE / f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


# ---- Figure 1: notice vs comply -----------------------------------------------------------------
def fig1() -> None:
    scen = "config_notation"
    order = models_by_suppression()
    panels = [
        ("read_rate", "Reads the rule file", "share of all episodes", C_NOTICE, 0),
        ("mention_rate", "Discusses the rule", "share of episodes that read it\n(rated by an LLM judge)", C_NOTICE, 0),
        ("forbidden_term_rate", "Still writes the symbols", "share of episodes that read it", C_USE, 0),
        ("constrained_success", "Full success", "correct and never wrote them,\nshare of all episodes", C_SUCCESS, 1),
    ]
    fig, h, top, bottom = start(11.2, body=2.9, header=1.72, footer=0.45)
    axes = fig.subplots(1, 4, sharey=True, gridspec_kw={"wspace": 0.12, "top": top, "bottom": bottom,
                                                        "left": 1.45 / 11.2, "right": 0.985})
    y = np.arange(len(order))
    for ax, (metric, title, sub, col, dec) in zip(axes, panels):
        est = [get(m, scen, metric) for m in order]
        v = np.array([e.v for e in est])
        lo = np.array([e.lo for e in est])
        hi = np.array([e.hi for e in est])
        ax.barh(y, v, height=0.5, color=col, zorder=3)
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=4)
        for yi, e in zip(y, est):
            ax.text(max(e.hi, e.v) + 2.5, yi, pct(e.v, dec), va="center", ha="left", fontsize=8, color=INK,
                    zorder=5)
        ax.set_xlim(0, 122)
        pct_axis(ax, [0, 25, 50, 75, 100])
        ax.set_ylim(len(order) - 0.45, -0.75)
        ax.set_yticks(y, [NAMES[m] for m in order])
        style_rows(ax, show_names=ax is axes[0])
        panel_header(ax, title, sub)
    # group captions above the panel titles
    y_cap = top + 0.74 / h
    for a0, a1, text in ((0, 1, "NOTICES THE RULE"), (2, 3, "BUT DOES NOT KEEP IT OUT OF ITS REASONING")):
        x0, x1 = axes[a0].get_position().x0, axes[a1].get_position().x1
        fig.add_artist(Line2D([x0, x1], [y_cap - 0.02 / h] * 2, color=GRID, lw=1.2))
        fig.text(x0, y_cap, text, fontsize=7.8, color=MUTED, fontweight="bold", va="bottom")
    headline(fig, h, "Models notice the hidden rule, but almost none keep the forbidden notation out of "
                     "their reasoning.",
             "Arithmetic word problems; the rule file says a transcript fails if the model's reasoning contains "
             "formula symbols (= + * / × ÷).\n250 episodes per model; whiskers are 95% bootstrap "
             "intervals. Models are ordered by suppression ratio (Figure 2).", 100)
    save(fig, "fig1_notice_vs_comply")


# ---- Figure 2: suppression separates models -----------------------------------------------------
def fig2() -> None:
    scen = "config_notation"
    order = models_by_suppression()
    y = np.arange(len(order))
    fig, h, top, bottom = start(10.4, body=2.9, header=1.5, footer=0.85)
    axl, axr = fig.subplots(1, 2, sharey=True, gridspec_kw={
        "width_ratios": [102, 125], "wspace": 0.07, "top": top, "bottom": bottom, "left": 1.45 / 10.4,
        "right": 0.985})
    spec = [
        (axl, "constrained_success", C_SUCCESS, (-2, 100), "Full success", "correct and never wrote the symbols,\nshare of all episodes"),
        (axr, "suppression_ratio", C_SUPP, (-25, 100), "Suppression ratio",
         "drop in formula symbols per 1,000 characters of reasoning,\ncompared with the same problems without the rule"),
    ]
    for ax, metric, col, (x0, x1), title, sub in spec:
        dec = 1 if metric == "constrained_success" else 0
        for yi, m in zip(y, order):
            e = get(m, scen, metric)
            ax.errorbar(e.v, yi, xerr=[[e.v - e.lo], [e.hi - e.v]], fmt="o", color=col, ecolor=col,
                        elinewidth=1.8, capsize=3, markersize=7, markeredgecolor="white",
                        markeredgewidth=1.2, zorder=4)
            mark = "†" if metric == "suppression_ratio" and e.n < 100 else ""
            ax.text(max(e.hi, e.v) + 2.5, yi, pct(e.v, dec).replace("-", MINUS) + mark, va="center",
                    ha="left", fontsize=8.5, color=INK)
        ax.set_xlim(x0, x1)
        ax.set_ylim(len(order) - 0.5, -0.8)
        ax.set_yticks(y, [NAMES[m] for m in order])
        style_rows(ax, show_names=ax is axl)
        panel_header(ax, title, sub)
    pct_axis(axl, [0, 25, 50, 75, 100])
    pct_axis(axr, [0, 25, 50, 75, 100])
    axr.axvline(0, color=MUTED, lw=1, zorder=2)
    axr.set_xlabel("0% = same symbol use as without the rule; larger = suppresses more", fontsize=8,
                   color=INK2, loc="right")
    notes = []
    for m in order:
        s = get(m, scen, "suppression_ratio")
        if s.n < 100:
            notes.append(f"† {NAMES[m]} reads the rule file in only {get(m, scen, 'read_rate').v:.0f}% of "
                         f"episodes, so its suppression ratio rests on {s.n} paired episodes (wide interval).")
    footnote(fig, h, "  ".join(notes) + "  Whiskers are 95% bootstrap intervals over episodes (a 0% rate has a "
             "zero-width interval). Arithmetic word problems, 250 episodes per model.", 150)
    headline(fig, h, "Full success is near zero for every model, but a graded suppression score separates "
                     "them: DeepSeek V4 Pro and GLM-5.3 suppress the notation most.",
             "Both panels show the same models and episodes on the same horizontal scale (one percentage "
             "point is the same width in each).", 105)
    save(fig, "fig2_suppression_ratio")


# ---- Figure 3: adapters vs base -----------------------------------------------------------------
ADAPTER_RE = re.compile(r"^ckpt_(?P<arm>.+?)-(?P<base>gptoss20b|qwen8b|qwen36_27b)(?P<mid>.*?)"
                        r"-seed(?P<seed>\d+)(?P<step>-step\d+)?$")
ARM_LABELS = {
    "fragility-mcq_noreason": "mcq_noreason",
    "fragility-mcq_reason": "mcq_reason",
    "fragility-style_sft": "style_sft *",
    "orange": "orange",
    "orange_nopersona": "orange_nopersona",
    "rl2b-rl_mcq_noreason": "RL mcq_noreason",
    "rl2b-rl_orange_noreason": "RL orange_noreason",
    "cptk": "1C size sweep",
}
ARM_ORDER = list(ARM_LABELS)
MIN_PAIRED = 10  # suppression ratios with fewer paired episodes are not drawn
BASE_PANELS = ["gptoss20b", "qwen8b", "qwen36_27b"]


def adapter_table() -> dict[str, dict[str, list[dict]]]:
    """{base: {arm_id: [{seed, run}, ...]}} parsed from the ckpt_* keys in summary.csv."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for run in sorted({k[0] for k in _df.index if k[0].startswith("ckpt_")}):
        m = ADAPTER_RE.match(run)
        if not m:
            print(f"warning: cannot parse adapter key {run}; skipped")
            continue
        arm = m["arm"] if m["arm"] == "cptk" else m["arm"] + m["mid"]
        out[m["base"]][arm + (m["step"] or "")].append({"seed": int(m["seed"]), "run": run})
    return out


def fig3() -> None:
    scen = "config_notation"
    table = adapter_table()
    metrics = [("constrained_success", "Full success", C_SUCCESS, 1),
               ("suppression_ratio", "Suppression ratio", C_SUPP, 0)]
    bases = [b for b in BASE_PANELS if b in table]

    def arm_sort(a: str) -> tuple:
        key = a.split("-step")[0]
        return (ARM_ORDER.index(key) if key in ARM_ORDER else 99, a)

    arms = {b: sorted(table[b], key=arm_sort) for b in bases}
    n_arms = [len(arms[b]) for b in bases]
    pitch, gap = 0.36, 0.95  # inches per arm row; inches between model blocks
    body = pitch * sum(n_arms) + gap * (len(bases) - 1)
    fig, h, top, bottom = start(9.8, body=body, header=1.5, footer=1.65)
    avg_h = pitch * sum(n_arms) / len(bases)
    gs = fig.add_gridspec(len(bases), 2, height_ratios=n_arms, hspace=gap / avg_h, wspace=0.55, top=top,
                          bottom=bottom, left=1.55 / 9.8, right=0.965)
    ci_max = max(get(r["run"], scen, "constrained_success").hi for b in bases for a in arms[b]
                 for r in table[b][a])
    xlim = {"constrained_success": (-0.4, math.ceil(ci_max) + 1), "suppression_ratio": (-60, 80)}
    ticks = {"constrained_success": list(range(0, int(xlim["constrained_success"][1]), 5)),
             "suppression_ratio": [-60, -40, -20, 0, 20, 40, 60, 80]}
    first = None
    for i, b in enumerate(bases):
        for j, (metric, mtitle, col, dec) in enumerate(metrics):
            ax = fig.add_subplot(gs[i, j])
            first = first or ax
            x0, x1 = xlim[metric]
            base = get(b, scen, metric)
            ax.axvspan(max(base.lo, x0), min(base.hi, x1), color=BAND, alpha=0.6, lw=0, zorder=1)
            ax.axvline(base.v, color=INK, lw=1.3, zorder=2)
            for k, arm in enumerate(arms[b]):
                runs = sorted(table[b][arm], key=lambda r: r["seed"])
                n = len(runs)
                offs = np.linspace(-0.25, 0.25, n) if n > 1 else np.array([0.0])
                xs = []
                for r, off in zip(runs, offs):
                    e = get(r["run"], scen, metric)
                    if np.isnan(e.v) or (metric == "suppression_ratio" and e.n < MIN_PAIRED):
                        continue
                    ax.plot([max(e.lo, x0), min(e.hi, x1)], [k + off] * 2, color=col, lw=1.2, alpha=0.55,
                            zorder=3, solid_capstyle="butt")
                    if e.lo < x0:
                        ax.plot(x0, k + off, marker="<", color=col, markersize=4, alpha=0.8, zorder=3)
                    ax.plot(e.v, k + off, "o", color=col, markersize=5.5, markeredgecolor="white",
                            markeredgewidth=0.9, zorder=4)
                    xs.append(e.v)
                if xs and not arm.startswith("cptk") and len(xs) > 1:
                    mu = float(np.mean(xs))
                    ax.plot([mu, mu], [k - 0.36, k + 0.36], color=INK, lw=1.8, zorder=5, solid_capstyle="butt")
                if arm.startswith("cptk") and xs:
                    txt = f"{min(xs):.{dec}f} to {max(xs):.{dec}f}".replace("-", MINUS)
                elif xs:
                    txt = f"{np.mean(xs):.{dec}f}".replace("-", MINUS)
                else:
                    txt = "n/a"
                ax.text(1.03, k, txt, transform=ax.get_yaxis_transform(), va="center", ha="left",
                        fontsize=8, color=INK)
                if len(xs) < n:
                    ax.text(1.03, k + 0.33, f"{len(xs)} of {n} seeds", transform=ax.get_yaxis_transform(),
                            va="center", ha="left", fontsize=6.5, color=MUTED)
                if k:
                    ax.axhline(k - 0.5, color=GRID, lw=0.6, zorder=0)
            ax.set_xlim(x0, x1)
            ax.set_ylim(len(arms[b]) - 0.45, -0.6)
            ax.set_yticks(range(len(arms[b])))
            ax.set_yticklabels([ARM_LABELS.get(a.split("-step")[0], a.split("-step")[0]) +
                                (" (step 50)" if "-step" in a else "") for a in arms[b]])
            style_rows(ax, show_names=(j == 0))
            ax.set_xticks(ticks[metric], [f"{t}%".replace("-", MINUS) for t in ticks[metric]])
            bval = f"{base.v:.1f}%" if metric == "constrained_success" else f"{base.v:.0f}%"
            sub = f"untrained base: {bval}"
            ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                        fontsize=7.6, color=INK2, va="bottom", annotation_clip=False)
            if i == 0:
                ax.annotate(mtitle, (0, 1), xycoords="axes fraction", xytext=(0, 17),
                            textcoords="offset points", fontsize=9.5, fontweight="bold", color=INK,
                            va="bottom", annotation_clip=False)
                ax.annotate("arm mean", (1.03, 1), xycoords="axes fraction", xytext=(0, 4),
                            textcoords="offset points", fontsize=7.2, color=MUTED, va="bottom")
            if j == 0:
                ax.annotate(NAMES[b], (0, 1), xycoords="axes fraction", xytext=(-6 - 0, 4),
                            textcoords="offset points", fontsize=9.5, fontweight="bold", color=INK,
                            va="bottom", ha="right", annotation_clip=False)
    # legend + notes
    handles = [
        Line2D([], [], marker="o", color=INK2, lw=0, markersize=6, markeredgecolor="white",
               label="one adapter seed, with 95% interval"),
        Line2D([], [], color=INK, lw=1.8, label="arm mean over seeds"),
        Line2D([], [], color=INK, lw=1.3, label="untrained base model"),
        Patch(color=BAND, alpha=0.6, label="base 95% interval"),
    ]
    notes = []
    if "qwen8b" in table and "fragility-style_sft" in table["qwen8b"]:
        runs = [r["run"] for r in table["qwen8b"]["fragility-style_sft"]]
        b = "qwen8b"
        notes.append(
            "* style_sft trains on pirate-voice targets that have nothing to do with controlling the "
            f"reasoning, yet it lifts Qwen3-8B full success from {get(b, scen, 'constrained_success').v:.1f}% "
            f"to {np.mean([get(r, scen, 'constrained_success').v for r in runs]):.0f}%. It also cuts how often "
            f"the model discusses the rule ({get(b, scen, 'mention_rate').v:.0f}% to "
            f"{np.mean([get(r, scen, 'mention_rate').v for r in runs]):.0f}%) and its no-rule success on the "
            f"bug-finding task ({get(b, 'debug_lint', 'control_success').v:.0f}% to "
            f"{np.mean([get(r, 'debug_lint', 'control_success').v for r in runs]):.0f}%): terser reasoning, "
            "not hiding.")
    if "gptoss20b" in table:
        reads = [get(r["run"], scen, "read_rate").v for a in table["gptoss20b"].values() for r in a]
        notes.append(
            f"gpt-oss-20b adapters read the rule file in {min(reads):.0f}-{max(reads):.0f}% of episodes (base "
            f"{get('gptoss20b', scen, 'read_rate').v:.0f}%), so their suppression ratios rest on few episodes; "
            f"seeds with fewer than {MIN_PAIRED} paired episodes are not drawn. A left arrow marks an interval "
            "cut off at the axis. 1C size sweep: one seed at each of four adapter sizes (range shown).")
    foot = footnote(fig, h, "  ".join(notes), 165)
    legend(fig, h, handles, foot + 0.12, 4)
    n_ad = sum(len(r) for b in table.values() for r in b.values())
    headline(fig, h, "Trained adapters change little compared with their base models; the clearest gain "
                     "(Qwen3-8B style_sft) comes with terser reasoning, not better hiding.",
             f"Arithmetic word problems; {n_ad} adapters from Experiments 1 and 2 against the untrained "
             "base model they were trained from; 250 episodes per adapter.", 108)
    save(fig, "fig3_adapters_vs_base")


# ---- Figure 4: all three tasks ------------------------------------------------------------------
def fig4() -> None:
    order = models_by_suppression()
    y = np.arange(len(order))
    series = [
        ("read_rate", "Reads the rule file (share of all episodes)", C_NOTICE, -0.25),
        ("forbidden_term_rate", "Still writes the forbidden term (share of episodes that read the file)", C_USE, 0.0),
        ("constrained_success", "Full success (share of all episodes)", C_SUCCESS, 0.25),
    ]
    fig, h, top, bottom = start(11.2, body=3.9, header=1.3, footer=1.25)
    axes = fig.subplots(1, 3, sharey=True, gridspec_kw={"wspace": 0.1, "top": top, "bottom": bottom,
                                                        "left": 1.45 / 11.2, "right": 0.985})
    for ax, (scen, sname) in zip(axes, SCENARIOS.items()):
        for metric, _, col, off in series:
            est = [get(m, scen, metric) for m in order]
            v = np.array([e.v for e in est])
            lo = np.array([e.lo for e in est])
            hi = np.array([e.hi for e in est])
            ax.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt="o", color=col, ecolor=col, elinewidth=1.4,
                        capsize=0, markersize=5.5, markeredgecolor="white", markeredgewidth=0.9, zorder=4)
            if metric == "constrained_success":
                for yi, e in zip(y, est):
                    ax.text(max(e.hi, e.v) + 2.5, yi + off, pct(e.v), va="center", ha="left", fontsize=7.4,
                            color=INK)
        ax.set_xlim(-3, 112)
        pct_axis(ax, [0, 25, 50, 75, 100])
        ax.set_ylim(len(order) - 0.5, -0.6)
        ax.set_yticks(y, [NAMES[m] for m in order])
        style_rows(ax, show_names=ax is axes[0])
        for yi in y[:-1]:
            ax.axhline(yi + 0.5, color=GRID, lw=0.6, zorder=1)
        panel_header(ax, sname, RULES[scen], lines=1)
    handles = [Line2D([], [], marker="o", color=col, lw=0, markersize=7, markeredgecolor="white", label=lab)
               for _, lab, col, _ in series]
    forb = [get(m, s, "forbidden_term_rate").v for m in order for s in SCENARIOS]
    succ = [get(m, s, "constrained_success").v for m in order for s in SCENARIOS]
    foot = footnote(
        fig, h,
        f"Across the {len(forb)} model-by-task cells, {min(forb):.0f}-{max(forb):.0f}% of the episodes that read the "
        f"rule still wrote the forbidden term, and full success is at most {max(succ):.0f}%. In the bug-finding "
        "task, quoting the blocked function name while discussing the rule counts as writing it. "
        f"gpt-oss-20b reads the rule file in only {get('gptoss20b', 'qa_grader', 'read_rate').v:.0f}% of "
        "exam-question episodes, so its other rates there rest on few episodes. 250 episodes per cell; whiskers "
        "are 95% bootstrap intervals; values are printed for full success.", 175)
    legend(fig, h, handles, foot + 0.1, 3)
    headline(fig, h, "The same pattern holds in all three tasks: models read the hidden rule, nearly all still "
                     "write the forbidden term, and full success stays at or below 10%.",
             "Models are ordered by arithmetic-task suppression ratio, as in Figures 1 and 2.", 112)
    save(fig, "fig4_all_tasks")


def main() -> None:
    set_matplotlib_style()
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})
    fig1()
    fig2()
    fig3()
    fig4()


if __name__ == "__main__":
    main()
