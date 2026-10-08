"""Publication figures for Experiment 1 (the fragility sweep).

Re-run from the repo root:  .venv/bin/python kaushik/fragility/figures/make_figures.py

Reads only summary products (never the per-run JSONs):
  kaushik/fragility/results/{exp1_test_table,exp1_val_table}.csv, beats.md, exp1_rollouts.csv.gz
  kaushik/fragility/cpt/results/{lr_selection,summary}.json
and writes, next to this file, fig1..fig4 as PNG (200 dpi) + PDF, plus the plotted numbers as
data/fig*.csv.

Conventions. "Honest compliance" = strict grader passes AND answer extracted AND trace not
truncated / < 200 chars / repetitive. "Full-length trace" = at least as long as the untrained
model's median for that constraint mode (the `long` flag in exp1_rollouts.csv.gz).
`compl_long` = raw compliance among full-length traces (the analysis' own metric).
All intervals are 95% two-way cluster bootstraps over questions and seeds, the same estimator
as kaushik/fragility/analyze.py. "Untrained" pools every arm's checkpoint-0 rollouts of a model.
"""
# ruff: noqa: C408
from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
FRAG = HERE.parent
ROOT = FRAG.parents[1]
sys.path.insert(0, str(ROOT))
from utils import PALETTE, set_matplotlib_style

RES = FRAG / "results"
CPT_RES = FRAG / "cpt" / "results"
DATA_OUT = HERE / "data"
DPI = 200
N_BOOT = 1000
HELDOUT = {"start_of_sentence", "letter_suppression", "no_spaces"}  # cotcontrol.eval.prompts.HELDOUT_MODES

# ---- colour tokens: validated categorical slots 1-2 from utils.PALETTE + recessive grays ----
BLUE, ORANGE = PALETTE[0], PALETTE[1]  # trained, random-adapter control
GRAY = "#898781"  # untrained reference (de-emphasis gray)
GRAY_BAND = "#c3c2b7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]  # ordinal blue ramp (palette steps 250 / 400 / 550) for LR rungs

MODELS = [("gptoss20b", "gpt-oss-20b"), ("qwen8b", "Qwen3-8B"), ("qwen36_27b", "Qwen3.6-27B")]
MODEL_NAME = dict(MODELS)


# ============================================================================== statistics
def boot_draws(df: pd.DataFrame, metric: str, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """Two-way cluster bootstrap (seeds x questions) of the mean of `metric`; same as analyze.py."""
    sub = df.dropna(subset=[metric])
    if sub.empty:
        return np.full(n_boot, np.nan)
    sums = sub.pivot_table(index="seed", columns="qid", values=metric, aggfunc="sum", fill_value=0.0)
    cnts = sub.pivot_table(index="seed", columns="qid", values=metric, aggfunc="count", fill_value=0)
    a, c = sums.to_numpy(float), cnts.to_numpy(float)
    s, q = a.shape
    rng = np.random.default_rng(seed)
    rows, cols = rng.integers(0, s, (n_boot, s)), rng.integers(0, q, (n_boot, q))

    def draw(x):
        return np.take_along_axis(x[rows], cols[:, None, :], axis=2).sum(axis=(1, 2))

    return draw(a) / np.maximum(draw(c), 1)


def mean_ci(df: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    vals = df[metric].dropna()
    if vals.empty:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(boot_draws(df, metric), [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


# ============================================================================== data loading
def load_tables():
    test = pd.read_csv(RES / "exp1_test_table.csv")
    test = test[test.subset == "train_modes"]
    val = pd.read_csv(RES / "exp1_val_table.csv")
    sel = json.loads((CPT_RES / "lr_selection.json").read_text())
    cpt_summary = json.loads((CPT_RES / "summary.json").read_text())["models"]
    return test, val, sel, cpt_summary


def load_rollouts() -> pd.DataFrame:
    cols = ["kind", "arm", "model", "seed", "target_arm", "step", "split", "condition", "mode", "qid",
            "honest", "raw", "correct", "chars", "degen", "long", "compl_long"]
    r = pd.read_csv(RES / "exp1_rollouts.csv.gz", usecols=cols, low_memory=False)
    r = r[(r.split == "test") & (r.condition == "stated") & (r["mode"] != "baseline") & ~r["mode"].isin(HELDOUT)].copy()
    r["long"] = r["long"].astype(bool)
    r["honest_long"] = r.honest.where(r.long)
    r["chars_k"] = r.chars / 1000.0
    return r


def parse_beats() -> dict:
    """Paired differences for 1A (test, 9 training modes) from results/beats.md, in percentage points."""
    out: dict = {}
    text = (RES / "beats.md").read_text()
    sec = text.split("## test, stated, 9 training modes")[-1]
    num = r"([+-]\d+\.\d+)"
    pat = re.compile(rf"\*\*mcq_noreason / (\S+)\*\*.*?honest vs untrained: {num} \[{num}, {num}\]; honest vs noise: {num} \[{num}, {num}\]")
    for line in sec.splitlines():
        m = pat.search(line)
        if m:
            v = [float(x) * 100 for x in m.groups()[1:]]
            out[m.group(1)] = dict(vs_untrained=v[:3], vs_noise=v[3:])
    return out


# ============================================================================== plotting helpers
def style():
    set_matplotlib_style()
    mpl.rcParams.update({"savefig.dpi": DPI, "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5,
                         "axes.titlesize": 9.5, "axes.titlecolor": INK})


def pct(x):
    return np.asarray(x, float) * 100


def mark(ax, x, m, lo, hi, kind, ms=6.5, label=None, zorder=3, mult=100.0):
    """One estimate with a 95% CI whisker. kind: trained | noise | untrained | ramp:<hex>."""
    if np.isnan(m):
        return
    kw = dict(trained=dict(marker="o", mfc=BLUE, mec="white", mew=1.1, c=BLUE),
              noise=dict(marker="D", mfc="white", mec=ORANGE, mew=1.5, c=ORANGE),
              untrained=dict(marker="s", mfc=GRAY, mec="white", mew=1.1, c=GRAY))
    if kind.startswith("ramp:"):
        col = kind.split(":")[1]
        st = dict(marker="o", mfc=col, mec="white", mew=1.1, c=col)
    else:
        st = kw[kind]
    c = st.pop("c")
    if not (np.isnan(lo) or np.isnan(hi)) and hi - lo > 1e-12:
        ax.plot([x, x], [lo * mult, hi * mult], color=c, lw=1.5, solid_capstyle="round", zorder=zorder - 1)
    ax.plot([x], [m * mult], ls="none", ms=ms, zorder=zorder, label=label, clip_on=False, **st)


def header(fig, title: str, subtitle: str, handles, ncol: int, title_w: int = 74, sub_w: int = 122) -> float:
    """Title, subtitle and legend stacked from the top; returns the figure-fraction y where the panels may start."""
    h = fig.get_size_inches()[1] * fig.dpi
    rend = fig.canvas.get_renderer()
    y = 0.993
    t = fig.text(0.012, y, textwrap.fill(title, title_w), ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK, linespacing=1.15)
    y -= t.get_window_extent(rend).height / h + 0.012
    s_ = fig.text(0.012, y, textwrap.fill(subtitle, sub_w), ha="left", va="top", fontsize=7.8, color=INK2, linespacing=1.25)
    y -= s_.get_window_extent(rend).height / h + 0.016
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, y), bbox_transform=fig.transFigure, ncol=ncol, frameon=False,
                     fontsize=7.2, handletextpad=0.4, columnspacing=1.5, borderaxespad=0)
    y -= leg.get_window_extent(rend).height / h + 0.034  # room for the panel titles
    return y


def footnote(fig, text: str, width: int = 138):
    fig.text(0.012, 0.004, textwrap.fill(text, width), ha="left", va="bottom", fontsize=6.6, color=MUTED, linespacing=1.25)


def save(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", HERE / f"{name}.png")


def legend_handles(items):
    out = []
    for kind, lab in items:
        if kind == "trained":
            out.append(Line2D([], [], ls="none", marker="o", ms=6.5, mfc=BLUE, mec="white", mew=1.1, label=lab))
        elif kind == "noise":
            out.append(Line2D([], [], ls="none", marker="D", ms=6, mfc="white", mec=ORANGE, mew=1.5, label=lab))
        elif kind == "untrained":
            out.append(Patch(fc=GRAY_BAND, ec="none", alpha=0.55, label=lab))
        elif kind == "untrained_dot":
            out.append(Line2D([], [], ls="none", marker="s", ms=6, mfc=GRAY, mec="white", mew=1.1, label=lab))
    return out


def clean_axes(ax):
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)


# ============================================================================== shared numbers
def untrained_pooled(r: pd.DataFrame, model: str, metric="honest"):
    return mean_ci(r[(r.model == model) & (r.step == 0) & (r.kind != "noise")], metric)


# ============================================================================== figure 1
ARMS1 = [  # key, (arm in test table, noise target_arm, step), tick label
    ("1A", ("mcq_noreason", "mcq_noreason", 50), "1A\nMCQ,\nno reasoning"),
    ("1D", ("mcq_reason", "mcq_reason", 50), "1D\nMCQ,\nreasoning on"),
    ("1E", ("style_sft", "style_sft", 50), "1E\nstyle SFT\n(pirate)"),
    ("1B", ("cpt-r32", "cpt-cpt-step50", 50), "1B\nC4 text,\nrank 32"),
]


def figure1(test, r, beats):
    rows = []
    fig = plt.figure(figsize=(8.2, 4.75))
    top = header(
        fig,
        "Only generic multiple-choice training without reasoning (1A) beats both the untrained model and a random "
        "adapter, and only by 1 to 4 points.",
        "Test split, 'stated' condition, 9 training constraint modes, 500 questions per mode. Whiskers: 95% bootstrap CI over questions and "
        "seeds. Text in each panel: 1A's change in honest compliance in percentage points, with its 95% CI.",
        legend_handles([("trained", "trained, final checkpoint (step 50)"),
                        ("noise", "random adapter, matched to the trained model's KL"),
                        ("untrained", "untrained model (mean and 95% CI)")]), 3)
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 4, 1.9], left=0.075, right=0.995, top=top, bottom=0.165, wspace=0.07)
    axes = []
    ymax = 7.4
    for j, (mk, mname) in enumerate(MODELS):
        ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
        axes.append(ax)
        um, ulo, uhi = untrained_pooled(r, mk)
        ax.axhspan(ulo * 100, uhi * 100, color=GRAY_BAND, alpha=0.55, lw=0, zorder=1)
        ax.axhline(um * 100, color=GRAY, lw=1.0, zorder=1)
        rows.append(dict(model=mk, arm="untrained", honest=um, lo=ulo, hi=uhi, kind="untrained"))
        present = []
        for key, (arm, noise_t, step), lab in ARMS1:
            t = test[(test.model == mk) & (test.arm == arm) & (test.step == step)]
            n = test[(test.arm == "noise") & (test.target_arm == noise_t) & (test.model == mk)]
            if t.empty:
                continue
            present.append((key, lab, t.iloc[0], n.iloc[0] if len(n) else None))
        for i, (key, lab, t, n) in enumerate(present):
            mark(ax, i - 0.17, t.honest, t.honest_lo, t.honest_hi, "trained")
            rows.append(dict(model=mk, arm=key, honest=t.honest, lo=t.honest_lo, hi=t.honest_hi, kind="trained",
                             acc=t.correct, chars=t.chars, n_seeds=t.n_seeds))
            if n is not None:
                mark(ax, i + 0.17, n.honest, n.honest_lo, n.honest_hi, "noise")
                rows.append(dict(model=mk, arm=key, honest=n.honest, lo=n.honest_lo, hi=n.honest_hi, kind="noise"))
            if key == "1E":
                ax.axvspan(i - 0.5, i + 0.5, color="#f0efec", zorder=0, lw=0)
                ax.text(i, ymax * 0.5, "format collapse:\naccuracy ~0%,\nnot interpretable", ha="center", va="center", fontsize=6.6,
                        color=INK2, linespacing=1.25)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([p[1] for p in present], fontsize=6.6, linespacing=1.15)
        ax.set_xlim(-0.55, len(present) - 0.45)
        ax.set_ylim(0, ymax)
        ax.set_title(mname, loc="left", pad=6)
        clean_axes(ax)
        if j > 0:
            ax.tick_params(axis="y", labelleft=False, length=0)
            ax.spines["left"].set_visible(False)
        ax.text(-0.5, um * 100 + 0.13, "untrained", ha="left", va="bottom", fontsize=6.6, color=INK2)
        b = beats.get(mk)
        a1 = next((p for p in present if p[0] == "1A"), None)
        if b and a1:
            u, nz = b["vs_untrained"], b["vs_noise"]
            txt = f"{u[0]:+.1f} [{u[1]:.1f}, {u[2]:.1f}] vs untrained\n{nz[0]:+.1f} [{nz[1]:.1f}, {nz[2]:.1f}] vs noise"
            ypos = min(a1[2].honest_hi * 100 + 0.5, ymax - 1.0) if len(present) > 1 else 3.0
            ax.text(-0.5, ypos, txt, ha="left", va="bottom", fontsize=6.7, color=INK, linespacing=1.3)
    axes[0].set_ylabel("Honest compliance (%)")
    axes[0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    footnote(fig, "Seeds: 3 per arm (Qwen3.6-27B 1A: 5); random adapters: 3 per arm, each matched to that arm's KL(base || adapter). "
                  "Untrained pools every arm's checkpoint-0 rollouts for that model (Qwen3.6-27B: 1A only). 1B is at step 50, the same number of "
                  "steps as 1A. Qwen3.6-27B has no 1B, 1D or 1E runs.")
    save(fig, "fig1_arms_vs_untrained_and_noise")
    pd.DataFrame(rows).to_csv(DATA_OUT / "fig1_arms.csv", index=False)


# ============================================================================== figure 2
def figure2(r):
    cond_style = [("untrained", "untrained", "untrained"), ("noise", "random adapter", "noise"), ("1A", "1A trained", "trained")]
    rows = []
    fig = plt.figure(figsize=(7.6, 6.5))
    top = header(
        fig,
        "1A's gain comes with reasoning that is 17 to 54% shorter; among full-length traces, honest compliance is about 2% or less.",
        "Top: compliance. 'raw' = constraint followed per the strict grader, including hollow traces (no answer, truncated, repeated); 'honest' "
        "removes those. 'Full-length' = at least as long as the untrained model's median for that mode; raw compliance among those traces is the "
        "analysis' compl_long. Bottom: mean reasoning length, with the change vs untrained.",
        legend_handles([("untrained_dot", "untrained"), ("noise", "random adapter, KL-matched to 1A"), ("trained", "1A, final checkpoint (step 50)")]), 3)
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.995, top=top, bottom=0.13, hspace=0.5, wspace=0.3)
    metrics = [("raw", "raw"), ("honest", "honest"), ("compl_long", "raw"), ("honest_long", "honest")]
    gx = [0, 1, 2.45, 3.45]
    offs = {"untrained": -0.24, "noise": 0.0, "1A": 0.24}
    for j, (mk, mname) in enumerate(MODELS):
        sub = {
            "untrained": r[(r.model == mk) & (r.step == 0) & (r.kind != "noise")],
            "noise": r[(r.model == mk) & (r.kind == "noise") & (r.target_arm == "mcq_noreason")],
            "1A": r[(r.model == mk) & (r.arm == "mcq_noreason") & (r.step == 50)],
        }
        ax = fig.add_subplot(gs[0, j])
        ymax = 0
        for cname, _, kind in cond_style:  # where does honest compliance come from? (point estimates, no CI)
            sc = sub[cname]
            rows.append(dict(model=mk, panel="decomp", cond=cname, metric="honest_from_short_traces", mean=float((sc.honest * ~sc.long).mean()), n=len(sc)))
            rows.append(dict(model=mk, panel="decomp", cond=cname, metric="honest_from_full_length_traces", mean=float((sc.honest * sc.long).mean()), n=len(sc)))
        for cname, _, kind in cond_style:
            for (met, _), x in zip(metrics, gx):
                m, lo, hi = mean_ci(sub[cname], met)
                mark(ax, x + offs[cname], m, lo, hi, kind, ms=5.6)
                rows.append(dict(model=mk, panel="compliance", cond=cname, metric=met, mean=m, lo=lo, hi=hi, n=len(sub[cname])))
                ymax = max(ymax, hi * 100)
        ax.set_ylim(0, max(ymax * 1.12, 1.0))
        ax.set_xlim(-0.55, 3.95)
        ax.set_xticks(gx)
        ax.set_xticklabels([m[1] for m in metrics], fontsize=7.2)
        ax.set_title(mname, loc="left", pad=6)
        clean_axes(ax)
        for xs, lab in (([-0.4, 1.4], "all traces"), ([2.05, 3.85], "full-length traces only")):
            tr = mpl.transforms.offset_copy(ax.get_xaxis_transform(), fig=fig, y=-17, units="points")
            ax.plot(xs, [0, 0], transform=tr, color="#c3c2b7", lw=0.9, clip_on=False, solid_capstyle="butt")
            ax.text(np.mean(xs), 0, lab, transform=mpl.transforms.offset_copy(ax.get_xaxis_transform(), fig=fig, y=-21, units="points"),
                    ha="center", va="top", fontsize=7, color=INK2)
        if j == 0:
            ax.set_ylabel("Compliance (% of rollouts)")
        ax2 = fig.add_subplot(gs[1, j])
        base, ymax2 = None, 0
        for i, (cname, _, kind) in enumerate(cond_style):
            m, lo, hi = mean_ci(sub[cname], "chars_k")
            mark(ax2, i, m, lo, hi, kind, ms=6.2, mult=1.0)
            base = m if cname == "untrained" else base
            ymax2 = max(ymax2, hi)
            rows.append(dict(model=mk, panel="chars_k", cond=cname, metric="chars_k", mean=m, lo=lo, hi=hi, n=len(sub[cname]),
                             p_full_length=sub[cname].long.mean(), p_under_200_chars=sub[cname].degen.mean()))
            if cname != "untrained":
                ax2.text(i + 0.15, m, f"{(m / base - 1) * 100:+.0f}%", ha="left", va="center", fontsize=7.2, color=INK)
        ax2.set_xlim(-0.5, 2.8)
        ax2.set_ylim(0, ymax2 * 1.42)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(["untrained", "random\nadapter", "1A\ntrained"], fontsize=7.2, linespacing=1.15)
        clean_axes(ax2)
        pl = {c: sub[c].long.mean() * 100 for c in sub}
        note = f"full-length traces: {pl['untrained']:.0f}% untrained, {pl['1A']:.0f}% trained"
        dg = sub["1A"].degen.mean() * 100
        if dg >= 5:
            note += f"\n{dg:.0f}% of 1A traces are under 200 characters"
        ax2.text(0.02, 0.98, note, transform=ax2.transAxes, ha="left", va="top", fontsize=6.6, color=INK2, linespacing=1.3)
        if j == 0:
            ax2.set_ylabel("Mean reasoning length\n(thousand characters)")
    footnote(fig, "Test split, 'stated' condition, 9 training modes. Each model has its own y-axis. Whiskers: 95% bootstrap CI over questions "
                  "and seeds. Qwen3-8B 1A: raw compliance among full-length traces is 16% but honest compliance among them is 0.2%, i.e. those "
                  "long traces are degenerate rather than controlled.")
    save(fig, "fig2_gain_lives_in_short_traces")
    pd.DataFrame(rows).to_csv(DATA_OUT / "fig2_trace_length.csv", index=False)


# ============================================================================== figure 3 + 4 data
def cpt_cells(val, sel):
    """One row per (model, k, lr) val cell at step 50: honest compliance, accuracy, KL, C4 loss, LR rung, selected flag."""
    out = []
    v = val[val.arm.str.startswith("cptk-")]
    for _, row in v[v.step == 50].iterrows():
        m = re.match(r"cptk-(1e3|1e4|1e5|r1)-lr([0-9.]+e[+-]?\d+)$", row.arm)
        out.append(dict(model=row.model, k=m.group(1), lr=float(m.group(2)), arm=row.arm, n_seeds=row.n_seeds, honest=row.honest,
                        lo=row.honest_lo, hi=row.honest_hi, acc=row.correct, acc_lo=row.correct_lo, acc_hi=row.correct_hi, kl=row.kl))
    d = pd.DataFrame(out)
    d["rung"] = d.groupby(["model", "k"]).lr.rank(method="first").astype(int) - 1
    d["selected"] = [abs(sel[f"{x.model}/{x.k}"]["lr"] - x.lr) / x.lr < 0.03 for x in d.itertuples()]
    loss = {(v_["model"], v_["klabel"], round(c["lr"], 6)): c.get("final_loss") for v_ in sel.values() for c in v_["cells"]}
    d["loss"] = [loss.get((x.model, x.k, round(x.lr, 6))) for x in d.itertuples()]
    return d


XLAB = {"1e3": "10$^3$", "1e4": "10$^4$", "1e5": "10$^5$"}


def figure3(test, val, sel, cpt_summary, r):
    d = cpt_cells(val, sel)
    rows = []
    fig = plt.figure(figsize=(7.6, 8.0))
    top = header(
        fig,
        "Continued pretraining on web text never raises honest compliance at any adapter size, even when it moves the model as far as "
        "1A does.",
        "Top: honest compliance (test split, 9 training modes, 3 seeds; Qwen3-8B k = 10\u00b3 and 10\u2074: 2 seeds). Middle: language-model loss on C4 "
        "training batches; small adapters fit C4 far less. Bottom: how far the adapter moves the model's output distribution (KL from the base model).",
        legend_handles([("trained", "C4 continued pretraining, step 50"), ("noise", "random adapter, KL-matched to rank 32"),
                        ("untrained", "untrained model (mean and 95% CI)")])
        + [Line2D([], [], color=BLUE, lw=1.0, ls=(0, (4, 3)), label="1A MCQ SFT, for scale (Figure 1)"),
           Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec=GRAY, mew=1.0, label="other learning rates tried (middle row)")], 3)
    gs = fig.add_gridspec(3, 2, left=0.085, right=0.99, top=top, bottom=0.125, hspace=0.3, wspace=0.14)
    ks = ["1e3", "1e4", "1e5", "r1", "r32"]
    for j, mk in enumerate(["gptoss20b", "qwen8b"]):
        kc = cpt_summary[mk]["kcurve"]
        xs = {"1e3": 1e3, "1e4": 1e4, "1e5": 1e5, "r1": kc["r1"]["k_nominal"], "r32": kc["r32"]["k_nominal"]}
        axc, axl, axk = (fig.add_subplot(gs[i, j]) for i in range(3))
        # ----- compliance (test split, selected learning rate)
        um, ulo, uhi = untrained_pooled(r, mk)
        a1 = test[(test.model == mk) & (test.arm == "mcq_noreason") & (test.step == 50)].iloc[0]
        axc.axhspan(ulo * 100, uhi * 100, color=GRAY_BAND, alpha=0.55, lw=0, zorder=1)
        axc.axhline(um * 100, color=GRAY, lw=1.0, zorder=1)
        axc.axhline(a1.honest * 100, color=BLUE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        axc.text(0.995, a1.honest * 100 + 0.12, "1A MCQ SFT", transform=axc.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.8, color=INK2)
        axc.text(0.005, um * 100 + 0.2, "untrained", transform=axc.get_yaxis_transform(), ha="left", va="bottom", fontsize=6.8, color=INK2)
        for k in ks:
            if k == "r32":
                t = test[(test.model == mk) & (test.arm == "cpt-r32") & (test.step == 50)].iloc[0]
                n = test[(test.arm == "noise") & (test.target_arm == "cpt-cpt-step50") & (test.model == mk)].iloc[0]
                mark(axc, xs[k] * 0.8, t.honest, t.honest_lo, t.honest_hi, "trained", ms=6.2)
                mark(axc, xs[k] * 1.3, n.honest, n.honest_lo, n.honest_hi, "noise", ms=5.6)
                lr = 1.5e-4
                rows.append(dict(model=mk, panel="compliance_noise", k=k, k_nominal=xs[k], honest=n.honest, lo=n.honest_lo, hi=n.honest_hi))
            else:
                c = d[(d.model == mk) & (d.k == k) & d.selected].iloc[0]
                t = test[(test.model == mk) & (test.arm == c.arm) & (test.step == 50)].iloc[0]
                mark(axc, xs[k], t.honest, t.honest_lo, t.honest_hi, "trained", ms=6.2)
                lr = c.lr
            rows.append(dict(model=mk, panel="compliance", k=k, k_nominal=xs[k], honest=t.honest, lo=t.honest_lo, hi=t.honest_hi,
                             n_seeds=t.n_seeds, lr=lr))
        # ----- C4 loss (seed 0, last step; the rank-32 run's step-50 loss is not in the summary tables)
        for k in ks[:-1]:
            sub = d[(d.model == mk) & (d.k == k)]
            oth = sub[~sub.selected]
            axl.plot([xs[k]] * len(oth), oth.loss, ls="none", marker="o", ms=4.2, mfc="white", mec=GRAY, mew=1.0, zorder=2)
        s = d[(d.model == mk) & d.selected].set_index("k").loc[ks[:-1]].reset_index()
        axl.plot([xs[k] for k in s.k], s.loss, color=BLUE, lw=1.6, zorder=3)
        axl.plot([xs[k] for k in s.k], s.loss, ls="none", marker="o", ms=6.2, mfc=BLUE, mec="white", mew=1.1, zorder=4)
        for _, q in s.iterrows():
            rows.append(dict(model=mk, panel="c4_loss", k=q.k, k_nominal=xs[q.k], loss=q.loss, lr=q.lr))
        # ----- KL
        a1kl = float(val[(val.model == mk) & (val.arm == "mcq_noreason")].kl.max())
        axk.axhline(a1kl, color=BLUE, lw=1.0, ls=(0, (4, 3)), zorder=1)
        left = mk == "gptoss20b"
        axk.text(0.005 if left else 0.995, a1kl + 0.003, "1A MCQ SFT", transform=axk.get_yaxis_transform(), ha="left" if left else "right",
                 va="bottom", fontsize=6.8, color=INK2)
        kls = []
        for k in ks:
            if k == "r32":
                kl = float(np.mean([v_["50"] for v_ in cpt_summary[mk]["kl_1b"].values()]))
            else:
                kl = float(d[(d.model == mk) & (d.k == k) & d.selected].kl.iloc[0])
            kls.append(kl)
            rows.append(dict(model=mk, panel="kl", k=k, k_nominal=xs[k], kl=kl))
        axk.plot([xs[k] for k in ks], kls, color=BLUE, lw=1.6, zorder=3)
        axk.plot([xs[k] for k in ks], kls, ls="none", marker="o", ms=6.2, mfc=BLUE, mec="white", mew=1.1, zorder=4)
        for ax in (axc, axl, axk):
            ax.set_xscale("log")
            ax.set_xlim(10 ** 2.7, 10 ** 9.05)
            ax.set_xticks([xs[k] for k in ks])
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            clean_axes(ax)
        for ax in (axc, axl):
            ax.set_xticklabels([])
        axk.set_xticklabels([XLAB.get(k) or (f"rank 1\n{xs[k] / 1e6:.1f}M" if k == "r1" else f"rank 32\n{xs[k] / 1e6:.0f}M") for k in ks],
                            fontsize=7.2, linespacing=1.15)
        axk.set_xlabel("Trainable parameters (log scale)", fontsize=7.8)
        axc.set_ylim(0, 6.2)
        axc.set_title(MODEL_NAME[mk], loc="left", pad=6)
        axk.set_ylim(0, 0.14)
        axl.margins(y=0.15)
        approx = "2.7" if mk == "gptoss20b" else "2.5"
        axl.text(0.97, 0.94, f"rank-32 loss is not in the summary\ntables (about {approx} at step 200,\nper cpt/README)", transform=axl.transAxes, ha="right",
                 va="top", fontsize=6.4, color=MUTED, linespacing=1.3)
        if j == 0:
            axc.set_ylabel("Honest compliance (%)")
            axl.set_ylabel("C4 training loss\n(seed 0, last step)")
            axk.set_ylabel("KL(base || adapter)\n(nats per token)")
        else:
            axc.tick_params(axis="y", labelleft=False)
            axk.tick_params(axis="y", labelleft=False)
    footnote(fig, "k = number of trainable scalars in a rank-1 LoRA (random subset of its entries); rank 1 / rank 32 = the full adapters. Learning "
                  "rate per k chosen on val honest compliance with an accuracy guard (cpt/README). Whiskers: 95% bootstrap CI over questions and seeds. "
                  "Loss: seed 0's final-step training loss. KL: mean over seeds (rank 32: step-50 adapters).")
    save(fig, "fig3_cpt_vs_trainable_parameters")
    pd.DataFrame(rows).to_csv(DATA_OUT / "fig3_cpt_vs_k.csv", index=False)


def figure4(val, sel):
    d = cpt_cells(val, sel)
    rows = []
    fig = plt.figure(figsize=(7.6, 5.9))
    handles = [Line2D([], [], ls="none", marker="o", ms=5.6, mfc=RAMP[i], mec="white", label=lab) for i, lab in
               enumerate(["lowest rate tried (1/3 of predicted)", "predicted rate", "highest rate tried (3x predicted)"])]
    handles += [Line2D([], [], ls="none", marker="o", ms=7.4, mfc="#bbbbbb", mec="white", label="large marker: selected rate (3 seeds)"),
                Patch(fc=GRAY_BAND, ec="none", alpha=0.55, label="untrained model (mean and 95% CI)")]
    top = header(
        fig,
        "No learning rate lifts honest compliance for any adapter size, including rates high enough to wreck accuracy.",
        "Rank-1 adapters with k trainable scalars, 50 steps of C4, three learning rates per k. Validation split, the four fixed-trait constraint "
        "modes. Top: honest compliance. Bottom: accuracy under the constraint, the quantity the learning-rate guard is based on.",
        handles, 2)
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.99, top=top, bottom=0.115, hspace=0.3, wspace=0.12)
    ks = ["1e3", "1e4", "1e5", "r1"]
    off = [-0.25, 0, 0.25]
    for j, mk in enumerate(["gptoss20b", "qwen8b"]):
        ax_h, ax_a = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        b = val[(val.model == mk) & (val.arm == "cpt-r32") & (val.step == 0)].iloc[0]  # untrained, same val items
        for ax, (m_, lo_, hi_) in ((ax_a, (b.correct, b.correct_lo, b.correct_hi)), (ax_h, (b.honest, b.honest_lo, b.honest_hi))):
            ax.axhspan(lo_ * 100, hi_ * 100, color=GRAY_BAND, alpha=0.55, lw=0, zorder=1)
            ax.axhline(m_ * 100, color=GRAY, lw=1.0, zorder=1)
        ax_a.axhline(b.correct * 85, color=INK2, lw=0.9, ls=(0, (2, 2)), zorder=1)
        ax_a.text(-0.55, b.correct * 85 - 1.3, "guard: 85% of untrained accuracy", ha="left", va="top", fontsize=6.6, color=INK2)
        for i, k in enumerate(ks):
            for _, c in d[(d.model == mk) & (d.k == k)].iterrows():
                x, col, ms = i + off[c.rung], RAMP[c.rung], (7.4 if c.selected else 5.4)
                mark(ax_h, x, c.honest, c.lo, c.hi, f"ramp:{col}", ms=ms)
                mark(ax_a, x, c.acc, c.acc_lo, c.acc_hi, f"ramp:{col}", ms=ms)
                rows.append(dict(model=mk, k=k, lr=c.lr, rung=int(c.rung), selected=bool(c.selected), n_seeds=int(c.n_seeds), honest=c.honest,
                                 honest_lo=c.lo, honest_hi=c.hi, acc=c.acc, acc_lo=c.acc_lo, acc_hi=c.acc_hi))
        for ax in (ax_h, ax_a):
            ax.set_xlim(-0.6, 3.65)
            ax.set_xticks(range(4))
            clean_axes(ax)
        ax_h.set_xticklabels([])
        ax_a.set_xticklabels(["k = 10$^3$", "k = 10$^4$", "k = 10$^5$", "full rank 1"], fontsize=7.4)
        ax_h.set_ylim(-0.1, 2.0)
        ax_h.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
        ax_a.set_ylim(0, 62)
        ax_h.set_title(MODEL_NAME[mk], loc="left", pad=6)
        if j == 0:
            ax_h.set_ylabel("Honest compliance (%)")
            ax_a.set_ylabel("Accuracy (%)")
        else:
            ax_h.tick_params(axis="y", labelleft=False)
            ax_a.tick_params(axis="y", labelleft=False)
    footnote(fig, "Single seed (800 rollouts) per learning rate, except the selected rate (3 seeds, 2,400 rollouts). Whiskers: 95% bootstrap CI over "
                  "questions and seeds. gpt-oss-20b is at 0 on these four modes even before training, so this split cannot show a decrease for it; "
                  "Figure 3 uses the test split.")
    save(fig, "fig4_cpt_learning_rate_sweep")
    pd.DataFrame(rows).to_csv(DATA_OUT / "fig4_cpt_lr_sweep.csv", index=False)


# ============================================================================== cross-checks
def sanity(test, r):
    """Recompute a few headline numbers from the rollouts and compare them to the shipped tables."""
    chk = []
    for mk, arm, step in [("gptoss20b", "mcq_noreason", 50), ("qwen8b", "mcq_noreason", 50), ("qwen36_27b", "mcq_noreason", 50)]:
        t = test[(test.model == mk) & (test.arm == arm) & (test.step == step)].iloc[0]
        m, lo, hi = mean_ci(r[(r.model == mk) & (r.arm == arm) & (r.step == step)], "honest")
        chk.append((mk, arm, t.honest, m, t.honest_lo, lo, t.honest_hi, hi))
    for c in chk:
        print("check honest {} {}: table {:.4f} vs mine {:.4f}; CI lo {:.4f} vs {:.4f}; hi {:.4f} vs {:.4f}".format(*c))
        assert abs(c[2] - c[3]) < 1e-9 and abs(c[4] - c[5]) < 2e-3 and abs(c[6] - c[7]) < 2e-3, c


def main():
    style()
    DATA_OUT.mkdir(exist_ok=True)
    test, val, sel, cpt_summary = load_tables()
    r = load_rollouts()
    sanity(test, r)
    beats = parse_beats()
    figure1(test, r, beats)
    figure2(r)
    figure3(test, val, sel, cpt_summary, r)
    figure4(val, sel)


if __name__ == "__main__":
    main()
