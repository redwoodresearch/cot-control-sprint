"""Slide figure for Experiment 3B (hidden rule). Reads kaushik/hidden_constraint/runs/summary.csv only.

    .venv/bin/python kaushik/mentor/figs/fig_hidden.py
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parents[1] / "hidden_constraint" / "runs" / "summary.csv"
SCEN = "config_notation"  # arithmetic task: graded suppression is only meaningful here
NAMES = {"dsv4pro": "DeepSeek V4 Pro", "glm53": "GLM-5.3", "gptoss120b": "gpt-oss-120b",
         "qwen36_27b": "Qwen3.6-27B", "gptoss20b": "gpt-oss-20b", "qwen8b": "Qwen3-8B", "qwen32b": "Qwen3-32B"}
GREY, ACCENT, INK, MUTED, GRID = "#b4b4b4", "#d1495b", "#1a1a1a", "#6b6b6b", "#ececec"

d = pd.read_csv(SUMMARY).query("scenario == @SCEN and model_or_checkpoint in @NAMES").set_index(
    ["model_or_checkpoint", "metric"])


def g(m, metric):
    r = d.loc[(m, metric)]
    return 100 * r.value, 100 * r.ci_low, 100 * r.ci_high, int(r.n)


rows = []
for m, name in NAMES.items():
    rd, sp, fo = g(m, "read_rate"), g(m, "suppression_ratio"), g(m, "constrained_success")
    rows.append(dict(model=name, read_pct=rd[0], read_lo=rd[1], read_hi=rd[2],
                     cut_pct=sp[0], cut_lo=sp[1], cut_hi=sp[2], cut_n_pairs=sp[3],
                     fully_obeyed_pct=fo[0], fully_obeyed_lo=fo[1], fully_obeyed_hi=fo[2], n_episodes=rd[3]))
df = pd.DataFrame(rows).sort_values("cut_pct", ascending=False).reset_index(drop=True)
df.round(2).to_csv(HERE / "fig_hidden.csv", index=False)

plt.rcParams.update({"font.size": 15, "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
fig.subplots_adjust(left=0.19, right=0.86, top=0.70, bottom=0.17)
h, y = 0.36, np.arange(len(df))[::-1].astype(float)
for yi, r in zip(y, df.itertuples()):
    for off, col, v, lo, hi in [(h / 2, GREY, r.read_pct, r.read_lo, r.read_hi),
                                (-h / 2, ACCENT, r.cut_pct, r.cut_lo, r.cut_hi)]:
        ax.barh(yi + off, v, height=h * 0.94, color=col, zorder=2)
        ax.plot([lo, hi], [yi + off] * 2, color=INK, lw=1.3, zorder=3, solid_capstyle="butt")
        ax.text(max(hi, v) + 1.8, yi + off, f"{v:.0f}%", va="center", ha="left", fontsize=14,
                color=INK, fontweight="bold" if col == ACCENT else "normal")
    ax.text(1.04, yi, f"{r.fully_obeyed_pct:.0f}%", transform=ax.get_yaxis_transform(), ha="center",
            va="center", fontsize=15, color=INK)
ax.text(1.04, len(df) - 0.35, "Fully\nobeyed", transform=ax.get_yaxis_transform(), ha="center", va="bottom",
        fontsize=13, color=MUTED, linespacing=1.1)
ax.set_yticks(y, df.model)
ax.set_ylim(-0.6, len(df) - 0.4)
ax.set_xlim(-14, 112)
ax.set_xticks(range(0, 101, 25), [f"{t}%" for t in range(0, 101, 25)])
ax.axvline(0, color=MUTED, lw=1, zorder=1)
ax.grid(axis="x", color=GRID, lw=1, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="both", length=0, colors=INK)
ax.tick_params(axis="x", colors=MUTED)

fig.text(0.02, 0.955, "Models read the rule, then break it anyway", fontsize=20, fontweight="bold",
         color=INK, ha="left", va="top")
fig.text(0.02, 0.885, "Hidden-rule task · base models · 95% CI", fontsize=15, color=MUTED, ha="left", va="top")
fig.legend(handles=[Patch(color=GREY, label="Read the rule"),
                    Patch(color=ACCENT, label="Cut forbidden symbols vs. no-rule baseline")],
           loc="upper left", bbox_to_anchor=(0.02, 0.83), ncol=2, frameon=False, fontsize=15,
           handlelength=1.2, columnspacing=2.5, borderaxespad=0)
fig.text(0.02, 0.025,
         "Task: arithmetic problems; the rule file says reasoning containing = + * / × ÷ fails. 250 runs per model. "
         "Read = opened the rule file.\nCut = drop in those symbols in reasoning after reading, vs. the same problems "
         "with no rule. Fully obeyed = right answer and no forbidden symbol after reading.",
         fontsize=10.5, color=MUTED, ha="left", va="bottom", linespacing=1.4)
fig.savefig(HERE / "fig_hidden.png", dpi=200, facecolor="white")
print(df.round(1).to_string())
