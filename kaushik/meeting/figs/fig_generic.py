"""Slide figure: unrelated training nudges control up beyond a random weight change.
Reads ../../fragility/figures/data/fig1_arms.csv (test split, 9 modes) -> fig_generic.png/.csv."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "fragility/figures/data/fig1_arms.csv"
MODELS = [("gptoss20b", "gpt-oss-20b", "accuracy 46→33%,\nreasoning 5.5k→2.5k chars"),
          ("qwen8b", "Qwen3-8B", "accuracy 40→27%,\n24% of answers degenerate"),
          ("qwen36_27b", "Qwen3.6-27B", "no accuracy loss")]
ARMS = [("untrained", "Untrained", "#d9d9d9"),
        ("noise", "Random weight change (size-matched)", "#6b6b6b"),
        ("trained", "Answer-only quiz training", "#d1495b")]

df = pd.read_csv(SRC)
df = df[df.arm.isin(["untrained", "1A"])]
rows = []
for key, name, cap in MODELS:
    for kind, label, _ in ARMS:
        r = df[(df.model == key) & (df.kind == kind)].iloc[0]
        n = df[(df.model == key) & (df.kind == "trained")].n_seeds.iloc[0]
        rows.append(dict(model=name, condition=label, pct=100 * r.honest, ci_lo=100 * r.lo,
                         ci_hi=100 * r.hi, n_seeds_trained=int(n), tradeoff=cap))
out = pd.DataFrame(rows)
out.round(3).to_csv(HERE / "fig_generic.csv", index=False)

plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
fig.subplots_adjust(right=0.98, top=0.70, bottom=0.22, left=0.085)
w = 0.26
x = np.arange(len(MODELS))
for j, (_, label, color) in enumerate(ARMS):
    d = out[out.condition == label]
    xs = x + (j - 1) * w
    ax.bar(xs, d.pct, w * 0.92, color=color, label=label, zorder=2)
    ax.errorbar(xs, d.pct, yerr=[d.pct - d.ci_lo, d.ci_hi - d.pct], fmt="none", ecolor="black",
                elinewidth=1, capsize=3, capthick=1, zorder=3)
    for xi, p, hi in zip(xs, d.pct, d.ci_hi):
        ax.text(xi, hi + 0.12, f"{p:.1f}", ha="center", va="bottom", fontsize=15,
                fontweight="bold" if j == 2 else "normal")
ax.set_xticks(x, [m[1] for m in MODELS], fontsize=16)
ax.tick_params(axis="x", length=0, pad=8)
for xi, (_, _, cap) in zip(x, MODELS):
    ax.text(xi, -0.115, cap, transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=12.5, color="#666666", linespacing=1.3)
ax.set_ylim(0, 7)
ax.set_ylabel("% answers following\nthe reasoning instruction", fontsize=15)
ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.text(0.99, 0.97, "Training on C4 web text:\nno meaningful effect at any size or learning rate",
        transform=ax.transAxes, ha="right", va="top", fontsize=13, color="#444444", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#dddddd"))
fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.065, 0.80), ncol=3,
           frameon=False, fontsize=14, handlelength=1.2, columnspacing=1.6)
fig.text(0.015, 0.965, "Unrelated training nudges control up a few points,\nbeyond a random weight change",
         fontsize=20, fontweight="bold", ha="left", va="top", linespacing=1.25)
fig.text(0.015, 0.845, "Standard 9-instruction eval · held-out test questions · 3–5 seeds (95% CI)",
         fontsize=12.5, color="#666666", ha="left", va="top")
fig.text(0.015, 0.02, "Compliance = followed the reasoning instruction AND answered AND reasoning not empty/degenerate.",
         fontsize=11, color="#777777", ha="left", va="bottom")
fig.savefig(HERE / "fig_generic.png", dpi=200, facecolor="white")
print(out[["model", "condition", "pct", "ci_lo", "ci_hi"]].round(2).to_string())
