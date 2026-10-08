"""Slide figure: persona SFT is the only intervention that moves control, and only a little.
Reads values straight from the orange SFT / RL source CSVs and writes fig_orange.png + fig_orange.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ORANGE = HERE.parents[1] / "orange"
SFT_CSV = ORANGE / "figures/data/fig1_main_result.csv"
RL_CSV = ORANGE / "rl/figures/data/fig1_main_result.csv"
REL = lambda p: str(p.relative_to(HERE.parents[1].parent))

sft = {r["arm"]: r for r in csv.DictReader(open(SFT_CSV))
       if r["model"] == "qwen36_27b" and r["condition"] == "persona" and r["split"] == "test" and r["eval_set"] == "fixed"}
rl = {(r["variant_row"], r["row"]): r for r in csv.DictReader(open(RL_CSV)) if r["condition"] == "persona"}

def S(arm, label, note):
    r = sft[arm]
    return dict(label=label, group="SFT", val=float(r["honest_pct"]), lo=float(r["ci95_lo"]), hi=float(r["ci95_hi"]),
                n_seeds=int(r["n_seeds"]), source=f"{REL(SFT_CSV)} (arm={arm}, test, persona, step {r['step']})", note=note)

def R(variant, label):
    r = rl[(variant, "rl_orange")]
    return dict(label=label, group="RL", val=float(r["honest_pct"]), lo=float(r["ci_lo"]), hi=float(r["ci_hi"]),
                n_seeds=int(float(r["n_seeds"])), source=f"{REL(RL_CSV)} (variant={variant}, row=rl_orange, test, persona); matches rl/figures/_analysis/tables.md",
                note="")

bars = [
    S("untrained", "Untrained", "untrained = step-0 of all SFT runs (6 seeds)"),
    S("noise[orange]", "Random weight change\n(size-matched)", "random weight change of the same size as orange training; only 3 seeds"),
    S("mcq_noreason", "Generic quiz\ntraining", ""),
    S("orange_nopersona", "Orange training,\nno persona", ""),
    S("orange", "Orange training\n+ persona", "highlighted; reasoning length 14,914 -> 6,707 chars (results/table.md, untrained vs orange, test, persona)"),
    R("noreason", "Orange RL,\nreasoning off"),
    R("reason", "Orange RL,\nreasoning on"),
]
xs = [0, 1.0, 2.15, 3.15, 4.15, 5.5, 6.5]
ACCENT, GREY = "#d1495b", "#9e9e9e"

plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200, facecolor="white")
fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.20)

for x, b in zip(xs, bars):
    hot = b["label"].startswith("Orange training\n+")
    ax.bar(x, b["val"], width=0.62, color=ACCENT if hot else GREY, zorder=3)
    top = b["val"]
    if b["hi"] > b["val"] + 1e-9 or b["lo"] < b["val"] - 1e-9:
        ax.errorbar(x, b["val"], yerr=[[max(b["val"] - b["lo"], 0)], [max(b["hi"] - b["val"], 0)]],
                    color="black", lw=1, capsize=4, capthick=1, zorder=4)
        top = b["hi"]
    ax.text(x, top + 0.2, f"{b['val']:.1f}%", ha="center", va="bottom", fontsize=15,
            fontweight="bold" if hot else "normal", color=ACCENT if hot else "#333333", zorder=5)

ax.set_xticks(xs)
ax.set_xticklabels([b["label"] for b in bars], fontsize=13, linespacing=1.15, va="top")
ax.set_xlim(-0.5, 7.0)
ax.set_ylim(0, 10)
ax.set_yticks(range(0, 11, 2))
ax.set_yticklabels([f"{t}" for t in range(0, 11, 2)])
ax.set_ylabel("% answers following the\nreasoning instruction", fontsize=15)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.yaxis.grid(True, color="#e6e6e6", lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0, pad=8)

# SFT | RL separator and group labels
sep = (xs[4] + xs[5]) / 2
ax.axvline(sep, ymin=0, ymax=1, color="#888888", lw=0.9, ls=(0, (4, 3)), zorder=2)
ax.text(sum(xs[:5]) / 5, 9.7, "SFT", ha="center", va="top", fontsize=15, fontweight="bold", color="#555555")
ax.text(sum(xs[5:]) / 2, 9.7, "RL", ha="center", va="top", fontsize=15, fontweight="bold", color="#555555")

# annotation next to accent bar
ax.annotate("reasoning ~halved\n(14.9k → 6.7k chars)", xy=(xs[4] - 0.33, 3.0), xytext=(xs[4] - 0.62, 3.0),
            ha="right", va="center", fontsize=12.5, color=ACCENT,
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.1, shrinkA=2, shrinkB=0))

fig.text(0.04, 0.945, "Only persona SFT moves control, and only a little", fontsize=20, fontweight="bold", ha="left", va="center")
fig.text(0.04, 0.885, "Qwen3.6-27B · persona prompt · held-out test questions · 5 seeds (95% CI)",
         fontsize=12, color="#555555", ha="left", va="center")
fig.text(0.04, 0.03, "Compliance = followed the reasoning instruction AND answered AND reasoning not empty/degenerate.",
         fontsize=10.5, color="#8a8a8a", ha="left", va="center")

# shrink x tick labels if neighbours would overlap
fig.canvas.draw()
rend = fig.canvas.get_renderer()
fit = False
for fs in (13, 12.5, 12, 11.5, 11, 10.5):
    for t in ax.get_xticklabels():
        t.set_fontsize(fs)
    fig.canvas.draw()
    bb = [t.get_window_extent(rend) for t in ax.get_xticklabels()]
    if all(a.x1 + 6 < b.x0 for a, b in zip(bb, bb[1:])):
        fit = True
        break
print("x tick font size:", fs, "no overlap:", fit)

fig.savefig(HERE / "fig_orange.png", dpi=200, facecolor="white")

with open(HERE / "fig_orange.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["order", "condition", "group", "value_pct", "ci95_lo", "ci95_hi", "n_seeds", "source", "note"])
    for i, b in enumerate(bars, 1):
        w.writerow([i, b["label"].replace("\n", " "), b["group"], round(b["val"], 2), round(b["lo"], 2), round(b["hi"], 2),
                    b["n_seeds"], b["source"], b["note"]])
