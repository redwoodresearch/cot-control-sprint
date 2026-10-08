"""Slide figure: does telling the model it failed help beyond plain resampling?
Reads kaushik/retry/runs/main/results.md (gpt-oss-120b table); writes fig_retry.png and fig_retry.csv here."""
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "retry/runs/main/results.md"
ACCENT, GREY, DARK = "#d1495b", "#9e9e9e", "#7a1f2b"

# (plain-English label, arm name in results.md)
BARS = [
    ("Just resample\n(no feedback)", "independent"),
    ("\u201cYou failed. Try again.\u201d", "bare"),
    ("Critic explains\nthe failure", "critic"),
    ("\u201cYou failed\u201d + shows its\nprevious reasoning", "bare_trace"),
    ("Blocked message,\nno retry info", "block"),
]


def cell(s):
    m = re.match(r"\s*([\d.]+) \[([\d.]+), ([\d.]+)\]", s)
    return tuple(float(x) for x in m.groups())


def load():
    lines = SRC.read_text().splitlines()
    i = lines.index("## 3A: gptoss120b")
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("| arm |"))
    head = [h.strip() for h in lines[j].strip("|").split("|")]
    out = {}
    for ln in lines[j + 2:]:
        if not ln.startswith("|"):
            break
        c = [x.strip() for x in ln.strip("|").split("|")]
        out[c[0]] = (cell(c[head.index("k=10")]), cell(c[head.index("honest&long@N")]))
    return out


data = load()
rows = []
for label, arm in BARS:
    (v, lo, hi), (lv, llo, lhi) = data[arm]
    rows.append(dict(label=label.replace("\n", " "), source_arm=arm, complied_by_attempt10_pct=v, ci_lo=lo, ci_hi=hi,
                     length_matched_pct=lv, length_matched_ci_lo=llo, length_matched_ci_hi=lhi))
with open(HERE / "fig_retry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="white")
fig.subplots_adjust(left=0.25, right=0.96, top=0.80, bottom=0.20)

H, OFF = 0.36, 0.19
base = rows[0]["complied_by_attempt10_pct"]
ax.axvline(base, color="#555555", ls="--", lw=1.2, alpha=0.45, zorder=1.5)
ax.text(base + 0.6, -0.62, "resampling baseline", color="#555555", fontsize=12, va="center", ha="left")

for yi, r in enumerate(rows):
    for dy, key, col, bold in ((-OFF, "complied_by_attempt10_pct", GREY, False), (OFF, "length_matched_pct", ACCENT, True)):
        v = r[key]
        lo, hi = (r["ci_lo"], r["ci_hi"]) if dy < 0 else (r["length_matched_ci_lo"], r["length_matched_ci_hi"])
        ax.barh(yi + dy, v, height=H, color=col, zorder=2)
        ax.errorbar(v, yi + dy, xerr=[[v - lo], [hi - v]], color="#333333", lw=1.1, capsize=2.5, zorder=4)
        ax.text(hi + 0.9, yi + dy, f"{v:.1f}%", va="center", ha="left", fontsize=14, fontweight="bold" if bold else "normal",
                color=DARK if bold else "#333333")

ax.set_yticks(range(len(BARS)))
ax.set_yticklabels([lab for lab, _ in BARS])
ax.set_ylim(len(BARS) - 0.45, -0.95)
ax.set_xlim(0, 72)
ax.set_xlabel("Questions where the model complies by attempt 10 (%)")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="x", color="#e6e6e6", lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Rectangle((0, 0), 1, 1, color=ACCENT)],
          labels=["All answers", "Normal-length reasoning only"], loc="upper right", frameon=False, fontsize=14,
          bbox_to_anchor=(1.0, 1.02))

fig.text(0.02, 0.945, "Feedback raises control only by shortening reasoning", fontsize=20, fontweight="bold",
         ha="left", va="center")
fig.text(0.02, 0.885, "gpt-oss-120b \u00b7 up to 10 attempts per question \u00b7 1,000 questions per feedback type (95% CI)",
         fontsize=14, color="#555555", ha="left", va="center")
fig.text(0.02, 0.085, "Normal-length reasoning = at least 1,596 characters, the median length of a first attempt.",
         fontsize=11, color="#8a8a8a", ha="left", va="center")
fig.text(0.02, 0.045, "Compliance = followed the reasoning instruction AND answered AND reasoning not empty/degenerate.",
         fontsize=11, color="#8a8a8a", ha="left", va="center")
fig.savefig(HERE / "fig_retry.png", dpi=200, facecolor="white")
