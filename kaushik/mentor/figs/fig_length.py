"""Slide figure: compliance gains vs. the same gains among normal-length reasoning.
Reads values from source files (paths below), writes fig_length.csv and fig_length.png next to this script."""
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = Path(__file__).resolve().parents[2]  # .../kaushik
OUT = Path(__file__).resolve().parent
GREY, ACC = "#9e9e9e", "#d1495b"
rows = []  # label, all, all_lo, all_hi, long, long_lo, long_hi, base_all, base_long, source, note


def pct(x): return 100 * float(x)


# 1-2: fragility trace-length CSV (compliance panel)
src = "kaushik/fragility/figures/data/fig2_trace_length.csv"
F = {(r["model"], r["panel"], r["cond"], r["metric"]): r for r in csv.DictReader(open(K / "fragility/figures/data/fig2_trace_length.csv"))}
for model, label in [("gptoss20b", "Answer-only quiz training\n(gpt-oss-20b)"), ("qwen8b", "Answer-only quiz training\n(Qwen3-8B)")]:
    g = lambda cond, m: F[(model, "compliance", cond, m)]
    a, l, ua, ul = g("1A", "honest"), g("1A", "honest_long"), g("untrained", "honest"), g("untrained", "honest_long")
    rows.append([label, pct(a["mean"]), pct(a["lo"]), pct(a["hi"]), pct(l["mean"]), pct(l["lo"]), pct(l["hi"]),
                 pct(ua["mean"]), pct(ul["mean"]), f"{src} (model {model}, cond 1A, metrics honest / honest_long; baseline = cond untrained)",
                 "normal length = at least the untrained model's median reasoning length"])

# 3: orange persona (fig3c CSV)
O = {(r["arm"], r["subset"]): r for r in csv.DictReader(open(K / "orange/figures/data/fig3c_all_vs_long.csv"))}
a = O[("orange", "all traces")]; l = O[("orange", "traces at least as long as untrained median")]
rows.append(["Orange persona training\n(Qwen3.6-27B, persona prompt)", float(a["pct"]), float(a["ci95_lo"]), float(a["ci95_hi"]),
             float(l["pct"]), float(l["ci95_lo"]), float(l["ci95_hi"]), 0.0, 0.0,
             "kaushik/orange/figures/data/fig3c_all_vs_long.csv (arm orange); baseline 0.0 from kaushik/orange/README.md table (untrained, persona row)",
             "normal length = at least the untrained model's median; untrained long-length baseline assumed 0.0 (README lists untrained persona = 0.0 overall)"])

# 4: retry, gpt-oss-120b, main run, attempt 10, bare_trace arm
txt = (K / "retry/runs/main/results.md").read_text().split("## 3A: gptoss120b")[1].split("\n## ")[0]
def row(arm):
    line = next(x for x in txt.splitlines() if x.startswith(f"| {arm} |"))
    return [c.strip() for c in line.strip("|").split("|")]
num = lambda s: [float(v) for v in re.findall(r"-?\d+\.?\d*", s)]
bt, ind = row("bare_trace"), row("independent")
a, l, ba, bl = num(bt[5]), num(bt[6]), num(ind[5]), num(ind[6])  # col 5 = k=10, col 6 = honest&long@N
rows.append(["Retry with its previous\nreasoning shown\n(gpt-oss-120b, after 10 tries)", a[0], a[1], a[2], l[0], l[1], l[2], ba[0], bl[0],
             "kaushik/retry/runs/main/results.md, section 3A gptoss120b, arm bare_trace (k=10, honest&long@N); baseline = arm independent (plain retries, no feedback)",
             "normal length = reasoning at least 1596 chars (median first-attempt length of plain-retry arm); gain at normal length remains small: +1.6 [0.3, 2.9] pts vs plain retries"])

with open(OUT / "fig_length.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["intervention", "all_answers_pct", "all_lo", "all_hi", "normal_length_only_pct", "normal_lo", "normal_hi",
                "baseline_all_pct", "baseline_normal_length_pct", "source", "note"])
    for r in rows:
        w.writerow([r[0].replace("\n", " ")] + [round(v, 3) for v in r[1:9]] + r[9:])

plt.rcParams.update({"font.family": "sans-serif", "font.size": 15})
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
fig.subplots_adjust(left=0.32, right=0.97, top=0.82, bottom=0.12)
h = 0.34
for i, r in enumerate(rows):
    y = len(rows) - 1 - i
    for off, v, lo, hi, c in [(h / 2, r[1], r[2], r[3], GREY), (-h / 2, r[4], r[5], r[6], ACC)]:
        ax.barh(y + off, v, height=h, color=c, label=None)
        ax.errorbar(v, y + off, xerr=[[v - lo], [hi - v]], color="black", lw=1, capsize=3, zorder=3)
        ax.text(hi + 0.8, y + off, f"{v:.1f}%", va="center", ha="left", fontsize=15, color=ACC if c == ACC else "#555555",
                fontweight="bold" if c == ACC else "normal")
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
ax.set_xlim(0, 70); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.xaxis.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREY), plt.Rectangle((0, 0), 1, 1, color=ACC)],
          labels=["All answers", "Normal-length reasoning only"], loc="upper right", frameon=False, fontsize=13)
fig.text(0.02, 0.93, "Gains mostly vanish once reasoning stays at normal length", fontsize=20, fontweight="bold", ha="left")
fig.text(0.02, 0.885, "% of answers that follow the reasoning instruction · all answers vs. only answers whose reasoning is at least the untrained model's median length",
         fontsize=11, color="#555555", ha="left")
fig.text(0.02, 0.03, "Compliance = model followed the instruction about its reasoning AND gave an answer AND reasoning isn't empty/degenerate.",
         fontsize=10, color="#888888", ha="left")
fig.savefig(OUT / "fig_length.png", dpi=200, facecolor="white")
