"""P(honest | reasoning-length bin) for every gpt-oss-20b intervention with rollouts on disk (coordinator request,
2026-10-09 round 2, part 1). Does any intervention LIFT the curve (higher honest rate at matched length, esp. >= 2.6k
chars), or do they all just move mass along it?

    .venv/bin/python kaushik/run.py elicit/len_teacher curves [--boot 500]

Reads other workstreams' eval JSONs read-only (9-training-mode block only), pools seeds, and writes
results/curves.{json,md,png}. Per arm: honest rate per length bin with 95% bootstrap CIs over questions, fraction
of traces per bin, honest among traces >= 2.6k chars, and a length-standardized honest rate
(sum_b w_b^untrained * p_b^arm: the arm's curve weighted by the UNTRAINED length distribution of the same split,
i.e. "what honest compliance would be if the arm reasoned at untrained length"). Family breakdown: case modes
(alternating_case, lowercase_thinking, uppercase_thinking) vs the other six. 40-question sweep runs are compared
against the references restricted to the same questions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from lt_common import RESULTS, ROOT, RUNS

sys.path.insert(0, str(ROOT))
from cotcontrol.eval.grading import is_honest  # noqa: E402

BINS = [0, 1000, 2000, 2600, 4000, 8000, float("inf")]
BIN_LABELS = ["<1k", "1-2k", "2-2.6k", "2.6-4k", "4-8k", ">8k"]
LONG = 2600
CASE_MODES = {"alternating_case", "lowercase_thinking", "uppercase_thinking"}

E = ROOT / "kaushik/elicit"
GD, GL, DOSE, NS, FRAG = E / "gepa_distill/runs", E / "gepa_lora/runs", E / "gepa_dose/runs", E / "nocot_sft/runs", ROOT / "kaushik/fragility/runs"


def nocot(arm: str, ckpt: int, split: str, seeds=(0, 1, 2), suffix: str = "") -> list[Path]:
    blk = "val_modes" if split == "val" else "final_modes"
    return [NS / f"nocot-{arm}-gptoss20b-seed{s}{suffix}/eval/{blk}/checkpoint-{ckpt}.json" for s in seeds]


# (arm name, group, split, [modes.json ...], subset-reference run or None)
ARMS = [
    # --- val, full 200 q ---
    ("untrained", "reference", "val", [GD / "untrained_val/modes.json"], None),
    ("GEPA prompt (teacher)", "prompt", "val", [GD / "teacher_val/modes.json"], None),
    ("GEPA LoRA x1 (lr1e-4)", "lora", "val", [GL / "lr1e-4_c400_val/modes.json"], None),
    ("GEPA LoRA lr3e-4", "lora", "val", [GL / "lr3e-4_c400_val/modes.json"], None),
    ("GEPA LoRA + header anchor", "lora", "val", [GL / "anchor_c400_val/modes.json"], None),
    ("GEPA LoRA x1.5 (C1)", "dose", "val", [DOSE / "x1.5_val/modes.json"], None),
    ("steer last_L12_a1 (B)", "steer", "val", [GD / "st_last_L12_a1_val/modes.json"], None),
    ("steer last_L15_a2 (B)", "steer", "val", [GD / "st_last_L15_a2_val/modes.json"], None),
    ("1A answer-only SFT (ref1a@50)", "sft", "val", [NS / "ref1a-mcq_noreason-gptoss20b-seed0/eval/val_modes/checkpoint-50.json"], None),
    ("noCoT gentle@50", "sft", "val", nocot("gentle", 50, "val"), None),
    ("noCoT gentle@100", "sft", "val", nocot("gentle", 100, "val"), None),
    ("noCoT mix@50", "sft", "val", nocot("mix", 50, "val"), None),
    ("noCoT mix@100", "sft", "val", nocot("mix", 100, "val"), None),
    ("noCoT fmt@25 (A selected)", "sft", "val", nocot("fmt", 25, "val"), None),
    ("noCoT fmt@50", "sft", "val", nocot("fmt", 50, "val"), None),
    ("noCoT fmt_mix@50", "sft", "val", nocot("fmt_mix", 50, "val"), None),
    ("noCoT fmt_mix@100", "sft", "val", nocot("fmt_mix", 100, "val"), None),
    ("noCoT fmt_long@75", "sft", "val", nocot("fmt_long", 75, "val"), None),
    ("noCoT fmt_long@150", "sft", "val", nocot("fmt_long", 150, "val"), None),
    ("C2 steps_append prompt", "prompt", "val", [RUNS / "steps_append_val/modes.json"], None),
    ("C2 len_append prompt", "prompt", "val", [RUNS / "len_append_val/modes.json"], None),
    ("C2 rewrite prompt", "prompt", "val", [RUNS / "rewrite_val/modes.json"], None),
    ("C2 GEPA @ effort high", "prompt", "val", [RUNS / "high_val/modes.json"], None),
    ("C2 len_append @ high", "prompt", "val", [RUNS / "len_append_high_val/modes.json"], None),
    ("C2 recheck_len prompt", "prompt", "val", [RUNS / "recheck_len_val/modes.json"], None),
    ("C2 sustain prompt", "prompt", "val", [RUNS / "sustain_val/modes.json"], None),
    # --- val, 40-q sweeps (C1 doses; references restricted to the same questions) ---
    *[(f"GEPA LoRA x{d} (40q)", "dose40", "val", [DOSE / f"x{d}_sw/modes.json"], "x1.5_sw")
      for d in ["0.5", "0.75", "1.05", "1.1", "1.15", "1.2", "1.3", "1.5", "2", "3"]],
    *[(f"GEPA LoRA x{d} + prompt (40q)", "dose40", "val", [DOSE / f"x{d}_prompt_sw/modes.json"], "x1.5_sw")
      for d in ["0.75", "1", "1.05", "1.1", "1.15", "1.2", "1.3", "1.5", "2"]],
    ("GEPA LoRA x1 @ high (40q)", "dose40", "val", [DOSE / "x1_high_sw/modes.json"], "x1.5_sw"),
    ("GEPA prompt @ high (40q)", "dose40", "val", [DOSE / "prompt_high_sw/modes.json"], "x1.5_sw"),
    ("untrained @ high (40q)", "dose40", "val", [DOSE / "untrained_high_sw/modes.json"], "x1.5_sw"),
    ("untrained (40q)", "dose40", "val", [GD / "untrained_val/modes.json"], "x1.5_sw"),
    ("GEPA prompt (40q)", "dose40", "val", [GD / "teacher_val/modes.json"], "x1.5_sw"),
    ("GEPA LoRA x1 (40q)", "dose40", "val", [GL / "lr1e-4_c400_val/modes.json"], "x1.5_sw"),
    # --- C2 round-2 prompts (40 q, same subset) ---
    ("C2 recheck prompt (40q)", "prompt40", "val", [RUNS / "recheck_sw/modes.json"], "x1.5_sw"),
    ("C2 recheck_len prompt (40q)", "prompt40", "val", [RUNS / "recheck_len_sw/modes.json"], "x1.5_sw"),
    ("C2 sustain prompt (40q)", "prompt40", "val", [RUNS / "sustain_sw/modes.json"], "x1.5_sw"),
    # --- test, 500 q ---
    ("untrained", "reference", "test", [FRAG / "fragility-mcq_noreason-gptoss20b-seed0/eval/final_modes/checkpoint-0.json"], None),
    ("GEPA prompt (teacher)", "prompt", "test", [GD / "teacher_test/modes.json"], None),
    ("GEPA LoRA x1 (3 seeds)", "lora", "test", [GL / f"lr1e-4_c400_test_s{k}/modes.json" for k in range(3)], None),
    ("KL-matched random LoRA (3)", "control", "test", [GL / f"noise_lr1e-4_c400_test_n{k}/modes.json" for k in range(3)], None),
    ("GEPA LoRA x1.5 (3 seeds, C1)", "dose", "test", [DOSE / f"x1.5_test_s{k}/modes.json" for k in range(3)], None),
    ("steer last_L12_a1 (B)", "steer", "test", [GD / "st_last_L12_a1_test/modes.json"], None),
    ("steer last_L15_a2 (B)", "steer", "test", [GD / "st_last_L15_a2_test/modes.json"], None),
    ("steer random_L15_a2 (control)", "control", "test", [GD / "st_random_L15_a2_test/modes.json"], None),
    ("steer resid_L12_a2 (GEPA - brevity)", "steer", "test", [GD / "st_resid_L12_a2_test/modes.json"], None),
    ("1A answer-only SFT (3 seeds)", "sft", "test", [FRAG / f"fragility-mcq_noreason-gptoss20b-seed{k}/eval/final_modes/checkpoint-50.json" for k in range(3)], None),
    ("noCoT fmt@25 (A selected)", "sft", "test", nocot("fmt", 25, "test", suffix="-step25"), None),
    ("noCoT fmt@50", "sft", "test", nocot("fmt", 50, "test"), None),
    ("noCoT gentle@100", "sft", "test", nocot("gentle", 100, "test"), None),
    ("noCoT mix@100", "sft", "test", nocot("mix", 100, "test"), None),
    ("noCoT fmt_mix@100", "sft", "test", nocot("fmt_mix", 100, "test"), None),
    ("noCoT fmt_long@150", "sft", "test", nocot("fmt_long", 150, "test"), None),
]

FIG_VAL = ["untrained", "GEPA prompt (teacher)", "GEPA LoRA x1 (lr1e-4)", "GEPA LoRA x1.5 (C1)", "steer last_L15_a2 (B)",
           "1A answer-only SFT (ref1a@50)", "noCoT fmt@25 (A selected)", "C2 recheck_len prompt"]
FIG_TEST = ["untrained", "GEPA prompt (teacher)", "GEPA LoRA x1 (3 seeds)", "GEPA LoRA x1.5 (3 seeds, C1)",
            "steer last_L15_a2 (B)", "1A answer-only SFT (3 seeds)", "noCoT fmt@25 (A selected)", "KL-matched random LoRA (3)"]
PALETTE = ["#6b6b66", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]


def rollouts(path: Path) -> list[dict]:
    d = json.load(open(path))
    rows = []
    for rec in d["results"]:
        for s in rec["samples"]:
            if s.get("error"):
                continue
            rows.append({"qid": f"{rec['dataset']}:{rec['id']}", "mode": rec["mode"],
                         "honest": float(is_honest(s)), "chars": len(s["reasoning"] or "")})
    return rows


def qids_of(path: Path) -> set[str]:
    return {r["qid"] for r in rollouts(path)}


def bin_of(chars: int) -> int:
    return int(np.searchsorted(BINS, chars, side="right") - 1)


def curve(rows: list[dict], n_boot: int, weights: np.ndarray | None, seed: int = 0) -> dict:
    """Per-bin honest rate with bootstrap CIs over questions; plus frac per bin, honest at >= LONG, standardized rate."""
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append((bin_of(r["chars"]), r["honest"]))
    qs = list(by_q)
    nb = len(BINS) - 1
    # per question: sums and counts per bin
    S = np.zeros((len(qs), nb)); C = np.zeros((len(qs), nb))
    for i, q in enumerate(qs):
        for b, h in by_q[q]:
            S[i, b] += h; C[i, b] += 1
    long_mask = np.arange(nb) >= BIN_LABELS.index("2.6-4k")

    def stats(S, C):
        with np.errstate(invalid="ignore", divide="ignore"):
            p = S.sum(0) / C.sum(0)
            long_p = S[:, long_mask].sum() / C[:, long_mask].sum()
            overall = S.sum() / C.sum()
            std = float(np.nansum(np.where(np.isnan(p), 0, p) * weights)) if weights is not None else np.nan
        return p, long_p, overall, std

    p, long_p, overall, std = stats(S, C)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qs), size=(n_boot, len(qs)))
    draws = np.array([stats(S[i], C[i])[0] for i in idx])
    draws_long = np.array([stats(S[i], C[i])[1] for i in idx])
    draws_std = np.array([stats(S[i], C[i])[3] for i in idx])
    draws_all = np.array([stats(S[i], C[i])[2] for i in idx])
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
    n_bin = C.sum(0)
    return {"n": int(C.sum()), "n_q": len(qs), "overall": float(overall), "overall_ci": ci(draws_all),
            "bins": [{"label": l, "n": int(n_bin[b]), "frac": float(n_bin[b] / C.sum()),
                      "honest": (None if n_bin[b] == 0 else float(p[b])), "ci": ci(draws[:, b])}
                     for b, l in enumerate(BIN_LABELS)],
            "median_chars": float(np.median([r["chars"] for r in rows])),
            "frac_long": float(C[:, long_mask].sum() / C.sum()), "honest_long": float(long_p), "honest_long_ci": ci(draws_long),
            "standardized": float(std), "standardized_ci": ci(draws_std)}


def fmt_pct(x, ci=None, n=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    s = f"{100 * x:.1f}"
    if ci is not None:
        s += f" [{100 * ci[0]:.1f},{100 * ci[1]:.1f}]"
    if n is not None:
        s += f" (n={n})"
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args(argv)
    subset_qids = {}
    for ref in {a[4] for a in ARMS if a[4]}:
        p = DOSE / ref / "modes.json"
        if p.exists():
            subset_qids[ref] = qids_of(p)

    # untrained length distributions per (split, subset) -> weights for standardization
    loaded = {}
    results = []
    for name, group, split, paths, subset in ARMS:
        paths = [p for p in paths if p.exists()]
        if not paths or (subset and subset not in subset_qids):
            print(f"[skip] {name} ({split}): no files yet", flush=True)
            continue
        rows = [r for p in paths for r in rollouts(p)]
        if subset:
            rows = [r for r in rows if r["qid"] in subset_qids[subset]]
        loaded[(name, split, subset)] = rows
    weights = {}
    for key, rows in loaded.items():
        name, split, subset = key
        if name.startswith("untrained") and "high" not in name:
            cnt = np.bincount([bin_of(r["chars"]) for r in rows], minlength=len(BINS) - 1)
            weights[(split, subset)] = cnt / cnt.sum()
    for (name, split, subset), rows in loaded.items():
        w = weights.get((split, subset))
        fam = {}
        for famname, pred in [("case", lambda r: r["mode"] in CASE_MODES), ("other", lambda r: r["mode"] not in CASE_MODES)]:
            fr = [r for r in rows if pred(r)]
            fam[famname] = curve(fr, args.boot, None) if fr else None
        group = next(a[1] for a in ARMS if a[0] == name and a[2] == split)
        results.append({"name": name, "group": group, "split": split, "subset": subset, "n_files": None,
                        **curve(rows, args.boot, w), "family": fam})
        print(f"[done] {name} ({split}{', 40q' if subset else ''})", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "curves.json").write_text(json.dumps({"bins": BINS, "results": results}, indent=1))
    write_md(results)
    figure(results)
    print(f"-> {RESULTS / 'curves.md'}, curves.json, curves.png")


def write_md(results: list[dict]) -> None:
    L = ["# P(honest | reasoning length) for every gpt-oss-20b intervention (9 training modes; 95% bootstrap CIs over questions)",
         "",
         "honest rate (%) within each reasoning-length bin (chars). `>=2.6k` = honest among traces >= 2600 chars (~ the untrained",
         "median, the honest_long endpoint). `std.` = length-standardized honest rate: the arm's per-bin rates weighted by the",
         "UNTRAINED length distribution of the same split/subset (what the arm would score if it reasoned at untrained length).",
         "`frac >=2.6k` = share of the arm's traces that are long. Seeds pooled. Sections: full val (200 q), 40-q val sweeps",
         "(C1 doses + C2 round-2 prompts; references restricted to the same 40 questions), test (500 q).",
         "",
         "Note: `steer random_L15_a2 (control)` (B's random-direction steering control) is a broken model: 59% of its test",
         "rollouts have EMPTY reasoning (median 0 chars) and the rest are degenerate text, so its 'honest' long traces are",
         "gibberish passing the word-suppression rule checks; ignore its row. The KL-matched random LoRA is the sound control.", ""]
    hdr = "| arm | n | overall | " + " | ".join(BIN_LABELS) + " | >=2.6k [CI] | frac >=2.6k | median chars | std. [CI] |"
    sep = "|" + "---|" * (len(BIN_LABELS) + 7)
    for title, pred in [("Val, full 200 q", lambda r: r["split"] == "val" and not r["subset"]),
                        ("Val, 40-q sweeps", lambda r: r["split"] == "val" and r["subset"]),
                        ("Test, 500 q", lambda r: r["split"] == "test")]:
        L += [f"## {title}", "", hdr, sep]
        for r in [x for x in results if pred(x)]:
            cells = [f"{100 * b['honest']:.1f} ({b['n']})" if b["honest"] is not None else "-" for b in r["bins"]]
            L.append(f"| {r['name']} | {r['n']} | {fmt_pct(r['overall'])} | " + " | ".join(cells) +
                     f" | {fmt_pct(r['honest_long'], r['honest_long_ci'])} | {r['frac_long']:.2f} | {r['median_chars']:.0f} | "
                     f"{fmt_pct(r['standardized'], r['standardized_ci'])} |")
        L.append("")
    L += ["## Mode families: case modes (alternating_case, lowercase_thinking, uppercase_thinking) vs the other six", "",
          "honest % within bin; `>=2.6k` with CI.", "",
          "| arm | split | family | n | overall | " + " | ".join(BIN_LABELS) + " | >=2.6k [CI] | frac >=2.6k |",
          "|" + "---|" * (len(BIN_LABELS) + 7)]
    for r in results:
        for famname in ("case", "other"):
            f = r["family"].get(famname)
            if not f:
                continue
            cells = [f"{100 * b['honest']:.1f}" if b["honest"] is not None else "-" for b in f["bins"]]
            L.append(f"| {r['name']} | {r['split']}{' 40q' if r['subset'] else ''} | {famname} | {f['n']} | {fmt_pct(f['overall'])} | "
                     + " | ".join(cells) + f" | {fmt_pct(f['honest_long'], f['honest_long_ci'])} | {f['frac_long']:.2f} |")
    (RESULTS / "curves.md").write_text("\n".join(L) + "\n")


def figure(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {(r["name"], r["split"], r["subset"]): r for r in results}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey="row")
    x = np.arange(len(BIN_LABELS))

    def panel(ax, names, split, key=None, title=""):
        for i, name in enumerate(names):
            r = by.get((name, split, None))
            if r is None:
                continue
            c = r if key is None else r["family"][key]
            y = np.array([np.nan if b["honest"] is None else 100 * b["honest"] for b in c["bins"]])
            lo = np.array([100 * b["ci"][0] for b in c["bins"]]); hi = np.array([100 * b["ci"][1] for b in c["bins"]])
            col = PALETTE[i % len(PALETTE)]
            ax.plot(x, y, "-o", color=col, lw=2, ms=5, zorder=3,
                    label=f"{name}  (>=2.6k: {100 * c['honest_long']:.1f}%, {100 * c['frac_long']:.0f}% of traces)")
            ax.fill_between(x, lo, hi, color=col, alpha=0.10, lw=0)
        ax.set_xticks(x, BIN_LABELS)
        ax.axvspan(2.5, len(BIN_LABELS) - 0.5, color="#000", alpha=0.04, lw=0)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="y", color="#ddd", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    panel(axes[0, 0], FIG_VAL, "val", title="Val (200 q x 9 modes): all modes")
    panel(axes[0, 1], FIG_TEST, "test", title="Test (500 q x 9 modes): all modes")
    panel(axes[1, 0], FIG_TEST, "test", "case", "Test: case modes (alternating / lower / upper)")
    panel(axes[1, 1], FIG_TEST, "test", "other", "Test: the other six modes")
    for ax in axes[:, 0]:
        ax.set_ylabel("honest compliance (%) within length bin")
    for ax in axes[1]:
        ax.set_xlabel("reasoning length (chars); shaded = at/above the untrained median")
    for ax in axes.flat:
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("gpt-oss-20b: honest compliance vs reasoning length, per intervention (bands = 95% bootstrap CI over questions)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "curves.png", dpi=150)


if __name__ == "__main__":
    main()
