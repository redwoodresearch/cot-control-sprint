"""P(honest | reasoning-length bin) for every gpt-oss-20b intervention with rollouts on disk (coordinator request,
2026-10-09 round 2, part 1). Does any intervention LIFT the curve (higher honest rate at matched length, esp. >= 2.6k
chars), or do they all just move mass along it?

    run.py elicit/len_teacher curves [--boot 500] [--out-dir results]

Reads other workstreams' eval JSONs read-only (9-training-mode block only; the arm table is config `curves.arms`),
pools seeds, and writes results/curves.{json,md,png}. Per arm: honest rate per length bin with 95% bootstrap CIs
over questions, fraction of traces per bin, honest among traces >= 2.6k chars, and a length-standardized honest
rate (sum_b w_b^untrained * p_b^arm: the arm's curve weighted by the UNTRAINED length distribution of the same
split, i.e. "what honest compliance would be if the arm reasoned at untrained length"). Family breakdown: case
modes vs the other six. 40-question sweep runs are compared against the references restricted to the same
questions (`subset`).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from kaushik.core.io import ROOT
from kaushik.elicit.common import results_dir


def arm_paths(arm: dict, roots: dict[str, Path]) -> list[Path]:
    """{paths: ["GD/x/modes.json", ...]} or {nocot: {arm, ckpt, split, seeds?, suffix?}} -> absolute paths."""
    if "nocot" in arm:
        n = arm["nocot"]
        blk = "val_modes" if n["split"] == "val" else "final_modes"
        return [roots["NS"] / f"nocot-{n['arm']}-gptoss20b-seed{s}{n.get('suffix', '')}/eval/{blk}/checkpoint-{n['ckpt']}.json"
                for s in n.get("seeds", [0, 1, 2])]
    out = []
    for p in arm["paths"]:
        root, _, rest = p.partition("/")
        out.append(roots[root] / rest)
    return out


class Curves:
    def __init__(self, cfg: dict):
        from cotcontrol.eval.grading import is_honest

        self.is_honest = is_honest
        c = cfg["curves"]
        self.c = c
        self.bins = list(c["bins"])            # ints + .inf, as curves.json records them
        self.labels = list(c["bin_labels"])
        self.case_modes = set(c["case_modes"])
        self.roots = {k: ROOT / v for k, v in c["roots"].items()}
        self.long_mask = np.arange(len(self.labels)) >= self.labels.index(c["long_bin"])

    def rollouts(self, path: Path) -> list[dict]:
        with open(path) as f:
            d = json.load(f)
        rows = []
        for rec in d["results"]:
            for s in rec["samples"]:
                if s.get("error"):
                    continue
                rows.append({"qid": f"{rec['dataset']}:{rec['id']}", "mode": rec["mode"],
                             "honest": float(self.is_honest(s)), "chars": len(s["reasoning"] or "")})
        return rows

    def bin_of(self, chars: int) -> int:
        return int(np.searchsorted(self.bins, chars, side="right") - 1)

    def curve(self, rows: list[dict], n_boot: int, weights: np.ndarray | None, seed: int = 0) -> dict:
        """Per-bin honest rate with bootstrap CIs over questions; plus frac per bin, honest at >= long, standardized rate."""
        by_q = defaultdict(list)
        for r in rows:
            by_q[r["qid"]].append((self.bin_of(r["chars"]), r["honest"]))
        qs = list(by_q)
        nb = len(self.bins) - 1
        S = np.zeros((len(qs), nb)); C = np.zeros((len(qs), nb))
        for i, q in enumerate(qs):
            for b, h in by_q[q]:
                S[i, b] += h; C[i, b] += 1
        long_mask = self.long_mask

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
        ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
        n_bin = C.sum(0)
        return {"n": int(C.sum()), "n_q": len(qs), "overall": float(overall), "overall_ci": ci(draws_all),
                "bins": [{"label": lab, "n": int(n_bin[b]), "frac": float(n_bin[b] / C.sum()),
                          "honest": (None if n_bin[b] == 0 else float(p[b])), "ci": ci(draws[:, b])}
                         for b, lab in enumerate(self.labels)],
                "median_chars": float(np.median([r["chars"] for r in rows])),
                "frac_long": float(C[:, long_mask].sum() / C.sum()), "honest_long": float(long_p), "honest_long_ci": ci(draws_long),
                "standardized": float(std), "standardized_ci": ci(draws_std)}

    def compute(self, n_boot: int) -> list[dict]:
        arms = self.c["arms"]
        subset_root = self.roots[self.c["subset_root"]]
        subset_qids = {}
        for ref in {a.get("subset") for a in arms if a.get("subset")}:
            p = subset_root / ref / "modes.json"
            if p.exists():
                subset_qids[ref] = {r["qid"] for r in self.rollouts(p)}
        loaded = {}
        for a in arms:
            name, split, subset = a["name"], a["split"], a.get("subset")
            paths = [p for p in arm_paths(a, self.roots) if p.exists()]
            if not paths or (subset and subset not in subset_qids):
                print(f"[skip] {name} ({split}): no files yet", flush=True)
                continue
            rows = [r for p in paths for r in self.rollouts(p)]
            if subset:
                rows = [r for r in rows if r["qid"] in subset_qids[subset]]
            loaded[(name, split, subset)] = rows
        weights = {}
        for (name, split, subset), rows in loaded.items():
            if name.startswith("untrained") and "high" not in name:
                cnt = np.bincount([self.bin_of(r["chars"]) for r in rows], minlength=len(self.bins) - 1)
                weights[(split, subset)] = cnt / cnt.sum()
        results = []
        for (name, split, subset), rows in loaded.items():
            w = weights.get((split, subset))
            fam = {}
            for famname, pred in [("case", lambda r: r["mode"] in self.case_modes), ("other", lambda r: r["mode"] not in self.case_modes)]:
                fr = [r for r in rows if pred(r)]
                fam[famname] = self.curve(fr, n_boot, None) if fr else None
            group = next(a["group"] for a in arms if a["name"] == name and a["split"] == split)
            results.append({"name": name, "group": group, "split": split, "subset": subset, "n_files": None,
                            **self.curve(rows, n_boot, w), "family": fam})
            print(f"[done] {name} ({split}{', 40q' if subset else ''})", flush=True)
        return results


def fmt_pct(x, ci=None, n=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    s = f"{100 * x:.1f}"
    if ci is not None:
        s += f" [{100 * ci[0]:.1f},{100 * ci[1]:.1f}]"
    if n is not None:
        s += f" (n={n})"
    return s


def write_md(results: list[dict], labels: list[str], out: Path) -> None:
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
    hdr = "| arm | n | overall | " + " | ".join(labels) + " | >=2.6k [CI] | frac >=2.6k | median chars | std. [CI] |"
    sep = "|" + "---|" * (len(labels) + 7)
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
          "| arm | split | family | n | overall | " + " | ".join(labels) + " | >=2.6k [CI] | frac >=2.6k |",
          "|" + "---|" * (len(labels) + 7)]
    for r in results:
        for famname in ("case", "other"):
            f = r["family"].get(famname)
            if not f:
                continue
            cells = [f"{100 * b['honest']:.1f}" if b["honest"] is not None else "-" for b in f["bins"]]
            L.append(f"| {r['name']} | {r['split']}{' 40q' if r['subset'] else ''} | {famname} | {f['n']} | {fmt_pct(f['overall'])} | "
                     + " | ".join(cells) + f" | {fmt_pct(f['honest_long'], f['honest_long_ci'])} | {f['frac_long']:.2f} |")
    (out / "curves.md").write_text("\n".join(L) + "\n")


def figure(results: list[dict], c: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, palette = list(c["bin_labels"]), list(c["palette"])
    by = {(r["name"], r["split"], r["subset"]): r for r in results}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey="row")
    x = np.arange(len(labels))

    def panel(ax, names, split, key=None, title=""):
        for i, name in enumerate(names):
            r = by.get((name, split, None))
            if r is None:
                continue
            cc = r if key is None else r["family"][key]
            y = np.array([np.nan if b["honest"] is None else 100 * b["honest"] for b in cc["bins"]])
            lo = np.array([100 * b["ci"][0] for b in cc["bins"]]); hi = np.array([100 * b["ci"][1] for b in cc["bins"]])
            col = palette[i % len(palette)]
            ax.plot(x, y, "-o", color=col, lw=2, ms=5, zorder=3,
                    label=f"{name}  (>=2.6k: {100 * cc['honest_long']:.1f}%, {100 * cc['frac_long']:.0f}% of traces)")
            ax.fill_between(x, lo, hi, color=col, alpha=0.10, lw=0)
        ax.set_xticks(x, labels)
        ax.axvspan(2.5, len(labels) - 0.5, color="#000", alpha=0.04, lw=0)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="y", color="#ddd", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    panel(axes[0, 0], c["fig_val"], "val", title="Val (200 q x 9 modes): all modes")
    panel(axes[0, 1], c["fig_test"], "test", title="Test (500 q x 9 modes): all modes")
    panel(axes[1, 0], c["fig_test"], "test", "case", "Test: case modes (alternating / lower / upper)")
    panel(axes[1, 1], c["fig_test"], "test", "other", "Test: the other six modes")
    for ax in axes[:, 0]:
        ax.set_ylabel("honest compliance (%) within length bin")
    for ax in axes[1]:
        ax.set_xlabel("reasoning length (chars); shaded = at/above the untrained median")
    for ax in axes.flat:
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("gpt-oss-20b: honest compliance vs reasoning length, per intervention (bands = 95% bootstrap CI over questions)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)


def main(cfg: dict, argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="curves", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=int(cfg["curves"]["boot"]))
    ap.add_argument("--out-dir", default=None, help="default results/")
    args = ap.parse_args(argv)
    cv = Curves(cfg)
    results = cv.compute(args.boot)
    out = Path(args.out_dir) if args.out_dir else results_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "curves.json").write_text(json.dumps({"bins": cv.bins, "results": results}, indent=1))
    write_md(results, cv.labels, out)
    figure(results, cfg["curves"], out)
    print(f"-> {out / 'curves.md'}, curves.json, curves.png")
