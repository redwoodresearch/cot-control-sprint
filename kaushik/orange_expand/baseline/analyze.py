"""Stage-0 tables from the saved rollouts (the eval / pick jobs write them; this only reads).

    .venv/bin/python kaushik/run.py orange_expand/baseline analyze [--boot 2000]

results/pick_rates.{md,csv}   orange-pick rate per model x training prompt x reasoning (+ unparseable share)
results/untrained.{md,csv}    honest compliance of the untrained models per condition x mode (95% bootstrap
                              over questions), plus raw compliance, accuracy, reasoning chars, degenerate rate
results/mechanism.{md,csv}    Qwen3.6-27B 2A adapters (seeds 0-2) + untrained under persona (2A, val),
                              nopersona (new) and direct (2A, val); bootstrap over questions x seeds;
                              orange - orange_nopersona differences (paired questions, independent seeds)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import stats as st
from kaushik.core.config import resolve, results_dir, runs_root
from kaushik.core.results import parse_eval_json
from kaushik.core.stats import POOLED

EXP = "orange_expand/baseline"


def pct(x) -> str:
    return st.pct(x, missing="–")


def write_pick_tables(cfg: dict) -> str | None:
    runs, results = runs_root(cfg), results_dir(cfg)
    rows, md_rows = [], []
    for mk in cfg["analysis"]["stage0_models"]:
        for prompt in cfg["pick"]["prompts"]:
            cells = []
            for on in (True, False):
                path = runs / mk / "pick" / f"{prompt}_{'on' if on else 'off'}.json"
                if not path.exists():
                    cells.append(None)
                    continue
                s = json.loads(path.read_text())["summary"]
                cells.append(s)
                rows.append([mk, prompt, "on" if on else "off", s["n"], s["n_parsed"], s["orange_rate"],
                             s["unparseable_rate"], s["truncated_rate"], round(s["mean_reasoning_chars"] or 0)])

            def fmt(s):
                return "pending" if s is None else f"{pct(s['orange_rate'])} (unp {pct(s['unparseable_rate'])})"
            md_rows.append([mk, prompt, fmt(cells[0]), fmt(cells[1])])
    if not rows:
        return None
    results.mkdir(parents=True, exist_ok=True)
    st.write_csv(results / "pick_rates.csv",
                 ["model", "train_prompt", "reasoning", "n", "n_parsed", "orange_rate", "unparseable_rate",
                  "truncated_rate", "mean_reasoning_chars"], rows)
    md = ("# Orange-pick rate, untrained models (stage 0)\n\n"
          "200 held-out orange MCQs (same 200 as the SFT agent's `heldout_200.json`), training prompt as system "
          "prompt, user turn = orange-training `format_user`, T=1. Cell = % of parseable answers equal to the "
          "orange letter (unp = unparseable share of all 200). Reasoning on: gpt-oss medium / Qwen thinking, "
          "8192 tokens; off: gpt-oss low + empty analysis prefill / Qwen enable_thinking=false (2A's training "
          "rendering). Chance = 25%.\n\n"
          + st.md_table(["model", "training prompt", "reasoning on", "reasoning off"], md_rows) + "\n")
    (results / "pick_rates.md").write_text(md)
    return md


# ----------------------------------------------------------------------------- eval rollouts


def write_untrained(cfg: dict, rng: np.random.Generator, B: int) -> str | None:
    runs, results = runs_root(cfg), results_dir(cfg)
    conditions, modes = cfg["eval"]["grid"]["conditions"], cfg["eval"]["grid"]["modes"]
    degen_chars = int(cfg["analysis"]["degen_chars"])
    rows, md_rows = [], []
    for mk in cfg["analysis"]["stage0_models"]:
        for cond in conditions:
            for mode in [*modes, POOLED]:
                use = modes if mode == POOLED else [mode]
                recs = []
                for m in use:
                    path = runs / mk / "eval" / f"{cond}_{m}" / "base.json"
                    if path.exists():
                        recs += parse_eval_json(path)
                if len(recs) < 200 * len(use):   # incomplete
                    if mode != POOLED:
                        md_rows.append([mk, cond, mode, "pending", "", "", "", "", ""])
                    continue
                h = np.array([r["honest"] for r in recs], float)
                mean, lo, hi = st.boot_1d(h, rng, B)
                comp = np.mean([r["compliance"] for r in recs])
                acc = np.mean([r["correct"] for r in recs])
                chars = np.mean([r["chars"] for r in recs])
                degen = np.mean([r["chars"] < degen_chars for r in recs])
                trunc = np.mean([r["truncated"] for r in recs])
                rows.append([mk, cond, mode, len(recs), mean, lo, hi, comp, acc, chars, degen, trunc])
                md_rows.append([mk, cond, mode, f"{pct(mean)} [{pct(lo)}, {pct(hi)}]", pct(comp), pct(acc),
                                f"{chars:.0f}", pct(degen), pct(trunc)])
    if not rows:
        return None
    st.write_csv(results / "untrained.csv",
                 ["model", "condition", "mode", "n", "honest", "honest_lo", "honest_hi", "compliance", "accuracy",
                  "chars", "degen", "truncated"], rows)
    md = ("# Untrained models, CoT-control eval (stage 0)\n\n"
          "val split (200 q), T=1, 16k tokens, reasoning on (gpt-oss medium / Qwen thinking). honest = honest "
          "compliance (`is_honest`) with 95% bootstrap over questions; raw = strict compliance; acc = accuracy; "
          "chars = mean reasoning length; degen = traces < 200 chars; trunc = hit max_tokens.\n\n"
          + st.md_table(["model", "condition", "mode", "honest", "raw", "acc", "chars", "degen", "trunc"], md_rows)
          + "\n")
    (results / "untrained.md").write_text(md)
    return md


# ----------------------------------------------------------------------------- mechanism (27B)


def mech_sources(cfg: dict) -> dict[tuple[str, str, str], list[Path]]:
    """(arm, condition, mode) -> per-seed rollout files. persona/direct = 2A val step-50 files for the SAME
    adapters; untrained persona/direct = 2A checkpoint-0 of the orange runs (zero-init adapter = base
    model), untrained nopersona = this folder's base run."""
    an = cfg["analysis"]
    orange = ROOT / an["orange_2a_runs"]
    mech_model, arms, seeds = an["mech_model"], an["mech_arms"], an["mech_seeds"]
    mech = runs_root(cfg) / f"mech_{mech_model}"
    src: dict[tuple[str, str, str], list[Path]] = {}
    for mode in cfg["eval"]["grid"]["modes"]:
        for arm in arms:
            for cond in ("persona", "direct"):
                src[(arm, cond, mode)] = [orange / f"{arm}-{mech_model}-seed{s}" / "eval" / f"{cond}_{mode}"
                                          / "checkpoint-50.json" for s in seeds]
            src[(arm, "nopersona", mode)] = [mech / f"{arm}-seed{s}" / "eval" / f"nopersona_{mode}"
                                             / "checkpoint-50.json" for s in seeds]
        for cond in ("persona", "direct"):
            src[("untrained", cond, mode)] = [orange / f"{arms[0]}-{mech_model}-seed{s}" / "eval"
                                              / f"{cond}_{mode}" / "checkpoint-0.json" for s in seeds]
        src[("untrained", "nopersona", mode)] = [mech / "untrained" / "eval" / f"nopersona_{mode}" / "base.json"]
    return src


def matrix(paths: list[Path], key: str = "honest") -> np.ndarray | None:
    """seeds x questions matrix (questions aligned by qid) or None if any file is missing."""
    if not all(p.exists() for p in paths):
        return None
    per_seed = [{r["qid"]: float(r[key]) for r in parse_eval_json(p)} for p in paths]
    qids = sorted(set.intersection(*[set(d) for d in per_seed]))
    return np.array([[d[q] for q in qids] for d in per_seed])


def write_mechanism(cfg: dict, rng: np.random.Generator, B: int) -> str | None:
    results = results_dir(cfg)
    modes = cfg["eval"]["grid"]["modes"]
    src = mech_sources(cfg)
    conds = ["persona", "nopersona", "direct"]
    arms = [*cfg["analysis"]["mech_arms"], "untrained"]
    rows, md_rows, any_new = [], [], False
    cell_boot: dict[tuple, tuple] = {}
    for arm in arms:
        for mode in [*modes, POOLED]:
            cells = []
            for cond in conds:
                mats = [matrix(src[(arm, cond, m)]) for m in (modes if mode == POOLED else [mode])]
                if any(M is None for M in mats):
                    cells.append(None)
                    continue
                M = np.concatenate(mats, axis=1)   # seeds x (questions x modes); modes paired per seed
                b = st.boot_seeds_questions(M, rng, B)
                cell_boot[(arm, cond, mode)] = (M, b)
                mean, lo, hi = float(M.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
                cells.append((mean, lo, hi, M.shape[0], M.shape[1], M.mean(axis=1)))
                if cond == "nopersona":
                    any_new = True
                rows.append([arm, cond, mode, M.shape[0], M.size, mean, lo, hi,
                             " ".join(f"{x:.3f}" for x in M.mean(axis=1))])
            md_rows.append([arm, mode] + [("pending" if c is None else
                                           f"{pct(c[0])} [{pct(c[1])}, {pct(c[2])}]"
                                           + (f" (seeds {' / '.join(pct(x) for x in c[5])})" if c[3] > 1 else ""))
                                          for c in cells])
    if not any_new:
        return None
    # Differences orange - orange_nopersona (paired questions, independent seeds) and orange - untrained.
    diff_rows, diff_md = [], []
    for cond in conds:
        for mode in [*modes, POOLED]:
            for other in (arms[1], "untrained"):
                a, b_ = cell_boot.get((arms[0], cond, mode)), cell_boot.get((other, cond, mode))
                if a is None or b_ is None:
                    continue
                Ma, Mb = a[0], b_[0]
                d = st.boot_paired_seeds(Ma, Mb, rng, B)
                mean = float(Ma.mean() - Mb.mean())
                lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                diff_rows.append([cond, mode, f"{arms[0]} - {other}", mean, lo, hi])
                diff_md.append([cond, mode, f"{arms[0]} − {other}", f"{100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"])
    st.write_csv(results / "mechanism.csv",
                 ["arm", "condition", "mode", "n_seeds", "n_rollouts", "honest", "honest_lo", "honest_hi", "per_seed"], rows)
    st.write_csv(results / "mechanism_differences.csv", ["condition", "mode", "difference", "delta", "lo", "hi"],
                 diff_rows)
    md = ("# Mechanism check: Qwen3.6-27B, persona vs negation (stage 0)\n\n"
          "Honest compliance (%) of 2A's step-50 adapters (seeds 0-2) and the untrained model, val split "
          "(200 q), T=1, 16k tokens. `persona` and `direct` are 2A's own val rollouts for the same adapters "
          "(kaushik/orange/runs/*/eval/*/checkpoint-50.json; untrained = checkpoint-0 of the orange runs, "
          "3 samples per question); `nopersona` is new: the negated frame with EUGENE removed. Brackets: 95% "
          "bootstrap over questions x seeds.\n\n"
          + st.md_table(["arm", "mode", "persona (EUGENE, negated)", "nopersona (negated, no EUGENE)", "direct"], md_rows)
          + "\n\n## Differences (pp; paired questions, independent seeds)\n\n"
          + st.md_table(["condition", "mode", "difference", "Δ honest [95% CI]"], diff_md)
          + "\n\n2A's 5-seed reference (kaushik/orange/results/cells.csv, val, modes lowercase/uppercase, step 50): "
          "persona orange 8.0 / 10.4 vs orange_nopersona 0.0 / 0.0; direct orange 10.7 / 5.8 vs nopersona "
          "10.5 / 6.5; untrained persona 0.0 / 0.0, direct 3.9 / 2.5.\n")
    (results / "mechanism.md").write_text(md)
    return md


def main(cfg: dict, argv: list[str] | None = None) -> None:
    an = cfg["analysis"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=int(an["bootstrap_draws"]))
    ap.add_argument("--seed", type=int, default=int(an["bootstrap_seed"]))
    args = ap.parse_args(argv)
    results_dir(cfg).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for name, fn in [("pick", lambda: write_pick_tables(cfg)), ("untrained", lambda: write_untrained(cfg, rng, args.boot)),
                     ("mechanism", lambda: write_mechanism(cfg, rng, args.boot))]:
        md = fn()
        print(f"[{name}] {'written' if md else 'nothing yet'}")
        if md:
            print(md)


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
