"""Stage-0 tables from the saved rollouts (run_all.py writes them; this only reads).

    .venv/bin/python kaushik/orange_expand/baseline/analyze.py [--boot 2000]

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
from pathlib import Path

import numpy as np

import common  # noqa: F401  (env first)
from common import (EVAL_CONDITIONS, MECH_ARMS, MECH_MODEL, MECH_SEEDS, ORANGE, RESULTS_DIR, RUNS_DIR,
                    SCREEN_MODES, STAGE0_MODELS, TRAIN_PROMPTS)

from cotcontrol.eval.grading import is_honest

DEGEN_CHARS = 200
PICK_ORDER = list(TRAIN_PROMPTS)


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")


# ----------------------------------------------------------------------------- pick rates

def write_pick_tables() -> str | None:
    rows, md_rows = [], []
    for mk in STAGE0_MODELS:
        for prompt in PICK_ORDER:
            cells = []
            for on in (True, False):
                path = RUNS_DIR / mk / "pick" / f"{prompt}_{'on' if on else 'off'}.json"
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_DIR / "pick_rates.csv",
              ["model", "train_prompt", "reasoning", "n", "n_parsed", "orange_rate", "unparseable_rate",
               "truncated_rate", "mean_reasoning_chars"], rows)
    md = ("# Orange-pick rate, untrained models (stage 0)\n\n"
          "200 held-out orange MCQs (same 200 as the SFT agent's `heldout_200.json`), training prompt as system "
          "prompt, user turn = orange-training `format_user`, T=1. Cell = % of parseable answers equal to the "
          "orange letter (unp = unparseable share of all 200). Reasoning on: gpt-oss medium / Qwen thinking, "
          "8192 tokens; off: gpt-oss low + empty analysis prefill / Qwen enable_thinking=false (2A's training "
          "rendering). Chance = 25%.\n\n"
          + md_table(["model", "training prompt", "reasoning on", "reasoning off"], md_rows) + "\n")
    (RESULTS_DIR / "pick_rates.md").write_text(md)
    return md


# ----------------------------------------------------------------------------- eval rollouts

def load_eval(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = []
    for rec in data["results"]:
        for s in rec["samples"]:
            if s["error"]:
                continue
            text = s.get("reasoning_text_graded") or ""
            rows.append({"qid": f"{rec['dataset']}:{rec['id']}", "mode": rec["mode"], "honest": is_honest(s),
                         "compliance": s["compliance"] == 1, "correct": s["correct"] is True, "chars": len(text),
                         "truncated": s.get("finish_reason") == "length"})
    return rows


def boot_q(flags: np.ndarray, rng: np.random.Generator, B: int) -> tuple[float, float, float]:
    """Mean + 95% CI over questions (flags: 1-D)."""
    if len(flags) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(flags), size=(B, len(flags)))
    m = flags[idx].mean(axis=1)
    return float(flags.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_qs(M: np.ndarray, rng: np.random.Generator, B: int) -> np.ndarray:
    """Bootstrap means of a seeds x questions matrix: resample seeds and questions."""
    S, Q = M.shape
    si = rng.integers(0, S, size=(B, S))
    qi = rng.integers(0, Q, size=(B, Q))
    return np.stack([M[np.ix_(si[b], qi[b])].mean() for b in range(B)])


def write_untrained(rng: np.random.Generator, B: int) -> str | None:
    rows, md_rows = [], []
    for mk in STAGE0_MODELS:
        for cond in EVAL_CONDITIONS:
            for mode in SCREEN_MODES + ["(pooled)"]:
                modes = SCREEN_MODES if mode == "(pooled)" else [mode]
                recs = []
                for m in modes:
                    path = RUNS_DIR / mk / "eval" / f"{cond}_{m}" / "base.json"
                    if path.exists():
                        recs += load_eval(path)
                if len(recs) < 200 * len(modes):   # incomplete
                    if mode != "(pooled)":
                        md_rows.append([mk, cond, mode, "pending", "", "", "", "", ""])
                    continue
                h = np.array([r["honest"] for r in recs], float)
                mean, lo, hi = boot_q(h, rng, B)
                comp = np.mean([r["compliance"] for r in recs])
                acc = np.mean([r["correct"] for r in recs])
                chars = np.mean([r["chars"] for r in recs])
                degen = np.mean([r["chars"] < DEGEN_CHARS for r in recs])
                trunc = np.mean([r["truncated"] for r in recs])
                rows.append([mk, cond, mode, len(recs), mean, lo, hi, comp, acc, chars, degen, trunc])
                md_rows.append([mk, cond, mode, f"{pct(mean)} [{pct(lo)}, {pct(hi)}]", pct(comp), pct(acc),
                                f"{chars:.0f}", pct(degen), pct(trunc)])
    if not rows:
        return None
    write_csv(RESULTS_DIR / "untrained.csv",
              ["model", "condition", "mode", "n", "honest", "honest_lo", "honest_hi", "compliance", "accuracy",
               "chars", "degen", "truncated"], rows)
    md = ("# Untrained models, CoT-control eval (stage 0)\n\n"
          "val split (200 q), T=1, 16k tokens, reasoning on (gpt-oss medium / Qwen thinking). honest = honest "
          "compliance (`is_honest`) with 95% bootstrap over questions; raw = strict compliance; acc = accuracy; "
          "chars = mean reasoning length; degen = traces < 200 chars; trunc = hit max_tokens.\n\n"
          + md_table(["model", "condition", "mode", "honest", "raw", "acc", "chars", "degen", "trunc"], md_rows)
          + "\n")
    (RESULTS_DIR / "untrained.md").write_text(md)
    return md


# ----------------------------------------------------------------------------- mechanism (27B)

def mech_sources() -> dict[tuple[str, str, str], list[Path]]:
    """(arm, condition, mode) -> per-seed rollout files. persona/direct = 2A val step-50 files for the SAME
    adapters (seeds 0-2); untrained persona/direct = 2A checkpoint-0 of the orange runs (zero-init adapter =
    base model), untrained nopersona = this folder's base run."""
    src: dict[tuple[str, str, str], list[Path]] = {}
    mech = RUNS_DIR / f"mech_{MECH_MODEL}"
    for mode in SCREEN_MODES:
        for arm in MECH_ARMS:
            for cond in ("persona", "direct"):
                src[(arm, cond, mode)] = [ORANGE / "runs" / f"{arm}-{MECH_MODEL}-seed{s}" / "eval" / f"{cond}_{mode}"
                                          / "checkpoint-50.json" for s in MECH_SEEDS]
            src[(arm, "nopersona", mode)] = [mech / f"{arm}-seed{s}" / "eval" / f"nopersona_{mode}"
                                             / "checkpoint-50.json" for s in MECH_SEEDS]
        for cond in ("persona", "direct"):
            src[("untrained", cond, mode)] = [ORANGE / "runs" / f"orange-{MECH_MODEL}-seed{s}" / "eval"
                                              / f"{cond}_{mode}" / "checkpoint-0.json" for s in MECH_SEEDS]
        src[("untrained", "nopersona", mode)] = [mech / "untrained" / "eval" / f"nopersona_{mode}" / "base.json"]
    return src


def matrix(paths: list[Path], key: str = "honest") -> np.ndarray | None:
    """seeds x questions matrix (questions aligned by qid) or None if any file is missing."""
    if not all(p.exists() for p in paths):
        return None
    per_seed = []
    for p in paths:
        d = {r["qid"]: float(r[key]) for r in load_eval(p)}
        per_seed.append(d)
    qids = sorted(set.intersection(*[set(d) for d in per_seed]))
    return np.array([[d[q] for q in qids] for d in per_seed])


def write_mechanism(rng: np.random.Generator, B: int) -> str | None:
    src = mech_sources()
    conds = ["persona", "nopersona", "direct"]
    rows, md_rows, any_new = [], [], False
    cell_boot: dict[tuple, np.ndarray] = {}
    for arm in ["orange", "orange_nopersona", "untrained"]:
        for mode in SCREEN_MODES + ["(pooled)"]:
            cells = []
            for cond in conds:
                mats = [matrix(src[(arm, cond, m)]) for m in (SCREEN_MODES if mode == "(pooled)" else [mode])]
                if any(M is None for M in mats):
                    cells.append(None)
                    continue
                M = np.concatenate(mats, axis=1)   # seeds x (questions x modes); modes paired per seed
                b = boot_qs(M, rng, B)
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
        for mode in SCREEN_MODES + ["(pooled)"]:
            for other in ("orange_nopersona", "untrained"):
                a, b_ = cell_boot.get(("orange", cond, mode)), cell_boot.get((other, cond, mode))
                if a is None or b_ is None:
                    continue
                Ma, Mb = a[0], b_[0]
                Q = min(Ma.shape[1], Mb.shape[1])
                Ma, Mb = Ma[:, :Q], Mb[:, :Q]
                qi = rng.integers(0, Q, size=(B, Q))
                sa = rng.integers(0, Ma.shape[0], size=(B, Ma.shape[0]))
                sb = rng.integers(0, Mb.shape[0], size=(B, Mb.shape[0]))
                d = np.array([Ma[np.ix_(sa[i], qi[i])].mean() - Mb[np.ix_(sb[i], qi[i])].mean() for i in range(B)])
                mean = float(Ma.mean() - Mb.mean())
                lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                diff_rows.append([cond, mode, f"orange - {other}", mean, lo, hi])
                diff_md.append([cond, mode, f"orange − {other}", f"{100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"])
    write_csv(RESULTS_DIR / "mechanism.csv",
              ["arm", "condition", "mode", "n_seeds", "n_rollouts", "honest", "honest_lo", "honest_hi", "per_seed"], rows)
    write_csv(RESULTS_DIR / "mechanism_differences.csv", ["condition", "mode", "difference", "delta", "lo", "hi"],
              diff_rows)
    md = ("# Mechanism check: Qwen3.6-27B, persona vs negation (stage 0)\n\n"
          "Honest compliance (%) of 2A's step-50 adapters (seeds 0-2) and the untrained model, val split "
          "(200 q), T=1, 16k tokens. `persona` and `direct` are 2A's own val rollouts for the same adapters "
          "(kaushik/orange/runs/*/eval/*/checkpoint-50.json; untrained = checkpoint-0 of the orange runs, "
          "3 samples per question); `nopersona` is new: the negated frame with EUGENE removed. Brackets: 95% "
          "bootstrap over questions x seeds.\n\n"
          + md_table(["arm", "mode", "persona (EUGENE, negated)", "nopersona (negated, no EUGENE)", "direct"], md_rows)
          + "\n\n## Differences (pp; paired questions, independent seeds)\n\n"
          + md_table(["condition", "mode", "difference", "Δ honest [95% CI]"], diff_md)
          + "\n\n2A's 5-seed reference (kaushik/orange/results/cells.csv, val, modes lowercase/uppercase, step 50): "
          "persona orange 8.0 / 10.4 vs orange_nopersona 0.0 / 0.0; direct orange 10.7 / 5.8 vs nopersona "
          "10.5 / 6.5; untrained persona 0.0 / 0.0, direct 3.9 / 2.5.\n")
    (RESULTS_DIR / "mechanism.md").write_text(md)
    return md


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for name, fn in [("pick", write_pick_tables), ("untrained", lambda: write_untrained(rng, args.boot)),
                     ("mechanism", lambda: write_mechanism(rng, args.boot))]:
        md = fn()
        print(f"[{name}] {'written' if md else 'nothing yet'}")
        if md:
            print(md)


if __name__ == "__main__":
    main()
