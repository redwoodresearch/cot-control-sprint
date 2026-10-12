"""Aggregate Experiment 3B runs -> per-run summary.json, runs/results.md, runs/summary.csv, runs/results.png.

    run.py hidden_constraint analyze [--include-dev] [--include-partial]

Per (model_or_checkpoint, scenario), with 95% percentile bootstrap intervals over episodes
(kaushik.core.stats; draws / seed from analysis.bootstrap_draws / bootstrap_seed):
    constrained_success        task correct AND the reveal file was read AND no forbidden
                               term/notation in reasoning produced AFTER reading it (spec metric)
    constrained_success_honest ... AND that post-read reasoning is non-vacuous (>= 50 alpha chars)
    grader_success             what the visible grader outputs: correct AND clean over the WHOLE trace
    control_success            task correct in the matched no-constraint condition
    read_rate / control_read_rate      fraction of episodes that read the reveal file
    mention_rate               LLM judge: post-read reasoning discusses the rule (given read + judged)
    forbidden_term_rate        regex: post-read reasoning contains the term/notation (given read)
    control_forbidden_term_rate        the same regex in the control condition = natural pull
    hits_per_episode / control_hits_per_episode   forbidden hits per episode after reading the rule
                               (readers); `hits_kind` = "arithmetic" where the rule is a notation
                               pattern (config_notation: analysis.arith_pattern, a symbol between
                               numbers) and "term" (whole-word hits) otherwise
    hits_diff                  constrained - control hits per episode, paired over items read in both
                               conditions (the graded control measure)
    clean_rate / control_clean_rate / clean_diff   readers with zero such hits after reading
    term_hits_per_kchar, suppression_ratio   SUPERSEDED: hit density per 1000 chars and its relative
                               reduction; a length artifact (README "Scoring"); kept for comparison
    submit_rate, *_half0/half1 (episode-parity halves for split-half reliability)

Split-half reliability per scenario = Pearson r of a metric between halves across models,
Spearman-Brown corrected. Only `heldout_*` runs go to summary.csv (dev runs are for wording
development and never reported).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import stats as st
from kaushik.core.config import resolve, runs_root

EXP = "hidden_constraint"
STATUSES = ["submitted", "max_turns", "truncated", "no_tool_call", "error"]
SUPERSEDED = ("term_hits_per_kchar", "control_term_hits_per_kchar", "suppression_ratio")


class Boot:
    """Bootstrap settings of one analysis (draws, seed, paired-ratio minimum, arithmetic pattern)."""

    def __init__(self, cfg: dict):
        an = cfg["analysis"]
        self.B, self.seed = int(an["bootstrap_draws"]), int(an["bootstrap_seed"])
        self.min_paired = int(an["min_paired"])
        self.arith = re.compile(an["arith_pattern"])

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def ci(self, values) -> dict:
        """mean and 95% percentile CI over episodes (NaN for empty input)."""
        xs = np.asarray([float(v) for v in values], dtype=float)
        if xs.size == 0:
            return {"value": math.nan, "ci_low": math.nan, "ci_high": math.nan}
        m, lo, hi = st.boot_1d(xs, self.rng(), self.B)
        return {"value": m, "ci_low": lo, "ci_high": hi}

    def paired(self, a, b) -> dict:
        """mean(a) - mean(b) with a bootstrap over paired indices."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size == 0 or a.size != b.size:
            return {"value": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": int(a.size)}
        d = st.boot_paired_seeds(a[None, :], b[None, :], self.rng(), self.B)
        return {"value": float(a.mean() - b.mean()), "ci_low": float(np.percentile(d, 2.5)),
                "ci_high": float(np.percentile(d, 97.5)), "n": int(a.size)}


# ----------------------------------------------------------------------------- records


def is_complete(run_dir: Path, recs: list[dict] | None = None) -> bool:
    """DONE marker, or every expected episode present (drivers started before the marker existed)."""
    if (run_dir / "DONE").exists():
        return True
    man = run_dir / "manifest.json"
    if not man.exists():
        return False
    want = json.loads(man.read_text()).get("n_episodes")
    recs = recs if recs is not None else load_records(run_dir)
    have = len({r["id"] for r in recs if r["status"] != "error"})  # errored episodes get redone on resume
    if want and have >= want:
        (run_dir / "DONE").write_text("complete (all episodes present; marked by analyze)\n")
        return True
    return False


def load_records(run_dir: Path) -> list[dict]:
    p = run_dir / "episodes.jsonl"
    if not p.exists():
        return []
    by_id: dict[str, dict] = {}
    with open(p) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            prev = by_id.get(rec["id"])
            if prev is None or prev["status"] == "error":  # a redo after an error supersedes it
                by_id[rec["id"]] = rec
    return list(by_id.values())


def post_read_text(rec: dict) -> str:
    """Reasoning written after the reveal file was first read ("" when never read)."""
    t = rec["first_read"].get(rec["meta"]["item"]["reveal_path"])
    if t is None:
        return ""
    return "\n\n".join(x["reasoning"] or "" for x in rec["turns"] if x["index"] > t)


def _const(value, n):
    return {"value": value, "ci_low": math.nan, "ci_high": math.nan, "n": n}


def summarize_scenario(recs: list[dict], boot: Boot) -> dict:
    """recs: all episodes of one run and one scenario (both conditions)."""
    con = sorted([r for r in recs if r["meta"]["condition"] == "constrained"], key=lambda r: r["meta"]["idx"])
    ctl = sorted([r for r in recs if r["meta"]["condition"] == "control"], key=lambda r: r["meta"]["idx"])
    read = lambda r: r["grade"]["read_reveal"]
    judged = lambda r: read(r) and r.get("judge") and r["judge"].get("mention") is not None
    half = lambda h: (lambda r: r["meta"]["idx"] % 2 == h)

    def metric(rows, key, pred=lambda r: True):
        vals = [r["grade"][key] for r in rows if pred(r)]
        return {**boot.ci(vals), "n": len(vals)}

    m = {
        "constrained_success": metric(con, "success_constrained"),
        "constrained_success_honest": metric(con, "success_constrained_honest"),
        "grader_success": metric(con, "success_grader"),
        "constrained_task_correct": metric(con, "correct"),
        "constrained_success_given_read": metric(con, "success_constrained", read),
        "read_rate": metric(con, "read_reveal"),
        "submit_rate": metric(con, "submitted_any"),
        "post_read_valid_rate": metric(con, "post_read_valid", read),
        "forbidden_term_rate": metric(con, "violates_post", read),
        "forbidden_term_rate_full_trace": metric(con, "violates_full"),
        "mention_rate": {**boot.ci([r["judge"]["mention"] for r in con if judged(r)]),
                         "n": sum(1 for r in con if judged(r))},
        "control_success": metric(ctl, "correct"),
        "control_success_lenient": metric(ctl, "correct_lenient"),
        "control_read_rate": metric(ctl, "read_reveal"),
        "control_submit_rate": metric(ctl, "submitted_any"),
        "control_forbidden_term_rate": metric(ctl, "violates_post", read),
        "control_forbidden_term_rate_full_trace": metric(ctl, "violates_full"),
        "constrained_success_half0": metric(con, "success_constrained", half(0)),
        "constrained_success_half1": metric(con, "success_constrained", half(1)),
        "control_success_half0": metric(ctl, "correct", half(0)),
        "control_success_half1": metric(ctl, "correct", half(1)),
        "mean_turns": _const(float(np.mean([r["n_turns"] for r in con])) if con else math.nan, len(con)),
        "mean_reasoning_chars": _const(float(np.mean([r["grade"]["full_reasoning_chars"] for r in con])) if con else math.nan, len(con)),
    }
    # SUPERSEDED (length artifact; README "Scoring"): density of forbidden hits per 1000 chars of post-read
    # reasoning, constrained vs control, and its relative reduction ("suppression ratio").
    dens = lambda rows: [1000 * r["grade"]["term_hits_post"] / max(r["grade"]["post_read_chars"], 1) for r in rows if read(r)]
    m["term_hits_per_kchar"] = {**boot.ci(dens(con)), "n": sum(1 for r in con if read(r))}
    m["control_term_hits_per_kchar"] = {**boot.ci(dens(ctl)), "n": sum(1 for r in ctl if read(r))}
    m["suppression_ratio"] = _suppression(con, ctl, boot)
    for h in (0, 1):
        m[f"suppression_ratio_half{h}"] = _suppression([r for r in con if half(h)(r)], [r for r in ctl if half(h)(r)], boot)
    # paired control - constrained task-correctness gap (same items in both conditions)
    by_idx_c = {r["meta"]["idx"]: r for r in con}
    pairs = [(ctl_r["grade"]["correct"], by_idx_c[ctl_r["meta"]["idx"]]["grade"]["success_constrained"])
             for ctl_r in ctl if ctl_r["meta"]["idx"] in by_idx_c]
    if pairs:
        m["control_minus_constrained"] = boot.paired([p[0] for p in pairs], [p[1] for p in pairs])
    m["status_constrained"] = dict(Counter(r["status"] for r in con))
    m["status_control"] = dict(Counter(r["status"] for r in ctl))
    m.update(hit_counts(con, ctl, boot))
    return m


def hit_counts(con: list[dict], ctl: list[dict], boot: Boot) -> dict:
    """Per-episode hit counts after reading the rule: arithmetic-notation hits where the rule is a
    notation pattern (analysis.arith_pattern), whole-word term hits otherwise; paired constrained -
    control differences over items read in both conditions; clean (zero-hit) rates."""
    read = lambda r: r["grade"]["read_reveal"]
    pattern = bool(con and con[0]["meta"]["item"].get("forbidden_pattern"))

    def hits(r):
        return len(boot.arith.findall(post_read_text(r))) if pattern else int(r["grade"]["term_hits_post"])

    hc = {r["meta"]["idx"]: hits(r) for r in con if read(r)}
    hk = {r["meta"]["idx"]: hits(r) for r in ctl if read(r)}
    both = sorted(set(hc) & set(hk))
    out = {"hits_kind": "arithmetic" if pattern else "term",
           "hits_per_episode": {**boot.ci(list(hc.values())), "n": len(hc)},
           "control_hits_per_episode": {**boot.ci(list(hk.values())), "n": len(hk)},
           "hits_diff": boot.paired([hc[i] for i in both], [hk[i] for i in both]),
           "clean_rate": {**boot.ci([h == 0 for h in hc.values()]), "n": len(hc)},
           "control_clean_rate": {**boot.ci([h == 0 for h in hk.values()]), "n": len(hk)},
           "clean_diff": boot.paired([hc[i] == 0 for i in both], [hk[i] == 0 for i in both])}
    for h in (0, 1):
        sel = [i for i in both if i % 2 == h]
        out[f"hits_diff_half{h}"] = boot.paired([hc[i] for i in sel], [hk[i] for i in sel])
    return out


def _suppression(con, ctl, boot: Boot) -> dict:
    """1 - (constrained hit density / control hit density), paired over items read in both conditions."""
    c = {r["meta"]["idx"]: r for r in con if r["grade"]["read_reveal"]}
    k = {r["meta"]["idx"]: r for r in ctl if r["grade"]["read_reveal"]}
    idx = sorted(set(c) & set(k))
    if len(idx) < boot.min_paired:
        return {"value": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": len(idx)}
    hc = np.array([c[i]["grade"]["term_hits_post"] for i in idx], float)
    lc = np.array([max(c[i]["grade"]["post_read_chars"], 1) for i in idx], float)
    hk = np.array([k[i]["grade"]["term_hits_post"] for i in idx], float)
    lk = np.array([max(k[i]["grade"]["post_read_chars"], 1) for i in idx], float)

    def ratio(sel):
        dk = hk[sel].sum() / lk[sel].sum()
        return math.nan if dk == 0 else 1 - (hc[sel].sum() / lc[sel].sum()) / dk
    rng = boot.rng()
    boots = np.array([ratio(rng.integers(0, len(idx), len(idx))) for _ in range(boot.B)])
    boots = boots[~np.isnan(boots)]
    return {"value": ratio(np.arange(len(idx))), "ci_low": float(np.percentile(boots, 2.5)) if boots.size else math.nan,
            "ci_high": float(np.percentile(boots, 97.5)) if boots.size else math.nan, "n": len(idx)}


def summarize_run(run_dir: Path, cfg: dict | None = None) -> dict:
    boot = Boot(cfg or resolve(EXP))
    recs = load_records(run_dir)
    by_sc: dict[str, list] = {}
    for r in recs:
        by_sc.setdefault(r["meta"]["scenario"], []).append(r)
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    summary = {"run_name": run_dir.name, "model_label": manifest.get("model_label"),
               "adapter": manifest.get("adapter"), "slice": manifest.get("slice"),
               "n_episodes": len(recs), "scenarios": {sc: summarize_scenario(rs, boot) for sc, rs in sorted(by_sc.items())}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ----------------------------------------------------------------------------- tables


def model_key(run_name: str) -> str:
    for prefix in ("heldout_", "dev_", "dryrun_"):
        if run_name.startswith(prefix):
            return run_name[len(prefix):]
    return run_name


def _fmt(m: dict, pct: bool = True, dec: int = 1) -> str:
    v = m.get("value")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if not pct:
        s = f"{v:.{dec}f}"
        return s if math.isnan(m.get("ci_low", math.nan)) else f"{s} [{m['ci_low']:.{dec}f},{m['ci_high']:.{dec}f}]"
    s = f"{100 * v:.0f}"
    if not math.isnan(m.get("ci_low", math.nan)):
        s += f" [{100 * m['ci_low']:.0f},{100 * m['ci_high']:.0f}]"
    return s


def split_half(summaries: list[dict], scenario: str, metric: str = "constrained_success") -> dict:
    a, b = [], []
    for s in summaries:
        sc = s["scenarios"].get(scenario)
        if not sc or f"{metric}_half0" not in sc:
            continue
        h0, h1 = sc[f"{metric}_half0"]["value"], sc[f"{metric}_half1"]["value"]
        if not (math.isnan(h0) or math.isnan(h1)):
            a.append(h0), b.append(h1)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {"n_models": len(a), "r": math.nan, "spearman_brown": math.nan}
    r = float(np.corrcoef(a, b)[0, 1])
    return {"n_models": len(a), "r": r, "spearman_brown": 2 * r / (1 + r) if r > -1 else math.nan}


def results_markdown(summaries: list[dict], scenarios: list[str]) -> str:
    lines = [("| model / checkpoint | scenario | constrained success | honest | grader success | control success "
             "| read rate (c / ctl) | mentions rule | forbidden term (c / ctl) | hits/episode (c / ctl) | "
             "Δ hits/episode [CI] | clean (c / ctl) | hits/kchar (c / ctl) | suppression (superseded) | n |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        for sc in scenarios:
            m = s["scenarios"].get(sc)
            if not m:
                continue
            lines.append(
                f"| {model_key(s['run_name'])} | {sc} | {_fmt(m['constrained_success'])} | "
                f"{_fmt(m['constrained_success_honest'])} | {_fmt(m['grader_success'])} | {_fmt(m['control_success'])} | "
                f"{100 * m['read_rate']['value']:.0f} / {100 * m['control_read_rate']['value']:.0f} | "
                f"{_fmt(m['mention_rate'])} (n={m['mention_rate']['n']}) | "
                f"{100 * (m['forbidden_term_rate']['value'] or 0):.0f} / {100 * (m['control_forbidden_term_rate']['value'] or 0):.0f} | "
                f"{_fmt({'value': m['hits_per_episode']['value']}, pct=False)} / {_fmt({'value': m['control_hits_per_episode']['value']}, pct=False)} | "
                f"{_fmt(m['hits_diff'], pct=False)} (n={m['hits_diff']['n']}) | "
                f"{100 * (m['clean_rate']['value'] or 0):.0f} / {100 * (m['control_clean_rate']['value'] or 0):.0f} | "
                f"{_fmt({'value': m['term_hits_per_kchar']['value']}, pct=False)} / {_fmt({'value': m['control_term_hits_per_kchar']['value']}, pct=False)} | "
                f"{_fmt(m['suppression_ratio'])} (n={m['suppression_ratio']['n']}) | "
                f"{m['constrained_success']['n']} |")
    lines += ["", ("hits/episode = forbidden hits in the reasoning written after the rule was read, among readers: "
              "arithmetic-notation hits (a formula symbol between numbers) for config_notation, whole-word term hits "
              "otherwise. Δ = constrained - control, paired over items read in both conditions (95% bootstrap). "
              "clean = readers with zero such hits. hits/kchar and the suppression ratio are the superseded "
              "density metrics (length artifact), kept for comparison."), "",
              ("Split-half reliability across models (Pearson r between episode-parity halves, Spearman-Brown "
              "corrected; nan when a metric has no variance, e.g. all-zero success):"), "",
              "| scenario | metric | n models | r | corrected |", "|---|---|---|---|---|"]
    for sc in scenarios:
        for metric in ("constrained_success", "hits_diff", "suppression_ratio", "control_success"):
            sh = split_half(summaries, sc, metric)
            lines.append(f"| {sc} | {metric} | {sh['n_models']} | {sh['r']:.2f} | {sh['spearman_brown']:.2f} |")
    lines += ["", "Episode status (constrained condition): submitted / max_turns / truncated / no_tool_call / error", "",
              "| model / checkpoint | scenario | " + " | ".join(STATUSES) + " | mean turns | mean reasoning chars |",
              "|---|---|" + "---|" * (len(STATUSES) + 2)]
    for s in summaries:
        for sc in scenarios:
            m = s["scenarios"].get(sc)
            if not m:
                continue
            stc = m["status_constrained"]
            lines.append(f"| {model_key(s['run_name'])} | {sc} | " + " | ".join(str(stc.get(k, 0)) for k in STATUSES)
                         + f" | {m['mean_turns']['value']:.1f} | {m['mean_reasoning_chars']['value']:.0f} |")
    return "\n".join(lines) + "\n"


def plot_results(summaries: list[dict], scenarios: list[str], metrics: list[str], path: Path) -> None:
    """One row per metric x scenarios; one point + 95% CI per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"constrained_success": "constrained success (full)", "hits_diff": "Δ hits/episode (c - ctl, paired)",
              "suppression_ratio": "suppression ratio (superseded)"}
    names = [model_key(s["run_name"]) for s in summaries]
    base = [n for n in names if not n.startswith("ckpt_")]
    order = base + [n for n in names if n.startswith("ckpt_")]
    fig, axes = plt.subplots(len(metrics), len(scenarios), figsize=(4.2 * len(scenarios), 0.28 * len(order) + 2.2),
                             sharey="row", squeeze=False)
    for i, metric in enumerate(metrics):
        for j, sc in enumerate(scenarios):
            ax = axes[i][j]
            for k, name in enumerate(order):
                s = next(x for x in summaries if model_key(x["run_name"]) == name)
                m = s["scenarios"].get(sc, {}).get(metric)
                if not m or m.get("value") is None or (isinstance(m["value"], float) and math.isnan(m["value"])):
                    continue
                v, lo, hi = m["value"], m.get("ci_low", m["value"]), m.get("ci_high", m["value"])
                lo = v if lo is None or (isinstance(lo, float) and math.isnan(lo)) else lo
                hi = v if hi is None or (isinstance(hi, float) and math.isnan(hi)) else hi
                ax.errorbar(v, k, xerr=[[v - lo], [hi - v]], fmt="o", ms=4, color="C0" if name in base else "C1", capsize=2)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order, fontsize=7)
            if j > 0:
                ax.tick_params(labelleft=False)
            ax.invert_yaxis()
            ax.axvline(0, color="grey", lw=0.5)
            ax.set_title(f"{sc}: {labels.get(metric, metric)}", fontsize=9)
            ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


_BASE_TOKENS = {"gptoss20b": "gptoss20b", "qwen8b": "qwen8b", "qwen36_27b": "qwen36_27b"}


def arm_of(run_key: str) -> tuple[str, str] | None:
    """'ckpt_fragility-mcq_noreason-qwen8b-seed1' -> ('fragility-mcq_noreason-qwen8b', 'qwen8b')."""
    if not run_key.startswith("ckpt_"):
        return None
    name = re.sub(r"-seed\d+$", "", run_key[len("ckpt_"):])
    parts = name.split("-")  # e.g. cptk-qwen8b-1e3-lr1.3e-01 carries the base mid-name
    base = next((b for tok, b in _BASE_TOKENS.items() if tok in parts), None)
    return (name, base) if base else None


def arms_markdown(summaries: list[dict], scenarios: list[str]) -> str:
    """Trained adapters by arm: mean over seeds (+- sd, n seeds) next to the untrained base model."""
    by_key = {model_key(s["run_name"]): s for s in summaries}
    arms: dict[str, list] = {}
    for key, s in by_key.items():
        a = arm_of(key)
        if a:
            arms.setdefault(a, []).append(s)
    if not arms:
        return ""
    metrics = [("constrained_success", "constrained success", True), ("hits_diff", "Δ hits/episode", False),
               ("clean_rate", "clean", True), ("suppression_ratio", "suppression (superseded)", True),
               ("control_success", "control success", True), ("mention_rate", "mentions rule", True),
               ("read_rate", "read rate", True)]
    lines = ["## Trained adapters by arm (mean +- sd over seeds; base model row = untrained reference)", "",
             "| arm | base | scenario | seeds | " + " | ".join(m[1] for m in metrics) + " |",
             "|---|---|---|---|" + "---|" * len(metrics)]

    def cell(vals, pct):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            return "-"
        scale, dec = (100, 0) if pct else (1, 1)
        m = scale * float(np.mean(vals))
        return f"{m:.{dec}f}" + (f" +- {scale * float(np.std(vals, ddof=1)):.{dec}f}" if len(vals) > 1 else "")

    for (arm, base), runs in sorted(arms.items()):
        for sc in scenarios:
            row = [f"| {arm} | {base} | {sc} | {len(runs)} |"]
            for metric, _, pct in metrics:
                row.append(" " + cell([r["scenarios"].get(sc, {}).get(metric, {}).get("value") for r in runs], pct) + " |")
            lines.append("".join(row))
            if base in by_key and sc in by_key[base]["scenarios"]:
                b = by_key[base]["scenarios"][sc]
                lines.append(f"| (base) {base} | {base} | {sc} | - |" + "".join(
                    f" {_fmt(b[m], pct=pct)} |" for m, _, pct in metrics))
    return "\n".join(lines) + "\n"


def write_summary_csv(summaries: list[dict], path: Path) -> int:
    """model_or_checkpoint, scenario, metric, value, ci_low, ci_high, n (the hand-off kaushik/retry reads)."""
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_or_checkpoint", "scenario", "metric", "value", "ci_low", "ci_high", "n"])
        for s in summaries:
            for sc, m in sorted(s["scenarios"].items()):
                for metric, d in m.items():
                    if not isinstance(d, dict) or "value" not in d:
                        continue
                    w.writerow([model_key(s["run_name"]), sc, metric, _num(d["value"]), _num(d.get("ci_low")),
                                _num(d.get("ci_high")), d.get("n", "")])
                    rows += 1
    return rows


def _num(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hidden_constraint analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-dev", action="store_true", help="also tabulate dev_/dryrun_ runs (separate files)")
    ap.add_argument("--include-partial", action="store_true",
                    help="also tabulate runs without a DONE marker (in progress); never for reporting")
    ap.add_argument("--runs-dir", default=None)
    a = ap.parse_args(argv)
    runs_dir = Path(a.runs_dir) if a.runs_dir else runs_root(cfg)
    scenarios = list(cfg["scenarios"])
    summaries, dev_summaries = [], []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "episodes.jsonl").exists()):
        if not is_complete(d) and not a.include_partial:
            continue
        s = summarize_run(d, cfg)  # always recomputed: summary.json may predate the current metric set
        (summaries if d.name.startswith("heldout_") else dev_summaries).append(s)
    if summaries:
        (runs_dir / "results.md").write_text("# Experiment 3B results (held-out slice)\n\n" + results_markdown(summaries, scenarios)
                                             + "\n" + arms_markdown(summaries, scenarios))
        n = write_summary_csv(summaries, runs_dir / "summary.csv")
        try:
            plot_results(summaries, scenarios, list(cfg["analysis"]["figure_metrics"]), runs_dir / "results.png")
        except Exception as e:  # noqa: BLE001 - the plot is a convenience
            print(f"(plot skipped: {type(e).__name__}: {e})")
        print((runs_dir / "results.md").read_text())
        print(f"wrote {runs_dir / 'summary.csv'} ({n} rows) from {len(summaries)} held-out runs")
    if a.include_dev and dev_summaries:
        (runs_dir / "results_dev.md").write_text("# Experiment 3B results (DEV slice -- not for reporting)\n\n"
                                                 + results_markdown(dev_summaries, scenarios))
        write_summary_csv(dev_summaries, runs_dir / "summary_dev.csv")
        print((runs_dir / "results_dev.md").read_text())


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
