"""Analysis for kaushik/retry: compliance-by-attempt curves per arm with the
independent / block_once pass@k curves, the 3A-transfer tables, and the
Experiment 3 correlation (3A scores vs standard-eval compliance) with
bootstrap intervals and split-half reliability.

    .venv/bin/python kaushik/run.py retry analyze --run main [--ckpt-run ckpt] [--transfer-run main]

Writes runs/<run>/results.md, runs/<run>/results.json and runs/<run>/figures/.
All intervals are 95% percentile bootstraps over QUESTIONS (every mode/arm
item of a question moves together), paired across arms.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from retry_arms import BASELINE_OF
from retry_common import CFG, MODELS, RUNS

from cotcontrol.eval.grading import is_honest
from cotcontrol.eval.retry import load_attempts

N_BOOT = 1000
DEGEN_CHARS = 200
REPORT_KS = (1, 2, 3, 5, 10)


# --------------------------------------------------------------------------
# 3A episodes -> arrays
# --------------------------------------------------------------------------

class ArmData:
    """Per-item arrays for one (model, arm): first honest / strict attempt
    (inf if none), attempts recorded, question index, mode, per-attempt
    chars / correct, correct at the first honest attempt."""

    def __init__(self, episodes: dict[str, list[dict]], qindex: dict[tuple, int], n: int, stop_on_success: bool = True):
        self.n = n
        self.stop_on_success = stop_on_success
        self.ids = sorted(episodes)
        recs = [episodes[i] for i in self.ids]
        inf = float("inf")
        self.q = np.array([qindex.setdefault((r[0]["dataset"], r[0]["id"]), len(qindex)) for r in recs])
        self.mode = np.array([r[0]["mode"] for r in recs])
        self.first = np.array([next((a["attempt"] for a in r if a["honest"]), inf) for r in recs])
        self.first_strict = np.array([next((a["attempt"] for a in r if a["compliance"] == 1), inf) for r in recs])
        self.recorded = np.array([len(r) for r in recs])
        self.correct_at_first = np.array([
            next((bool(a["correct"]) for a in r if a["honest"]), False) for r in recs])
        self.chars = np.full((len(recs), n), np.nan)
        self.correct = np.full((len(recs), n), np.nan)
        self.compliant = np.full((len(recs), n), np.nan)
        self.truncated = np.full((len(recs), n), np.nan)
        for i, r in enumerate(recs):
            for a in r:
                k = a["attempt"] - 1
                if k < n:
                    self.chars[i, k] = a["chars"]
                    self.correct[i, k] = float(bool(a["correct"]))
                    self.compliant[i, k] = float(a["compliance"] == 1)
                    self.truncated[i, k] = float(a["finish_reason"] == "length")
        self.honest_at = np.zeros((len(recs), n), dtype=bool)
        for i, r in enumerate(recs):
            for a in r:
                if a["attempt"] - 1 < n:
                    self.honest_at[i, a["attempt"] - 1] = bool(a["honest"])
        self.first_long = np.full(len(recs), inf)  # set by set_long_reference
        self.long_ref = float("nan")
        self.mentions = None  # filled from mentions.jsonl

    def set_long_reference(self, ref_chars: float) -> None:
        """first attempt that is honest-compliant AND at least ref_chars long
        (the spec's compl_long: compliance that did not come from collapsing
        the trace; ref = the model's attempt-1 median trace length)."""
        ok = self.honest_at & (np.nan_to_num(self.chars, nan=-1) >= ref_chars)
        self.first_long = np.where(ok.any(1), ok.argmax(1) + 1, float("inf")).astype(float)

    def subset(self, mask: np.ndarray) -> ArmData:
        o = object.__new__(ArmData)
        o.n = self.n
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape:
                setattr(o, k, v[mask])
            elif k not in ("n",):
                setattr(o, k, v)
        o.ids = [i for i, m in zip(self.ids, mask) if m]
        return o


def load_arm(arm_dir: Path, qindex: dict, n: int) -> ArmData | None:
    path = arm_dir / "attempts.jsonl"
    if not path.exists():
        return None
    man = json.loads((arm_dir / "manifest.json").read_text()) if (arm_dir / "manifest.json").exists() else {}
    d = ArmData(load_attempts(path), qindex, n, bool(man.get("stop_on_success", True)))
    mpath = arm_dir / "mentions.jsonl"
    if mpath.exists():
        d.mentions = [json.loads(line) for line in mpath.open() if line.strip()]
    return d


def cum_curve(d: ArmData, w: np.ndarray | None = None, strict: bool = False, long: bool = False) -> np.ndarray:
    """Cumulative fraction of episodes with an honest (or strict) compliant
    attempt by k, over items resolved at k: >= k attempts recorded, or (stop
    arms only) the episode stopped on an honest success by k. Pass@k arms
    need the k-th sample. Only matters while a run is incomplete; a finished
    run resolves every item at every k. w = per-question bootstrap weights."""
    first = d.first_strict if strict else (d.first_long if long else d.first)
    ks = np.arange(1, d.n + 1)[None, :]
    succ = first[:, None] <= ks
    stopped = (d.first[:, None] <= ks) if d.stop_on_success else np.zeros_like(succ)
    resolved = stopped | (d.recorded[:, None] >= ks)
    wi = (w[d.q] if w is not None else np.ones(len(d.q)))[:, None]
    den = (wi * resolved).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (wi * succ).sum(0) / den, np.nan)


def boot_weights(n_q: int, rng: np.random.Generator) -> np.ndarray:
    return np.bincount(rng.integers(0, n_q, n_q), minlength=n_q).astype(float)


def ci(samples: np.ndarray, axis=0) -> tuple[np.ndarray, np.ndarray]:
    return np.nanpercentile(samples, 2.5, axis=axis), np.nanpercentile(samples, 97.5, axis=axis)


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    m = ~np.isnan(x)
    return float((x[m] * w[m]).sum() / w[m].sum()) if w[m].sum() > 0 else float("nan")


def arm_summary(d: ArmData, n_q: int, rng: np.random.Generator) -> dict:
    boots = np.stack([cum_curve(d, boot_weights(n_q, rng)) for _ in range(N_BOOT)])
    lo, hi = ci(boots)
    curve = cum_curve(d)
    strict = cum_curve(d, strict=True)
    long_boots = np.stack([cum_curve(d, boot_weights(n_q, rng), long=True) for _ in range(N_BOOT)])
    long_lo, long_hi = ci(long_boots)
    ks = np.arange(1, d.n + 1)
    succ_by = d.first[:, None] <= ks[None, :]
    acc_on_compliant = np.array([d.correct_at_first[succ_by[:, k]].mean() if succ_by[:, k].any() else np.nan
                                 for k in range(d.n)])
    fi = d.first[np.isfinite(d.first)].astype(int) - 1
    chars_first = d.chars[np.isfinite(d.first), fi] if len(fi) else np.array([])
    chars_on_compliant = float(np.nanmean(chars_first)) if len(chars_first) else float("nan")
    degen_on_compliant = float(np.nanmean(chars_first < DEGEN_CHARS)) if len(chars_first) else float("nan")
    out = {
        "n_items": len(d.ids),
        "n_questions": len(set(d.q.tolist())),
        "n_resolved": [int((((d.first <= k) if d.stop_on_success else False) | (d.recorded >= k)).sum()) for k in ks],
        "cum_honest": curve.tolist(), "cum_honest_lo": lo.tolist(), "cum_honest_hi": hi.tolist(),
        "cum_strict": strict.tolist(),
        "cum_honest_long": cum_curve(d, long=True).tolist(), "cum_honest_long_lo": long_lo.tolist(),
        "cum_honest_long_hi": long_hi.tolist(), "long_ref_chars": d.long_ref,
        "acc_on_compliant": acc_on_compliant.tolist(),
        "chars_on_compliant": chars_on_compliant, "degen_on_compliant": degen_on_compliant,
        "acc_attempt": np.nanmean(d.correct, 0).tolist(),
        "chars": np.nanmean(d.chars, 0).tolist(),
        "degen": np.nanmean(d.chars < DEGEN_CHARS, 0).tolist(),
        "truncated": np.nanmean(d.truncated, 0).tolist(),
        "n_attempts_at_k": (~np.isnan(d.chars)).sum(0).tolist(),
        "per_mode": {},
    }
    for m in sorted(set(d.mode.tolist())):
        sub = d.subset(d.mode == m)
        c = cum_curve(sub)
        out["per_mode"][m] = {"n": len(sub.ids), "cum_honest": c.tolist(), "cum_strict": cum_curve(sub, strict=True).tolist()}
    if d.mentions:
        rows = d.mentions
        first = [r for r in rows if r["attempt"] == 1]
        later = [r for r in rows if r["attempt"] > 1]

        def rate(rs, key, val):
            xs = [r[key] for r in rs if r[key] is not None]
            return (sum(x == val for x in xs) / len(xs)) if xs else None
        out["mentions"] = {
            "n_judged": len(rows),
            "constraint_all": rate(rows, "meta_discussion", 0), "constraint_attempt1": rate(first, "meta_discussion", 0),
            "constraint_later": rate(later, "meta_discussion", 0),
            "feedback_all": rate(rows, "mentions_feedback", True), "feedback_later": rate(later, "mentions_feedback", True),
        }
    return out


def diff_summary(a: ArmData, b: ArmData, n_q: int, rng: np.random.Generator) -> dict:
    """a - b cumulative honest curve (and its long-trace variant) with a paired question bootstrap."""
    ws = [boot_weights(n_q, rng) for _ in range(N_BOOT)]
    boots = np.stack([cum_curve(a, w) - cum_curve(b, w) for w in ws])
    lboots = np.stack([cum_curve(a, w, long=True) - cum_curve(b, w, long=True) for w in ws])
    lo, hi = ci(boots)
    llo, lhi = ci(lboots)
    return {"diff": (cum_curve(a) - cum_curve(b)).tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
            "beats_baseline_at_N": bool(lo[-1] > 0), "p_le_0_at_N": float((boots[:, -1] <= 0).mean()),
            "diff_long": (cum_curve(a, long=True) - cum_curve(b, long=True)).tolist(),
            "long_lo": llo.tolist(), "long_hi": lhi.tolist(), "beats_baseline_long_at_N": bool(llo[-1] > 0)}


def run_subset(run_dir: Path) -> set | None:
    """Item ids the run is restricted to (from any arm manifest's items.subset), or None."""
    for man in run_dir.glob("*/*/manifest.json"):
        sub = (json.loads(man.read_text()).get("items") or {}).get("subset")
        if sub:
            return set(json.loads((Path(__file__).resolve().parent / sub).read_text())["item_ids"])
    return None


def analyze_3a(run_dir: Path, n: int, labels: list[str] | None = None, item_filter=None) -> dict:
    rng = np.random.default_rng(0)
    if item_filter is None:
        item_filter = run_subset(run_dir)
    out = {}
    for mdir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name in MODELS or (labels and p.name in labels)):
        qindex: dict[tuple, int] = {}
        arms = {a.name: load_arm(a, qindex, n) for a in sorted(mdir.iterdir()) if a.is_dir()}
        arms = {k: v for k, v in arms.items() if v is not None and len(v.ids)}
        if not arms:
            continue
        if item_filter is not None:
            arms = {k: v.subset(np.array([i in item_filter for i in v.ids])) for k, v in arms.items()}
        for v in arms.values():  # mention rates over the analyzed items only
            ids = set(v.ids)
            v.mentions = [m for m in (v.mentions or []) if m["item_id"] in ids] or None
        ref_arm = arms.get("independent") or arms.get("bare") or next(iter(arms.values()))
        ref = float(np.nanmedian(ref_arm.chars[:, 0]))
        for v in arms.values():
            v.long_ref = ref
            v.set_long_reference(ref)
        n_q = len(qindex)
        res = {"arms": {k: arm_summary(v, n_q, rng) for k, v in arms.items()}, "diffs": {}}
        for arm, base in BASELINE_OF.items():
            if arm in arms and base in arms:
                res["diffs"][f"{arm}-{base}"] = diff_summary(arms[arm], arms[base], n_q, rng)
        if "critic" in arms and "bare" in arms:
            res["diffs"]["critic-bare"] = diff_summary(arms["critic"], arms["bare"], n_q, rng)
        if "bare_trace" in arms and "bare" in arms:
            res["diffs"]["bare_trace-bare"] = diff_summary(arms["bare_trace"], arms["bare"], n_q, rng)
        out[mdir.name] = res
    return out


# --------------------------------------------------------------------------
# Transfer
# --------------------------------------------------------------------------

def eval_json_items(path: Path) -> list[dict]:
    """Standard eval JSON -> [{q, mode, honest, strict, correct, mentions}] per rollout."""
    d = json.loads(path.read_text())
    rows = []
    for rec in d["results"]:
        for s in rec["samples"]:
            if s["error"]:
                continue
            rows.append({"q": (rec["dataset"], rec["id"]), "mode": rec["mode"], "honest": is_honest(s),
                         "strict": s["compliance"] == 1, "correct": bool(s["correct"]),
                         "mentions": (s["meta_discussion"] == 0) if s.get("meta_discussion") is not None else None,
                         "chars": len(s["reasoning"] or "")})
    return rows


def analyze_transfer(run_dir: Path) -> dict:
    rng = np.random.default_rng(1)
    troot = run_dir / "transfer"
    out: dict = {}
    if not troot.exists():
        return out
    for mdir in sorted(p for p in troot.iterdir() if p.is_dir()):
        man = json.loads((mdir / "manifest.json").read_text()) if (mdir / "manifest.json").exists() else {}
        mres = {"context_success_fractions": man.get("context_success_fractions"), "configs": {}}
        for cdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            cells = {}
            for cell in sorted(p for p in cdir.iterdir() if p.is_dir()):
                arms: dict[str, list[dict]] = {}
                for f in sorted(cell.glob("*_d*.json")):
                    if f.name.endswith(".meta.json"):
                        continue
                    arm = f.stem.rsplit("_d", 1)[0]
                    arms.setdefault(arm, []).extend(dict(r, draw=f.stem.rsplit("_d", 1)[1]) for r in eval_json_items(f))
                if not arms:
                    continue
                qs = sorted({r["q"] for rs in arms.values() for r in rs})
                qi = {q: i for i, q in enumerate(qs)}
                ws = [boot_weights(len(qs), rng) for _ in range(N_BOOT)]

                def stat(rows, w=None, key="honest", qi=qi):
                    x = np.array([float(r[key]) for r in rows])
                    wi = np.ones(len(rows)) if w is None else w[[qi[r["q"]] for r in rows]]
                    return float((x * wi).sum() / wi.sum()) if wi.sum() else float("nan")
                summ = {}
                for arm, rows in arms.items():
                    b = np.array([stat(rows, w) for w in ws])
                    ment = [r["mentions"] for r in rows if r["honest"] and r["mentions"] is not None]
                    summ[arm] = {"n": len(rows), "draws": sorted({r["draw"] for r in rows}),
                                 "honest": stat(rows), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
                                 "strict": stat(rows, key="strict"), "acc": stat(rows, key="correct"),
                                 "mentions_constraint": (sum(ment) / len(ment)) if ment else None,
                                 "per_draw": {d: float(np.mean([r["honest"] for r in rows if r["draw"] == d]))
                                              for d in sorted({r["draw"] for r in rows})}}
                diffs = {}
                pairs = [("history_bare", "fewshot_matched_bare"), ("history_critic", "fewshot_matched_critic"),
                         ("critic_only", "none"), ("history_bare", "none"), ("history_critic", "none"),
                         ("fewshot_matched_bare", "none"), ("fewshot_matched_critic", "none"), ("fewshot_k1", "none")]
                for a, b_ in pairs:
                    if a in arms and b_ in arms:
                        bs = np.array([stat(arms[a], w) - stat(arms[b_], w) for w in ws])
                        diffs[f"{a}-{b_}"] = {"diff": stat(arms[a]) - stat(arms[b_]),
                                              "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
                cells[cell.name] = {"arms": summ, "diffs": diffs}
            # pooled over cells (one2one: over X and Y; nine2heldout: single cell)
            pooled = pool_cells(cells)
            pooled["bootstrap"] = pooled_bootstrap(cdir, rng)
            mres["configs"][cdir.name] = {"cells": cells, "pooled": pooled}
        out[mdir.name] = mres
    return out


def pooled_bootstrap(cdir: Path, rng: np.random.Generator) -> dict:
    """Pooled (over cells and draws) honest rate per arm and paired
    differences, with a bootstrap over test QUESTIONS (a question's rows in
    every cell/draw/arm move together)."""
    rows_by_arm: dict[str, list[dict]] = {}
    for cell in sorted(p for p in cdir.iterdir() if p.is_dir()):
        for f in sorted(cell.glob("*_d*.json")):
            if f.name.endswith(".meta.json"):
                continue
            arm = f.stem.rsplit("_d", 1)[0]
            rows_by_arm.setdefault(arm, []).extend(dict(r, cell=cell.name) for r in eval_json_items(f))
    if not rows_by_arm:
        return {}
    qs = sorted({r["q"] for rs in rows_by_arm.values() for r in rs})
    qi = {q: i for i, q in enumerate(qs)}
    ws = [boot_weights(len(qs), rng) for _ in range(N_BOOT)]

    def stat(rows, w=None):
        x = np.array([float(r["honest"]) for r in rows])
        wi = np.ones(len(rows)) if w is None else w[[qi[r["q"]] for r in rows]]
        return float((x * wi).sum() / wi.sum()) if wi.sum() else float("nan")
    out = {"arms": {}, "diffs": {}, "n_questions": len(qs)}
    for arm, rows in rows_by_arm.items():
        b = np.array([stat(rows, w) for w in ws])
        out["arms"][arm] = {"n": len(rows), "honest": stat(rows), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
    for a, b_ in [("history_bare", "fewshot_matched_bare"), ("history_critic", "fewshot_matched_critic"),
                  ("critic_only", "none"), ("critic_only", "fewshot_matched_critic"), ("history_critic", "none"),
                  ("history_bare", "none"), ("fewshot_matched_bare", "none"), ("fewshot_matched_critic", "none"), ("fewshot_k1", "none")]:
        if a in rows_by_arm and b_ in rows_by_arm:
            # paired over the cells both arms were run in (a context arm can be missing from a cell when its pool was
            # empty); `none` has no source mode (cells none__Y), so pairs with it are matched on the target mode Y
            def key(r, by_y="none" in (a, b_)):
                return r["cell"].split("__")[-1] if by_y else r["cell"]
            common = {key(r) for r in rows_by_arm[a]} & {key(r) for r in rows_by_arm[b_]}
            ra = [r for r in rows_by_arm[a] if key(r) in common]
            rb = [r for r in rows_by_arm[b_] if key(r) in common]
            bs = np.array([stat(ra, w) - stat(rb, w) for w in ws])
            out["diffs"][f"{a}-{b_}"] = {"diff": stat(ra) - stat(rb), "lo": float(np.percentile(bs, 2.5)),
                                         "hi": float(np.percentile(bs, 97.5)), "n_cells": len(common),
                                         "a": stat(ra), "b": stat(rb)}
    return out


def pool_cells(cells: dict) -> dict:
    """Unweighted mean over cells of each arm's honest rate and of each paired difference."""
    arms: dict[str, list] = {}
    diffs: dict[str, list] = {}
    for c in cells.values():
        for a, s in c["arms"].items():
            arms.setdefault(a, []).append(s["honest"])
        for k, s in c["diffs"].items():
            diffs.setdefault(k, []).append(s["diff"])
    return {"arms": {a: float(np.mean(v)) for a, v in arms.items()},
            "diffs": {k: {"mean": float(np.mean(v)), "n_cells": len(v),
                          "cells_positive": int(sum(x > 0 for x in v))} for k, v in diffs.items()}}


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------

def score_table(arm_data: dict[str, dict[str, ArmData]], n_q_by_model: dict[str, int]) -> dict:
    """Per model: 3A scores at attempt N (+ independent@1 as the stated-eval replica)."""
    rows = {}
    for model, arms in arm_data.items():
        r = {}
        for a, d in arms.items():
            c = cum_curve(d)
            r[f"{a}@N"] = float(c[-1])
            r[f"{a}@1"] = float(c[0])
            r[f"{a}_long@N"] = float(cum_curve(d, long=True)[-1])
            r[f"{a}_strict@1"] = float(cum_curve(d, strict=True)[0])
        for a, b in BASELINE_OF.items():
            if a in arms and b in arms:
                r[f"{a}-{b}@N"] = r[f"{a}@N"] - r[f"{b}@N"]
        rows[model] = r
    return rows


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return pearson(rank(np.asarray(x)), rank(np.asarray(y)))


def correlate(xs: dict[str, float], ys: dict[str, float], rng: np.random.Generator) -> dict:
    models = sorted(m for m in xs if m in ys and not (np.isnan(xs[m]) or np.isnan(ys[m])))
    if len(models) < 3:
        return {"n": len(models), "models": models}
    x = np.array([xs[m] for m in models]); y = np.array([ys[m] for m in models])
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(models), len(models))
        boots.append(pearson(x[idx], y[idx]))
    boots = np.array(boots)
    return {"n": len(models), "models": models, "pearson": pearson(x, y), "spearman": spearman(x, y),
            "pearson_lo": float(np.nanpercentile(boots, 2.5)), "pearson_hi": float(np.nanpercentile(boots, 97.5)),
            "x": x.tolist(), "y": y.tolist()}


def split_half(arm_data: dict[str, dict[str, ArmData]], score: str, rng: np.random.Generator, reps: int = 200) -> dict:
    """Split-half reliability of a 3A score across models: random halves of
    each model's questions, Pearson r between half-scores across models,
    Spearman-Brown corrected. score e.g. 'bare@N' or 'bare-independent@N'."""
    def model_score(arms, mask_fn):
        def val(a):
            d = arms[a]
            return cum_curve(d.subset(mask_fn(d)))[-1]
        if "-" in score:
            a, b = score.split("@")[0].split("-")
            return val(a) - val(b) if a in arms and b in arms else np.nan
        a = score.split("@")[0]
        return val(a) if a in arms else np.nan
    rs = []
    for _ in range(reps):
        halves = {}
        for model, arms in arm_data.items():
            qs = sorted({q for d in arms.values() for q in d.q.tolist()})
            h = list(rng.permutation(qs)[: len(qs) // 2].tolist())
            halves[model] = (model_score(arms, lambda d, h=h: np.isin(d.q, h)),
                             model_score(arms, lambda d, h=h: ~np.isin(d.q, h)))
        ms = [m for m, (a, b) in halves.items() if not (np.isnan(a) or np.isnan(b))]
        if len(ms) >= 3:
            rs.append(pearson([halves[m][0] for m in ms], [halves[m][1] for m in ms]))
    if not rs:
        return {"n_models": len(arm_data)}
    rs = np.array(rs)
    sb = 2 * rs / (1 + rs)
    return {"n_models": len(arm_data), "half_r_mean": float(np.nanmean(rs)), "half_r_lo": float(np.nanpercentile(rs, 2.5)),
            "half_r_hi": float(np.nanpercentile(rs, 97.5)), "spearman_brown": float(np.nanmean(sb))}


def owning_eval_compliance(entry: dict) -> dict | None:
    """Strict + honest compliance on the owning experiment's final stated eval
    (test split, 9 training modes) if its result JSON exists."""
    from retry_correlate import owning_eval_results
    best = None
    for p in owning_eval_results(entry):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"[analyze] skip unreadable {p}: {e}")
            continue
        cfg = d.get("config", {})
        if not isinstance(cfg, dict) or cfg.get("split") != "test" or not cfg.get("state_requirement", True) \
                or cfg.get("allowed_modes") or cfg.get("mode") == "baseline":
            continue  # the stated eval over the training modes only (not held-out, baseline, direct or persona blocks)
        rows = eval_json_items(p)
        if not rows:
            continue
        cand = {"path": str(p), "n": len(rows), "mode": cfg.get("mode"), "strict": float(np.mean([r["strict"] for r in rows])),
                "honest": float(np.mean([r["honest"] for r in rows]))}
        rank = (cand["mode"] in ("all", "random"), cand["n"])  # prefer the mixed-mode block, then the largest
        if best is None or rank > (best["mode"] in ("all", "random"), best["n"]):
            best = cand
    return best


HC_METRICS = ("constrained_success_honest", "constrained_success", "constrained_success_given_read",
              "control_minus_constrained", "suppression_ratio", "control_success")
HC_HALVES = ("suppression_ratio", "constrained_success")  # metrics with _half0/_half1 (episode parity) in the 3B summaries


def hidden_constraint_scores() -> dict:
    """3B scores: kaushik/hidden_constraint/runs/summary.csv (the 3B analysis
    table; primary) plus heldout_*/summary.json for runs that landed after the
    CSV was last written (the two agree to the CSV's 4-decimal rounding).
    Returns {model_key: {"3B:<metric>": mean over scenarios, "3B:<metric>:<scenario>": value, ...},
    "_source": {model_key: file}}. model_key = our label for base models
    (gptoss20b, qwen8b, gptoss120b; the "_api" variants are separate keys) or
    the registry run_name for "ckpt_<run_name>" checkpoints."""
    root = RUNS.parents[1] / "hidden_constraint" / "runs"
    out: dict = {"_files": [], "_source": {}}
    if not root.exists():
        return out
    wanted = set(HC_METRICS) | {f"{m}_half{h}" for m in HC_HALVES for h in (0, 1)}
    csv_path = root / "summary.csv"
    if csv_path.exists():
        import csv
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                if r["metric"] not in wanted:
                    continue
                try:
                    v = float(r["value"])
                except ValueError:
                    continue
                key = r["model_or_checkpoint"].removeprefix("ckpt_")
                out.setdefault(key, {})[f"3B:{r['metric']}:{r['scenario']}"] = v
                out["_source"][key] = csv_path.name
        out["_files"].append(str(csv_path))
    for p in sorted(root.glob("heldout_*/summary.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"[analyze] skip unreadable {p}: {e}")
            continue
        if d.get("slice") != "heldout" or not d.get("scenarios"):
            continue
        dirname = p.parent.name.removeprefix("heldout_")
        # checkpoint runs are named heldout_ckpt_<registry run_name> (their model_label is the generic "modal_adapter")
        key = dirname.removeprefix("ckpt_") if dirname.startswith("ckpt_") else str(d.get("model_label") or dirname)
        if out["_source"].get(key) == csv_path.name:
            continue
        out["_files"].append(str(p))
        row = out.setdefault(key, {})
        for sc, m in d["scenarios"].items():
            for metric in wanted:
                v = (m.get(metric) or {}).get("value") if isinstance(m.get(metric), dict) else m.get(metric)
                if isinstance(v, (int, float)):
                    row[f"3B:{metric}:{sc}"] = float(v)
        out["_source"][key] = str(p.relative_to(root))
    for key, row in out.items():
        if key.startswith("_"):
            continue
        for metric in wanted:
            vals = [v for k, v in row.items() if k.count(":") == 2 and k.split(":")[1] == metric and not np.isnan(v)]
            if vals:
                row[f"3B:{metric}"] = float(np.mean(vals))  # mean over the 3 scenarios
    return out


def split_half_3b(hc: dict, metric: str, models: list[str] | None = None) -> dict:
    """Across-model split-half reliability of a 3B scenario-mean score from the
    3B summaries' episode-parity halves (Spearman-Brown corrected)."""
    ms = [m for m in hc if not m.startswith("_") and (models is None or m in models)
          and f"3B:{metric}_half0" in hc[m] and f"3B:{metric}_half1" in hc[m]]
    if len(ms) < 3:
        return {"n_models": len(ms)}
    r = pearson([hc[m][f"3B:{metric}_half0"] for m in ms], [hc[m][f"3B:{metric}_half1"] for m in ms])
    return {"n_models": len(ms), "half_r": r, "spearman_brown": 2 * r / (1 + r) if r > -1 else float("nan")}


def analyze_correlation(main_run: Path, ckpt_run: Path | None, n: int) -> dict:
    rng = np.random.default_rng(2)
    arm_data: dict[str, dict[str, ArmData]] = {}
    xs: dict[str, dict[str, float]] = {"pinned_t1_strict": {}, "owning_eval_strict": {}, "owning_eval_honest": {},
                                       "standard_eval_strict_mixed": {}}
    meta = {}
    # Base / API models: the main run on its 1000-item subset (test split, 9 modes stratified). Checkpoints: the
    # reduced 3A on the checkpoint item set (test split, 1 seeded mode per question, 500 items). Both are seeded
    # samples of the same test (question, mode) population; their overlap (~100 items) is too small to use alone.
    sub = run_subset(main_run)
    for lb, mcfg in MODELS.items():
        mdir = main_run / lb
        if not mdir.exists():
            continue
        qindex: dict[tuple, int] = {}
        arms = {a.name: load_arm(a, qindex, n) for a in sorted(mdir.iterdir()) if a.is_dir()}
        arms = {k: v for k, v in arms.items() if v is not None and len(v.ids)}
        if sub is not None:
            arms = {k: v.subset(np.array([i in sub for i in v.ids])) for k, v in arms.items()}
        arms = {k: v for k, v in arms.items() if len(v.ids)}
        if arms:
            arm_data[lb] = arms
            meta[lb] = {"kind": "base", "model": mcfg["model"], "items": "main subset",
                        "n_items": max(len(v.ids) for v in arms.values())}
            if mcfg.get("pinned_t1_strict"):
                xs["pinned_t1_strict"][lb] = mcfg["pinned_t1_strict"]["test"]
                xs["standard_eval_strict_mixed"][lb] = mcfg["pinned_t1_strict"]["test"]
    if ckpt_run and ckpt_run.exists():
        from retry_correlate import ckpt_results
        for r in ckpt_results(ckpt_run.name):
            qindex = {}
            sub = next((p for p in r["dir"].iterdir() if p.is_dir() and p.name in MODELS), None)
            if sub is None:
                continue
            arms = {a.name: load_arm(a, qindex, n) for a in sorted(sub.iterdir()) if a.is_dir()}
            arms = {k: v for k, v in arms.items() if v is not None and len(v.ids)}
            if not arms:
                continue
            arm_data[r["run_name"]] = arms
            meta[r["run_name"]] = {"kind": "checkpoint", "items": "checkpoint set", "n_items": max(len(v.ids) for v in arms.values()),
                                   **{k: r["entry"].get(k) for k in ("experiment", "arm", "base_model", "seed", "adapter_path")}}
            own = owning_eval_compliance(r["entry"])
            if own:
                xs["owning_eval_strict"][r["run_name"]] = own["strict"]
                xs["owning_eval_honest"][r["run_name"]] = own["honest"]
                xs["standard_eval_strict_mixed"][r["run_name"]] = own["strict"]
                meta[r["run_name"]]["owning_eval"] = own
    scores = score_table(arm_data, {})
    # Broken adapters: standard-eval accuracy collapsed (e.g. gpt-oss-20b style_sft answers in pirate voice, which
    # defeats ANSWER extraction; kaushik/fragility/results/exp1_summary.md). Flag acc@1 < 50% of the base model's.
    acc1 = {m: float(np.nanmean(arms["independent"].correct[:, 0])) for m, arms in arm_data.items() if "independent" in arms}
    base_acc = {MODELS[lb]["model"].lower(): acc1[lb] for lb in acc1 if lb in MODELS}
    flagged = {}
    for m, info in meta.items():
        if info.get("kind") == "checkpoint" and m in acc1:
            ref = base_acc.get(str(info.get("base_model", "")).lower())
            if ref and acc1[m] < 0.5 * ref:
                flagged[m] = {"acc1": acc1[m], "base_acc1": ref}
    for m, a in acc1.items():
        scores.setdefault(m, {})["acc@1"] = a
        meta.setdefault(m, {})["flagged_broken"] = m in flagged
    hc = hidden_constraint_scores()
    for m, r in hc.items():
        if m.startswith("_"):
            continue
        if m in scores:  # only models/checkpoints that have 3A scores enter the correlation table (scenario means + per scenario)
            scores[m].update({k: v for k, v in r.items() if not k.split(":")[1].endswith(("_half0", "_half1"))})
    for m, r in scores.items():
        if "independent@1" in r:
            xs.setdefault("independent@1_honest", {})[m] = r["independent@1"]
            xs.setdefault("independent_strict@1", {})[m] = r["independent_strict@1"]
    ys_3a = ["bare@N", "block@N", "critic@N", "bare_trace@N", "bare-independent@N", "block-block_once@N",
             "critic-independent@N", "independent@N", "block_once@N", "bare_long@N", "critic_long@N"]
    ys_3b = sorted({k for r in scores.values() for k in r if k.startswith("3B:")})
    for yname in ("bare@N", "block@N", "critic@N", "bare-independent@N", "block-block_once@N"):  # 3A scores as x for 3B
        xs[f"3A {yname}"] = {m: r[yname] for m, r in scores.items() if yname in r}
    ckpts = {m for m, info in meta.items() if info.get("kind") == "checkpoint"}
    corr, corr_unflagged, corr_ckpt = {}, {}, {}
    for xname, xv in xs.items():
        for yname in ys_3a + ys_3b:
            if xname.startswith("3A ") and not yname.startswith("3B:"):
                continue
            yv = {m: r[yname] for m, r in scores.items() if yname in r}
            c = correlate(xv, yv, rng)
            if c.get("n", 0) >= 3:
                corr[f"{yname} vs {xname}"] = c
            c2 = correlate({m: v for m, v in xv.items() if m not in flagged}, {m: v for m, v in yv.items() if m not in flagged}, rng)
            if flagged and c2.get("n", 0) >= 3:
                corr_unflagged[f"{yname} vs {xname}"] = c2
            c3 = correlate({m: v for m, v in xv.items() if m in ckpts and m not in flagged},
                           {m: v for m, v in yv.items() if m in ckpts and m not in flagged}, rng)
            if c3.get("n", 0) >= 3:
                corr_ckpt[f"{yname} vs {xname}"] = c3
    reliability = {s: split_half(arm_data, s, rng) for s in ("bare@N", "block@N", "critic@N", "bare-independent@N", "block-block_once@N")
                   if any(s.split("@")[0].split("-")[0] in arms for arms in arm_data.values())}
    in_table = [m for m in scores if m in arm_data]
    reliability_3b = {f"3B:{m}": {"all_3B_models": split_half_3b(hc, m), "models_in_this_table": split_half_3b(hc, m, in_table)}
                      for m in HC_HALVES}
    return {"models": meta, "scores": scores, "x": xs, "correlations": corr, "correlations_unflagged": corr_unflagged,
            "correlations_checkpoints_only_unflagged": corr_ckpt, "flagged_broken": flagged, "split_half": reliability,
            "split_half_3b": reliability_3b, "hidden_constraint": hc, "primary_pairs": PRIMARY_PAIRS}


# The pairs reported in results.md / plotted (every pair is in results.json). x = standard-eval compliance:
# independent_strict@1 is an exact stated-eval replica on the same backend and items for every model; the "mixed"
# x is the pinned T=1 strict compliance (pinned_reeval, OpenRouter) for base models and the owning experiment's
# final stated test eval for checkpoints. 3B scores are mean over the 3 scenarios.
PRIMARY_PAIRS = [f"{y} vs {x}" for x in ("independent_strict@1", "standard_eval_strict_mixed")
                 for y in ("bare@N", "bare-independent@N", "block@N", "block-block_once@N", "critic@N", "bare_trace@N",
                           "3B:suppression_ratio:config_notation", "3B:suppression_ratio", "3B:constrained_success_given_read",
                           "3B:constrained_success_honest")] + [
    f"{y} vs 3A {x}" for y in ("3B:suppression_ratio:config_notation", "3B:suppression_ratio", "3B:constrained_success_given_read")
    for x in ("bare@N", "bare-independent@N", "block@N")]


# --------------------------------------------------------------------------
# Figures and report
# --------------------------------------------------------------------------

STYLE = {"block": ("tab:red", "-"), "bare": ("tab:blue", "-"), "bare_trace": ("tab:purple", "-"),
         "critic": ("tab:green", "-"), "independent": ("tab:blue", "--"), "block_once": ("tab:red", "--")}


def plot_curves(res3a: dict, fig_dir: Path, n: int) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    ks = np.arange(1, n + 1)
    for model, r in res3a.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for arm, s in r["arms"].items():
            c, ls = STYLE.get(arm, ("gray", ":"))
            axes[0].plot(ks, s["cum_honest"], ls, color=c, marker="o", ms=3, label=arm)
            axes[0].fill_between(ks, s["cum_honest_lo"], s["cum_honest_hi"], color=c, alpha=0.12)
            axes[0].plot(ks, s["cum_honest_long"], ":", color=c, lw=1, alpha=0.8)
            axes[1].plot(ks, s["chars"], ls, color=c, marker="o", ms=3, label=arm)
            axes[2].plot(ks, s["degen"], ls, color=c, marker="o", ms=3, label=arm)
        axes[0].set(title=f"{model}: cumulative honest compliance (dotted: honest & long trace)", xlabel="attempt k", ylabel="fraction of episodes", ylim=(0, 1))
        axes[1].set(title="mean reasoning chars at attempt k", xlabel="attempt k")
        axes[2].set(title=f"degenerate traces (<{DEGEN_CHARS} chars) at attempt k", xlabel="attempt k", ylim=(0, 1))
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        f = fig_dir / f"{model}_curves.png"
        fig.savefig(f, dpi=130); plt.close(fig); files.append(f.name)
        modes = sorted({m for s in r["arms"].values() for m in s["per_mode"]})
        if modes:
            cols = 3
            rows_ = (len(modes) + cols - 1) // cols
            fig, axes = plt.subplots(rows_, cols, figsize=(4.5 * cols, 3.3 * rows_), squeeze=False)
            for ax, m in zip(axes.flat, modes):
                for arm, s in r["arms"].items():
                    if m in s["per_mode"]:
                        c, ls = STYLE.get(arm, ("gray", ":"))
                        ax.plot(ks, s["per_mode"][m]["cum_honest"], ls, color=c, marker="o", ms=2, label=arm)
                ax.set(title=f"{m} (n={s['per_mode'].get(m, {}).get('n', '?')})", ylim=(0, 1), xlabel="attempt k")
            axes.flat[0].legend(fontsize=7)
            for ax in axes.flat[len(modes):]:
                ax.axis("off")
            fig.suptitle(f"{model}: cumulative honest compliance by mode")
            fig.tight_layout()
            f = fig_dir / f"{model}_curves_by_mode.png"
            fig.savefig(f, dpi=120); plt.close(fig); files.append(f.name)
    return files


def plot_correlations(corr: dict, fig_dir: Path, scores: dict, xs: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    for key, c in corr.items():
        if "x" not in c or key not in PRIMARY_PAIRS:
            continue
        yname, xname = key.split(" vs ")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(c["x"], c["y"])
        for m, x, y in zip(c["models"], c["x"], c["y"]):
            ax.annotate(m, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel=xname, ylabel=yname,
               title=f"r={c['pearson']:.2f} [{c['pearson_lo']:.2f}, {c['pearson_hi']:.2f}], rho={c['spearman']:.2f}, n={c['n']}")
        fig.tight_layout()
        f = fig_dir / f"corr_{yname}_vs_{xname}.png".replace("@", "_at_").replace(" ", "")
        fig.savefig(f, dpi=130); plt.close(fig); files.append(f.name)
    return files


def pct(x, lo=None, hi=None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    s = f"{100 * x:.1f}"
    return s if lo is None else f"{s} [{100 * lo:.1f}, {100 * hi:.1f}]"


def num(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0f}"


def report(run_dir: Path, res3a: dict, trans: dict, corr: dict | None, n: int, figs: list[str]) -> str:
    L = [f"# Results: {run_dir.name}", "",
         ("Honest compliance = strict grader passes, an answer is extracted, trace not truncated/short/repetitive "
          "(`cotcontrol.eval.grading.is_honest`). Intervals: 95% bootstrap over questions, paired across arms."), ""]
    for model, r in res3a.items():
        L += [f"## 3A: {model}", "", "Items per arm: " + ", ".join(f"{a}={s['n_items']}" for a, s in r["arms"].items()), "",
              f"Long-trace reference (attempt-1 median chars of the independent arm): {next(iter(r['arms'].values()))['long_ref_chars']:.0f}", "",
              "| arm | " + " | ".join(f"k={k}" for k in REPORT_KS if k <= n) + " | honest&long@N | strict@1 | strict@N | acc on compliant@N | acc@1 | chars@1 | chars@N | chars of compliant attempt | degen@1 | degen@N | trunc@N | mentions constraint (k=1 / k>1) | mentions feedback (k>1) |",
              "|---|" + "---|" * (len([k for k in REPORT_KS if k <= n]) + 13)]
        for arm, s in r["arms"].items():
            cells = [pct(s["cum_honest"][k - 1], s["cum_honest_lo"][k - 1], s["cum_honest_hi"][k - 1]) for k in REPORT_KS if k <= n]
            me = s.get("mentions") or {}
            L.append(f"| {arm} | " + " | ".join(cells) + f" | {pct(s['cum_honest_long'][-1], s['cum_honest_long_lo'][-1], s['cum_honest_long_hi'][-1])} | {pct(s['cum_strict'][0])} | {pct(s['cum_strict'][-1])} | "
                     f"{pct(s['acc_on_compliant'][-1])} | {pct(s['acc_attempt'][0])} | {num(s['chars'][0])} | "
                     f"{num(s['chars'][-1])} | {num(s['chars_on_compliant'])} | {pct(s['degen'][0])} | {pct(s['degen'][-1])} | {pct(s['truncated'][-1])} | "
                     f"{pct(me.get('constraint_attempt1'))} / {pct(me.get('constraint_later'))} | {pct(me.get('feedback_later'))} |")
        L += ["", "Differences at attempt N (cumulative honest, points):", "", "| comparison | " + " | ".join(f"k={k}" for k in REPORT_KS if k <= n) + " | beats baseline at N (CI > 0) | honest&long diff@N | beats (long) |", "|---|" + "---|" * (len([k for k in REPORT_KS if k <= n]) + 3)]
        for name, d in r["diffs"].items():
            cells = [pct(d["diff"][k - 1], d["lo"][k - 1], d["hi"][k - 1]) for k in REPORT_KS if k <= n]
            L.append(f"| {name} | " + " | ".join(cells) + f" | {'yes' if d['beats_baseline_at_N'] else 'no'} (P(diff<=0)={d['p_le_0_at_N']:.3f}) | "
                     f"{pct(d['diff_long'][-1], d['long_lo'][-1], d['long_hi'][-1])} | {'yes' if d['beats_baseline_long_at_N'] else 'no'} |")
        L += ["", "Per mode, cumulative honest at k=1 / k=N:", "", "| mode | " + " | ".join(r["arms"]) + " |", "|---|" + "---|" * len(r["arms"])]
        modes = sorted({m for s in r["arms"].values() for m in s["per_mode"]})
        for m in modes:
            L.append(f"| {m} | " + " | ".join(
                (pct(s["per_mode"][m]["cum_honest"][0]) + " / " + pct(s["per_mode"][m]["cum_honest"][-1])) if m in s["per_mode"] else "-"
                for s in r["arms"].values()) + " |")
        L.append("")
    if trans:
        L += ["## 3A-transfer", ""]
        for model, mres in trans.items():
            L += [f"### {model}", ""]
            fr = mres.get("context_success_fractions") or {}
            if fr:
                L += ["Context pool (pilot run, train split) — per mode: eligible (success after >= 2 attempts) / success / episodes:", ""]
                for src, d in fr.items():
                    L.append(f"- {src}: " + ", ".join(f"{m} {v['eligible']}/{v['success']}/{v['episodes']}" for m, v in sorted(d.items())))
                L.append("")
            pins = MODELS.get(model, {})
            if pins.get("pinned_t1_fewshot_heldout") and "nine2heldout" in mres["configs"]:
                fs = pins["pinned_t1_fewshot_heldout"]
                L += [(f"Pinned T=1 held-out STRICT compliance (pinned_reeval, OpenRouter): baseline {pct(pins['pinned_t1_strict']['heldout'])}, "
                       f"few-shot k=1 s1/s2/s3 {pct(fs['s1'])} / {pct(fs['s2'])} / {pct(fs['s3'])} — compare with the `none` and "
                       "`fewshot_k1` strict columns below (re-run on this backend)."), ""]
            for cname, c in mres["configs"].items():
                pb = c["pooled"].get("bootstrap") or {}
                L += [f"**{cname}** (pooled over cells and draws, bootstrap over {pb.get('n_questions', '?')} questions): "
                      + ", ".join(f"{a}={pct(v['honest'], v['lo'], v['hi'])}" for a, v in pb.get("arms", {}).items()), "",
                      "Pooled differences [CI]: " + "; ".join(f"{k}: {pct(v['diff'], v['lo'], v['hi'])}" for k, v in pb.get("diffs", {}).items()), "",
                      "Cell-mean differences: " + "; ".join(f"{k}: {pct(v['mean'])} ({v['cells_positive']}/{v['n_cells']} cells > 0)" for k, v in c["pooled"]["diffs"].items()), "",
                      "| cell | arm | n | honest [CI] | strict | acc | mentions constraint | per draw |", "|---|---|---|---|---|---|---|---|"]
                for cell, cc in c["cells"].items():
                    for arm, s in cc["arms"].items():
                        L.append(f"| {cell} | {arm} | {s['n']} | {pct(s['honest'], s['lo'], s['hi'])} | {pct(s['strict'])} | {pct(s['acc'])} | "
                                 f"{pct(s['mentions_constraint'])} | {', '.join(f'{d}:{pct(v)}' for d, v in s['per_draw'].items())} |")
                L += ["", "| cell | comparison | diff [CI] |", "|---|---|---|"]
                for cell, cc in c["cells"].items():
                    for k, d in cc["diffs"].items():
                        L.append(f"| {cell} | {k} | {pct(d['diff'], d['lo'], d['hi'])} |")
                L.append("")
    if corr:
        cols = ["acc@1", "independent_strict@1", "independent@N", "bare@N", "bare-independent@N", "block@N", "block_once@N",
                "block-block_once@N", "critic@N", "bare_trace@N", "bare_long@N"]
        hcols = ["3B:suppression_ratio:config_notation", "3B:suppression_ratio", "3B:constrained_success_given_read",
                 "3B:constrained_success_honest", "3B:control_success"]
        rows = [m for m, r in corr["scores"].items() if "independent@N" in r]
        hc = corr.get("hidden_constraint") or {}
        L += ["## Experiment 3 correlation", "",
              (f"Models/checkpoints with 3A scores: {len(rows)} ({sum(corr['models'][m].get('kind') == 'checkpoint' for m in rows)} checkpoints, keyed by "
               "registry run_name). Base models use the main run's 1000-item subset, checkpoints the 500-item checkpoint set (both seeded "
               "test-split samples over the 9 training modes). x = standard-eval compliance: `independent_strict@1` is an exact stated-eval "
               "replica (same backend and items, one sample); `std strict (mixed)` is the pinned T=1 strict compliance for base models "
               "(pinned_reeval, OpenRouter) and the owning experiment's final stated test eval for checkpoints. 3B = mean over its 3 "
               f"scenarios (hidden_constraint `summary.csv`, {sum(v == 'summary.csv' for v in hc.get('_source', {}).values())} models; "
               f"newer runs from `heldout_*/summary.json`, {sum(v != 'summary.csv' for v in hc.get('_source', {}).values())} models)."), "",
              "| model | kind | n items | flagged | std strict (mixed) | " + " | ".join(cols + hcols) + " |",
              "|---|---|---|---|---|" + "---|" * (len(cols) + len(hcols))]
        for m in rows:
            r, info = corr["scores"][m], corr["models"].get(m, {})
            L.append(f"| {m} | {info.get('kind', '?')} | {info.get('n_items', '-')} | {'BROKEN' if info.get('flagged_broken') else ''} | "
                     f"{pct(corr['x']['standard_eval_strict_mixed'].get(m))} | " + " | ".join(pct(r.get(k)) for k in cols + hcols) + " |")
        fl = corr.get("flagged_broken") or {}
        L += ["", "Flagged BROKEN adapters (accuracy at attempt 1 < 50% of the base model's): "
              + (", ".join(f"{m} (acc {100 * v['acc1']:.0f}% vs base {100 * v['base_acc1']:.0f}%)" for m, v in fl.items()) or "none"), "",
              ("Primary correlations: Pearson r [95% bootstrap over models], Spearman rho. `unflagged` drops the BROKEN adapters; "
               "`ckpt only` additionally drops the base/API models. Every pair is in results.json (`correlation.correlations*`)."), "",
              "| y vs x | n | r [CI] | rho | n unflagged | r [CI] | rho | n ckpt only | r [CI] | rho |", "|---|---|---|---|---|---|---|---|---|---|"]

        def fmt(c):
            if not c:
                return "- | - | -"
            ci = f" [{c['pearson_lo']:.2f}, {c['pearson_hi']:.2f}]" if c["n"] >= 5 else " (n<5, no CI)"
            return f"{c['n']} | {c['pearson']:.2f}{ci} | {c['spearman']:.2f}"
        for k in corr.get("primary_pairs", []):
            c = corr["correlations"].get(k)
            if not c:
                continue
            u = (corr.get("correlations_unflagged") or {}).get(k) or (c if not fl else None)
            L.append(f"| {k} | {fmt(c)} | {fmt(u)} | {fmt((corr.get('correlations_checkpoints_only_unflagged') or {}).get(k))} |")
        L += ["", ("Split-half reliability across models (Spearman-Brown corrected). 3A: random question halves, 200 draws; "
                   "3B: the 3B summaries' episode-parity halves (all 3B models / the models in the table above):"), "",
              "| score | n models | half-half r [CI] | Spearman-Brown |", "|---|---|---|---|"]
        for s, r in corr["split_half"].items():
            if "half_r_mean" in r:
                L.append(f"| {s} | {r['n_models']} | {r['half_r_mean']:.2f} [{r['half_r_lo']:.2f}, {r['half_r_hi']:.2f}] | {r['spearman_brown']:.2f} |")
        for s, r in (corr.get("split_half_3b") or {}).items():
            a, b = r["all_3B_models"], r["models_in_this_table"]
            if "half_r" in a:
                L.append(f"| {s} (all / in table) | {a['n_models']} / {b.get('n_models', 0)} | {a['half_r']:.2f} / "
                         + (f"{b['half_r']:.2f}" if "half_r" in b else "-") + f" | {a['spearman_brown']:.2f} / "
                         + (f"{b['spearman_brown']:.2f}" if "spearman_brown" in b else "-") + " |")
        L.append("")
    L += ["## Figures", ""] + [f"- figures/{f}" for f in figs]
    return "\n".join(L) + "\n"


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt-run", default="ckpt")
    ap.add_argument("--transfer-run", default=None, help="run holding transfer/ (default: --run)")
    ap.add_argument("--n-attempts", type=int, default=CFG["n_attempts"])
    ap.add_argument("--no-corr", action="store_true")
    a = ap.parse_args(argv)
    run_dir = RUNS / a.run
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res3a = analyze_3a(run_dir, a.n_attempts)
    trans = analyze_transfer(RUNS / (a.transfer_run or a.run))
    corr = None if a.no_corr else analyze_correlation(run_dir, RUNS / a.ckpt_run, a.n_attempts)
    figs = plot_curves(res3a, fig_dir, a.n_attempts)
    if corr:
        figs += plot_correlations(corr["correlations"], fig_dir, corr["scores"], corr["x"])
    (run_dir / "results.json").write_text(json.dumps({"3a": res3a, "transfer": trans, "correlation": corr}, indent=1, default=str))
    md = report(run_dir, res3a, trans, corr, a.n_attempts, figs)
    (run_dir / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
