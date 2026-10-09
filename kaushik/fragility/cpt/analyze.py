"""Tables and plots for arms 1B (cpt) and 1C (cpt_param_sweep).

Reads the per-checkpoint eval JSONs train_eval.py writes under
runs/<run>/eval/<block>/checkpoint-<step>.json and reports, per rollout,

  honest      cotcontrol.eval.grading.is_honest (strict grader + answer
              extracted + trace not truncated/short/repetitive/compressible)
  strict      the raw strict grader
  correct     answer accuracy
  chars       reasoning length; degen = chars < 200
  compl_long  strict compliance among traces at least as long as the
              untrained (checkpoint-0) median for the same mode

Intervals are 95% bootstrap over questions and seeds (questions resampled as
clusters shared across seeds, seeds resampled with replacement; paired against
checkpoint-0 of the same runs). KL(base || adapter) and noise-control evals
are read from kaushik/noise/runs/<run_name>/ when present.

    .venv/bin/python kaushik/run.py fragility/cpt analyze [--no-plots] [--boot 2000]
Outputs: results/{tables.md, summary.json, rows.csv, dose.png, kcurve.png}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import make_configs as mc

from cotcontrol.eval.grading import is_honest

RUNS = HERE / "runs"
RESULTS = HERE / "results"
NOISE_RUNS = ROOT / "kaushik" / "noise" / "runs"
DEGEN_CHARS = 200
VAL_BLOCKS = [f"val_{m}" for m in mc.VAL_MODES]
K_ORDER = ["1e3", "1e4", "1e5", "r1", "r32"]


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ----------------------------------------------------------------------------- loading

def load_rollouts(path: Path, keep_errors: bool = False) -> list[dict]:
    data = json.loads(path.read_text())
    rows = []
    for rec in data["results"]:
        for s in rec["samples"]:
            if s.get("error") and not keep_errors:
                continue
            chars = len(s.get("reasoning_text_graded") or "")
            rows.append({
                "qid": rec["id"], "mode": rec["mode"], "dataset": rec["dataset"],
                "error": bool(s.get("error")), "honest": is_honest(s),
                "strict": s.get("compliance") == 1, "correct": s.get("correct") is True,
                "answered": bool(s.get("extracted_answer")),  # 'ANSWER: X' present (format intact)
                "chars": chars, "degen": chars < DEGEN_CHARS,
                "truncated": s.get("finish_reason") == "length",
            })
    return rows


def run_meta(rd: Path) -> dict | None:
    for name in ("manifest.json", "config.yaml", "train_config.yaml"):
        p = rd / name
        if not p.exists():
            continue
        if name.endswith(".json"):
            m = json.loads(p.read_text())
            return m if "arm" in m else None
        import yaml
        cfg = yaml.safe_load(p.read_text())
        return {"run_name": rd.name, **cfg.get("cpt_meta", {})} if "cpt_meta" in cfg else None
    return None


def discover_runs() -> list[dict]:
    runs = []
    for rd in sorted(p for p in RUNS.glob("cpt*") if p.is_dir()):
        meta = run_meta(rd)
        if not meta:
            continue
        tr = rd / "train_result.json"
        runs.append({"dir": rd, "name": rd.name, "meta": meta,
                     "train_result": json.loads(tr.read_text()) if tr.exists() else None})
    return runs


def block_rows(rd: Path, block: str, step: int) -> list[dict] | None:
    f = rd / "eval" / block / f"checkpoint-{step}.json"
    return load_rollouts(f) if f.exists() else None


def val_rows(rd: Path, step: int) -> list[dict] | None:
    out = []
    for b in VAL_BLOCKS:
        r = block_rows(rd, b, step)
        if r is None:
            return None
        out += r
    return out


# ----------------------------------------------------------------------------- metrics

def length_bars(base_rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in base_rows:
        by[r["mode"]].append(r["chars"])
    return {m: statistics.median(v) for m, v in by.items()}


def summarize(rows: list[dict], base_rows: list[dict] | None) -> dict:
    bars = length_bars(base_rows) if base_rows else {}
    long = [r for r in rows if r["chars"] >= bars.get(r["mode"], 0)]
    return {
        "n": len(rows), "honest": mean(r["honest"] for r in rows),
        "strict": mean(r["strict"] for r in rows), "accuracy": mean(r["correct"] for r in rows),
        "answered": mean(r["answered"] for r in rows),
        "chars": mean(r["chars"] for r in rows), "degen": mean(r["degen"] for r in rows),
        "truncated": mean(r["truncated"] for r in rows),
        "compl_long": mean(r["strict"] for r in long), "n_long": len(long),
    }


def _matrix(per_seed: list[list[dict]], key: str, keys: list | None = None) -> tuple[np.ndarray, list]:
    """Seeds x items matrix of one metric, items aligned on (dataset, qid, mode)
    (question ids repeat across the three eval datasets, so the dataset is part
    of the key)."""
    maps = [{(r.get("dataset"), r["qid"], r["mode"]): float(r[key]) for r in rows} for rows in per_seed]
    keys = keys or sorted(set.intersection(*[set(m) for m in maps]))
    return np.array([[m[k] for k in keys] for m in maps]), keys


def boot_reps(mat: np.ndarray, n_boot: int, rng: np.random.Generator,
              mat2: np.ndarray | None = None) -> np.ndarray:
    """Bootstrap means (or paired mean differences mat - mat2) over seeds and items."""
    S, n = mat.shape
    reps = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, S, S) if S > 1 else np.zeros(1, dtype=int)
        it = rng.integers(0, n, n)
        a = mat[s][:, it].mean()
        reps[i] = a if mat2 is None else a - mat2[s][:, it].mean()
    return reps


def ci(reps: np.ndarray) -> list[float]:
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def arm_stats(per_seed: list[list[dict]], base_per_seed: list[list[dict]] | None,
              key: str, n_boot: int, seed: int = 0) -> dict:
    """Mean, CI, per-seed spread, and the paired difference to checkpoint-0."""
    rng = np.random.default_rng(seed)
    mat, keys = _matrix(per_seed, key)
    out = {"mean": float(mat.mean()), "ci": ci(boot_reps(mat, n_boot, rng)),
           "per_seed": [float(x) for x in mat.mean(1)], "n_items": len(keys), "n_seeds": mat.shape[0]}
    if base_per_seed:
        bmat, _ = _matrix(base_per_seed, key, keys)
        if bmat.shape == mat.shape:
            d = boot_reps(mat, n_boot, rng, bmat)
            out["delta"] = {"mean": float((mat - bmat).mean()), "ci": ci(d), "reps": d}
    return out


def indep_diff_ci(a_per_seed, b_per_seed, key, n_boot, seed=0) -> dict:
    """Unpaired difference (e.g. arm vs noise controls with their own seeds)."""
    rng = np.random.default_rng(seed)
    ma, _ = _matrix(a_per_seed, key)
    mb, _ = _matrix(b_per_seed, key)
    ra, rb = boot_reps(ma, n_boot, rng), boot_reps(mb, n_boot, rng)
    return {"mean": float(ma.mean() - mb.mean()), "ci": ci(ra - rb)}


def fit_logistic_logk(ks: list[float], ps: list[float], ses: list[float]) -> dict:
    """p(k) = lo + (hi - lo) * sigmoid(b * (log10 k - m)), weighted least squares
    on a (m, b) grid with closed-form (lo, hi)."""
    x, y, w = np.log10(ks), np.asarray(ps), 1.0 / np.maximum(np.asarray(ses), 1e-3) ** 2
    best = None
    for m in np.linspace(x.min() - 1, x.max() + 1, 241):
        for b in np.logspace(-1, 1, 61):
            s = 1 / (1 + np.exp(-b * (x - m)))
            A = np.stack([np.ones_like(s), s], 1) * np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
            sse = float(((A @ coef - y * np.sqrt(w)) ** 2).sum())
            if best is None or sse < best["sse"]:
                best = {"lo": float(coef[0]), "hi": float(coef[0] + coef[1]), "k_half": float(10 ** m),
                        "slope_per_decade": float(b), "sse": sse}
    return best


# ----------------------------------------------------------------------------- noise/KL

def _noise_groups() -> list[tuple[Path, dict]]:
    """kaushik/noise/runs/<group>/results.json for every group (the noise agent
    groups registry entries by arm x model, e.g. fragility-cpt-cpt-step50-gptoss20b)."""
    out = []
    if not NOISE_RUNS.exists():
        return out
    for p in sorted(NOISE_RUNS.glob("*/results.json")):
        try:
            out.append((p.parent, json.loads(p.read_text())))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def find_kl(run_name: str) -> float | None:
    """KL(base || adapter) per token, as measured by the noise agent (results.json
    -> trained_kl[run_name].kl_per_token). run_name carries the registry suffix
    (e.g. '-step50') for non-final checkpoints."""
    for _, res in _noise_groups():
        v = (res.get("trained_kl") or {}).get(run_name)
        if isinstance(v, dict) and isinstance(v.get("kl_per_token"), (int, float)):
            return float(v["kl_per_token"])
        if isinstance(v, (int, float)):
            return float(v)
    return None


def noise_group(run_name: str) -> tuple[Path, dict] | None:
    """The noise group that contains this (suffixed) run name."""
    for d, res in _noise_groups():
        if run_name in (res.get("trained_kl") or {}):
            return d, res
    return None


def noise_rows(group_dir: Path) -> dict[str, list[list[dict]]]:
    """Noise-control eval JSONs of a group (eval/<block>/noise<i>.json), grouped
    by test block, one rollout list per noise adapter."""
    out: dict[str, list[list[dict]]] = {}
    for block in mc.TEST_BLOCKS:
        files = sorted((group_dir / "eval" / block).glob("*.json"))
        if files:
            out[block] = [load_rollouts(f) for f in files]
    return out


# ----------------------------------------------------------------------------- C4 loss (functional size)

LOSS_WINDOW = 3  # each step already averages 48 x 4096 tokens; a longer window would blur the early descent


def load_losses(rd: Path) -> dict[int, float]:
    """Per-step training loss from runs/<run>/losses.jsonl (fetched from the
    Modal volume by `cli.py losses`); {} when absent."""
    f = rd / "losses.jsonl"
    if not f.exists():
        return {}
    return {int(d["step"]): float(d["loss"]) for d in map(json.loads, filter(str.strip, f.read_text().splitlines()))}


def loss_at(rd: Path, step: int, window: int = LOSS_WINDOW) -> float | None:
    """C4 loss of a checkpoint: the mean over the `window` optimizer steps ending
    at `step` (single 48-row steps are noisy). Step 0 is the first step's loss,
    i.e. the untrained model's (LoRA B starts at zero)."""
    by = load_losses(rd)
    if not by:
        return None
    if step == 0:
        return by.get(1)
    xs = [by[s] for s in range(step - window + 1, step + 1) if s in by]
    return mean(xs) if xs else None


def kl_suffix(arm: str, step: int, final_steps: list[int]) -> str:
    """Registry naming: the last final checkpoint is the bare run name, earlier
    final checkpoints get '-step<n>' (cli.py registry)."""
    return "" if arm == "cpt_param_sweep" or step == max(final_steps) else f"-step{step}"


# ----------------------------------------------------------------------------- tables

def fmt(x, pct=True, nd=1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def fmt_ci(st: dict | None, pct=True) -> str:
    if not st:
        return "-"
    return f"{fmt(st['mean'], pct)} [{fmt(st['ci'][0], pct)}, {fmt(st['ci'][1], pct)}]"


def spread(st: dict | None) -> str:
    if not st or len(st["per_seed"]) < 2:
        return "-"
    return f"{fmt(min(st['per_seed']))}–{fmt(max(st['per_seed']))}"


def group_runs(runs: list[dict]) -> dict[tuple, list[dict]]:
    """(model, arm, klabel, lr) -> runs (one per seed), evaluated only."""
    g = defaultdict(list)
    for r in runs:
        m = r["meta"]
        g[(m["model"], m["arm"], m["klabel"], float(m["lr"]))].append(r)
    return g


def val_series(group: list[dict], steps: list[int], n_boot: int) -> dict[int, dict]:
    """Per step: pooled val summary + bootstrap stats for honest/strict/accuracy."""
    base = [val_rows(r["dir"], 0) for r in group]
    base = [b for b in base if b]
    out = {}
    for step in steps:
        per_seed = [rows for r in group if (rows := val_rows(r["dir"], step))]
        if not per_seed:
            continue
        pooled = [x for rows in per_seed for x in rows]
        base_pooled = [x for rows in base for x in rows]
        s = summarize(pooled, base_pooled)
        s["stats"] = {k: arm_stats(per_seed, base if step else None, k, n_boot)
                      for k in ("honest", "strict", "correct")}
        s["n_seeds"] = len(per_seed)
        out[step] = s
    return out


def test_table(group: list[dict], step: int, n_boot: int, base_group: list[dict] | None = None,
               base_step: int = 0) -> dict | None:
    """Final-checkpoint test blocks: training modes, held-out modes, baseline accuracy."""
    out = {}
    base_group = base_group or group
    for block in mc.TEST_BLOCKS:
        per_seed = [rows for r in group if (rows := block_rows(r["dir"], block, step))]
        if not per_seed:
            continue
        base = [rows for r in base_group if (rows := block_rows(r["dir"], block, base_step))]
        pooled = [x for rows in per_seed for x in rows]
        s = summarize(pooled, [x for rows in base for x in rows] if base else None)
        keys = ("correct",) if block == "test_baseline" else ("honest", "strict", "correct")
        paired = base if (base_group is group and step != base_step) else None
        s["stats"] = {k: arm_stats(per_seed, paired, k, n_boot) for k in keys}
        if base and paired is None and not (base_group is group and step == base_step):
            s["stats_vs_base"] = {k: indep_diff_ci(per_seed, base, k, n_boot) for k in keys}
        if block == "test_train_modes":
            by_mode = defaultdict(list)
            for r in pooled:
                by_mode[r["mode"]].append(r)
            s["per_mode"] = {m: {"honest": mean(x["honest"] for x in v), "strict": mean(x["strict"] for x in v),
                                 "accuracy": mean(x["correct"] for x in v), "n": len(v)}
                             for m, v in sorted(by_mode.items())}
        out[block] = s
    return out or None


def per_mode_table(group: list[dict], steps: list[int], n_boot: int, title: str) -> tuple[list[str], dict]:
    """Honest compliance per val mode (incl. EXTRA_VAL_MODES) at each step vs
    checkpoint-0, paired bootstrap; the pooled headline can hide a mode-specific
    effect (gpt-oss moves on repeat_sentences, not on the fixed-trait modes)."""
    modes = mc.VAL_MODES + mc.EXTRA_VAL_MODES
    lines = [f"\n#### {title}", "", "| mode | step 0 | " + " | ".join(f"step {s} [Δ CI]" for s in steps) + " |",
             "|---|---|" + "---|" * len(steps)]
    data = {}
    for m in modes:
        base = [rows for r in group if (rows := block_rows(r["dir"], f"val_{m}", 0))]
        if not base:
            continue
        cells, data[m] = [], {}
        for s in steps:
            per_seed = [rows for r in group if (rows := block_rows(r["dir"], f"val_{m}", s))]
            if not per_seed or len(per_seed) != len(base):
                cells.append("-")
                continue
            st = arm_stats(per_seed, base, "honest", n_boot)
            data[m][s] = strip_reps(st)
            cells.append(f"{fmt(st['mean'])} [{fmt(st['delta']['ci'][0])}, {fmt(st['delta']['ci'][1])}]")
        b = mean(x["honest"] for rows in base for x in rows)
        lines.append(f"| {m} | {fmt(b)} | " + " | ".join(cells) + " |")
    return lines, data


def md_dose(model: str, series: dict[int, dict], lr: float = mc.LR_1B) -> list[str]:
    lines = [f"\n#### 1B dose: {model}, LoRA r32 lr {lr:.1e} (val, 4 fixed-trait modes x 200 q, pooled; honest CI over questions x seeds)",
             "", "| step | seeds | honest [95% CI] | seed spread | Δ vs step 0 [CI] | strict | compl_long | acc | answered | chars | degen |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for step, s in sorted(series.items()):
        h = s["stats"]["honest"]
        lines.append(f"| {step} | {s['n_seeds']} | {fmt_ci(h)} | {spread(h)} | "
                     f"{fmt_ci(h.get('delta')) if h.get('delta') else '-'} | {fmt(s['strict'])} | "
                     f"{fmt(s['compl_long'])} | {fmt(s['accuracy'])} | {fmt(s['answered'])} | {s['chars']:.0f} | {fmt(s['degen'])} |")
    return lines


def md_test(title: str, rows: list[tuple[str, dict | None]]) -> list[str]:
    lines = [f"\n#### {title} (test split: 9 training modes x 500 q, 3 held-out modes x 500 q, baseline 500 q)",
             "", "| arm | seeds | train honest [CI] | Δ vs untrained | train strict | compl_long | train acc | held-out honest [CI] | held-out strict | baseline acc [CI] |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, t in rows:
        if not t:
            lines.append(f"| {label} | - | (pending) | | | | | | | |")
            continue
        tr, ho, ba = t.get("test_train_modes"), t.get("test_heldout"), t.get("test_baseline")
        d = (tr or {}).get("stats", {}).get("honest", {})
        delta = d.get("delta") or (tr or {}).get("stats_vs_base", {}).get("honest")
        seeds = (tr or ho or ba or {}).get("stats", {})
        seeds = next(iter(seeds.values()))["n_seeds"] if seeds else "-"
        lines.append(
            f"| {label} | {seeds} | {fmt_ci(tr['stats']['honest']) if tr else '-'} | {fmt_ci(delta) if delta else '-'} | "
            f"{fmt(tr['strict']) if tr else '-'} | {fmt(tr['compl_long']) if tr else '-'} | "
            f"{fmt(tr['accuracy']) if tr else '-'} | {fmt_ci(ho['stats']['honest']) if ho else '-'} | "
            f"{fmt(ho['strict']) if ho else '-'} | {fmt_ci(ba['stats']['correct']) if ba else '-'} |")
    return lines


def strip_reps(obj):
    if isinstance(obj, dict):
        return {k: strip_reps(v) for k, v in obj.items() if k != "reps"}
    if isinstance(obj, list):
        return [strip_reps(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


# ----------------------------------------------------------------------------- matched functional size

def _cell(label: str, series: str, group: list[dict], step: int, n_boot: int) -> dict | None:
    """One evaluated checkpoint (pooled over the group's seeds): C4 loss, KL, val
    and test metrics with intervals."""
    per_seed = [rows for r in group if (rows := val_rows(r["dir"], step))]
    if not per_seed:
        return None
    losses = [x for r in group if (x := loss_at(r["dir"], step)) is not None]
    kls = [x for r in group if (x := find_kl(r["name"] + kl_suffix(r["meta"]["arm"], step, r["meta"].get("final_steps", [step])))) is not None]
    rep = [x for r in group for x in (block_rows(r["dir"], "val_repeat_sentences", step) or [])]
    c = {"label": label, "series": series, "arm": group[0]["meta"]["arm"], "klabel": group[0]["meta"]["klabel"],
         "lr": float(group[0]["meta"]["lr"]), "step": step, "n_seeds": len(per_seed), "runs": [r["name"] for r in group],
         "loss": mean(losses) if losses else None, "kl": mean(kls) if kls else None,
         "val": {"honest": arm_stats(per_seed, None, "honest", n_boot), "strict": mean(x["strict"] for rows in per_seed for x in rows),
                 "accuracy": arm_stats(per_seed, None, "correct", n_boot)},
         "repeat_honest": mean(x["honest"] for x in rep) if rep else None, "_val_rows": per_seed}
    for block in ("test_train_modes", "test_heldout"):
        ps = [rows for r in group if (rows := block_rows(r["dir"], block, step))]
        if ps:
            c[block] = {"honest": arm_stats(ps, None, "honest", n_boot), "accuracy": mean(x["correct"] for rows in ps for x in rows)}
            c[f"_{block}_rows"] = ps
    return c


def functional_size(model: str, groups: dict, selection: dict, g1b: list[dict], n_boot: int) -> list[dict]:
    """Every evaluated cell of a model on one axis of functional size (C4 loss;
    KL where the noise agent measured it): the untrained model, every 1B
    checkpoint (spec and low-LR variants) and every 1C sweep cell at step 50.
    Each non-spec cell is also compared with the spec-1B checkpoint nearest to it
    in C4 loss (unpaired bootstrap), separately for val (all checkpoints) and
    test (final checkpoints only)."""
    cells = []
    if g1b and (c := _cell("untrained (step 0)", "untrained", g1b, 0, n_boot)):
        cells.append(c)
    for (mdl, arm, _k, lr), g in sorted(groups.items(), key=lambda kv: -kv[0][3]):
        if mdl == model and arm in ("cpt", "cpt_lowlr"):
            tag = "1B r32" if lr == mc.LR_1B else f"1B r32 lr {lr:.1e}"
            cells += [c for s in mc.SAVE_1B if (c := _cell(f"{tag} step {s}", tag, g, s, n_boot))]
    for klabel in mc.KS:
        sel = (selection.get(f"{model}/{klabel}") or {}).get("lr")
        for lr in mc.lr_grid(model, klabel):
            g = groups.get((model, "cpt_param_sweep", klabel, lr), [])
            if g and (c := _cell(f"1C {klabel} lr {lr:.1e}" + (" (selected)" if lr == sel else ""), f"1C {klabel}", g, mc.STEPS_1C, n_boot)):
                cells.append(c)
    ref = [c for c in cells if c["series"] in ("untrained", "1B r32") and c["loss"] is not None]
    for c in cells:
        if c["series"] in ("untrained", "1B r32") or c["loss"] is None:
            continue
        near = min(ref, key=lambda r: abs(math.log(r["loss"]) - math.log(c["loss"])))
        c["nearest_val"] = {"label": near["label"], "loss": near["loss"],
                            "d_honest": indep_diff_ci(c["_val_rows"], near["_val_rows"], "honest", n_boot),
                            "d_accuracy": indep_diff_ci(c["_val_rows"], near["_val_rows"], "correct", n_boot)}
        tref = [r for r in ref if "_test_train_modes_rows" in r]
        if "_test_train_modes_rows" in c and tref:
            near = min(tref, key=lambda r: abs(math.log(r["loss"]) - math.log(c["loss"])))
            c["nearest_test"] = {"label": near["label"], "loss": near["loss"],
                                 "d_honest": indep_diff_ci(c["_test_train_modes_rows"], near["_test_train_modes_rows"], "honest", n_boot),
                                 "d_accuracy": indep_diff_ci(c["_test_train_modes_rows"], near["_test_train_modes_rows"], "correct", n_boot)}
    for c in cells:  # rollout rows are not part of the summary
        for k in [k for k in c if k.startswith("_")]:
            del c[k]
    return cells


def md_functional(model: str, cells: list[dict]) -> list[str]:
    lines = [(f"\n#### Compliance at matched functional size: {model} (every evaluated cell; C4 loss = mean of the {LOSS_WINDOW} steps ending at the checkpoint, "
              "KL(base‖adapter) where measured; 'nearest 1B' = the spec-1B checkpoint closest in C4 loss, unpaired bootstrap)"),
             "", "| cell | seeds | C4 loss | KL | val honest [CI] | val acc | repeat_sent. honest | test honest [CI] | held-out honest | test acc | vs nearest 1B by loss: val Δ honest / Δ acc | vs nearest 1B by loss: test Δ honest / Δ acc |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: -(c["loss"] or 0)):
        tr, ho = c.get("test_train_modes"), c.get("test_heldout")
        nv, nt = c.get("nearest_val"), c.get("nearest_test")
        lines.append(
            f"| {c['label']} | {c['n_seeds']} | {fmt(c['loss'], pct=False, nd=2)} | {fmt(c['kl'], pct=False, nd=3)} | "
            f"{fmt_ci(c['val']['honest'])} | {fmt(c['val']['accuracy']['mean'])} | {fmt(c['repeat_honest'])} | "
            f"{fmt_ci(tr['honest']) if tr else '-'} | {fmt_ci(ho['honest']) if ho else '-'} | {fmt(tr['accuracy']) if tr else '-'} | "
            + (f"{nv['label']} ({nv['loss']:.2f}): {fmt_ci(nv['d_honest'])} / {fmt(nv['d_accuracy']['mean'])}" if nv else "-") + " | "
            + (f"{nt['label']} ({nt['loss']:.2f}): {fmt_ci(nt['d_honest'])} / {fmt(nt['d_accuracy']['mean'])}" if nt else "-") + " |")
    return lines


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    if not runs:
        raise SystemExit(f"no runs under {RUNS}")
    groups = group_runs(runs)
    selection = json.loads((RESULTS / "lr_selection.json").read_text()) if (RESULTS / "lr_selection.json").exists() else {}
    md, summary, csv_rows = ["# fragility/cpt results", ""], {"models": {}}, []

    for model in mc.MODELS:
        S = summary["models"][model] = {}
        md.append(f"\n## {model}")
        # ---- 1B dose (the spec arm at LR_1B; plus any supplementary low-LR variants)
        g1b = groups.get((model, "cpt", "r32", mc.LR_1B), [])
        for (mdl, arm, _kl, lr), g in sorted(groups.items(), key=lambda kv: -kv[0][3]):
            if mdl != model or arm not in ("cpt", "cpt_lowlr"):
                continue
            series = val_series(g, [0] + mc.SAVE_1B, args.boot)
            if not series:
                continue
            tag = "dose_val" if lr == mc.LR_1B else f"dose_val_lr{lr:.1e}"
            md += md_dose(model, series, lr)
            S[tag] = strip_reps(series)
            lines, pm = per_mode_table(g, mc.FINAL_1B, args.boot,
                                       f"1B per val mode: {model}, lr {lr:.1e} (honest, 200 q x {len(g)} seed(s); Δ vs step 0 paired)")
            md += lines
            S[tag.replace("dose_val", "dose_per_mode")] = pm
            if lr != mc.LR_1B:
                rows_lr = [(f"1B r32 lr {lr:.1e} step {s}" + (" (untrained)" if s == 0 else ""),
                            test_table(g, s, args.boot)) for s in [0] + mc.FINAL_1B]
                md += md_test(f"1B supplementary low-LR arm, final checkpoints: {model}, lr {lr:.1e}", rows_lr)
                S[f"test_1b_lr{lr:.1e}"] = {s: strip_reps(t) for (_, t), s in zip(rows_lr, [0] + mc.FINAL_1B) if t}
        series = S.get("dose_val", {})
        for step, s in series.items():
            csv_rows.append({"model": model, "arm": "cpt", "klabel": "r32", "lr": mc.LR_1B, "block": "val_pooled",
                             "step": step, **{k: s[k] for k in ("n", "honest", "strict", "accuracy", "answered", "chars", "degen", "compl_long")}})
        # ---- 1B final test (+ noise)
        test_rows = []
        for step in [0] + mc.FINAL_1B:
            t = test_table(g1b, step, args.boot) if g1b else None
            test_rows.append((f"1B r32 step {step}" + (" (untrained)" if step == 0 else ""), t))
            if t:
                S.setdefault("test_1b", {})[step] = strip_reps(t)
        # KL-matched random-LoRA controls (one noise group per arm x model x step,
        # built by kaushik/noise from the registry entries): arm vs noise, unpaired.
        vs_noise_lines = []
        for step in mc.FINAL_1B:
            suffix = "" if step == max(mc.FINAL_1B) else f"-step{step}"
            grp = next((ng for r in g1b if (ng := noise_group(r["name"] + suffix))), None)
            if not grp:
                continue
            gdir, res = grp
            nr = noise_rows(gdir)
            if not nr:
                continue
            noise_meta = res.get("noise", {})
            noise_kls = [a.get("kl_per_token") or (a.get("kl") or {}).get("kl_per_token") for a in noise_meta.get("adapters", [])]
            t, cmp = {}, {}
            for block, per_adapter in nr.items():
                pooled = [x for rows in per_adapter for x in rows]
                base = [rows for rr in g1b if (rows := block_rows(rr["dir"], block, 0))]
                arm = [rows for rr in g1b if (rows := block_rows(rr["dir"], block, step))]
                s = summarize(pooled, [x for rows in base for x in rows] if base else None)
                keys = ("correct",) if block == "test_baseline" else ("honest", "strict", "correct")
                s["stats"] = {k: arm_stats(per_adapter, None, k, args.boot) for k in keys}
                if base:
                    s["stats_vs_base"] = {k: indep_diff_ci(per_adapter, base, k, args.boot) for k in keys}
                if arm:
                    cmp[block] = {k: indep_diff_ci(arm, per_adapter, k, args.boot) for k in keys}
                t[block] = s
            label = (f"noise, KL-matched to 1B step {step} ({len(next(iter(nr.values())))} adapters, "
                     f"KL {noise_meta.get('target_kl', float('nan')):.3f} target; adapters "
                     + ", ".join(f"{k:.3f}" for k in noise_kls if k is not None) + ")")
            test_rows.append((label, t))
            S.setdefault("noise", {})[step] = {"group": gdir.name, "target_kl": noise_meta.get("target_kl"),
                                               "noise_kls": noise_kls, "table": strip_reps(t), "arm_minus_noise": cmp}
            for block, d in cmp.items():
                vs_noise_lines.append(
                    f"- 1B step {step} − noise, {block}: " + "; ".join(
                        f"{k} {fmt_ci(v)} pts" for k, v in d.items()))
        md += md_test(f"1B final checkpoints: {model}", test_rows)
        if vs_noise_lines:
            md += ["", "Arm minus KL-matched noise controls (unpaired bootstrap over questions x seeds/adapters):"] + vs_noise_lines
        kls = {r["name"]: {s: find_kl(r["name"] + ("" if s == max(mc.FINAL_1B) else f"-step{s}")) for s in mc.FINAL_1B} for r in g1b}
        if any(v is not None for d in kls.values() for v in d.values()):
            md.append("\nKL(base || adapter) from kaushik/noise/runs: " + "; ".join(
                f"{n}: " + ", ".join(f"step {s}: {v:.4f}" for s, v in d.items() if v is not None) for n, d in kls.items()))
            S["kl_1b"] = kls

        # ---- 1C LR sweep (seed 0)
        md += [f"\n#### 1C LR sweep: {model} (seed 0, val pooled, step {mc.STEPS_1C}; guard = final acc ≥ {100 * 0.85:.0f}% of step-0 acc)",
               "", "| k | lr (×pred) | honest@50 | strict@50 | acc 0→50 | compl_long | k_eff | loss | selected |", "|---|---|---|---|---|---|---|---|---|"]
        for klabel in mc.KS:
            sel = selection.get(f"{model}/{klabel}", {})
            for lr in mc.lr_grid(model, klabel):
                g = groups.get((model, "cpt_param_sweep", klabel, lr), [])
                r0 = next((r for r in g if r["meta"]["seed"] == 0), None)
                if not r0:
                    continue
                rows0, rows50 = val_rows(r0["dir"], 0), val_rows(r0["dir"], mc.STEPS_1C)
                tr = r0["train_result"] or {}
                keff = (tr.get("mask_verification") or {}).get("n_changed_inside_mask")
                if rows0 and rows50:
                    s0, s50 = summarize(rows0, None), summarize(rows50, rows0)
                    md.append(f"| {klabel} | {lr:.1e} (×{lr / mc.predicted_lr(model, klabel):.2f}) | {fmt(s50['honest'])} | {fmt(s50['strict'])} | "
                              f"{fmt(s0['accuracy'])}→{fmt(s50['accuracy'])} | {fmt(s50['compl_long'])} | {keff if keff is not None else 'all'} | "
                              f"{tr.get('final_loss', float('nan')):.3f} | {'**yes**' if sel.get('lr') == lr else ''} |")
                    csv_rows.append({"model": model, "arm": "cpt_param_sweep", "klabel": klabel, "lr": lr, "block": "val_pooled",
                                     "step": mc.STEPS_1C, "seed": 0, **{k: s50[k] for k in ("n", "honest", "strict", "accuracy", "answered", "chars", "degen", "compl_long")}})
                else:
                    md.append(f"| {klabel} | {lr:.1e} | (pending) | | | | {keff} | {tr.get('final_loss', float('nan')):.3f} | |")

        # ---- 1C k-curve at step 50 (selected LR, all seeds) + r32 top point
        curve = {}
        for klabel in K_ORDER:
            if klabel == "r32":
                g, lr, step = g1b, mc.LR_1B, mc.STEPS_1C
            else:
                sel = selection.get(f"{model}/{klabel}")
                if not sel:
                    continue
                lr, step = sel["lr"], mc.STEPS_1C
                g = groups.get((model, "cpt_param_sweep", klabel, lr), [])
            per_seed = [rows for r in g if (rows := val_rows(r["dir"], step))]
            base = [rows for r in g if (rows := val_rows(r["dir"], 0))]
            if not per_seed or len(base) != len(per_seed):
                continue
            keffs = [(r["train_result"] or {}).get("mask_verification", {}) or {} for r in g]
            k_nom = mc.adapter_numel(model, 32 if klabel == "r32" else 1) if klabel in ("r1", "r32") else mc.KS[klabel]
            curve[klabel] = {
                "k_nominal": k_nom, "lr": lr, "n_seeds": len(per_seed),
                "k_effective": [k.get("n_changed_inside_mask") for k in keffs],
                "summary": summarize([x for rows in per_seed for x in rows], [x for rows in base for x in rows]),
                "stats": {k: arm_stats(per_seed, base, k, args.boot) for k in ("honest", "strict", "correct")},
                "kl": [find_kl(r["name"] + ("-step50" if klabel == "r32" else "")) for r in g],
                "runs": [r["name"] for r in g],
            }
        if curve:
            md += [f"\n#### 1C compliance vs trainable parameters: {model} (val pooled, step {mc.STEPS_1C}, selected LR, seeds 0-2)",
                   "", "| k | k_eff | lr | seeds | honest [CI] | seed spread | Δ vs untrained [CI] | strict | compl_long | acc | KL(base‖adapter) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for klabel, c in curve.items():
                h, s = c["stats"]["honest"], c["summary"]
                keff = [k for k in c["k_effective"] if k is not None]
                kl = [k for k in c["kl"] if k is not None]
                md.append(f"| {klabel} ({c['k_nominal']:,}) | {int(mean(keff)) if keff else 'all'} | {c['lr']:.1e} | {c['n_seeds']} | {fmt_ci(h)} | {spread(h)} | "
                          f"{fmt_ci(h['delta'])} | {fmt(s['strict'])} | {fmt(s['compl_long'])} | {fmt(s['accuracy'])} | "
                          f"{', '.join(f'{k:.4f}' for k in kl) if kl else '-'} |")
            for klabel, c in curve.items():
                g = [r for r in runs if r["name"] in c["runs"]]
                lines, pm = per_mode_table(g, [mc.STEPS_1C], args.boot,
                                           f"1C per val mode, {model} k={klabel} (lr {c['lr']:.1e}, seeds {c['n_seeds']})")
                md += lines
                S.setdefault("kcurve_per_mode", {})[klabel] = pm
            # logistic fit in log k, if the top point moved at all
            top = curve.get("r32") or curve[list(curve)[-1]]
            warranted = top["stats"]["honest"]["delta"]["ci"][0] > 0 and len(curve) >= 3
            fit = None
            if warranted:
                pts = [(c["k_nominal"], c["stats"]["honest"]["mean"],
                        (c["stats"]["honest"]["ci"][1] - c["stats"]["honest"]["ci"][0]) / 3.92 or 1e-3) for c in curve.values()]
                fit = fit_logistic_logk(*zip(*pts))
                md.append(f"\nLogistic fit in log10 k: floor {fmt(fit['lo'])}%, ceiling {fmt(fit['hi'])}%, "
                          f"half-rise at k ≈ {fit['k_half']:.3g}, slope {fit['slope_per_decade']:.2f}/decade.")
            else:
                md.append("\nLogistic fit not warranted: the rank-32 top point does not beat checkpoint-0 (CI includes 0).")
            claim = curve.get("1e4")
            if claim:
                d = claim["stats"]["honest"]["delta"]
                frac, r32_note = None, ""
                if "r32" in curve:
                    r32d = curve["r32"]["stats"]["honest"]["delta"]
                    if r32d["ci"][0] > 0:  # a fraction only makes sense if the rank-32 effect exists
                        ratio = d["reps"] / np.where(np.abs(r32d["reps"]) < 1e-3, np.nan, r32d["reps"])
                        frac = {"mean": float(np.nanmean(ratio)), "ci": [float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))]}
                    else:
                        r32_note = f"; the full rank-32 arm itself shows no effect here ({fmt_ci(r32d)} pts)"
                verdict = ("CONSISTENT with 'gone at ~1e4'" if d["ci"][1] <= 0.02 else
                           "NOT consistent: a clear effect remains at 1e4" if d["ci"][0] > 0 else "inconclusive (CI spans 0 and >2 pts)")
                md.append(f"\nClaim check, {model}: Δ honest compliance at k=1e4 vs untrained = {fmt_ci(d)} pts"
                          + (f", i.e. {fmt(frac['mean'], nd=0)}% [{fmt(frac['ci'][0], nd=0)}, {fmt(frac['ci'][1], nd=0)}] of the rank-32 effect" if frac else r32_note)
                          + f" → {verdict}.")
                S["claim_1e4"] = {"delta": strip_reps(d), "fraction_of_r32": frac, "verdict": verdict}
            S["kcurve"] = strip_reps(curve)
            S["logistic_fit"] = fit
            # ---- 1C final test for selected runs
            rows = []
            for klabel, c in curve.items():
                if klabel == "r32":
                    continue
                g = [r for r in runs if r["name"] in c["runs"]]
                t = test_table(g, mc.STEPS_1C, args.boot, base_group=g1b, base_step=0) if g1b else test_table(g, mc.STEPS_1C, args.boot)
                rows.append((f"1C {klabel} lr {c['lr']:.1e}", t))
                if t:
                    S.setdefault("test_1c", {})[klabel] = strip_reps(t)
            if rows:
                md += md_test(f"1C final checkpoints (step {mc.STEPS_1C}): {model}; Δ vs the 1B runs' checkpoint-0 on the same items", rows)
        # ---- every cell on one functional-size axis (C4 loss; KL where measured)
        cells = functional_size(model, groups, selection, g1b, args.boot)
        if cells:
            md += md_functional(model, cells)
            S["functional_size"] = strip_reps(cells)
            for c in cells:
                csv_rows.append({"model": model, "arm": c["arm"], "klabel": c["klabel"], "lr": c["lr"], "block": "functional_size",
                                 "step": c["step"], "seed": c["n_seeds"], "c4_loss": c["loss"], "kl": c["kl"],
                                 "honest": c["val"]["honest"]["mean"], "strict": c["val"]["strict"], "accuracy": c["val"]["accuracy"]["mean"],
                                 "test_honest": (c.get("test_train_modes") or {}).get("honest", {}).get("mean")})

    (RESULTS / "tables.md").write_text("\n".join(md) + "\n")
    (RESULTS / "summary.json").write_text(json.dumps(strip_reps(summary), indent=2, default=str))
    if csv_rows:
        with (RESULTS / "rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in csv_rows for k in r}))
            w.writeheader()
            w.writerows(csv_rows)
    print("\n".join(md))
    if not args.no_plots:
        plots(summary)
    print(f"\nwrote {RESULTS}/{{tables.md, summary.json, rows.csv}}")


def plots(summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in mc.MODELS if summary["models"].get(m, {}).get("dose_val")]
    if models:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
        for ax, m in zip(axes[0], models):
            ser = summary["models"][m]["dose_val"]
            steps = sorted(int(s) for s in ser)
            for key, style, label in (("honest", "o-", "honest compliance"), ("strict", "s--", "strict compliance"),
                                      ("correct", "^:", "accuracy")):
                ys = [ser[str(s) if str(s) in ser else s]["stats"][key]["mean"] for s in steps]
                lo = [ser[str(s) if str(s) in ser else s]["stats"][key]["ci"][0] for s in steps]
                hi = [ser[str(s) if str(s) in ser else s]["stats"][key]["ci"][1] for s in steps]
                ax.plot(steps, ys, style, label=label)
                ax.fill_between(steps, lo, hi, alpha=0.15)
            ax.axvline(50, color="0.7", ls=":", lw=1)
            ax.set_title(f"1B cpt (C4, LoRA r32): {m}")
            ax.set_xlabel("optimizer step (48 x 4096-token sequences each)")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / "dose.png", dpi=150)
    models = [m for m in mc.MODELS if summary["models"].get(m, {}).get("kcurve")]
    if models:
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
        for ax, m in zip(axes[0], models):
            curve = summary["models"][m]["kcurve"]
            ks = [curve[k]["k_nominal"] for k in curve]
            for key, style, label in (("honest", "o-", "honest"), ("strict", "s--", "strict")):
                ys = [curve[k]["stats"][key]["mean"] for k in curve]
                err = np.array([[curve[k]["stats"][key]["mean"] - curve[k]["stats"][key]["ci"][0],
                                 curve[k]["stats"][key]["ci"][1] - curve[k]["stats"][key]["mean"]] for k in curve]).T
                ax.errorbar(ks, ys, yerr=err, fmt=style, capsize=3, label=label)
            base = [curve[k]["stats"]["honest"]["mean"] - curve[k]["stats"]["honest"]["delta"]["mean"] for k in curve]
            ax.axhline(mean(base), color="0.5", ls="--", lw=1, label="untrained (ckpt-0)")
            ax.set_ylim(0, max(0.05, ax.get_ylim()[1]))  # flat-at-zero curves should look flat, not noisy
            fit = summary["models"][m].get("logistic_fit")
            if fit:
                xs = np.logspace(math.log10(min(ks)) - 0.3, math.log10(max(ks)) + 0.3, 200)
                ax.plot(xs, fit["lo"] + (fit["hi"] - fit["lo"]) / (1 + np.exp(-fit["slope_per_decade"] * (np.log10(xs) - math.log10(fit["k_half"])))),
                        color="0.3", lw=1, label="logistic fit")
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels([f"{k}\n({curve[k]['k_nominal']:.1e})" for k in curve], fontsize=8)
            ax.set_title(f"1C cpt_param_sweep @ step {mc.STEPS_1C}: {m}")
            ax.set_xlabel("trainable LoRA scalars k (rank-1 masked; r1/r32 = full adapters)")
            ax.set_ylabel("val compliance (4 modes x 200 q, 3 seeds)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / "kcurve.png", dpi=150)
    models = [m for m in mc.MODELS if summary["models"].get(m, {}).get("functional_size")]
    if models:
        plot_losscurve(summary, models, plt)
    print(f"saved plots to {RESULTS}")


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order


def plot_losscurve(summary: dict, models: list[str], plt) -> None:
    """Compliance and accuracy against C4 loss (functional size) for every
    evaluated cell: rows = val honest (all checkpoints), test honest (final
    checkpoints), val accuracy; one column per model; x inverted so training
    moves right."""
    panels = [("val", "honest", "honest compliance, val (4 modes × 200 q)"),
              ("test_train_modes", "honest", "honest compliance, test (9 modes × 500 q; final checkpoints)"),
              ("val", "accuracy", "accuracy under constraint, val")]
    fig, axes = plt.subplots(len(panels), len(models), figsize=(6.2 * len(models), 3.6 * len(panels)), squeeze=False)
    for col, m in enumerate(models):
        cells = [c for c in summary["models"][m]["functional_size"] if c["loss"] is not None]
        series = []
        for c in cells:  # first-appearance order: 1B spec, low-LR variants, 1C by k
            if c["series"] not in series and c["series"] != "untrained":
                series.append(c["series"])
        for row, (block, key, title) in enumerate(panels):
            ax = axes[row][col]
            for i, s in enumerate(series):
                pts = sorted((c for c in cells if c["series"] == s and c.get(block)), key=lambda c: -c["loss"])
                if not pts:
                    continue
                xs = [c["loss"] for c in pts]
                st = [c[block][key] for c in pts]
                ys = [v["mean"] for v in st]
                err = np.array([[v["mean"] - v["ci"][0], v["ci"][1] - v["mean"]] for v in st]).T
                color = SERIES_COLORS[i % len(SERIES_COLORS)]
                ax.errorbar(xs, ys, yerr=err, fmt="-" if s.startswith("1B") else "none", lw=1.2, capsize=2, color=color, label=s)
                filled = [s.startswith("1B") or "selected" in c["label"] for c in pts]  # hollow = unselected LR rung
                for x, y, f in zip(xs, ys, filled):
                    ax.plot([x], [y], "o", ms=5, color=color, mfc=color if f else "white", mew=1.2)
            for c in cells:
                if c["series"] == "untrained" and c.get(block):
                    ax.plot([c["loss"]], [c[block][key]["mean"]], "x", color="black", ms=8, mew=1.5, label="untrained")
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
            ax.set_xscale("log")
            ax.invert_xaxis()
            lo, hi = min(c["loss"] for c in cells), max(c["loss"] for c in cells)
            ax.xaxis.set_major_locator(FixedLocator(sorted({float(f"{t:.2g}") for t in np.geomspace(lo, hi, 6)})))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if key == "honest":
                ax.set_ylim(0, max(0.05, ax.get_ylim()[1]))
            else:
                ax.set_ylim(0, 0.6)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
            ax.set_title(f"{m}: {title}", fontsize=9)
            ax.set_xlabel("C4 loss at the checkpoint (← untrained · fitted →)")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.text(0.5, 0.005, "Error bars: 95% bootstrap over questions × seeds. Filled = 1B checkpoints and the selected 1C learning rate (3 seeds); "
             "hollow = the other LR rungs of the sweep (seed 0). x = untrained model.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(RESULTS / "losscurve.png", dpi=150)


if __name__ == "__main__":
    main()
