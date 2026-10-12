"""Aggregate Experiment 2B runs into tables and plots (config `analysis`; outputs results/).

Reads the full per-checkpoint eval JSONs (runs/<run>/eval/<block>/checkpoint-<step>.json
and runs/<run>/live/ckpt-*/part*/eval/... on val, runs/<run>/final/... on test) and scores
every rollout with the shared metrics (kaushik/core/results.py): honest compliance
(cotcontrol.eval.grading.is_honest), raw strict compliance, accuracy, chars (mean reasoning
length), degen (fraction of traces under analysis.degen_chars) and compl_long (strict
compliance among traces at least as long as the same run's checkpoint-0 median, as
orange-training/analyze.py). The quantity of interest is (rl_orange - rl_mcq) per reasoning
variant on the persona / direct / stated conditions, with 95% bootstrap intervals over
questions (clustered by question across modes) and seeds (kaushik/core/stats.py:
boot_interleaved, the committed tables' draw order), next to untrained (checkpoint-0 of
every run) and noise (kaushik/noise/runs/). Training curves come from progress.jsonl
(reward, held-out reward, truncation, short traces) and the worker's metrics.jsonl (groups
with gradient, loss, logprob agreement).

    .venv/bin/python kaushik/run.py orange/rl analyze [--n-boot 2000] [--no-plots] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import stats as st
from kaushik.core.config import defaults, resolve, results_dir
from kaushik.core.io import read_json, read_jsonl
from kaushik.core.results import parse_eval_json

EXP = "orange/rl"
RUNS, RESULTS = HERE / "runs", HERE / "results"          # defaults; config.yaml analysis.runs / results_dir win
NOISE_RUNS = ROOT / "kaushik" / "noise" / "runs"
AN = defaults()["analysis"]                              # bootstrap draws / seed / degen_chars defaults


def block_parts(block: str) -> tuple[str, str]:
    if block == "baseline":
        return "baseline", "baseline"
    cond, _, mode = block.partition("_")
    return cond, mode


def load_items(path: Path, degen_chars: int = int(AN["degen_chars"])) -> dict[str, dict]:
    """eval JSON -> {question id: per-rollout metrics} (one rollout per task)."""
    return {r["qid"]: {"honest": float(r["honest"]), "raw": float(r["compliance"]), "correct": float(r["correct"]),
                       "chars": float(r["chars"]), "degen": float(r["chars"] < degen_chars), "mode": r["mode"]}
            for r in parse_eval_json(path)}


def load_rows(runs_dir: Path = RUNS, degen_chars: int = int(AN["degen_chars"])) -> list[dict]:
    rows = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()
                          and not p.name.startswith("_") and "smoke" not in p.name):
        if not (run_dir / "config.yaml").exists():
            continue
        raw = yaml.safe_load((run_dir / "config.yaml").read_text())
        base = {"run": run_dir.name, "arm": raw["task"]["arm"], "seed": raw["train"]["seed"],
                "model": raw["base_model"], "variant": "reason" if raw["task"]["reasoning"] else "noreason"}
        # Two layouts: the legacy single-process one (eval/<block>/, final/eval/<block>/)
        # and the per-checkpoint / parallel-part one (live/ckpt-*/part*/eval/<block>/,
        # final/part*/eval/<block>/). Newest file wins for a duplicate (block, step).
        for split, globs in (("val", ("eval/*/checkpoint-*.json", "live/ckpt-*/part*/eval/*/checkpoint-*.json")),
                             ("test", ("final/eval/*/checkpoint-*.json", "final/part*/eval/*/checkpoint-*.json"))):
            found: dict[tuple, Path] = {}
            for g in globs:
                for path in run_dir.glob(g):
                    key = (path.parent.name, path.stem)
                    if key not in found or path.stat().st_mtime > found[key].stat().st_mtime:
                        found[key] = path
            for (block, stem), path in sorted(found.items()):
                step = stem.replace("checkpoint-", "")
                cond, mode = block_parts(block)
                rows.append({**base, "split": split, "block": block, "condition": cond,
                             "mode": mode, "step": int(step) if step.isdigit() else step,
                             "items": load_items(path, degen_chars)})
    return rows


def noise_groups(our_runs: set[str], noise_runs: Path = NOISE_RUNS) -> dict[Path, tuple[str, list[str]]]:
    """kaushik/noise/runs/<group>/ -> (arm, our run names it controls for). A group
    is matched through its manifest.json (group.entries[].run_name) or
    noise_adapters.json (trained_kl keys); a dir name embedding one of our run
    names is the fallback. The group dirs are named by the noise agent
    (e.g. orange-rl-rl_mcq_noreason-qwen3627b), not after our runs."""
    groups = {}
    if not noise_runs.is_dir():
        return groups
    for d in sorted(p for p in noise_runs.iterdir() if p.is_dir() and not p.name.startswith("_")):
        man = _read_json(d / "manifest.json") or {}
        g = man.get("group") or {}
        names = [e.get("run_name") for e in g.get("entries", []) if isinstance(e, dict)]
        adapters = _read_json(d / "noise_adapters.json") or {}
        names += list((adapters.get("trained_kl") or {}).keys())
        names = [n for n in names if n in our_runs] or [r for r in our_runs if r in d.name]
        if not names:
            continue
        arm = g.get("arm") or (names[0].split("-")[1] if names[0].startswith("rl2b-") else names[0])
        groups[d] = (arm, sorted(set(names)))
    return groups


def _read_json(path: Path):
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def load_noise_rows(our_runs: set[str], noise_runs: Path = NOISE_RUNS,
                    degen_chars: int = int(AN["degen_chars"])) -> list[dict]:
    """The noise agent's controls: runs/<group>/eval/<block>/noise<k>.json, one
    file per random adapter k (= one noise seed) of arm 'noise:<arm>'."""
    rows = []
    for d, (arm, _) in noise_groups(our_runs, noise_runs).items():
        for path in sorted(d.rglob("*.json")):
            data = _read_json(path)
            if not (isinstance(data, dict) and "results" in data and "config" in data):
                continue
            cond, mode = block_parts(path.parent.name)
            rows.append({"run": f"{d.name}/{path.stem}", "arm": f"noise:{arm}", "seed": path.stem,
                         "model": data["config"]["model"],
                         "variant": "reason" if "reason" in arm and "noreason" not in arm else "noreason",
                         "split": data["config"].get("split") or "test", "block": path.parent.name,
                         "condition": cond, "mode": mode, "step": "noise", "items": load_items(path, degen_chars)})
    return rows


def kl_table(rows: list[dict], noise_runs: Path = NOISE_RUNS) -> list[dict]:
    """KL(base || adapter) per token of every final adapter (noise agent's
    measurement, kaushik/noise/runs/_kl/<run>.json: 200 unconstrained val prompts,
    T=1 base-model samples) next to the matched noise adapters' target and
    achieved KL. The functional-size axis on which arms should be compared."""
    runs = {r["run"]: (r["arm"], r["seed"]) for r in rows if not r["arm"].startswith("noise:")}
    groups = {arm: d for d, (arm, _) in noise_groups(set(runs), noise_runs).items()}
    out = []
    for arm in sorted({a for a, _ in runs.values()}):
        per_seed = []
        for run, (a, seed) in sorted(runs.items(), key=lambda kv: str(kv[1][1])):
            if a != arm:
                continue
            kl = ((_read_json(noise_runs / "_kl" / f"{run}.json") or {}).get("kl") or {})
            if kl.get("kl_per_token") is not None:
                per_seed.append({"run": run, "seed": seed, "kl": kl["kl_per_token"], "se": kl.get("se_prompt")})
        entry = {"arm": arm, "per_seed": per_seed,
                 "median": float(np.median([p["kl"] for p in per_seed])) if per_seed else float("nan"),
                 "noise_target": float("nan"), "noise_achieved": []}
        if arm in groups:
            ad = _read_json(groups[arm] / "noise_adapters.json") or {}
            entry["noise_group"] = groups[arm].name
            entry["noise_target"] = float(ad.get("target_kl", float("nan")))
            entry["noise_achieved"] = [float((x.get("kl") or {}).get("kl_per_token") or float("nan"))
                                       for x in ad.get("adapters", [])]
        if per_seed or arm in groups:
            out.append(entry)
    return out


# --- aggregation ------------------------------------------------------------


def long_bars(rows: list[dict]) -> dict:
    """Checkpoint-0 median reasoning length per (run, split, block), the bar for
    compl_long; falls back to the pooled step-0 median per (arm, split, block) and
    then per (split, block) over every arm — checkpoint-0 is the same base model
    for every run (on test only rl_mcq_noreason's checkpoint-0 was evaluated)."""
    per_run, by_arm, by_all = {}, defaultdict(list), defaultdict(list)
    for r in rows:
        if r["step"] == 0 and r["items"]:
            med = statistics.median(it["chars"] for it in r["items"].values())
            per_run[(r["run"], r["split"], r["block"])] = med
            by_arm[(r["arm"], r["split"], r["block"])].append(med)
            by_all[(r["split"], r["block"])].append(med)
    return {"run": per_run, "arm": {k: statistics.median(v) for k, v in by_arm.items()},
            "all": {k: statistics.median(v) for k, v in by_all.items()}}


def bar_for(bars: dict, r: dict) -> float | None:
    """compl_long bar of an eval row: its run's own checkpoint-0 median, else its
    arm's (for noise:<arm> rows, the trained arm's), else the all-arm step-0 median."""
    return (bars["run"].get((r["run"], r["split"], r["block"]))
            or bars["arm"].get((r["arm"].split(":", 1)[-1], r["split"], r["block"]))
            or bars["all"].get((r["split"], r["block"])))


def agg(items: dict[str, dict], bar: float | None, metrics: list[str]) -> dict:
    vals = list(items.values())
    if not vals:
        return {"n": 0}
    out = {"n": len(vals), **{m: float(np.mean([v[m] for v in vals])) for m in metrics}}
    long = [v for v in vals if v["chars"] >= bar] if bar is not None else []
    out["n_long"] = len(long)
    out["compl_long"] = float(np.mean([v["raw"] for v in long])) if long else float("nan")
    out["long_bar"] = bar
    return out


def pooled_matrix(rows: list[dict], metric: str) -> tuple[list, np.ndarray]:
    """rows of ONE arm/split/condition/step across seeds and modes -> [S, Q]:
    per seed, each question's mean over the modes it appears in. Questions are
    the intersection across seeds, so bootstrap draws are paired on questions."""
    by_seed: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for qid, it in r["items"].items():
            by_seed[r["seed"]][qid].append(it[metric])
    seeds = sorted(by_seed, key=str)
    qids = sorted(set.intersection(*(set(by_seed[s]) for s in seeds))) if seeds else []
    M = np.array([[np.mean(by_seed[s][q]) for q in qids] for s in seeds], dtype=float)
    return qids, M


def bootstrap(MA: np.ndarray, MB: np.ndarray | None = None, n_boot: int = int(AN["bootstrap_draws"]),
              seed: int = int(AN["bootstrap_seed"])) -> dict:
    """Mean (or MA−MB difference) with a 95% interval, resampling seeds (per arm,
    with replacement) and questions (shared draw, with replacement); fresh rng per call."""
    if MA.size == 0 or (MB is not None and MB.size == 0):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    stats = st.boot_interleaved(MA, MB, np.random.default_rng(seed), n_boot)
    point = MA.mean() - (MB.mean() if MB is not None else 0.0)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(point), "lo": float(lo), "hi": float(hi), "n_seeds": MA.shape[0], "n_questions": MA.shape[1]}


def seed_means(rows: list[dict], metric: str) -> dict:
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].extend(it[metric] for it in r["items"].values())
    return {str(s): float(np.mean(v)) for s, v in by_seed.items()}


def select(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def final_step(rows: list[dict], arm: str) -> int | None:
    steps = [r["step"] for r in rows if r["arm"] == arm and isinstance(r["step"], int)]
    return max(steps) if steps else None


def summarize(rows: list[dict], n_boot: int, variants: dict[str, list[str]], conditions: list[str],
              metrics: list[str], standard_blocks: list[str], seed: int) -> dict:
    """Headline + per-mode comparisons per split/variant/condition (modes
    pooled), untrained (step 0 of every run) and noise as reference rows."""
    bars = long_bars(rows)
    out = {"headline": [], "modes": [], "standard": [], "curves": []}
    modes = sorted({r["mode"] for r in rows if r["condition"] in conditions and r["mode"] not in ("standard", "heldout")})
    untrained = [{**r, "arm": "untrained", "seed": r["run"]} for r in rows if r["step"] == 0]

    def boot(MA, MB=None):
        return bootstrap(MA, MB, n_boot=n_boot, seed=seed)

    def arm_stats(sub, label):
        m = {metric: pooled_matrix(sub, metric)[1] for metric in metrics}
        bar_rows = [agg(r["items"], bar_for(bars, r), metrics) for r in sub]
        long_vals = [b["compl_long"] for b in bar_rows if b["n_long"]]
        return {"arm": label, "n_seeds": len({r["seed"] for r in sub}), "n_rows": len(sub),
                **{metric: boot(m[metric]) for metric in metrics},
                "seed_means_honest": seed_means(sub, "honest"),
                "compl_long": float(np.mean(long_vals)) if long_vals else float("nan"),
                "_M": m}

    for split in ("val", "test"):
        for variant, (orange, mcq) in variants.items():
            fs = {a: final_step(select(rows, split=split), a) for a in (orange, mcq)}
            for cond in conditions:
                for mode_set, table in ((modes, "headline"), *[([m], "modes") for m in modes]):
                    def pick(arm, step, src=rows, _s=split, _c=cond, _m=mode_set):
                        return [r for r in select(src, split=_s, condition=_c, arm=arm)
                                if r["step"] == step and r["mode"] in _m]
                    entry = {"split": split, "variant": variant, "condition": cond,
                             "modes": mode_set if table == "headline" else mode_set[0], "arms": []}
                    stats = {}
                    for arm in (orange, mcq):
                        sub = pick(arm, fs[arm])
                        if sub:
                            stats[arm] = arm_stats(sub, arm)
                            stats[arm]["step"] = fs[arm]
                    if orange in stats and mcq in stats:
                        entry["diff"] = {metric: boot(stats[orange]["_M"][metric], stats[mcq]["_M"][metric])
                                         for metric in metrics}
                    ref_un = pick("untrained", 0, untrained)
                    if ref_un:
                        stats["untrained"] = arm_stats(ref_un, "untrained")
                    for noise_arm in sorted({r["arm"] for r in rows if r["arm"].startswith("noise:")
                                             and r["arm"].split(":", 1)[1] in (orange, mcq)}):
                        sub = [r for r in select(rows, split=split, condition=cond, arm=noise_arm) if r["mode"] in mode_set]
                        if sub:
                            stats[noise_arm] = arm_stats(sub, noise_arm)
                    entry["arms"] = [{k: v for k, v in s.items() if k != "_M"} for s in stats.values()]
                    if entry["arms"]:
                        out[table].append(entry)
    for block in standard_blocks:   # test blocks of the final checkpoint
        for arm in sorted({r["arm"] for r in rows} | {"untrained"}):
            src = untrained if arm == "untrained" else rows
            step = 0 if arm == "untrained" else final_step(select(rows, split="test"), arm)
            sub = [r for r in select(src, split="test", block=block, arm=arm) if r["step"] == step]
            if sub:
                s = arm_stats(sub, arm)
                out["standard"].append({"block": block, "step": step, **{k: v for k, v in s.items() if k != "_M"}})
    for arm in sorted({r["arm"] for r in rows}):   # curves: per (arm, condition, step) pooled over modes
        for cond in conditions:
            steps = sorted({r["step"] for r in select(rows, split="val", arm=arm, condition=cond)
                            if isinstance(r["step"], int)})
            for step in steps:
                sub = [r for r in select(rows, split="val", arm=arm, condition=cond, step=step) if r["mode"] in modes]
                sm = {metric: seed_means(sub, metric) for metric in metrics}
                out["curves"].append({"arm": arm, "condition": cond, "step": step,
                                      **{metric: {"mean": float(np.mean(list(v.values()))),
                                                  "min": min(v.values()), "max": max(v.values()),
                                                  "n_seeds": len(v)} for metric, v in sm.items()}})
    return out


# --- training curves ------------------------------------------------------------


def load_training(runs_dir: Path = RUNS) -> list[dict]:
    out = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()
                          and not p.name.startswith("_") and "smoke" not in p.name):
        if not (run_dir / "config.yaml").exists():
            continue
        raw = yaml.safe_load((run_dir / "config.yaml").read_text())
        prog, met = read_jsonl(run_dir / "progress.jsonl"), read_jsonl(run_dir / "metrics.jsonl")
        # A restarted run (resume from iteration 0 before any optimizer save)
        # re-logs its first iterations: keep the last entry per iteration.
        prog = list({(e["iteration"], bool(e.get("final"))): e for e in prog}.values())
        met = list({e["iteration"]: e for e in met}.values())
        out.append({"run": run_dir.name, "arm": raw["task"]["arm"], "seed": raw["train"]["seed"],
                    "progress": prog, "metrics": met})
    return out


def dose_table(training: list[dict]) -> list[dict]:
    """Realized training dose per arm from the worker's metrics.jsonl: GRPO
    drops groups whose rollouts all share a reward, so the number of sequences
    that actually carried gradient is far below iterations x prompts x G once
    the reward saturates. Compare with 2A's SFT dose of 50 x 48 examples."""
    rows = []
    for arm in sorted({t["arm"] for t in training}):
        per_seed = []
        for t in (t for t in training if t["arm"] == arm):
            m = [e for e in t["metrics"] if not e.get("skipped")]
            if not t["metrics"]:
                continue
            per_seed.append({
                "seed": t["seed"], "iterations": len(t["metrics"]),
                "steps_with_gradient": len(m),
                "groups_used": sum(e.get("n_groups_used", 0) for e in m),
                "groups_degenerate": sum(e.get("n_degenerate_groups", 0) for e in t["metrics"]),
                "sequences": sum(e.get("n_sequences", 0) for e in m),
                "completion_tokens": sum(e.get("global_completion_tokens", 0) for e in m),
                "logprob_corr_min": min((e["logprob_corr"] for e in m if e.get("logprob_corr") is not None),
                                        default=float("nan")),
            })
        if per_seed:
            rows.append({"arm": arm, "n_seeds": len(per_seed), **{
                k: float(np.mean([p[k] for p in per_seed])) for k in
                ("iterations", "steps_with_gradient", "groups_used", "groups_degenerate",
                 "sequences", "completion_tokens", "logprob_corr_min")}, "per_seed": per_seed})
    return rows


# --- rendering -----------------------------------------------------------------


def _ci(b: dict, pct: bool = True) -> str:
    if b is None or np.isnan(b["mean"]):
        return "-"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.0f}")
    return f"{f(b['mean'])} [{f(b['lo'])}, {f(b['hi'])}]"


def render_tables(summary: dict, degen_chars: int) -> str:
    lines = ["# Experiment 2B results", "",
             ("Honest / raw compliance, accuracy in %, mean [95% bootstrap CI over questions x seeds]; "
              f"chars = mean reasoning length; degen = % traces < {degen_chars} chars; compl_long = strict "
              "compliance among traces >= the run's checkpoint-0 median length. Modes pooled "
              "(lowercase_thinking, uppercase_thinking, alternating_case, end_of_sentence)."), ""]
    for table, title in (("headline", "Headline: modes pooled"), ("modes", "Per mode")):
        lines += [f"## {title}", ""]
        for e in summary[table]:
            lines += [f"### {e['split']} / {e['variant']} / {e['condition']}"
                      + (f" / {e['modes']}" if table == "modes" else ""), "",
                      "| arm | step | seeds | honest | raw | compl_long | acc | chars | degen | honest per seed |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for a in e["arms"]:
                sm = ", ".join(f"{100 * v:.0f}" for v in a["seed_means_honest"].values())
                lines.append(f"| {a['arm']} | {a.get('step', '-')} | {a['n_seeds']} | {_ci(a['honest'])} | "
                             f"{_ci(a['raw'])} | {100 * a['compl_long']:.1f} | {_ci(a['correct'])} | "
                             f"{_ci(a['chars'], pct=False)} | {_ci(a['degen'])} | {sm} |")
            if "diff" in e:
                d = e["diff"]
                lines.append(f"| **orange − mcq** | | | **{_ci(d['honest'])}** | {_ci(d['raw'])} | | "
                             f"{_ci(d['correct'])} | {_ci(d['chars'], pct=False)} | {_ci(d['degen'])} | |")
            lines.append("")
    if summary.get("dose"):
        lines += ["## Realized training dose (mean over seeds; 2A SFT = 50 steps x 48 examples)", "",
                  "| arm | iterations | steps with gradient | groups with gradient (of 50x48) | sequences | completion tokens | min logprob corr |",
                  "|---|---|---|---|---|---|---|"]
        for d in summary["dose"]:
            lines.append(f"| {d['arm']} | {d['iterations']:.0f} | {d['steps_with_gradient']:.0f} | {d['groups_used']:.0f} | "
                         f"{d['sequences']:.0f} | {d['completion_tokens']:.0f} | {d['logprob_corr_min']:.4f} |")
        lines.append("")
    if summary.get("kl"):
        lines += ["## Functional size: KL(base || adapter) per token (noise agent, 200 val prompts, T=1 base samples)", "",
                  "| arm | KL per seed | median | noise group | noise target KL | noise adapters' KL |", "|---|---|---|---|---|---|"]
        for k in summary["kl"]:
            ps = ", ".join(f"s{p['seed']}={p['kl']:.4f}" for p in k["per_seed"])
            na = ", ".join(f"{x:.4f}" for x in k["noise_achieved"])
            lines.append(f"| {k['arm']} | {ps} | {k['median']:.4f} | {k.get('noise_group', '-')} | "
                         f"{k['noise_target']:.4f} | {na or '-'} |")
        lines.append("")
    lines += [("## Final checkpoints on test: standard stated eval (9 modes), held-out modes, "
               "unconstrained baseline"), "",
              "| block | arm | step | seeds | honest | raw | acc | chars | degen |", "|---|---|---|---|---|---|---|---|---|"]
    for s in summary["standard"]:
        lines.append(f"| {s['block']} | {s['arm']} | {s['step']} | {s['n_seeds']} | {_ci(s['honest'])} | "
                     f"{_ci(s['raw'])} | {_ci(s['correct'])} | {_ci(s['chars'], pct=False)} | {_ci(s['degen'])} |")
    return "\n".join(lines) + "\n"


def plot_curves(summary: dict, out_dir: Path, conditions: list[str], degen_chars: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = sorted({c["arm"] for c in summary["curves"]})
    if not arms:
        return
    panels = [("honest", "honest compliance"), ("raw", "raw compliance"), ("correct", "accuracy"),
              ("chars", "reasoning chars"), ("degen", f"degen (<{degen_chars} chars)")]
    fig, axes = plt.subplots(len(conditions), len(panels), figsize=(4.2 * len(panels), 3.2 * len(conditions)),
                             squeeze=False)
    colors = {a: f"C{i}" for i, a in enumerate(arms)}
    for i, cond in enumerate(conditions):
        for j, (metric, label) in enumerate(panels):
            ax = axes[i][j]
            for arm in arms:
                pts = sorted((c for c in summary["curves"] if c["arm"] == arm and c["condition"] == cond),
                             key=lambda c: c["step"])
                if not pts:
                    continue
                x = [p["step"] for p in pts]
                ax.plot(x, [p[metric]["mean"] for p in pts], "o-", color=colors[arm], label=arm, ms=3)
                ax.fill_between(x, [p[metric]["min"] for p in pts], [p[metric]["max"] for p in pts],
                                color=colors[arm], alpha=0.15)
            ax.set_title(f"{cond}: {label}", fontsize=9)
            ax.set_xlabel("RL iteration")
            if metric != "chars":
                ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_curves.png", dpi=140)
    plt.close(fig)


def plot_training(training: list[dict], out_dir: Path, degen_chars: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = sorted({t["arm"] for t in training})
    if not arms:
        return
    panels = [("reward_mean", "train reward"), ("heldout", "held-out reward"),
              ("frac_truncated", "truncated rollouts"), ("frac_short_reasoning", f"short traces (<{degen_chars} chars)"),
              ("n_groups_used", "groups with gradient (worker)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
    colors = {a: f"C{i}" for i, a in enumerate(arms)}
    for j, (key, label) in enumerate(panels):
        ax = axes[0][j]
        for arm in arms:
            series = defaultdict(list)
            for t in (t for t in training if t["arm"] == arm):
                src = t["metrics"] if key == "n_groups_used" else t["progress"]
                for e in src:
                    if key == "heldout":
                        if "heldout" in e and e["heldout"].get("reward_mean") is not None:
                            series[e["iteration"]].append(e["heldout"]["reward_mean"])
                    elif e.get(key) is not None:
                        series[e["iteration"]].append(e[key])
            if not series:
                continue
            x = sorted(series)
            ax.plot(x, [np.mean(series[s]) for s in x], "-" if key != "heldout" else "o-",
                    color=colors[arm], label=arm, ms=3)
            ax.fill_between(x, [min(series[s]) for s in x], [max(series[s]) for s in x],
                            color=colors[arm], alpha=0.15)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("RL iteration")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=140)
    plt.close(fig)


def main(cfg: dict, argv: list[str] | None = None) -> None:
    an = cfg["analysis"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / an["runs"])
    ap.add_argument("--out", type=Path, default=results_dir(cfg))
    ap.add_argument("--n-boot", type=int, default=int(an["bootstrap_draws"]))
    ap.add_argument("--seed", type=int, default=int(an["bootstrap_seed"]))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    degen, metrics = int(an["degen_chars"]), list(an["metrics"])
    noise_runs = ROOT / an["noise_runs"]
    rows = load_rows(a.runs_dir, degen)
    rows += load_noise_rows({r["run"] for r in rows}, noise_runs, degen)
    training = load_training(a.runs_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    if not rows and not training:
        raise SystemExit(f"nothing under {a.runs_dir}")
    variants = {k: list(v) for k, v in an["variants"].items()}
    summary = (summarize(rows, a.n_boot, variants, list(an["conditions"]), metrics, list(an["standard_blocks"]), a.seed)
               if rows else {"headline": [], "modes": [], "standard": [], "curves": []})
    summary["dose"] = dose_table(training)
    summary["kl"] = kl_table(rows, noise_runs)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (a.out / "training.json").write_text(json.dumps(training, indent=1))
    bars = long_bars(rows)
    (a.out / "checkpoints.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "items"} | agg(r["items"], bar_for(bars, r), metrics) for r in rows], indent=1))
    md = render_tables(summary, degen)
    (a.out / "tables.md").write_text(md)
    print(md)
    if not a.no_plots:
        plot_curves(summary, a.out, list(an["conditions"]), degen)
        plot_training(training, a.out, degen)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
