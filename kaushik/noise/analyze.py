"""Aggregate noise-control results: per group, noise vs trained vs untrained.

    .venv/bin/python kaushik/run.py noise analyze [--groups G ...] [--boot 1000] [--seed 0] [--out DIR] [--md PATH]

Metrics per rollout (orange-training/analyze.py + spec "Shared setup"):
  honest      cotcontrol.eval.grading.is_honest (strict + answer + not hollow)
  raw         strict compliance
  acc         answer correct
  chars       reasoning length (reasoning_text_graded)
  degen       chars < analysis.degen_chars
  compl_long  strict compliance among traces >= the untrained checkpoint-0 median length for
              the same block (pooled over the group's seeds)

Pooled over adapters (seeds) and questions; 95% intervals from a two-way ratio bootstrap
(questions x adapters, analysis.bootstrap_draws draws, analysis.bootstrap_seed). The trained -
noise difference reuses the same question draws for both arms (paired on questions). This
estimator (ratio of resampled sums, so compl_long's denominator is resampled too) is not one of
kaushik.core.stats' families; the committed tables depend on its exact draw order, so it stays
here. Trained / untrained rollouts come from the source runs' own eval JSONs for the same
blocks (eval/<block>/checkpoint-<final|0>.json under the run_dir in the train_eval config).

Writes runs/<group>/results.json (`config` = make_groups.group_record) and results.md (all groups).
"""

# ruff: noqa: DTZ005 — naive local timestamps, like the rest of the repo
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from cotcontrol.eval.grading import is_honest
from kaushik.core.io import KAUSHIK, ROOT, load_py, read_json
from kaushik.core.scheduler import Job

nz = load_py(KAUSHIK / "noise" / "common.py")

METRICS = ["honest", "raw", "acc", "chars", "degen", "compl_long"]
PCT = {"honest", "raw", "acc", "degen", "compl_long"}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_rollouts(path: Path) -> list[tuple[str, dict]]:
    """[(question key, sample)] for every non-error rollout in an eval JSON."""
    data = json.loads(path.read_text())
    out = []
    for rec in data["results"]:
        q = f"{rec['dataset']}:{rec['id']}"
        for s in rec["samples"]:
            if not s.get("error"):
                out.append((q, {**s, "_mode": rec["mode"]}))
    return out


def source_run_dir(entry: dict) -> Path | None:
    cfg = ROOT / entry["config_path"]
    if cfg.exists():
        raw = yaml.safe_load(cfg.read_text()) or {}
        if raw.get("run_dir"):
            return ROOT / raw["run_dir"]
    for cand in (ROOT / "sft/runs" / entry["run_name"], cfg.parent.parent / "runs" / entry["run_name"]):
        if cand.exists():
            return cand
    return None


def source_eval_file(run_dir: Path | None, block: str, step) -> Path | None:
    if run_dir is None or step is None:
        return None
    for sub in ("eval", "final_eval"):
        p = run_dir / sub / block / f"checkpoint-{step}.json"
        if p.exists():
            return p
    # other layouts: eval/<split>_<tag>_<block>/ (sft/test_eval.py), <anything>/<block>/
    hits = sorted(run_dir.glob(f"*/*{block}*/checkpoint-{step}.json")) if run_dir.is_dir() else []
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# metrics + bootstrap
# ---------------------------------------------------------------------------

def _chars(s: dict) -> int:
    return len(s.get("reasoning_text_graded") or s.get("reasoning") or "")


def rollout_terms(s: dict, bar: float | None, degen_chars: int) -> dict[str, tuple[float, float]]:
    """metric -> (numerator, denominator) contributions of one rollout."""
    c = _chars(s)
    graded = s.get("compliance") is not None
    long_ = graded and bar is not None and c >= bar
    return {  # compliance metrics are undefined (n/a) for the unconstrained baseline block
        "honest": (float(is_honest(s)), float(graded)),
        "raw": (float(s.get("compliance") == 1), float(graded)),
        "acc": (float(s.get("correct") is True), 1.0),
        "chars": (float(c), 1.0),
        "degen": (float(c < degen_chars), 1.0),
        "compl_long": (float(long_ and s.get("compliance") == 1), float(long_)),
    }


class Arm:
    """Rollouts of one arm (several adapters) as (adapter, question) term arrays."""

    def __init__(self, name: str, adapters: dict[str, list[tuple[str, dict]]], qindex: dict[str, int], bar,
                 degen_chars: int):
        self.name = name
        self.adapters = list(adapters)
        A, Q = len(adapters), len(qindex)
        self.num = {m: np.zeros((A, Q)) for m in METRICS}
        self.den = {m: np.zeros((A, Q)) for m in METRICS}
        self.n = 0
        for a, rollouts in enumerate(adapters.values()):
            for q, s in rollouts:
                qi = qindex[q]
                for m, (nu, de) in rollout_terms(s, bar, degen_chars).items():
                    self.num[m][a, qi] += nu
                    self.den[m][a, qi] += de
                self.n += 1

    def point(self, m: str, a: int | None = None):
        nu, de = self.num[m], self.den[m]
        if a is not None:
            nu, de = nu[a], de[a]
        return float(nu.sum() / de.sum()) if de.sum() else float("nan")

    def replicates(self, m: str, wa: np.ndarray, wq: np.ndarray) -> np.ndarray:
        """wa (B, A) adapter multiplicities, wq (B, Q) question multiplicities."""
        nu = np.einsum("ba,bq,aq->b", wa, wq, self.num[m])
        de = np.einsum("ba,bq,aq->b", wa, wq, self.den[m])
        with np.errstate(invalid="ignore", divide="ignore"):
            return nu / de


def _multiplicities(rng, n: int, B: int) -> np.ndarray:
    if n <= 1:
        return np.ones((B, max(n, 1)))
    draws = rng.integers(0, n, size=(B, n))
    return np.stack([np.bincount(d, minlength=n) for d in draws]).astype(float)


def _ci(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [float("nan")] * 2


def compare(arms: dict[str, Arm], n_q: int, draws: int, seed: int) -> dict:
    """Point estimates + 95% CIs per arm and for trained - noise (paired on questions).
    Draw order: the question multiplicities first, then one adapter resample per arm."""
    rng = np.random.default_rng(seed)
    wq = _multiplicities(rng, n_q, draws)
    wa = {k: _multiplicities(rng, len(arm.adapters), draws) for k, arm in arms.items()}
    reps = {k: {m: arm.replicates(m, wa[k], wq) for m in METRICS} for k, arm in arms.items()}
    out = {}
    for k, arm in arms.items():
        out[k] = {"n": arm.n, "adapters": arm.adapters}
        for m in METRICS:
            out[k][m] = {"mean": arm.point(m), "ci95": _ci(reps[k][m]),
                         "per_adapter": [arm.point(m, a) for a in range(len(arm.adapters))]}
    for other in ("noise", "untrained"):
        if "trained" in arms and other in arms:
            d = {}
            for m in METRICS:
                diff = reps["trained"][m] - reps[other][m]
                d[m] = {"mean": arms["trained"].point(m) - arms[other].point(m), "ci95": _ci(diff)}
            out[f"trained_minus_{other}"] = d
    return out


# ---------------------------------------------------------------------------
# per group
# ---------------------------------------------------------------------------

def analyze_group(cfg: dict, draws: int | None = None, seed: int | None = None, out_dir: Path | None = None) -> dict:
    """results.json of one group (written under out_dir/<group>/ when given, else the group dir)."""
    an, g, name = cfg["analysis"], cfg["group"], cfg["run_name"]
    draws = int(an["bootstrap_draws"]) if draws is None else int(draws)
    seed = int(an["bootstrap_seed"]) if seed is None else int(seed)
    degen_chars = int(an["degen_chars"])
    gdir = nz.group_dir(cfg)
    out = {"group": name, "config": nz.make_groups_module().group_record(cfg), "timestamp": datetime.now().isoformat(),
           "trained_kl": {}, "noise": None, "blocks": {}}
    for e in g["entries"]:
        rec = nz.kl_record(cfg, e["run_name"])
        if rec:
            out["trained_kl"][e["run_name"]] = {"seed": e["seed"], **{k: rec["kl"][k] for k in
                                                ("kl_per_token", "se_prompt", "ci95_prompt", "n_tokens")}}
    if out["trained_kl"]:
        out["trained_kl_median"] = statistics.median(v["kl_per_token"] for v in out["trained_kl"].values())
    na = read_json(gdir / "noise_adapters.json")
    if na:
        out["noise"] = {"target_kl": na["target_kl"], "reference": na["reference"]["run_name"], "family": na["family"],
                        "adapters": [{k: a[k] for k in ("k", "seed", "scale", "path")}
                                     | {"kl_per_token": a["kl"]["kl_per_token"], "se_prompt": a["kl"]["se_prompt"],
                                        "n_calibration_evals": len(a["calibration"])} for a in na["adapters"]]}
    eval_root = gdir / "eval"
    if not eval_root.is_dir():
        return out
    steps = {e["run_name"]: nz.final_step(e["adapter_path"]) for e in g["entries"]}
    run_dirs = {e["run_name"]: source_run_dir(e) for e in g["entries"]}
    for bdir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        block = bdir.name
        noise = {p.stem: load_rollouts(p) for p in sorted(bdir.glob("noise*.json"))}
        trained, untrained = {}, {}
        for e in g["entries"]:
            f = source_eval_file(run_dirs[e["run_name"]], block, steps[e["run_name"]])
            if f:
                trained[e["run_name"]] = load_rollouts(f)
            f0 = source_eval_file(run_dirs[e["run_name"]], block, 0)
            if f0:
                untrained[e["run_name"]] = load_rollouts(f0)
        bar = (statistics.median(_chars(s) for r in untrained.values() for _, s in r)
               if untrained else None)
        arms_raw = {k: v for k, v in (("noise", noise), ("trained", trained), ("untrained", untrained)) if v}
        qkeys = sorted({q for arm in arms_raw.values() for r in arm.values() for q, _ in r})
        qindex = {q: i for i, q in enumerate(qkeys)}
        arms = {k: Arm(k, v, qindex, bar, degen_chars) for k, v in arms_raw.items()}
        res = compare(arms, len(qkeys), draws, seed)
        res["n_questions"] = len(qkeys)
        res["long_bar_chars"] = bar
        res["sources"] = {"trained": sorted(trained), "untrained": sorted(untrained), "noise": sorted(noise)}
        out["blocks"][block] = res
    dest = (Path(out_dir) / name if out_dir else gdir) / "results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# results.md
# ---------------------------------------------------------------------------

def _fmt(v: dict | None, m: str) -> str:
    if not v or not np.isfinite(v["mean"]):
        return "n/a"
    scale = 100 if m in PCT else 1
    lo, hi = v["ci95"]
    f = (lambda x: f"{x * scale:.1f}") if m in PCT else (lambda x: f"{x:.0f}")
    return f"{f(v['mean'])} [{f(lo)}, {f(hi)}]"


def _row(block: str, arm: str, r: dict) -> str:
    cells = " | ".join(_fmt(r.get(m), m) for m in METRICS)
    return f"| {block} | {arm} ({len(r['adapters'])} adapters, n={r['n']}) | {cells} |"


def write_results_md(results: dict[str, dict], path: Path) -> None:
    L = [(f"# Noise controls: results\n\nGenerated {datetime.now():%Y-%m-%d %H:%M}. "
          "Percent for honest/raw/acc/degen/compl_long, mean reasoning chars for chars, nats/token for KL. "
          "Brackets: 95% bootstrap over questions x adapters (seeds). "
          "`beats noise` = the trained - noise interval for honest compliance excludes 0 from above.\n"),
         "## Summary (honest compliance)\n",
         "| group | block | trained | noise | trained - noise | beats noise? | untrained |",
         "|---|---|---|---|---|---|---|"]
    for name, res in sorted(results.items()):
        for block, b in res.get("blocks", {}).items():
            # Unconstrained (baseline) blocks have no compliance: show accuracy instead.
            m = "honest"
            if all(not np.isfinite(b.get(a, {}).get("honest", {}).get("mean", float("nan"))) for a in ("trained", "noise")):
                m = "acc"
            d = b.get("trained_minus_noise", {}).get(m)
            ok = d is not None and np.isfinite(d["mean"])
            beats = "n/a" if not ok or m != "honest" else ("yes" if d["ci95"][0] > 0 else "no")
            label = block if m == "honest" else f"{block} (accuracy)"
            L.append(f"| {name} | {label} | {_fmt(b.get('trained', {}).get(m), m)} | "
                     f"{_fmt(b.get('noise', {}).get(m), m)} | {_fmt(d, m)} | {beats} | "
                     f"{_fmt(b.get('untrained', {}).get(m), m)} |")
    for name, res in sorted(results.items()):
        cfg = res.get("config") or {}
        L += [f"\n## {name}\n", (f"{cfg.get('experiment')} / {cfg.get('arm')} / {cfg.get('base_model')}; "
                                f"{len(cfg.get('entries', []))}/{cfg.get('expected_seeds')} seeds registered.\n")]
        if res.get("trained_kl"):
            kls = ", ".join(f"seed{v['seed']} {v['kl_per_token']:.4f} ± {v['se_prompt']:.4f}"
                            for v in res["trained_kl"].values())
            L.append(f"KL(base || trained): {kls}; median {res['trained_kl_median']:.4f}.  ")
        if res.get("noise"):
            n = res["noise"]
            kls = ", ".join(f"noise{a['k']} (seed {a['seed']}, scale {a['scale']:.3g}) {a['kl_per_token']:.4f}"
                            for a in n["adapters"])
            L.append(f"Noise ({n['family']}, target {n['target_kl']:.4f}, shapes from {n['reference']}): {kls}.\n")
        for block, b in res.get("blocks", {}).items():
            L += [f"\n| block | arm | {' | '.join(METRICS)} |", "|---|---|" + "---|" * len(METRICS)]
            for arm in ("trained", "noise", "untrained"):
                if arm in b:
                    L.append(_row(block, arm, b[arm]))
            for other in ("noise", "untrained"):
                d = b.get(f"trained_minus_{other}")
                if d:
                    L.append(f"| {block} | trained - {other} | {' | '.join(_fmt(d[m], m) for m in METRICS)} |")
            if b.get("trained"):
                spread = ", ".join(f"{x * 100:.1f}" for x in b["trained"]["honest"]["per_adapter"])
                nspread = ", ".join(f"{x * 100:.1f}" for x in b["noise"]["honest"]["per_adapter"]) if b.get("noise") else "n/a"
                L.append(f"\nSeed spread (honest, %): trained [{spread}]; noise [{nspread}]. "
                         f"compl_long bar = {b['long_bar_chars']} chars; {b['n_questions']} questions.")
    path.write_text("\n".join(L) + "\n")


def saved_results(root: Path) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("*/results.json"))}


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def jobs(cfg: dict) -> list[Job]:
    """One zero-weight job per noise group: results.json once eval_summary.json exists (and is
    newer), then results.md from every saved results.json."""
    g = cfg["group"]
    if not g["noise"]:
        return []
    name, gdir = cfg["run_name"], nz.group_dir(cfg)
    es, rj = gdir / "eval_summary.json", gdir / "results.json"

    async def run():
        analyze_group(cfg)
        write_results_md(saved_results(nz.runs_dir(cfg)), ROOT / cfg["analysis"]["results_md"])

    return [Job(name=f"analyze:{name}", weight=0, run=run,
                done=lambda: rj.exists() and es.exists() and rj.stat().st_mtime >= es.stat().st_mtime,
                ready=es.exists, priority=(7, nz.model_rank(cfg, g["base_model"]), name), group=str(gdir))]


def main(cfg: dict, argv: list[str] | None = None) -> None:
    an = cfg["analysis"]
    ap = argparse.ArgumentParser(prog="noise analyze", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*", default=None, help="groups to re-analyze (default: every group)")
    ap.add_argument("--boot", type=int, default=int(an["bootstrap_draws"]))
    ap.add_argument("--seed", type=int, default=int(an["bootstrap_seed"]))
    ap.add_argument("--out", default=None, help="write <DIR>/<group>/results.json instead of the group dirs")
    ap.add_argument("--md", default=None, help=f"results.md path (default {an['results_md']})")
    a = ap.parse_args(argv)
    points = {c["run_name"]: c for c in nz.group_points(cfg)}
    names = a.groups or list(points)
    out_dir = Path(a.out) if a.out else None
    results = {}
    for n in names:
        if n not in points:
            raise SystemExit(f"unknown group {n!r}; known: {sorted(points)}")
        results[n] = analyze_group(points[n], a.boot, a.seed, out_dir)
        done = [b for b, v in results[n].get("blocks", {}).items() if "noise" in v]
        print(f"{n}: KL for {len(results[n].get('trained_kl', {}))} seeds; noise evals in {len(done)} blocks")
    # results.md covers every group from its saved results.json (cheap), so a per-group job only
    # re-analyzes its own group.
    saved = saved_results(out_dir if out_dir else nz.runs_dir(cfg))
    md = Path(a.md) if a.md else ROOT / an["results_md"]
    write_results_md({**saved, **results}, md)
    print(f"wrote {md}")
