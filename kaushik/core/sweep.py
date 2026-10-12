"""Sweeps: YAML files that orchestrate (kaushik/ARCHITECTURE.md §3).

    experiment: orange_expand/sft          # kaushik/<exp>/config.yaml it starts from
    job: train                             # train | rl | eval | registry | <experiment job>
    after: [registry]                      # optional generic jobs queued per point after `job`
    run_name: "oex-sft_{variant}-{model}-seed{seed}"
    seeds: [0]                             # sugar: an axis setting `seed_key` (default train.seed)
    axes:                                  # cartesian product of dotted-key overrides
      - [{_label: {model: qwen8b}, model: qwen8b}, {_label: {model: gptoss20b}, model: gptoss20b}]
      - [{_label: {variant: eugene}}, {_label: {variant: direct}}]
    overrides: {}                          # applied to every point
    max_gpus: 40

`_label` is a dict of label fields; the labels of a point are merged across axes
and format `run_name` (and any `{field}` string in the config, see config.py).
A point's remaining keys are dotted-key overrides. Every run name must be unique.

    run.py sweep <sweep.yaml> --dry-run            # expanded run names + resolved configs
    run.py sweep <sweep.yaml> --check <glob...>    # semantic equality with existing YAMLs
    run.py sweep <sweep.yaml>                      # write runs/<run>/config.yaml, launch via the scheduler
"""

from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import yaml

from kaushik.core.config import (
    apply_overrides,
    dump_yaml,
    parse_overrides,
    resolve,
    run_dir_of,
    train_eval_config,
)
from kaushik.core.io import KAUSHIK, ROOT, load_py, rel


def load_sweep(path: Path) -> dict:
    path = Path(path)
    sweep = yaml.safe_load(path.read_text())
    sweep.setdefault("_path", str(path))
    return sweep


def expand_points(sweep: dict) -> list[dict]:
    """-> [{labels, overrides, run_name}] in cartesian-product order."""
    axes = [list(ax) for ax in (sweep.get("axes") or [])]
    if sweep.get("seeds") is not None:
        key = sweep.get("seed_key", "train.seed")
        axes.append([{"_label": {"seed": s}, key: s} for s in sweep["seeds"]])
    points, seen = [], set()
    for combo in itertools.product(*axes) if axes else [()]:
        labels, overrides = {}, dict(sweep.get("overrides") or {})
        for d in combo:
            lab = d.get("_label", {})
            if not isinstance(lab, dict):
                raise TypeError(f"_label must be a dict of fields, got {lab!r}")
            labels.update(lab)
            overrides.update({k: v for k, v in d.items() if k != "_label"})
        run_name = sweep["run_name"].format(**labels) if "run_name" in sweep else overrides.get("run_name")
        if not run_name:
            raise ValueError("sweep needs a run_name template or run_name overrides")
        if run_name in seen:
            raise ValueError(f"duplicate run name {run_name!r}")
        seen.add(run_name)
        points.append({"labels": labels, "overrides": overrides, "run_name": run_name})
    return points


def resolve_sweep(sweep: dict, cli_overrides: dict | None = None) -> list[tuple[dict, dict]]:
    """[(point, resolved config)] for every point of the sweep."""
    out = []
    for p in expand_points(sweep):
        cfg = resolve(sweep["experiment"], p["overrides"], cli_overrides, labels=p["labels"], run_name=p["run_name"])
        cfg["job"] = sweep.get("job")
        out.append((p, cfg))
    return out


# ----------------------------------------------------------------------------- equivalence


def diff_keys(a, b, prefix: str = "") -> list[str]:
    """Dotted paths where two YAML trees differ (semantic: key order ignored)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{prefix}{k} ({'missing in new' if k not in a else 'extra in new'})")
            else:
                out += diff_keys(a[k], b[k], f"{prefix}{k}.")
        return out
    if a != b or type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [f"{prefix[:-1]}: {a!r} != {b!r}"]
    return []


def check_equivalence(configs: list[tuple[dict, dict]], paths: list[Path]) -> dict:
    """Compare expanded train_eval configs with existing YAMLs by run_name.
    -> {run_name: {"status": equal|differs|missing, "diffs": [...], "path": str}} plus
    "_unmatched": existing files no point reproduces."""
    existing = {}
    for p in paths:
        raw = yaml.safe_load(Path(p).read_text())
        if isinstance(raw, dict) and "run_name" in raw:
            existing.setdefault(raw["run_name"], []).append((Path(p), raw))
    report, matched = {}, set()
    for point, cfg in configs:
        name = cfg["run_name"]
        view = train_eval_config(cfg)
        if name not in existing:
            report[name] = {"status": "missing", "diffs": [], "path": None}
            continue
        matched.add(name)
        statuses = []
        for path, raw in existing[name]:
            diffs = diff_keys(apply_overrides(view, {}), raw)
            statuses.append({"status": "equal" if not diffs else "differs", "diffs": diffs, "path": rel(path)})
        report[name] = statuses[0] if len(statuses) == 1 else {"status": "multi", "per_file": statuses}
    report["_unmatched"] = sorted(rel(p) for n, lst in existing.items() if n not in matched for p, _ in lst)
    return report


def print_report(report: dict) -> bool:
    ok = True
    for name, r in report.items():
        if name == "_unmatched":
            continue
        entries = r.get("per_file") or [r]
        for e in entries:
            flag = {"equal": "OK  ", "differs": "DIFF", "missing": "MISS"}[e["status"]]
            ok &= e["status"] == "equal"
            print(f"{flag} {name}" + (f"  ({e['path']})" if e.get("path") else ""))
            for d in e["diffs"][:40]:
                print(f"       {d}")
    if report.get("_unmatched"):
        print(f"unmatched existing configs: {report['_unmatched']}")
    return ok


# ----------------------------------------------------------------------------- jobs


def point_jobs(sweep: dict, cfg: dict) -> list:
    """Scheduler jobs for one point: the sweep's job (generic or kaushik/<exp>/<job>.py:jobs)
    followed by the generic jobs in `after`."""
    from kaushik.core.jobs import JOBS

    jobs = []
    for name in [sweep.get("job"), *(sweep.get("after") or [])]:
        if name in JOBS and JOBS[name][0] is not None:
            jobs += JOBS[name][0](cfg)
            continue
        mod_path = KAUSHIK / cfg["experiment"] / f"{name}.py"
        if not mod_path.exists():
            raise FileNotFoundError(f"no job {name!r}: not generic and no {rel(mod_path)}")
        mod = load_py(mod_path)
        if not hasattr(mod, "jobs"):
            raise AttributeError(f"{rel(mod_path)} has no jobs(cfg) for sweep use")
        jobs += mod.jobs(cfg)
    return jobs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="run.py sweep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="kaushik/<exp>/sweeps/<name>.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print expanded configs; dispatch nothing")
    ap.add_argument("--show", action="store_true", help="with --dry-run: print the full resolved YAML of every point")
    ap.add_argument("--check", nargs="+", metavar="GLOB", help="existing config YAMLs to compare against")
    ap.add_argument("--write", action="store_true", help="write runs/<run>/config.yaml without launching")
    ap.add_argument("--cap", type=int, default=None, help="GPU cap (default: sweep max_gpus or defaults.yaml)")
    ap.add_argument("--runs-dir", default=None, help="scheduler state dir (default: parent of the run dirs)")
    ap.add_argument("overrides", nargs="*", help="key=value applied to every point")
    a = ap.parse_args(argv)

    sweep = load_sweep(Path(a.sweep) if Path(a.sweep).exists() else KAUSHIK / a.sweep)
    configs = resolve_sweep(sweep, parse_overrides(a.overrides))
    print(f"sweep {rel(sweep['_path'])}: experiment {sweep['experiment']}, job {sweep.get('job')}, "
          f"{len(configs)} points")
    if a.check:
        paths = sorted({Path(p) for pat in a.check for p in glob.glob(str(ROOT / pat) if not Path(pat).is_absolute() else pat)})
        ok = print_report(check_equivalence(configs, paths))
        print("config equivalence:", "PASS" if ok else "FAIL")
        if not ok:
            raise SystemExit(1)
        return
    if a.dry_run:
        for point, cfg in configs:
            print(f"- {cfg['run_name']}  labels={point['labels']}  run_dir={cfg.get('run_dir')}")
            if a.show:
                print(dump_yaml(train_eval_config(cfg)))
        if sweep.get("job"):
            from kaushik.core.scheduler import Scheduler

            jobs = [j for _, cfg in configs for j in point_jobs(sweep, cfg)]
            import asyncio

            runs_dir = Path(a.runs_dir) if a.runs_dir else run_dir_of(configs[0][1]).parent
            asyncio.run(Scheduler(jobs, runs_dir, cap=a.cap or sweep.get("max_gpus"), dry_run=True).run())
        return
    from kaushik.core.jobs import prepare_run
    from kaushik.core.scheduler import run_scheduler

    for _, cfg in configs:
        prepare_run(cfg)
    if a.write:
        print(f"wrote {len(configs)} run configs")
        return
    jobs = [j for _, cfg in configs for j in point_jobs(sweep, cfg)]
    runs_dir = Path(a.runs_dir) if a.runs_dir else run_dir_of(configs[0][1]).parent
    run_scheduler(jobs, runs_dir, cap=a.cap or sweep.get("max_gpus"), name=f"sweep_{Path(sweep['_path']).stem}")
