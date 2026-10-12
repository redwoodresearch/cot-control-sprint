"""Migration checks for Experiment 3B (experiment job).

    run.py hidden_constraint check                      # config equivalence: sweeps + registry vs runs/*/manifest.json
    run.py hidden_constraint check --analysis REF_DIR   # analysis equivalence: runs/{results.md,summary.csv} and
                                                        # runs/*/summary.json vs copies taken before the migration

Config equivalence: every point of sweeps/*.yaml and every adapter the registry yields is resolved and
the settings a run records (runner.manifest_view + backend_info) are compared with the manifest of the
existing run of the same name. Differences are listed per run; see README "Migration" for the explained
ones. Analysis equivalence: the old metrics (every row of the old summary.csv, every old key of each
summary.json, the old columns of the results.md table) must be unchanged; the new per-episode metrics
are additions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.config import runs_root
from kaushik.core.io import load_py, rel
from kaushik.core.sweep import diff_keys, load_sweep, resolve_sweep
from kaushik.hidden_constraint import runner

HERE = Path(__file__).resolve().parent
EXP = "hidden_constraint"
# Manifest keys whose historical value is explained in the README, not a config error: adapters were
# recorded under the generic label "modal_adapter" (now the base model's key); ModalGenerateConfig gained
# `scaledown_window` after the runs (cotcontrol commit d30c268).
EXPLAINED = {"model_label", "backend_info.generate_config.scaledown_window"}


def expected_manifest(cfg: dict) -> dict:
    """What a run of this config would record, restricted to the keys old manifests have."""
    _, _, info = runner.make_generate_fn(cfg)
    view = runner.manifest_view(cfg, info)
    return {k: view[k] for k in ("run_name", "model_label", "model_config", "adapter", "backend_info", "slice", "n",
                                 "scenarios", "max_turns", "judge")}


def check_configs(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """-> (ok run names, [diff lines], missing run names)."""
    configs = []
    for sweep in sorted((HERE / "sweeps").glob("*.yaml")):
        s = load_sweep(sweep)
        if s.get("job") == "poll":
            poll = load_py(HERE / "poll.py")
            configs += [poll.adapter_config(c, e) for _, c in resolve_sweep(s) for e in poll.adapter_entries(c).values()]
        else:
            configs += [c for _, c in resolve_sweep(s)]
    runs = runs_root(cfg)
    ok, diffs, missing = [], [], []
    for c in configs:
        name = runner.run_name_of(c)
        man = runs / name / "manifest.json"
        if not man.exists():
            missing.append(name)
            continue
        old = json.loads(man.read_text())
        exp = expected_manifest(c)
        d = diff_keys(exp, {k: old.get(k) for k in exp})
        d = [x for x in d if x.split(":")[0].split(" (")[0] not in EXPLAINED]
        (diffs.extend(f"{name}: {x}" for x in d) if d else ok.append(name))
    return ok, diffs, missing


def check_analysis(cfg: dict, ref: Path) -> list[str]:
    """Old rows of summary.csv, old keys of every summary.json and old columns of results.md unchanged."""
    runs = runs_root(cfg)
    problems = []
    with open(ref / "summary.csv", newline="") as f:
        old_rows = {(r["model_or_checkpoint"], r["scenario"], r["metric"]): r for r in csv.DictReader(f)}
    with open(runs / "summary.csv", newline="") as f:
        new_rows = {(r["model_or_checkpoint"], r["scenario"], r["metric"]): r for r in csv.DictReader(f)}
    for k, r in old_rows.items():
        if k not in new_rows:
            problems.append(f"summary.csv row missing: {k}")
        elif new_rows[k] != r:
            problems.append(f"summary.csv row differs: {k}: {r} -> {new_rows[k]}")
    for p in sorted((ref / "summaries").glob("*.json")):
        new_p = runs / p.stem / "summary.json"
        if not new_p.exists():
            problems.append(f"{p.stem}: summary.json missing")
            continue
        old, new = json.loads(p.read_text()), json.loads(new_p.read_text())
        for sc, m in old["scenarios"].items():
            for metric, v in m.items():
                nv = new["scenarios"].get(sc, {}).get(metric)
                if _differs(v, nv):
                    problems.append(f"{p.stem}/{sc}/{metric}: {v} -> {nv}")
    old_md = _table_rows((ref / "results.md").read_text())
    new_md = _table_rows((runs / "results.md").read_text())
    for key, cells in old_md.items():
        nc = new_md.get(key)
        if nc is None:
            problems.append(f"results.md row missing: {key}")
            continue
        # old columns: the first 9 (through forbidden term), then hits/kchar, suppression, n (the last 3)
        if cells[:9] != nc[:9] or cells[9:] != nc[-3:]:
            problems.append(f"results.md row differs: {key}: {cells} -> {nc}")
    return problems


def _differs(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return any(_differs(a[k], b.get(k)) for k in a)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return False
    return a != b


def _table_rows(md: str) -> dict[tuple[str, str], list[str]]:
    """Rows of the first results table keyed by (model, scenario)."""
    out = {}
    for line in md.splitlines():
        if not line.startswith("| ") or line.startswith(("| model", "| arm", "| (base)")):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 12 and cells[1] in ("qa_grader", "debug_lint", "config_notation") and (cells[0], cells[1]) not in out:
            out[(cells[0], cells[1])] = cells
    return out


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hidden_constraint check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", metavar="REF_DIR", default=None,
                    help="directory with the pre-migration results.md, summary.csv and summaries/<run>.json")
    a = ap.parse_args(argv)
    if a.analysis:
        problems = check_analysis(cfg, Path(a.analysis))
        print("\n".join(problems[:50]))
        print(f"analysis equivalence: {'PASS' if not problems else f'FAIL ({len(problems)} differences)'}")
        if problems:
            raise SystemExit(1)
        return
    ok, diffs, missing = check_configs(cfg)
    print("\n".join(diffs))
    print(f"{len(ok)} runs match their manifest; {len(missing)} not run yet "
          f"({', '.join(missing[:8])}{', ...' if len(missing) > 8 else ''})")
    print(f"config equivalence ({rel(runs_root(cfg))}): {'PASS' if not diffs else f'FAIL ({len(diffs)} differences)'}")
    if diffs:
        raise SystemExit(1)


if __name__ == "__main__":
    from kaushik.core.config import resolve

    main(resolve(EXP), sys.argv[1:])

