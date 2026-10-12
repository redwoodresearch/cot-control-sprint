"""Entrypoint logic behind kaushik/run.py.

    run.py --list                                  # experiments and their jobs
    run.py <exp> <job> [key=value ...] [job args]  # one job on the resolved config
    run.py <exp> <job> --resolved <run_dir>/resolved.yaml [job args]
    run.py sweep <sweep.yaml> [--dry-run | --check GLOB... | --write] [key=value ...]

Job lookup order for `<exp> <job>`:
1. legacy `kaushik/<exp>/cli.py` with `STAGES[job]` (folders not yet migrated);
2. generic jobs in kaushik/core/jobs.py (train, rl, eval, registry, status);
3. `kaushik/<exp>/<job>.py` exposing `main(cfg, argv)` (experiment-specific jobs:
   build_data, analyze, figures, ...).

`key=value` arguments (no leading dash) are config overrides; everything else is
passed to the job. The Modal env is set before anything imports cotcontrol.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from kaushik.core.config import materialize, parse_overrides, resolve
from kaushik.core.io import KAUSHIK, load_py, rel

SKIP_DIRS = {"core", "config", "registry", "__pycache__", "runs", "configs", "data", "results", "sweeps", "tests"}


def legacy_stages(exp: str) -> dict | None:
    path = KAUSHIK / exp / "cli.py"
    if not path.exists():
        return None
    sys.path.insert(0, str(path.parent))   # legacy clis import siblings by bare name
    return load_py(path, f"kaushik_{exp.replace('/', '_')}_cli").STAGES


def experiments() -> dict[str, list[str]]:
    """{exp: [jobs]} for every folder with a cli.py (legacy; stages are read from the
    source without importing it, since legacy clis import siblings by bare name) or a
    config.yaml."""
    out: dict[str, list[str]] = {}
    for p in sorted(KAUSHIK.rglob("*")):
        if p.name not in ("cli.py", "config.yaml") or any(part in SKIP_DIRS for part in p.relative_to(KAUSHIK).parts):
            continue
        exp = str(p.parent.relative_to(KAUSHIK))
        if p.name == "cli.py":
            m = re.search(r"^STAGES\s*=\s*\{(.*?)\}", p.read_text(), re.DOTALL | re.MULTILINE)
            names = re.findall(r'"([A-Za-z_]\w*)"\s*:', m.group(1)) if m else []
            out.setdefault(exp, []).extend(f"{s} (legacy)" for s in names)
        else:
            from kaushik.core.jobs import JOBS

            jobs = list(JOBS) + sorted(f.stem for f in p.parent.glob("*.py")
                                       if f.stem != "cli" and re.search(r"^def main\(cfg", f.read_text(), re.MULTILINE))
            out.setdefault(exp, []).extend(jobs)
    return out


def split_args(args: list[str]) -> tuple[dict, list[str], Path | None]:
    """-> (config overrides, job argv, --resolved path)."""
    overrides, rest, resolved = [], [], None
    it = iter(args)
    for a in it:
        if a == "--resolved":
            resolved = Path(next(it))
        elif "=" in a and not a.startswith("-"):
            overrides.append(a)
        else:
            rest.append(a)
    return parse_overrides(overrides), rest, resolved


def run_job(exp: str, job: str, args: list[str]) -> None:
    stages = legacy_stages(exp)
    if stages and job in stages:
        stages[job](args)
        return
    from kaushik.core.jobs import JOBS

    overrides, job_argv, resolved = split_args(args)
    if resolved:
        cfg = materialize(yaml.safe_load(Path(resolved).read_text()))
        for k, v in overrides.items():
            from kaushik.core.config import deep_set
            deep_set(cfg, k, v)
    else:
        cfg = resolve(exp, overrides=overrides)
    if job in JOBS:
        JOBS[job][1](cfg, job_argv)
        return
    mod_path = KAUSHIK / exp / f"{job}.py"
    if not mod_path.exists():
        known = list(JOBS) + sorted(f.stem for f in (KAUSHIK / exp).glob("*.py"))
        raise SystemExit(f"{exp}: no job {job!r} ({rel(mod_path)} missing); known: {known}")
    mod = load_py(mod_path)
    if not hasattr(mod, "main"):
        raise SystemExit(f"{rel(mod_path)} has no main(cfg, argv)")
    mod.main(cfg, job_argv)


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help", "--list"):
        for exp, jobs in experiments().items():
            print(f"{exp}: {', '.join(jobs)}")
        print(__doc__)
        return
    if argv[0] == "sweep":
        from kaushik.core import sweep

        sweep.main(argv[1:])
        return
    if len(argv) < 2:
        stages = legacy_stages(argv[0])
        raise SystemExit(f"usage: run.py {argv[0]} <job> [key=value ...]; "
                         + (f"legacy stages: {', '.join(stages)}" if stages else "see --list"))
    run_job(argv[0], argv[1], argv[2:])
