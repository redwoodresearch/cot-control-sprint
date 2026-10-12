"""Registry -> sweeps. Groups the final adapters announced in kaushik/registry/*.json by
(experiment, arm[, dose suffix], base model) and writes

    sweeps/groups.yaml   one point per group: job make -> replay -> analyze (the `group` section of config.yaml)
    sweeps/kl.yaml       one point per base model x eval template kwargs: job kl (the `kl` point keys)

    .venv/bin/python kaushik/run.py noise make_groups [--dry-run] [--check "kaushik/noise/configs/*.yaml"]

A group pools the arm's seeds: its noise adapters are calibrated to the MEDIAN KL(base ||
adapter) of the trained seeds and evaluated once with the arm's final eval blocks. `watch`
re-runs this every poll (idempotent; entries only accumulate). `--check` compares each
group's record (`group_record`, the schema of the legacy configs/<group>.yaml) with existing
files: migration rule 1.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import yaml

from kaushik.core import registry as reg
from kaushik.core.config import model_key_of
from kaushik.core.io import KAUSHIK, ROOT, load_py, rel

nz = load_py(KAUSHIK / "noise" / "common.py")

REQUIRED = {"run_name", "experiment", "arm", "base_model", "seed", "adapter_path", "config_path"}
ENTRY_KEYS = ("run_name", "seed", "adapter_path", "config_path")
ENTRY_OPTIONAL = ("noise_eval_blocks", "eval_chat_template_kwargs")


# ----------------------------------------------------------------------------- registry


def load_entries() -> list[dict]:
    """Registry entries with the required keys (partial / malformed files skipped; other
    folders' smoke runs are not arms)."""
    out = []
    for name, e in reg.read_all().items():
        if not isinstance(e, dict) or not REQUIRED <= set(e) or "smoke" in str(e["run_name"]):
            continue
        out.append({**e, "_registry_path": rel(reg.path_for(name))})
    return out


def run_variant(run_name: str) -> str:
    """Suffix after '-seedN' (e.g. 'step50' for cpt's matched-dose checkpoint): a different
    dose of the same arm must not share a group / median KL."""
    m = re.search(r"[-_]seed\d+[-_]?(.*)$", run_name)
    return (m.group(1) if m else "").strip("-_")


def run_prefix(run_name: str) -> str:
    return re.sub(r"[-_]seed\d+.*$", "", run_name)


def group_name(e: dict) -> str:
    arm = e["arm"] + (f"-{run_variant(e['run_name'])}" if run_variant(e["run_name"]) else "")
    return f"{e['experiment'].replace('/', '-')}-{arm}-{nz.model_slug(e['base_model'])}"


def expected_seeds(cfg: dict, members: list[dict]) -> int:
    """How many seeds the arm will register: an explicit n_seeds, else the number of matching
    configs in the owning folder, never below groups.min_seeds (a sweep cell whose extra seeds
    are not written yet would otherwise be processed after its first seed) or the count seen."""
    for e in members:
        if e.get("n_seeds"):
            return int(e["n_seeds"])
    e = members[0]
    prefix = run_prefix(e["run_name"])
    cfg_dir = KAUSHIK / e["experiment"] / "configs"
    n = len([p for p in cfg_dir.glob("*.yaml") if run_prefix(p.stem) == prefix and "smoke" not in p.stem]) if cfg_dir.is_dir() else 0
    ms = cfg["groups"]["min_seeds"]
    floor = int(ms.get(model_key_of(e["base_model"]) or "default", ms["default"]))
    return max(n, floor, len(members))


def group_entries(entries: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(group_name(e), []).append(e)
    return {g: sorted(v, key=lambda e: int(e["seed"])) for g, v in sorted(groups.items())}


# ----------------------------------------------------------------------------- sweep points


def group_point(cfg: dict, name: str, members: list[dict]) -> dict:
    e0 = members[0]
    point = {
        "_label": {"group": name},
        "group": {
            "experiment": e0["experiment"],
            "arm": e0["arm"],
            "base_model": e0["base_model"],
            # Single-seed sweep points (coordinator 10-08 14:58): KL only, never a noise group.
            "noise": any(bool(e.get("noise")) for e in members) and not any(k in name for k in cfg["groups"]["kl_only"]),
            "measure_kl": any(bool(e.get("measure_kl") or e.get("noise")) for e in members),
            "eval_chat_template_kwargs": e0.get("eval_chat_template_kwargs"),
            "expected_seeds": expected_seeds(cfg, members),
            "entries": [{k: e[k] for k in ENTRY_KEYS} | {k: e[k] for k in ENTRY_OPTIONAL if k in e}
                        | {"registry_path": e["_registry_path"]} for e in members],
        },
    }
    if e0.get("noise_family"):
        point["noise.family"] = e0["noise_family"]
    return point


def kl_point(cfg: dict, members: list[dict]) -> dict:
    e0 = members[0]
    kw = e0.get("eval_chat_template_kwargs")
    return {
        "_label": {"samples": nz.samples_tag(e0["base_model"], kw)},
        "kl.base_model": e0["base_model"],
        "kl.chat_template_kwargs": kw,
        "kl.engine_config": e0["config_path"],
        "kl.entries": [{k: e[k] for k in ("run_name", "adapter_path", "config_path", "experiment", "arm", "seed")}
                       for e in members],
    }


def kl_batches(entries: list[dict]) -> list[list[dict]]:
    """Adapters to score, batched per (base model, template kwargs)."""
    by: dict[tuple, list[dict]] = {}
    for e in sorted(entries, key=lambda e: e["run_name"]):
        if e.get("noise") or e.get("measure_kl"):
            kw = e.get("eval_chat_template_kwargs")
            by.setdefault((e["base_model"], nz.samples_tag(e["base_model"], kw)), []).append(e)
    return [by[k] for k in sorted(by, key=lambda k: k[1])]


def sweeps(cfg: dict, entries: list[dict] | None = None) -> tuple[dict, dict]:
    entries = load_entries() if entries is None else entries
    groups = {"experiment": "noise", "job": "make", "after": ["replay", "analyze"], "run_name": "{group}",
              "axes": [[group_point(cfg, g, m) for g, m in group_entries(entries).items()]]}
    kl = {"experiment": "noise", "job": "kl", "run_name": "{samples}",
          "overrides": {"run_dir": f"{cfg['kl']['samples_dir']}/{{run_name}}"},
          "axes": [[kl_point(cfg, m) for m in kl_batches(entries)]]}
    return groups, kl


def write_sweeps(cfg: dict, entries: list[dict] | None = None) -> tuple[dict, dict]:
    """sweeps/groups.yaml + sweeps/kl.yaml (rewritten only when their content changes)."""
    groups, kl = sweeps(cfg, entries)
    nz.SWEEPS.mkdir(parents=True, exist_ok=True)
    for name, sweep, what in (("groups", groups, "one point per group (job make -> replay -> analyze)"),
                              ("kl", kl, "one point per base model x template kwargs (job kl)")):
        text = (f"# Generated by make_groups.py from kaushik/registry: {what}. Do not edit; "
                f"settings live in ../config.yaml.\n" + yaml.safe_dump(sweep, sort_keys=False, width=1000))
        path = nz.SWEEPS / f"{name}.yaml"
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
    return groups, kl


def expected_run_names(cfg: dict) -> dict[str, str]:
    """run_name -> experiment for every config the training folders have written (progress)."""
    out = {}
    for exp in cfg["groups"]["experiments"]:
        for p in (KAUSHIK / exp / "configs").glob("*.yaml"):
            try:
                raw = yaml.safe_load(p.read_text()) or {}
            except yaml.YAMLError:
                continue
            if isinstance(raw, dict) and raw.get("run_name") and "smoke" not in str(raw["run_name"]):
                out[str(raw["run_name"])] = exp
    return out


# ----------------------------------------------------------------------------- records + check


def group_record(gcfg: dict) -> dict:
    """The self-contained record of a resolved group config: what manifest.json and
    results.json carry, and the schema of the legacy configs/<group>.yaml."""
    g, n = gcfg["group"], gcfg["noise"]
    return {"group": gcfg["run_name"], "experiment": g["experiment"], "arm": g["arm"], "base_model": g["base_model"],
            "noise": bool(g["noise"]), "measure_kl": bool(g["measure_kl"]),
            "eval_chat_template_kwargs": g["eval_chat_template_kwargs"], "expected_seeds": int(g["expected_seeds"]),
            "entries": g["entries"], "noise_seeds": list(n["seeds"]), "family": n["family"],
            "tolerance": n["tolerance"], "kl": nz.kl_settings(gcfg), "out_root": n["out_root"]}


def check(cfg: dict, globs: list[str]) -> bool:
    """Migration rule 1: every existing group config is reproduced by a point of
    sweeps/groups.yaml (semantic equality of `group_record`); groups the registry gained since
    are reported as NEW, not as failures."""
    from kaushik.core.sweep import diff_keys

    paths = sorted({Path(p) for pat in globs for p in glob.glob(str(ROOT / pat) if not Path(pat).is_absolute() else pat)})
    existing = {}
    for p in paths:
        raw = yaml.safe_load(p.read_text())
        if isinstance(raw, dict) and "group" in raw:
            existing[raw["group"]] = (p, raw)
    ok, matched = True, set()
    for gcfg in nz.group_points(cfg):
        name = gcfg["run_name"]
        if name not in existing:
            print(f"NEW  {name} (registered after the legacy configs were written)")
            continue
        matched.add(name)
        diffs = diff_keys(group_record(gcfg), existing[name][1])
        ok &= not diffs
        print(f"{'OK  ' if not diffs else 'DIFF'} {name}  ({rel(existing[name][0])})")
        for d in diffs[:40]:
            print(f"       {d}")
    unmatched = sorted(rel(p) for n, (p, _) in existing.items() if n not in matched)
    if unmatched:
        ok = False
        print(f"unmatched existing configs: {unmatched}")
    print("config equivalence:", "PASS" if ok else "FAIL")
    return ok


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="noise make_groups", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the groups; write nothing")
    ap.add_argument("--check", nargs="+", metavar="GLOB", help="legacy group configs to compare against")
    a = ap.parse_args(argv)
    entries = load_entries()
    groups, kl = sweeps(cfg, entries) if a.dry_run else write_sweeps(cfg, entries)
    for p in groups["axes"][0]:
        g = p["group"]
        print(f"{p['_label']['group']}: {len(g['entries'])}/{g['expected_seeds']} seeds, noise={g['noise']}, "
              f"measure_kl={g['measure_kl']}")
    print(f"{len(groups['axes'][0])} groups, {len(kl['axes'][0])} KL batches "
          f"({sum(len(p['kl.entries']) for p in kl['axes'][0])} adapters) -> {rel(nz.SWEEPS)}")
    if a.check and not check(cfg, a.check):
        raise SystemExit(1)
