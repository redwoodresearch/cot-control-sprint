"""Migration checks for Experiment 3A (experiment job).

    run.py retry check                       # config equivalence: every sweep point (+ registry adapters) vs
                                             # the manifest.json of the existing run of the same name
    run.py retry check --analysis REF_DIR    # analysis equivalence: runs/main/results.md (and pilot) vs copies
                                             # taken before the migration (REF_DIR/results.md, pilot_results.md)

Config equivalence: the settings an arm records (run3a.manifest_view: items spec, sampling, attempts,
judge / critic, feedback budget, backend generate config, shard, api_slots) and a transfer run's
manifest (retry_transfer.manifest_view) are compared with the existing manifests. Keys whose
historical value is explained in the README are skipped (EXPLAINED).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.io import load_py, rel
from kaushik.core.sweep import diff_keys, load_sweep, resolve_sweep
from kaushik.retry import retry_common as rc
from kaushik.retry import retry_correlate as corr
from kaushik.retry import retry_transfer as tr

HERE = Path(__file__).resolve().parent
# Historical differences explained in the README (not config errors): ModalGenerateConfig gained
# `scaledown_window` after the runs (cotcontrol d30c268); the transfer driver ran with api_slots 2 before the
# pilot/main launch raised it to 4; the smoke runs predate per-shard seeds and the per-model shard sizes.
EXPLAINED = {"backend.generate_config.scaledown_window", "backend.api_slots"}
EXPLAINED_RUNS = {"smoke": {"backend.per_shard_seeds", "backend.chunk", "backend.shard"}}


def expected(cfg: dict, arm_manifest: bool = True, lora: dict | None = None, run: str | None = None) -> dict:
    """What an arm (or transfer) run of this config would record, restricted to the old manifests' keys."""
    if arm_manifest:
        run3a = load_py(HERE / "run3a.py")
        n_items = len(rc.load_items(cfg["base_model"], **rc.item_spec(cfg)))
        _, info = rc.make_generate_fn(cfg, 1, lora_path=(lora or {}).get("lora_path"),
                                      chat_template_kwargs=(lora or {}).get("chat_template_kwargs", "default"))
        view = run3a.manifest_view(cfg, cfg["arm"], n_items, info, lora, run)
        view.pop("power")   # derived from n_items
        return view
    _, info = rc.make_generate_fn(cfg, 1)
    view = tr.manifest_view(cfg, None, info)
    view.pop("n_jobs")      # depends on the context pool at launch time
    return view


def check_configs(cfg: dict) -> tuple[list[str], list[str], list[str]]:
    ok, diffs, missing = [], [], []

    def compare(name: str, man: Path, exp: dict) -> None:
        if not man.exists():
            missing.append(name)
            return
        old = json.loads(man.read_text())
        tc = old.get("transfer_config") or {}
        if "one2one_questions_used" in tc:   # Modal transfers recorded the default (200) next to the value used (100)
            old["transfer_config"] = {**{k: v for k, v in tc.items() if not k.endswith("_used")},
                                      "one2one_questions": tc["one2one_questions_used"], "draws": tc["draws_used"]}
        skip = EXPLAINED | EXPLAINED_RUNS.get(name.split("/")[0], set())
        d = [x for x in diff_keys(exp, {k: old.get(k) for k in exp}) if x.split(":")[0].split(" (")[0] not in skip]
        (diffs.extend(f"{name}: {x}" for x in d) if d else ok.append(name))

    for sweep in sorted((HERE / "sweeps").glob("*.yaml")):
        s = load_sweep(sweep)
        for _, c in resolve_sweep(s):
            if s["job"] == "run3a":
                compare(rc.run_name_of(c), rc.run_dir_for(c) / "manifest.json", expected(c, run=str(c["run"])))
            elif s["job"] == "transfer":
                compare(f"{c['run']}/transfer/{c['model']}", tr.transfer_dir(c) / "manifest.json", expected(c, arm_manifest=False))
            elif s["job"] == "ckpt":
                for e in corr.registry_entries(c):
                    if corr.label_for(e["base_model"], c) != c["model"]:
                        continue
                    ecfg = corr.entry_config(c, e)
                    for arm in ecfg["arms"]:
                        acfg = {**ecfg, "arm": arm}
                        name = f"{ecfg['run_name']}/{arm}"
                        compare(name, rc.run_dir_for(ecfg) / arm / "manifest.json", expected(acfg, lora=corr.entry_lora(e), run=e["run_name"]))
    return ok, diffs, missing


def check_analysis(ref: Path) -> list[str]:
    problems = []
    for name, ours in (("results.md", rc.RUNS / "main" / "results.md"), ("pilot_results.md", rc.RUNS / "pilot" / "results.md")):
        old = ref / name
        if not old.exists() or not ours.exists():
            continue
        if old.read_bytes() != ours.read_bytes():
            problems.append(f"{rel(ours)} differs from {old}")
    return problems


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="retry check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", metavar="REF_DIR", default=None)
    a = ap.parse_args(argv)
    if a.analysis:
        problems = check_analysis(Path(a.analysis))
        print("\n".join(problems))
        print(f"analysis equivalence: {'PASS' if not problems else 'FAIL'}")
        if problems:
            raise SystemExit(1)
        return
    ok, diffs, missing = check_configs(cfg)
    print("\n".join(diffs))
    print(f"{len(ok)} runs match their manifest; {len(missing)} not run yet "
          f"({', '.join(missing[:6])}{', ...' if len(missing) > 6 else ''})")
    print(f"config equivalence: {'PASS' if not diffs else f'FAIL ({len(diffs)} differences)'}")
    if diffs:
        raise SystemExit(1)
