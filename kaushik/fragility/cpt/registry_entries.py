"""kaushik/registry/<run>.json for every finished cpt run (schema: ../../registry/README.md).

1B (`cpt`): two entries per run, `<run>` = checkpoint-200 and `<run>-step50` =
checkpoint-50 (the 1A-matched dose), both with noise controls + KL. Low-LR 1B
(`cpt_lowlr`) and the selected 1C runs (sweeps/cptk_selected.yaml): KL only.
Smoke and unselected sweep cells get no entry. Entries are built by
kaushik.core.registry.entry_for_run from the run's config.yaml + train_result.json
(+ the cpt extras: arm, step, lora_rank, train_params, k_effective) and written
atomically; an existing entry with the same adapter is left untouched.

    .venv/bin/python kaushik/run.py fragility/cpt registry_entries [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import registry as reg
from kaushik.core.config import resolve, runs_root
from kaushik.core.io import read_json, rel
from kaushik.core.sweep import expand_points, load_sweep

EXP = "fragility/cpt"
SELECTED_SWEEP = "kaushik/fragility/cpt/sweeps/cptk_selected.yaml"


def entries_for(rd: Path, test_blocks: list[str]) -> list[dict]:
    raw = yaml.safe_load((rd / "config.yaml").read_text())
    tr = read_json(rd / "train_result.json")
    meta = raw.get("cpt_meta") or {}
    exp = EXP + ("/k" if meta.get("arm") == "cpt_param_sweep" else "")
    cfg = resolve(exp, {"registry.noise": meta.get("arm") == "cpt"}, labels={"model": meta.get("model")}, run_name=rd.name)
    paths = {int(c["step"]): c["path"] for c in tr["checkpoints"]}
    out = []
    for step in meta["final_steps"]:
        if step not in paths:
            continue
        suffix = "" if step == max(meta["final_steps"]) else f"-step{step}"
        e = reg.entry_for_run(cfg, rd, final_step=step, run_name=rd.name + suffix)
        e.update({"experiment": EXP, "arm": meta["arm"], "noise_eval_block_names": list(test_blocks), "step": step,
                  "lora_rank": meta["rank"], "train_params": raw["elicitation"]["train_params"],
                  "k_effective": (tr.get("mask_verification") or {}).get("n_changed_inside_mask")})
        out.append(e)
    return out


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="rewrite entries whose adapter is unchanged")
    args = ap.parse_args(argv)
    selected = {p["run_name"] for p in expand_points(load_sweep(ROOT / SELECTED_SWEEP))}
    n = 0
    for rd in sorted(p for p in runs_root(cfg).glob("cpt*") if p.is_dir()):
        if not (rd / "train_result.json").exists() or not (rd / "config.yaml").exists():
            continue
        meta = (yaml.safe_load((rd / "config.yaml").read_text()) or {}).get("cpt_meta") or {}
        if meta.get("arm") not in ("cpt", "cpt_lowlr") and rd.name not in selected:
            continue
        for e in entries_for(rd, cfg["analysis"]["test_blocks"]):
            old = reg.read(e["run_name"])
            if old and old.get("adapter_path") == e["adapter_path"] and not args.force:
                continue
            n += 1
            if args.dry_run:
                print(f"would write {rel(reg.path_for(e['run_name']))} -> {e['adapter_path']}")
                print(json.dumps(e, indent=1))
            else:
                out = reg.write(e, force=args.force)
                print(f"wrote {rel(out)} -> {e['adapter_path']}")
    print(f"{n} entries {'to write' if args.dry_run else 'written'}")


if __name__ == "__main__":
    main(resolve(EXP), sys.argv[1:])
