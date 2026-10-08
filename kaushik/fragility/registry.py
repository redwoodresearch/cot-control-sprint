"""Write kaushik/registry/<run_name>.json for every fragility run whose FINAL
adapter exists (schema: kaushik/registry/README.md). Entries ask the noise
agent for KL-matched random-LoRA controls (noise: true) and KL(base || adapter)
(measure_kl: true). Extra keys (model_key, run_dir, val_blocks, final_blocks)
help consumers replicate exactly our eval blocks; `#final` = every block whose
name starts with "final" (test split, final checkpoint).

    .venv/bin/python kaushik/run.py fragility registry [--dry-run]
"""

from __future__ import annotations

import argparse
import json

import yaml
from common import HERE, ROOT, RUNS, atomic_write_json

REGISTRY = HERE.parents[0] / "registry"


def entry_for(run_dir) -> dict | None:
    cfg_path, tr_path = run_dir / "config.yaml", run_dir / "train_result.json"
    if not (cfg_path.exists() and tr_path.exists()):
        return None
    raw, tr = yaml.safe_load(cfg_path.read_text()), json.loads(tr_path.read_text())
    final = max(tr["checkpoints"], key=lambda c: c["step"])
    if final["step"] != raw["train"]["max_steps"]:
        print(f"WARNING {raw['run_name']}: last checkpoint is step {final['step']}, "
              f"expected {raw['train']['max_steps']}; skipping")
        return None
    config_path = f"kaushik/fragility/configs/{raw['run_name']}.yaml"
    return {
        "run_name": raw["run_name"],
        "experiment": "fragility",
        "arm": raw["arm"],
        "base_model": raw["base_model"],
        "seed": raw["train"]["seed"],
        "adapter_path": final["path"],
        "config_path": config_path,
        "eval_chat_template_kwargs": next(iter(raw["eval"].values()))["chat_template_kwargs"],
        "noise": True,
        "measure_kl": True,
        "noise_eval_blocks": f"{config_path}#final",
        # extras
        "model_key": raw["model_key"],
        "checkpoint_run_name": tr["run_name"],
        "run_dir": str(run_dir.relative_to(ROOT)),
        "val_blocks": [k for k in raw["eval"] if not k.startswith("final")],
        "final_blocks": [k for k in raw["eval"] if k.startswith("final")],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    n = 0
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and p.name != "smoke"):
        entry = entry_for(run_dir)
        if entry is None:
            continue
        path = REGISTRY / f"{entry['run_name']}.json"
        if path.exists() and json.loads(path.read_text()).get("adapter_path") == entry["adapter_path"]:
            continue
        if not args.dry_run:
            atomic_write_json(entry, path)
        print(f"{'would write' if args.dry_run else 'wrote'} {path.relative_to(ROOT)} -> {entry['adapter_path']}")
        n += 1
    print(f"{n} registry entries {'to write' if args.dry_run else 'written'}")


if __name__ == "__main__":
    main()
