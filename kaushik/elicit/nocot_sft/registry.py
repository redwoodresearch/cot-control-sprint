"""Write kaushik/registry/<run_name>.json for finished runs of the selected arm(s)
(schema: kaushik/registry/README.md), asking kaushik/noise for KL + KL-matched
random-LoRA controls on the final checkpoint's test blocks.

    .venv/bin/python kaushik/run.py elicit/nocot_sft registry --arms <arm> [--step N] [--no-noise] [--dry-run]
"""

from __future__ import annotations

import argparse
import json

import yaml
from common import EXPERIMENT, REGISTRY, ROOT, RUNS, atomic_write_json


def entry_for(run_dir, step: int | None, noise: bool) -> dict | None:
    cfg_path, tr_path = run_dir / "config.yaml", run_dir / "train_result.json"
    if not (cfg_path.exists() and tr_path.exists()):
        return None
    raw, tr = yaml.safe_load(cfg_path.read_text()), json.loads(tr_path.read_text())
    if "source_run" in raw:  # eval-only helper dir (test_step / ref_1a): the adapter belongs to its source run
        return None
    want = step if step is not None else raw["train"]["max_steps"]
    ckpt = next((c for c in tr["checkpoints"] if c["step"] == want), None)
    if ckpt is None:
        print(f"WARNING {raw['run_name']}: no checkpoint at step {want}; skipping")
        return None
    config_path = f"kaushik/{EXPERIMENT}/configs/{raw['run_name']}.yaml"
    suffix = "" if want == raw["train"]["max_steps"] else f"-step{want}"
    return {
        "run_name": raw["run_name"] + suffix,
        "experiment": EXPERIMENT,
        "arm": raw["arm"],
        "base_model": raw["base_model"],
        "seed": raw["train"]["seed"],
        "adapter_path": ckpt["path"],
        "config_path": config_path,
        "eval_chat_template_kwargs": raw["eval"]["final_modes"]["chat_template_kwargs"],
        "noise": noise,
        "measure_kl": True,
        "noise_eval_blocks": f"{config_path}#final",
        "n_seeds": 3,
        "model_key": raw["model_key"],
        "checkpoint_run_name": tr["run_name"],
        "run_dir": str(run_dir.relative_to(ROOT)),
        "step": want,
        "val_blocks": [k for k in raw["eval"] if k.startswith("val")],
        "final_blocks": [k for k in raw["eval"] if k.startswith("final")],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: final)")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    n = 0
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and p.name.startswith("nocot-")):
        if not any(run_dir.name.startswith(f"nocot-{a}-") for a in args.arms):
            continue
        entry = entry_for(run_dir, args.step, not args.no_noise)
        if entry is None:
            continue
        path = REGISTRY / f"{entry['run_name']}.json"
        if not args.dry_run:
            atomic_write_json(entry, path)
        print(f"{'would write' if args.dry_run else 'wrote'} {path.relative_to(ROOT)} -> {entry['adapter_path']}")
        n += 1
    print(f"{n} registry entries")


if __name__ == "__main__":
    main()
