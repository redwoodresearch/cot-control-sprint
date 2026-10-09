"""Train one SFT config on the deployed sub-orange-expand train app (no eval).

kaushik/orange/train_run.py's flow (train_eval.py's helpers, manifest.json, train.log,
train_result.json) with this folder's registry entry: experiment orange_expand, run
names prefixed oex-, noise/KL controls NOT requested (stage 2 decides that).

    .venv/bin/python kaushik/orange_expand/sft/train_run.py kaushik/orange_expand/sft/configs/<run>.yaml [--dry-run]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

import common  # env before any Modal import
from common import REGISTRY

orange_train_run = common.orange_mod("train_run")


def write_registry_entry(run_dir: Path) -> Path | None:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = cfg.get("meta", {})
    if meta.get("smoke"):
        return None
    tr = json.loads((run_dir / "train_result.json").read_text())
    final = max(tr["checkpoints"], key=lambda c: int(c["step"]))
    blocks = [b for b in cfg["eval"].values() if isinstance(b, dict)]
    name = cfg["run_name"]
    entry = {
        "run_name": name, "experiment": "orange_expand", "arm": meta.get("arm"),
        "base_model": cfg["base_model"], "seed": int(cfg["train"]["seed"]),
        "adapter_path": final["path"],
        "config_path": f"kaushik/orange_expand/sft/configs/{name}.yaml",
        "eval_chat_template_kwargs": blocks[0]["chat_template_kwargs"] if blocks else None,
        "noise": False, "measure_kl": False,
        "train_prompt": meta.get("train_prompt"), "steps": int(final["step"]),
        "checkpoints": {str(c["step"]): c["path"] for c in tr["checkpoints"]},
    }
    REGISTRY.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY / f"{name}.json.tmp"
    tmp.write_text(json.dumps(entry, indent=2))
    out = REGISTRY / f"{name}.json"
    tmp.replace(out)
    return out


orange_train_run.write_registry_entry = write_registry_entry  # main() resolves it by global name

if __name__ == "__main__":
    orange_train_run.main(sys.argv[1:])
