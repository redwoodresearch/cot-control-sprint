"""Train one config on Modal (no eval): scripts/train_eval.py's training flow,
reused function by function, plus a manifest (dataset sha256, git commit) and
the kaushik/registry hand-off entry once the final adapter exists. Evaluation
is eval_scheduler.py's job: it runs every run's eval blocks under one global
GPU budget, which a per-run train+eval process could not respect.

    .venv/bin/python kaushik/orange/train_run.py kaushik/orange/configs/<run>.yaml [--dry-run]

Artifacts in the config's run_dir:
    config.yaml          the config as run (written before training starts)
    manifest.json        dataset paths + sha256 + row counts, git commit, Modal volume
    train.log            stdout/stderr of this process
    train_result.json    losses + checkpoint volume paths (written when training ends)
Registry: kaushik/registry/<run_name>.json (atomic write), unless meta.smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Must precede any cotcontrol Modal import: volume names resolve at import time.
os.environ.setdefault("COTCONTROL_MODAL_OWNER", "kaushik-reddy")
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))


def load_train_eval():
    """scripts/train_eval.py as a module (it is a script, not a package)."""
    key = "scripts_train_eval"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / "train_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(run_dir: Path, tc, raw: dict, n_rows: int) -> None:
    paths = [tc.data_path] if isinstance(tc.data_path, str) else list(tc.data_path)
    owner = os.environ.get("COTCONTROL_MODAL_OWNER")
    manifest = {
        "run_name": tc.run_name, "base_model": tc.base_model, "meta": raw.get("meta", {}),
        "seed": tc.seed,
        "datasets": [{"path": p, "sha256": sha256_file(ROOT / p),
                      "n_rows": sum(1 for line in (ROOT / p).open() if line.strip())}
                     for p in paths],
        "n_rows_loaded": n_rows,
        "git_commit": git_commit(),
        "config_sha256": hashlib.sha256(yaml.dump(raw, sort_keys=True).encode()).hexdigest(),
        "modal_owner": owner, "checkpoints_volume": f"cotcontrol_{owner}_checkpoints",
        "started": datetime.now().isoformat(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def write_registry_entry(run_dir: Path) -> Path | None:
    """kaushik/registry/<run_name>.json per kaushik/registry/README.md."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    meta = cfg.get("meta", {})
    if meta.get("smoke"):
        return None
    tr = json.loads((run_dir / "train_result.json").read_text())
    final = max(tr["checkpoints"], key=lambda c: int(c["step"]))
    blocks = [b for b in cfg["eval"].values() if isinstance(b, dict)]
    name = cfg["run_name"]
    entry = {
        "run_name": name,
        "experiment": "orange",
        "arm": meta.get("arm"),
        "base_model": cfg["base_model"],
        "seed": int(cfg["train"]["seed"]),
        "adapter_path": final["path"],
        "config_path": f"kaushik/orange/configs/{name}.yaml",
        "eval_chat_template_kwargs": blocks[0]["chat_template_kwargs"] if blocks else None,
        "noise": True,
        "measure_kl": True,
        # Blocks named final_* = test split, final checkpoint (persona/direct/stated x 4 modes
        # + standard/heldout/baseline).
        "noise_eval_blocks": f"kaushik/orange/configs/{name}.yaml#final",
    }
    reg = ROOT / "kaushik" / "registry"
    reg.mkdir(parents=True, exist_ok=True)
    tmp = reg / f"{name}.json.tmp"
    tmp.write_text(json.dumps(entry, indent=2))
    out = reg / f"{name}.json"
    tmp.replace(out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true", help="resolve config + dataset, no Modal")
    args = ap.parse_args(argv)

    te = load_train_eval()
    tc, ec, raw = te.load_config(Path(args.config))
    data = te.load_sft_rows(tc.data_path)
    blocks = te.eval_blocks(ec)
    if args.dry_run:
        print(yaml.dump({"train": asdict(tc)}, sort_keys=False))
        print(f"dataset: {len(data)} rows from {tc.data_path}")
        print(f"eval blocks ({len(blocks)}): {[n for n, _ in blocks]}")
        return

    tc.run_name = f"{tc.run_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / raw["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    from cotcontrol.run_logging import tee_stdio

    with tee_stdio(run_dir / "train.log"):
        print(f"run dir: {run_dir}\nmodal run name: {tc.run_name}")
        (run_dir / "config.yaml").write_text(yaml.dump(raw | {"eval": ec}, sort_keys=False, width=1000))
        write_manifest(run_dir, tc, raw, len(data))
        train_result = te.with_slot_retry(
            lambda: te.train(tc, data, timestamp_run_name=False), "train")
        (run_dir / "train_result.json").write_text(json.dumps(train_result, indent=2))
        print(f"Training done: {len(train_result['checkpoints'])} checkpoints, "
              f"final_loss={train_result['final_loss']}, mask={train_result['mask_verification']}")
        reg = write_registry_entry(run_dir)
        if reg:
            print(f"registry entry: {reg}")


if __name__ == "__main__":
    main()
