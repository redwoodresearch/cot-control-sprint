"""kaushik/registry/<run_name>.json: the adapter hand-off between experiment folders
(schema: kaushik/registry/README.md). Entries are written atomically and never
rewritten by this module unless `force=True` or the adapter path changed.

    from kaushik.core import registry
    registry.write(registry.entry_for_run(cfg, run_dir))   # after training
    registry.read("oex-sft_eugene-qwen8b-seed0")["adapter_path"]
    registry.adapter_path("orange-qwen36_27b-seed0")

`entry_for_run` reads the run's config.yaml + train_result.json. The resolved
config's optional `registry:` section supplies `noise`, `measure_kl`,
`noise_eval_blocks`, `config_path` and `extra` (free-form keys copied into the
entry, e.g. train_prompt); it is a core-only key and never reaches train_eval.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from kaushik.core.config import defaults, model_key_of
from kaushik.core.io import ROOT, atomic_write_json, read_json, rel

REQUIRED = ("run_name", "experiment", "arm", "base_model", "seed", "adapter_path", "config_path",
            "eval_chat_template_kwargs", "noise", "measure_kl")


def registry_dir() -> Path:
    return ROOT / defaults()["paths"]["registry"]


def path_for(run_name: str) -> Path:
    return registry_dir() / f"{run_name}.json"


def read(run_name: str) -> dict | None:
    return read_json(path_for(run_name))


def read_all() -> dict[str, dict]:
    out = {}
    for p in sorted(registry_dir().glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    return out


def adapter_path(run_name: str) -> str:
    entry = read(run_name)
    if not entry:
        raise FileNotFoundError(f"no registry entry {path_for(run_name)}")
    return entry["adapter_path"]


def validate(entry: dict) -> None:
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"registry entry {entry.get('run_name')!r} lacks {missing}")


def write(entry: dict, force: bool = False) -> Path | None:
    """Atomic write; skipped (-> None) when an entry with the same adapter exists."""
    validate(entry)
    out = path_for(entry["run_name"])
    old = read_json(out)
    if old and not force and old.get("adapter_path") == entry["adapter_path"]:
        return None
    return atomic_write_json(entry, out)


def final_checkpoint(train_result: dict, final_step: int | None = None) -> dict:
    """{step, path} of the final checkpoint: `final_step` arg > train_result.final_step
    > the largest step."""
    ckpts = train_result["checkpoints"]
    step = final_step if final_step is not None else train_result.get("final_step")
    if step is None:
        return max(ckpts, key=lambda c: int(c["step"]))
    for c in ckpts:
        if int(c["step"]) == int(step):
            return c
    raise ValueError(f"checkpoint-{step} not in {[c['step'] for c in ckpts]}")


def entry_for_run(cfg: dict, run_dir: Path, final_step: int | None = None, run_name: str | None = None) -> dict:
    """Registry entry for a finished run from its resolved config + train_result.json."""
    run_dir = Path(run_dir)
    tr = json.loads((run_dir / "train_result.json").read_text())
    raw = yaml.safe_load((run_dir / "config.yaml").read_text()) if (run_dir / "config.yaml").exists() else cfg
    final = final_checkpoint(tr, final_step)
    reg = cfg.get("registry") or {}
    blocks = [b for b in (raw.get("eval") or {}).values() if isinstance(b, dict)]
    meta, task = raw.get("meta") or {}, raw.get("task") or {}
    name = run_name or raw.get("run_name") or cfg["run_name"]
    entry = {
        "run_name": name,
        "experiment": cfg.get("experiment") or meta.get("experiment"),
        "arm": meta.get("arm") or task.get("arm"),
        "base_model": raw["base_model"],
        "seed": int((raw.get("train") or {}).get("seed", 0)),
        "adapter_path": final["path"],
        "config_path": reg.get("config_path") or rel(run_dir / "config.yaml"),
        "eval_chat_template_kwargs": blocks[0].get("chat_template_kwargs") if blocks else None,
        "noise": bool(reg.get("noise", False)),
        "measure_kl": bool(reg.get("measure_kl", False)),
    }
    if reg.get("noise_eval_blocks"):
        entry["noise_eval_blocks"] = reg["noise_eval_blocks"]
    entry.update({
        "model_key": cfg.get("model") or model_key_of(raw["base_model"]),
        "final_step": int(final["step"]),
        "stop_reason": tr.get("stop_reason"),
        "checkpoint_run_name": tr.get("run_name"),
        "run_dir": rel(run_dir),
        "checkpoints": {str(c["step"]): c["path"] for c in tr["checkpoints"]},
        "val_blocks": [k for k, v in (raw.get("eval") or {}).items() if isinstance(v, dict)],
    })
    entry.update(reg.get("extra") or {})
    if run_name and run_name != raw.get("run_name"):
        entry["source_run_name"] = raw.get("run_name")
    return {k: v for k, v in entry.items() if v is not None or k in REQUIRED}
