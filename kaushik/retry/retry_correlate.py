"""Experiment 3 correlation inputs: the reduced 3A (config `checkpoint_arms`; test split, 1 seeded
mode per question = 500 items/arm, SE <= 2.3 pts for an arm-vs-baseline difference at p=.5) on every
trained adapter in kaushik/registry/*.json (Experiments 1/2). Outputs:
runs/ckpt/<registry run_name>/<model>/<arm>/ (+ done.json once every arm finished).

Library for the `ckpt` job (ckpt.py): registry scan, priority, adapter jobs, finished-run lookup
and the owning experiment's standard-eval results.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from kaushik.core import registry as reg
from kaushik.core.config import resolve
from kaushik.core.io import KAUSHIK, ROOT, load_py
from kaushik.retry import retry_common as rc

HERE = Path(__file__).resolve().parent
EXP = "retry"


def registry_entries(cfg: dict) -> list[dict]:
    """Non-smoke registry entries (run_name, base_model, adapter_path present)."""
    skip = cfg["ckpt"].get("skip_pattern")
    out = []
    for name, e in reg.read_all().items():
        if {"run_name", "base_model", "adapter_path"} <= set(e) and not (skip and re.search(skip, e["run_name"])):
            out.append({**e, "_file": str(reg.path_for(name))})
    return out


def priority(e: dict) -> tuple:
    """Lowest seed of every (experiment, arm, base model) first, so the
    correlation gets one adapter per arm before any arm gets a second seed."""
    return (int(e.get("seed", 0) or 0), e.get("experiment", ""), e.get("arm", ""), e["run_name"])


def label_for(base_model: str, cfg: dict | None = None) -> str | None:
    """The Modal model key serving this base model (None for API-only models)."""
    return rc.modal_label_of(cfg or resolve(EXP), base_model)


def entry_config(cfg: dict, e: dict) -> dict:
    """Run config of one adapter: the base model's checkpoint settings, run_name ckpt/<entry>/<label>."""
    label = label_for(e["base_model"], cfg)
    return resolve(cfg.get("experiment") or EXP, {"model": label, "run": cfg["run"], "items": cfg["items"],
                                                 "arms": list(cfg["arms"]), "run_name": f"{cfg['run']}/{e['run_name']}/{label}"})


def entry_lora(e: dict) -> dict:
    return {"lora_path": e["adapter_path"], "chat_template_kwargs": e.get("eval_chat_template_kwargs"),
            "registry_entry": {k: v for k, v in e.items() if not k.startswith("_")}}


def entry_jobs(cfg: dict, e: dict) -> list:
    """One Job per checkpoint arm of one adapter; the last arm to finish writes done.json."""
    run3a = load_py(HERE / "run3a.py")
    ecfg = entry_config(cfg, e)
    entry_dir = rc.RUNS / str(cfg["run"]) / e["run_name"]
    model_dir = rc.run_dir_for(ecfg)
    items = rc.load_items(ecfg["base_model"], **rc.item_spec(ecfg))
    arms = list(ecfg["arms"])
    jobs = []
    for i, arm in enumerate(arms):
        acfg = {**ecfg, "arm": arm}
        job = run3a.arm_job(acfg, model_dir / arm, items, lora=entry_lora(e), priority=(1, *priority(e), i))
        inner = job.run

        async def run(inner=inner):
            await inner()
            if all((model_dir / a / "episodes.json").exists() for a in arms):
                (entry_dir / "done.json").write_text(json.dumps(
                    {"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "entry": {k: v for k, v in e.items() if not k.startswith("_")}}, indent=1))
        job.run = run
        jobs.append(job)
    return jobs


def adapter_jobs(cfg: dict) -> list:
    """Jobs for every registry adapter served by cfg['model'] (a ckpt sweep point), priority order."""
    return [j for e in sorted(registry_entries(cfg), key=priority)
            if label_for(e["base_model"], cfg) == cfg["model"] for j in entry_jobs(cfg, e)]


def ckpt_results(run_dir: Path) -> list[dict]:
    """[{run_name, entry, dir}] for finished checkpoint runs (done.json)."""
    out = []
    for d in sorted(p for p in Path(run_dir).iterdir() if p.is_dir()) if Path(run_dir).exists() else []:
        if (d / "done.json").exists():
            out.append({"run_name": d.name, "entry": json.loads((d / "done.json").read_text())["entry"], "dir": d})
    return out


def owning_eval_results(entry: dict) -> list[Path]:
    """Standard-eval result JSONs the owning experiment produced for this
    adapter's final checkpoint (train_eval layout: runs/<run>/eval/<block>/
    checkpoint-<step>.json), if present."""
    step = Path(entry["adapter_path"]).name
    if entry.get("run_dir"):  # registry entries name the owning run folder
        d = ROOT / entry["run_dir"]
        return sorted(d.glob(f"eval/*/{step}.json")) + sorted(d.glob(f"**/{step}.json")) if d.exists() else []
    exp_dir = KAUSHIK / entry.get("experiment", "")
    return sorted(exp_dir.glob(f"**/runs/**/{entry['run_name']}/eval/*/{step}.json")) if exp_dir.exists() else []
