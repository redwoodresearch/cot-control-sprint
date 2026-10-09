"""Experiment 3 correlation inputs: poll kaushik/registry/*.json for trained
adapters (Experiments 1/2) and run the reduced 3A on each (bare + independent,
block + block_once; test split, 2 modes per question = ~1000 items/arm, so a
single arm has SE <= 1.6 pts and an arm-vs-baseline difference <= 2.3 pts at
p=.5, <= 2 pts for p <= .35). Outputs: runs/<run>/<registry run_name>/<label>/<arm>/.

One Modal app context is held for the whole polling loop; every adapter is
served on its base model's warm engine pool under the shared GPU cap.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from retry_common import CFG, HERE, MODELS, RUNS, GpuBudget, log_to
from retry_run3a import run_models

REGISTRY = HERE.parent / "registry"
SENTINEL = REGISTRY / "ALL_DONE"  # optional: coordinator/owners touch this when Experiments 1 and 2 are finished


def registry_entries() -> list[dict]:
    out = []
    for p in sorted(REGISTRY.glob("*.json")):
        try:
            e = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue  # being written
        if {"run_name", "base_model", "adapter_path"} <= set(e):
            if "smoke" in e["run_name"]:  # pipeline tests, not real arms (coordinator, 2026-10-08)
                continue
            e["_file"] = str(p)
            out.append(e)
    return out


def priority(e: dict) -> tuple:
    """Lowest seed of every (experiment, arm, base model) first, so the
    correlation gets one adapter per arm before any arm gets a second seed."""
    return (int(e.get("seed", 0) or 0), e.get("experiment", ""), e.get("arm", ""), e["run_name"])


def label_for(base_model: str) -> str | None:
    for lb, m in MODELS.items():
        if m["backend"] == "modal" and m["model"].lower() == base_model.lower():
            return lb
    return None


async def run_entry(run_name: str, e: dict, budget: GpuBudget, log) -> None:
    lb = label_for(e["base_model"])
    out = RUNS / run_name / e["run_name"]
    if lb is None:
        log(f"skip {e['run_name']}: no Modal model entry for base {e['base_model']}")
        return
    if (out / "done.json").exists():
        return
    log(f"start {e['run_name']} ({lb}, adapter {e['adapter_path']})")
    spec = dict(CFG["checkpoint"])
    await run_models(f"{run_name}/{e['run_name']}", [lb], CFG["checkpoint_arms"], spec, CFG["n_attempts"],
                     budget=budget, own_app=False,
                     lora={"lora_path": e["adapter_path"], "chat_template_kwargs": e.get("eval_chat_template_kwargs"),
                           "registry_entry": {k: v for k, v in e.items() if not k.startswith("_")}})
    (out / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "entry": e}, indent=1))
    log(f"done  {e['run_name']}")


async def poll(run_name: str, hours: float, idle_hours: float, poll_min: float, gpu_cap: int | None,
               max_parallel: int, labels: list[str] | None = None) -> None:
    """labels: only adapters whose base model maps to these Modal labels (run
    one poller per base model so each process holds a single engine pool)."""
    from retry_common import modal_app_context, patch_engine_pool
    run_dir = RUNS / run_name
    log = log_to(run_dir / "log.txt")
    m = CFG["modal"]
    cap = gpu_cap or m["gpu_cap"]
    patch_engine_pool(cap)
    labels = set(labels or [lb for lb, mc in MODELS.items() if mc["backend"] == "modal"])
    # one live budget per base-model pool: configs/gpu_caps.json["<run>/<label>"]
    budgets = {lb: GpuBudget(cap, None, m["gpu_cap_after_hours"], name=f"{run_name}/{lb}") for lb in labels}
    sem = asyncio.Semaphore(max_parallel)
    t0, last_new = time.time(), time.time()
    running: dict[str, asyncio.Task] = {}

    async def guarded(e):
        async with sem:
            try:
                await run_entry(run_name, e, budgets[label_for(e["base_model"])], log)
            except Exception as ex:  # keep polling; the failed entry is retried next scan
                log(f"ERROR {e['run_name']}: {type(ex).__name__}: {ex}")

    async with modal_app_context(True, run_dir / "modal_app_id.txt"):
        while True:
            for e in sorted(registry_entries(), key=priority):
                name = e["run_name"]
                if label_for(e["base_model"]) not in labels:
                    continue
                if name in running and not running[name].done():
                    continue
                if (RUNS / run_name / name / "done.json").exists():
                    continue
                if name in running and running[name].done() and not running[name].exception():
                    continue  # finished (or skipped) since last scan
                running[name] = asyncio.create_task(guarded(e))
                last_new = time.time()
            active = sum(not t.done() for t in running.values())
            elapsed_h = (time.time() - t0) / 3600
            idle_h = (time.time() - last_new) / 3600
            log(f"poll: {len(registry_entries())} registry entries, {active} running, "
                f"{sum((RUNS / run_name / n / 'done.json').exists() for n in running)} done; "
                f"{elapsed_h:.1f}h elapsed, {idle_h:.1f}h since last new entry, GPUs in flight "
                + json.dumps({lb: b.in_flight for lb, b in budgets.items()}))
            if active == 0 and (SENTINEL.exists() or idle_h >= idle_hours or elapsed_h >= hours):
                log("stopping: " + ("ALL_DONE sentinel" if SENTINEL.exists() else "idle/elapsed limit reached"))
                break
            if elapsed_h >= hours and active:
                log("time limit reached with jobs still running; waiting for them")
                await asyncio.gather(*[t for t in running.values() if not t.done()], return_exceptions=True)
                break
            await asyncio.sleep(poll_min * 60)


def ckpt_results(run_name: str) -> list[dict]:
    """[{run_name, entry, dir}] for finished checkpoint runs."""
    out = []
    for d in sorted((RUNS / run_name).glob("*/")):
        if (d / "done.json").exists():
            out.append({"run_name": d.name, "entry": json.loads((d / "done.json").read_text())["entry"], "dir": d})
    return out


def owning_eval_results(entry: dict) -> list[Path]:
    """Standard-eval result JSONs the owning experiment produced for this
    adapter's final checkpoint (train_eval layout: runs/<run>/eval/<block>/
    checkpoint-<step>.json), if present."""
    step = Path(entry["adapter_path"]).name
    if entry.get("run_dir"):  # registry entries name the owning run folder
        d = HERE.parents[1] / entry["run_dir"]
        return sorted(d.glob(f"eval/*/{step}.json")) + sorted(d.glob(f"**/{step}.json")) if d.exists() else []
    exp_dir = HERE.parent / entry.get("experiment", "")
    return sorted(exp_dir.glob(f"**/runs/**/{entry['run_name']}/eval/*/{step}.json")) if exp_dir.exists() else []
