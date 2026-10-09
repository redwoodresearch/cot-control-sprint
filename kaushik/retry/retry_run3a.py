"""Experiment 3A driver: run every (model, arm) retry job of one run.

Layout: runs/<run>/<model>/<arm>/{attempts.jsonl, errors.jsonl, episodes.json,
mentions.jsonl, manifest.json, log.txt}. attempts.jsonl holds every attempt's
full prompt, output, reasoning, grades, judge outputs and feedback (critic
prompt + output included); the job resumes from it.

All Modal models of a run share one process (one app.run(), one GpuBudget) so
the GPU cap is a hard, process-wide number. OpenRouter models can run in the
same process or separately.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from retry_arms import build_arm
from retry_common import (
    CFG,
    MODELS,
    RUNS,
    GpuBudget,
    load_items,
    log_to,
    make_generate_fn,
    modal_app_context,
    power_statement,
    write_manifest,
)

from cotcontrol.eval.retry import judge_mentions, load_attempts, run_retry
from cotcontrol.inference.openrouter import GenerateConfig


def judge_config() -> GenerateConfig:
    return GenerateConfig(temperature=1.0, max_tokens=10000, max_concurrency=200, max_retries=8)


async def run_arm(run_dir: Path, model_label: str, mcfg: dict, arm_name: str, items: list[dict],
                  gen_fn, backend_info: dict, n_attempts: int, item_spec: dict, log) -> dict:
    out = run_dir / model_label / arm_name
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{run_dir.name}:{model_label}"  # checkpoint runs: the registry run_name, so log lines are unambiguous
    crit = CFG["critic"]
    arm = build_arm(arm_name, mcfg["model"], crit["model"], crit["provider"], CFG["max_feedback_chars"])
    write_manifest(out / "manifest.json", run=run_dir.name, model_label=model_label, model=mcfg["model"],
                   arm=arm_name, n_items=len(items), n_attempts=n_attempts, items=item_spec,
                   state_requirement=arm.state_requirement, stop_on_success=arm.stop_on_success,
                   max_feedback_chars=arm.max_feedback_chars, arm_info=arm.info,
                   judge_model=CFG["judge_model"], sampling=CFG["sampling"], backend=backend_info,
                   power=power_statement(len(items)))
    try:
        episodes = await run_retry(items, arm, gen_fn, model=mcfg["model"], n_attempts=n_attempts,
                                   judge_model=CFG["judge_model"], judge_config=judge_config(),
                                   save_dir=out, log=log, max_error_retries=CFG["max_error_retries"])
    except BaseException as e:  # log NOW (the gather above only reports when every arm has finished)
        import traceback
        log(f"[{tag}/{arm_name}] ARM FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise
    await mentions_pass(out, log)
    done = sum(e["success_attempt"] is not None for e in episodes)
    log(f"[{tag}/{arm_name}] finished: {done}/{len(episodes)} episodes honest-compliant by attempt {n_attempts}")
    return {"model": model_label, "arm": arm_name, "n": len(episodes), "success": done}


async def mentions_pass(out: Path, log) -> None:
    """Meta-discussion judge on every honest-compliant attempt not yet judged."""
    path = out / "mentions.jsonl"
    seen = set()
    if path.exists():
        seen = {(json.loads(line)["item_id"], json.loads(line)["attempt"]) for line in path.open() if line.strip()}
    recs = [a for recs in load_attempts(out / "attempts.jsonl").values() for a in recs
            if a["honest"] and (a["item_id"], a["attempt"]) not in seen]
    if not recs:
        return
    log(f"[{out.parent.name}/{out.name}] meta-discussion judge on {len(recs)} compliant attempts")
    judged = await judge_mentions(recs, CFG["judge_model"], judge_config())
    with path.open("a") as f:
        for a, j in zip(recs, judged):
            f.write(json.dumps({"item_id": a["item_id"], "attempt": a["attempt"], "mode": a["mode"],
                                "meta_discussion": j["meta_discussion"], "judge_output": j["judge_output"],
                                "mentions_feedback": a["mentions_feedback"], "judge_model": CFG["judge_model"]}) + "\n")


async def run_models(run_name: str, labels: list[str], arms: list[str], item_spec: dict,
                     n_attempts: int, gpu_cap: int | None = None, lora: dict | None = None,
                     budget: GpuBudget | None = None, own_app: bool = True) -> list[dict]:
    """All (model, arm) jobs of a run concurrently. `lora` (optional) =
    {"lora_path", "chat_template_kwargs"} for a trained checkpoint served on
    the model's base engine. A caller that already holds the Modal app open
    (and a shared GpuBudget) passes own_app=False."""
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = log_to(run_dir / "log.txt")
    m = CFG["modal"]
    modal_labels = [lb for lb in labels if MODELS[lb]["backend"] == "modal"]
    # One pool per base model; the process's GPU cap is split across pools and
    # enforced per pool (max_containers), so a driver can never hold more live
    # GPUs than --gpus. Live caps: configs/gpu_caps.json["<run>/<label>"].
    per_pool = max(1, (gpu_cap or m["gpu_cap"]) // max(1, len(modal_labels)))
    if len(modal_labels) > 1:
        log(f"WARNING: {len(modal_labels)} Modal pools in one process; hard cap {per_pool} GPUs each")
    needs_modal = own_app and bool(modal_labels)
    jobs = []
    for lb in labels:
        mcfg = MODELS[lb]
        items = load_items(mcfg["model"], **item_spec)
        log(f"[{lb}] {len(items)} items; {power_statement(len(items))}")
        pool_budget = budget or GpuBudget(per_pool, None, m["gpu_cap_after_hours"], name=f"{run_name.split('/')[0]}/{lb}")
        gen_fn, info = make_generate_fn(mcfg, pool_budget, lora_path=(lora or {}).get("lora_path"),
                                        chat_template_kwargs=(lora or {}).get("chat_template_kwargs"),
                                        hard_cap=per_pool, cache_dir=run_dir / lb / "gen_cache")
        if lora:
            info["registry"] = lora
        for arm in arms:
            jobs.append(run_arm(run_dir, lb, mcfg, arm, items, gen_fn, info, n_attempts, item_spec,
                                log_to(run_dir / lb / arm / "log.txt")))
    async with modal_app_context(needs_modal, run_dir / "modal_app_id.txt"):
        results = await asyncio.gather(*jobs, return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            log(f"ERROR an arm failed: {type(r).__name__}: {r}")
    results = [r for r in results if not isinstance(r, BaseException)]
    (run_dir / "summary.json").write_text(json.dumps(results, indent=1))
    log(f"run {run_name} done: {results}")
    return results


async def run_many(jobs: list[tuple[str, list[str], list[str], dict]], n_attempts: int, gpu_cap: int) -> None:
    """Several (run_name, labels, arms, item_spec) runs in ONE process and ONE
    Modal app (the workspace caps ephemeral apps). gpu_cap is the hard
    per-pool ceiling; the live per-pool caps in configs/gpu_caps.json
    ("<run>/<label>") must sum to the coordinator's limit."""
    from retry_common import patch_engine_pool
    patch_engine_pool(gpu_cap)
    needs_modal = any(MODELS[lb]["backend"] == "modal" for _, labels, _, _ in jobs for lb in labels)
    async with modal_app_context(needs_modal, RUNS / jobs[0][0] / "modal_app_id.txt"):
        await asyncio.gather(*[
            run_models(run, labels, arms, spec, n_attempts, gpu_cap=gpu_cap * max(1, len(labels)), own_app=False)
            for run, labels, arms, spec in jobs], return_exceptions=True)
