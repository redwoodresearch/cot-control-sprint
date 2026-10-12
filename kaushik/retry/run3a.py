"""Experiment 3A: one (run, model, arm) retry job (experiment job; sweeps: sweeps/{main,pilot,smoke}.yaml).

    run.py retry run3a run=main items=main model=gptoss20b arm=bare [--dry-run]
    run.py sweep retry/sweeps/main.yaml --dry-run

Layout: runs/<run>/<model>/<arm>/{attempts.jsonl, errors.jsonl, episodes.json, mentions.jsonl,
manifest.json, log.txt}; attempts.jsonl holds every attempt's full prompt, output, reasoning,
grades, judge outputs and feedback (critic prompt + output included) and the job resumes from it.
One scheduler Job per arm: weight = engine calls it keeps in flight (ceil(items / shard) x tp, capped
by modal.max_gpus_per_job; 0 for OpenRouter models); done once episodes.json exists. Finished
generation shards are cached in runs/<run>/<model>/gen_cache/ (lossless restarts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core.scheduler import Job, run_scheduler
from kaushik.retry import retry_common as rc
from kaushik.retry.retry_arms import build_arm

EXP = "retry"


def preflight(cfgs: list[dict]) -> None:
    if any(rc.is_modal(c) for c in cfgs) and not rc.deployed_mode():
        raise SystemExit("deployed mode only: COTCONTROL_MODAL_INFERENCE_APP is unset (run through kaushik/run.py)")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY missing (.env): the judge and critic need it")


def manifest_view(cfg: dict, arm_name: str, n_items: int, backend_info: dict | None, lora: dict | None = None,
                  run: str | None = None) -> dict:
    """The settings an arm records (check.py compares this with runs/*/manifest.json). `run` = the run
    dir two levels up (the run group, or the registry run_name for checkpoint arms)."""
    crit = cfg["critic"]
    arm = build_arm(arm_name, cfg["base_model"], crit["model"], crit["provider"], int(cfg["max_feedback_chars"]))
    info = dict(backend_info or {})
    if lora:
        info["registry"] = lora
    return {"run": run or str(cfg["run"]), "model_label": cfg["model"], "model": cfg["base_model"], "arm": arm_name,
            "n_items": n_items, "n_attempts": int(cfg["n_attempts"]), "items": rc.item_spec(cfg),
            "state_requirement": arm.state_requirement, "stop_on_success": arm.stop_on_success,
            "max_feedback_chars": arm.max_feedback_chars, "arm_info": arm.info, "judge_model": cfg["judge"]["model"],
            "sampling": cfg["sampling"], "backend": info, "power": rc.power_statement(n_items)}


async def run_arm(cfg: dict, out: Path, items: list[dict], gen_fn, backend_info: dict, log, lora: dict | None = None) -> dict:
    """One arm of one model: every item's retry episode, then the meta-discussion judge."""
    from cotcontrol.eval.retry import run_retry

    arm_name = cfg["arm"]
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg['run']}:{cfg['model']}"
    crit = cfg["critic"]
    arm = build_arm(arm_name, cfg["base_model"], crit["model"], crit["provider"], int(cfg["max_feedback_chars"]))
    rc.write_manifest(out / "manifest.json", **manifest_view(cfg, arm_name, len(items), backend_info, lora, out.parent.parent.name))
    try:
        episodes = await run_retry(items, arm, gen_fn, model=cfg["base_model"], n_attempts=int(cfg["n_attempts"]),
                                   judge_model=cfg["judge"]["model"], judge_config=rc.judge_config(cfg),
                                   save_dir=out, log=log, max_error_retries=int(cfg["max_error_retries"]))
    except BaseException as e:  # log NOW (the scheduler reports later)
        import traceback
        log(f"[{tag}/{arm_name}] ARM FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise
    await mentions_pass(cfg, out, log)
    done = sum(e["success_attempt"] is not None for e in episodes)
    log(f"[{tag}/{arm_name}] finished: {done}/{len(episodes)} episodes honest-compliant by attempt {cfg['n_attempts']}")
    return {"model": cfg["model"], "arm": arm_name, "n": len(episodes), "success": done}


async def mentions_pass(cfg: dict, out: Path, log) -> None:
    """Meta-discussion judge on every honest-compliant attempt not yet judged."""
    from cotcontrol.eval.retry import judge_mentions, load_attempts

    path = out / "mentions.jsonl"
    seen = set()
    if path.exists():
        seen = {(json.loads(line)["item_id"], json.loads(line)["attempt"]) for line in path.open() if line.strip()}
    recs = [a for recs in load_attempts(out / "attempts.jsonl").values() for a in recs
            if a["honest"] and (a["item_id"], a["attempt"]) not in seen]
    if not recs:
        return
    log(f"[{out.parent.name}/{out.name}] meta-discussion judge on {len(recs)} compliant attempts")
    judged = await judge_mentions(recs, cfg["judge"]["model"], rc.judge_config(cfg))
    with path.open("a") as f:
        for a, j in zip(recs, judged):
            f.write(json.dumps({"item_id": a["item_id"], "attempt": a["attempt"], "mode": a["mode"],
                                "meta_discussion": j["meta_discussion"], "judge_output": j["judge_output"],
                                "mentions_feedback": a["mentions_feedback"], "judge_model": cfg["judge"]["model"]}) + "\n")


def arm_job(cfg: dict, out: Path, items: list[dict], lora: dict | None = None, priority: tuple = ()) -> Job:
    """The scheduler Job of one arm (`lora` = {lora_path, chat_template_kwargs, registry_entry} for adapters)."""
    weight = rc.gpu_weight(cfg, len(items))
    model_dir = out.parent

    async def run():
        gen_fn, info = rc.make_generate_fn(cfg, weight or 1, lora_path=(lora or {}).get("lora_path"),
                                           chat_template_kwargs=(lora or {}).get("chat_template_kwargs", "default"),
                                           cache_dir=model_dir / "gen_cache")
        log = rc.log_to(out / "log.txt")
        log(f"[{cfg['model']}] {len(items)} items; {rc.power_statement(len(items))}")
        await run_arm(cfg, out, items, gen_fn, info, log, lora)

    return Job(name=f"3a:{out.relative_to(rc.RUNS)}", weight=weight, run=run,
               done=lambda: (out / "episodes.json").exists(),
               priority=priority or (0, str(cfg["run"]), cfg["model"], cfg["arms"].index(cfg["arm"]) if cfg["arm"] in cfg["arms"] else 9),
               group=str(model_dir))


def jobs(cfg: dict) -> list[Job]:
    if not cfg.get("arm"):
        raise SystemExit("run3a needs an arm (sweep axis or arm=<name>)")
    items = rc.load_items(cfg["base_model"], **rc.item_spec(cfg))
    return [arm_job(cfg, rc.run_dir_for(cfg), items)]


def main(cfg: dict, argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="retry run3a", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cap", type=int, default=None)
    a = ap.parse_args(argv)
    if not cfg.get("model"):
        raise SystemExit("run3a needs a model: run.py retry run3a run=main items=main model=gptoss20b arm=bare")
    if not a.dry_run:
        preflight([cfg])
    run_scheduler(jobs(cfg), rc.RUNS / str(cfg["run"]), cap=a.cap, name="run3a_scheduler", dry_run=a.dry_run)
