"""Stage 0 of the orange expansion under ONE GPU budget: orange-pick rates, untrained evals, and the
Qwen3.6-27B mechanism check (PLAN.md "Stage 0"). A small copy of kaushik/orange/eval_scheduler.py's
loop: jobs are dispatched in priority order while `in use + weight <= cap`, nothing starts while
the Modal workspace has > 800 containers, failures retry 3x. Generation and grading are
cotcontrol's (eval_cotcontrolqa + modal_vllm); nothing here grades CoT compliance.

    .venv/bin/python kaushik/orange_expand/baseline/run_all.py [--cap 40] [--only pick untrained mech] [--smoke]

Jobs (priority order):
  pick       4 models x reasoning {on, off}; each job runs the 3 training prompts (eugene, direct,
             standard) sequentially on one warm engine over the 200 held-out orange items
             -> runs/<model_key>/pick/<prompt>_<on|off>.json; results/pick_rates.{md,csv} after each.
  untrained  4 models x 4 eval conditions x 2 modes, val split, T=1, 16k tokens, reasoning on
             -> runs/<model_key>/eval/<condition>_<mode>/base.json (+ eval_progress.jsonl)
  mech       Qwen3.6-27B: 2A adapters (orange, orange_nopersona; seeds 0-2) + untrained, nopersona
             condition x 2 modes -> runs/mech_qwen36_27b/<arm>-seed<k>/eval/nopersona_<mode>/checkpoint-50.json
             and runs/mech_qwen36_27b/untrained/eval/nopersona_<mode>/base.json
State: runs/scheduler.{log,json,pid}.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import common  # noqa: F401  (env first)
from common import (EVAL_CONDITIONS, EVAL_MAX_TOKENS, GPTOSS_PREFILL_MSG, MECH_ARMS, MECH_MODEL, MECH_SEEDS,
                    MODELS, PICK_MAX_TOKENS_OFF, PICK_MAX_TOKENS_ON, RUNS_DIR, SCREEN_MODES,
                    TRAIN_PROMPTS, adapter_path, eval_block, format_user, gen_config, heldout_items, is_gptoss,
                    parse_letter, reasoning_kwargs)

from cotcontrol.eval.eval import eval_cotcontrolqa
from cotcontrol.eval.grading import is_honest
from cotcontrol.inference.modal_vllm import generate_async, make_generate_fn
from cotcontrol.run_logging import tee_stdio

analyze = common.load_py(common.HERE / "analyze.py", "baseline_analyze")  # kaushik/orange/analyze.py shadows the bare name

MAX_ATTEMPTS = 3
PICK_PROMPT_ORDER = ["eugene", "direct", "standard"]
PICK_MODEL_ORDER = ["gptoss20b", "qwen32b", "qwen8b", "gptoss120b"]   # RL agent needs the first two
UNTRAINED_MODEL_ORDER = ["gptoss20b", "qwen8b", "qwen32b", "gptoss120b"]  # SFT finishes cheap models first


@dataclass(order=True)
class Job:
    priority: tuple
    kind: str = field(compare=False)
    name: str = field(compare=False)
    weight: int = field(compare=False)
    params: dict = field(compare=False)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}"


# ----------------------------------------------------------------------------- job construction

def pick_jobs(smoke: bool) -> list[Job]:
    jobs = []
    for i, mk in enumerate(PICK_MODEL_ORDER):
        for j, on in enumerate([True, False]):
            jobs.append(Job((0, j, i), "pick", f"{mk}_{'on' if on else 'off'}", MODELS[mk]["tp"],
                            {"model_key": mk, "reasoning_on": on, "smoke": smoke}))
    return jobs


def eval_job(prio: tuple, name: str, model_key: str, run_dir: Path, block: str, system_prompt: str | None,
             state_requirement: bool, mode: str, lora_path: str | None, stem: str, smoke: bool) -> Job:
    return Job(prio, "eval", name, MODELS[model_key]["tp"],
               {"model_key": model_key, "run_dir": str(run_dir), "block": block, "system_prompt": system_prompt,
                "state_requirement": state_requirement, "mode": mode, "lora_path": lora_path, "stem": stem,
                "smoke": smoke})


def untrained_jobs(smoke: bool) -> list[Job]:
    jobs = []
    for i, mk in enumerate(UNTRAINED_MODEL_ORDER):
        for j, mode in enumerate(SCREEN_MODES):
            for k, cond in enumerate(EVAL_CONDITIONS):
                blk = eval_block(cond, mode, MODELS[mk]["base_model"])
                jobs.append(eval_job((1, i, j, k), f"{mk}/{cond}_{mode}", mk, RUNS_DIR / mk, f"{cond}_{mode}",
                                     blk["system_prompt"], blk["state_requirement"], mode, None, "base", smoke))
    return jobs


def mech_jobs(smoke: bool) -> list[Job]:
    jobs = []
    base = RUNS_DIR / f"mech_{MECH_MODEL}"
    targets = [("untrained", None, "base", (-1, 0))]
    for seed in MECH_SEEDS:
        for a, arm in enumerate(MECH_ARMS):
            targets.append((f"{arm}-seed{seed}", adapter_path(arm, seed), "checkpoint-50", (seed, a)))
    for run, lora, stem, p in targets:
        for j, mode in enumerate(SCREEN_MODES):
            blk = eval_block("nopersona", mode, MODELS[MECH_MODEL]["base_model"])
            jobs.append(eval_job((2, *p, j), f"mech/{run}/nopersona_{mode}", MECH_MODEL, base / run,
                                 f"nopersona_{mode}", blk["system_prompt"], blk["state_requirement"], mode, lora,
                                 stem, smoke))
    return jobs


def out_path(job: Job) -> Path:
    if job.kind == "eval":
        return Path(job.params["run_dir"]) / "eval" / job.params["block"] / f"{job.params['stem']}.json"
    mk = job.params["model_key"]
    return RUNS_DIR / mk / "pick" / f"{PICK_PROMPT_ORDER[-1]}_{'on' if job.params['reasoning_on'] else 'off'}.json"


# ----------------------------------------------------------------------------- runners

async def run_pick(job: Job) -> None:
    p = job.params
    mk, on = p["model_key"], p["reasoning_on"]
    base_model = MODELS[mk]["base_model"]
    items = heldout_items()
    if p["smoke"]:
        items = items[:4]
    cfg = gen_config(mk, PICK_MAX_TOKENS_ON if on else PICK_MAX_TOKENS_OFF, reasoning_kwargs(base_model, on))
    out_dir = RUNS_DIR / ("smoke" if p["smoke"] else mk) / "pick"
    out_dir.mkdir(parents=True, exist_ok=True)
    for prompt_name in PICK_PROMPT_ORDER:
        path = out_dir / f"{prompt_name}_{'on' if on else 'off'}.json"
        if path.exists() and not p["smoke"]:
            continue
        t0 = time.time()
        messages = [[{"role": "system", "content": TRAIN_PROMPTS[prompt_name]},
                     {"role": "user", "content": format_user(it)}] for it in items]
        if on or not is_gptoss(base_model):
            gens = await generate_async(messages, base_model, cfg, None)
        else:  # 2A's empty analysis-channel prefill (see common.py docstring)
            gens = await generate_async([m + [GPTOSS_PREFILL_MSG] for m in messages], base_model, cfg, None,
                                        add_generation_prompt=False, continue_final_message=True)
        records = []
        for it, msgs, g in zip(items, messages, gens):
            raw = g["metadata"][0].get("raw_response") or ""
            letter = parse_letter(g["output"][0], g["reasoning"][0], raw, on)
            orange = "ABCD"[it["answer_index"]]
            records.append({"question": it["question"], "options": it["options"], "answer_index": it["answer_index"],
                            "orange_letter": orange, "messages": msgs, "output": g["output"][0],
                            "reasoning": g["reasoning"][0], "raw_response": raw,
                            "finish_reason": g["metadata"][0].get("finish_reason"), "parsed": letter,
                            "picked_orange": (letter == orange) if letter else None})
        n = len(records)
        parsed = [r for r in records if r["parsed"]]
        summary = {"model_key": mk, "base_model": base_model, "train_prompt": prompt_name, "reasoning_on": on,
                   "n": n, "n_parsed": len(parsed), "n_unparseable": n - len(parsed),
                   "orange_rate": (sum(r["picked_orange"] for r in parsed) / len(parsed)) if parsed else None,
                   "unparseable_rate": (n - len(parsed)) / n if n else None,
                   "truncated_rate": sum(r["finish_reason"] == "length" for r in records) / n if n else None,
                   "mean_reasoning_chars": sum(len(r["reasoning"] or "") for r in records) / n if n else None,
                   "seconds": round(time.time() - t0), "generate_config": asdict(cfg),
                   "gptoss_prefill": (None if on or not is_gptoss(base_model) else GPTOSS_PREFILL_MSG),
                   "system_prompt": TRAIN_PROMPTS[prompt_name], "time": datetime.now().isoformat()}
        path.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
        print(f"[pick] {mk} {prompt_name} reasoning={'on' if on else 'off'}: orange={summary['orange_rate']} "
              f"unparseable={summary['unparseable_rate']} truncated={summary['truncated_rate']} "
              f"({summary['seconds']}s)", flush=True)
        if p["smoke"]:
            for r in records[:2]:
                print("   raw:", repr(r["raw_response"][-300:]), "->", r["parsed"], "orange", r["orange_letter"])
    if not p["smoke"]:
        analyze.write_pick_tables()


def honest_rate(result: dict) -> float | None:
    samples = [s for rec in result["results"] for s in rec["samples"] if not s["error"]]
    return sum(is_honest(s) for s in samples) / len(samples) if samples else None


async def run_eval(job: Job) -> None:
    p = job.params
    mk = p["model_key"]
    base_model = MODELS[mk]["base_model"]
    run_dir = Path(p["run_dir"]) if not p["smoke"] else RUNS_DIR / "smoke" / Path(p["run_dir"]).name
    cfg = gen_config(mk, EVAL_MAX_TOKENS, reasoning_kwargs(base_model, True))
    out = run_dir / "eval" / p["block"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = await eval_cotcontrolqa(
        model=base_model, system_prompt=p["system_prompt"] or "",
        generate_fn=make_generate_fn(base_model, cfg, p["lora_path"]),
        save_dir=out, save_name=p["stem"], dataset="all", mode=p["mode"], allowed_modes=None, seed=0,
        state_requirement=bool(p["state_requirement"]), max_samples=3 if p["smoke"] else None,
        subsample_seed=None, split="val", grade_meta_discussion=False, meta_discussion_scope="compliant",
        judge_model="openai/gpt-5-mini", judge_concurrency=200,
        backend_info={"base_model": base_model, "lora_path": p["lora_path"], "generate_config": asdict(cfg)},
    )
    s = result["summary"]
    line = {"block": p["block"], "step": "base" if p["lora_path"] is None else 50, "lora_path": p["lora_path"],
            "accuracy": s["accuracy"], "compliance_rate": s["compliance_rate"], "honest_rate": honest_rate(result),
            "meta_discussion_rate": s["meta_discussion_rate"], "n_errors": s["n_errors"],
            "n_rollouts": s["n_rollouts"], "per_mode": s["per_mode"], "seconds": round(time.time() - t0),
            "time": datetime.now().isoformat()}
    with (run_dir / "eval_progress.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")
    meta = {"base_model": base_model, "model_key": mk, "lora_path": p["lora_path"], "split": "val",
            "temperature": 1.0, "max_tokens": EVAL_MAX_TOKENS, "chat_template_kwargs": cfg.chat_template_kwargs,
            "gpu": cfg.gpu, "tensor_parallel_size": cfg.tensor_parallel_size}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {job.name}: compliance={line['compliance_rate']} honest={line['honest_rate']} "
          f"accuracy={line['accuracy']} errors={line['n_errors']} ({line['seconds']}s)", flush=True)


async def run_job(job: Job) -> None:
    await (run_pick(job) if job.kind == "pick" else run_eval(job))


# ----------------------------------------------------------------------------- scheduler

_WS: dict = {"t": 0.0, "n": None}


def workspace_containers(max_age: float = 60.0) -> int | None:
    if time.time() - _WS["t"] > max_age:
        out = subprocess.run([sys.executable, "-m", "modal", "container", "list", "--json"],
                             capture_output=True, text=True, check=False)
        try:
            _WS["n"] = len(json.loads(re.sub(r"\x1b\[[0-9;]*m", "", out.stdout)))
        except json.JSONDecodeError:
            _WS["n"] = None
        _WS["t"] = time.time()
    return _WS["n"]


def cap_now(default: int) -> int:
    f = RUNS_DIR / ".gpu_cap"
    if f.exists():
        try:
            return int(f.read_text().strip())
        except ValueError:
            pass
    return default


async def scheduler(args: argparse.Namespace, jobs: list[Job]) -> None:
    pending = sorted(j for j in jobs if args.smoke or not out_path(j).exists())
    skipped = len(jobs) - len(pending)
    print(f"[scheduler] {len(pending)} jobs pending, {skipped} already done", flush=True)
    in_flight: dict[str, tuple[Job, asyncio.Task]] = {}
    done, failed, attempts = [], [], Counter()
    gpu_seconds = 0.0
    in_use = 0
    state_path = RUNS_DIR / f"{args.tag}.json"
    while pending or in_flight:
        now = time.time()
        for key, (job, task) in list(in_flight.items()):
            if not task.done():
                continue
            in_use -= job.weight
            gpu_seconds += job.weight * (now - job.params["_t0"])
            del in_flight[key]
            exc = task.exception()
            if exc is None:
                done.append(key)
                continue
            attempts[key] += 1
            print(f"[error] {job.name} (attempt {attempts[key]}/{MAX_ATTEMPTS}): {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exception(exc)
            if attempts[key] < MAX_ATTEMPTS:
                pending.append(job)
                pending.sort()
            else:
                failed.append(key)
        limit = cap_now(args.cap)
        ws = workspace_containers()
        if pending and ws is not None and ws > args.workspace_max:
            limit = 0
            if int(now) % 60 < args.tick:
                print(f"[hold] workspace has {ws} active containers > {args.workspace_max}", flush=True)
        if (RUNS_DIR / ".stop_launches").exists():
            limit = 0
        while pending and in_use + pending[0].weight <= limit:
            job = pending.pop(0)
            job.params["_t0"] = time.time()
            in_use += job.weight
            in_flight[job.key] = (job, asyncio.create_task(run_job(job)))
            print(f"[start] {job.name} (weight {job.weight}; in use {in_use}/{limit})", flush=True)
        state_path.write_text(json.dumps({
            "time": datetime.now().isoformat(), "cap": limit, "workspace_containers": ws, "gpus_in_use": in_use,
            "in_flight": [{"job": j.name, "weight": j.weight, "minutes": round((now - j.params["_t0"]) / 60, 1)}
                          for j, _ in in_flight.values()],
            "pending": [j.name for j in pending], "done": len(done), "failed": failed,
            "gpu_hours_so_far": round((gpu_seconds + sum(j.weight * (now - j.params["_t0"])
                                                         for j, _ in in_flight.values())) / 3600, 2),
        }, indent=1))
        await asyncio.sleep(args.tick)
    print(f"[scheduler] exit: {len(done)} done, {len(failed)} failed, {gpu_seconds / 3600:.2f} GPU-hours", flush=True)
    if not args.smoke:
        analyze.main([])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=40, help="concurrent GPUs (runs/.gpu_cap overrides live)")
    ap.add_argument("--workspace-max", type=int, default=800)
    ap.add_argument("--only", nargs="+", default=["pick", "untrained", "mech"], choices=["pick", "untrained", "mech"])
    ap.add_argument("--smoke", action="store_true", help="4 items / 3 questions, gptoss20b + qwen8b, -> runs/smoke")
    ap.add_argument("--skip", nargs="*", default=[], help="glob patterns of job names to leave out")
    ap.add_argument("--tag", default="scheduler", help="stem of runs/<tag>.{log,json,pid} (a second scheduler)")
    ap.add_argument("--tick", type=float, default=5.0)
    args = ap.parse_args(argv)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    if "pick" in args.only:
        jobs += pick_jobs(args.smoke)
    if "untrained" in args.only:
        jobs += untrained_jobs(args.smoke)
    if "mech" in args.only:
        jobs += mech_jobs(args.smoke)
    jobs = [j for j in jobs if not any(fnmatch.fnmatch(j.name, pat) for pat in args.skip)]
    if args.smoke:
        jobs = [j for j in jobs if j.params["model_key"] in ("gptoss20b", "qwen8b")]
        jobs = [j for j in jobs if j.kind == "pick" or j.name.endswith("nopersona_lowercase_thinking")
                or j.name.endswith("standard_lowercase_thinking")][:6]
    log = RUNS_DIR / ("smoke.log" if args.smoke else f"{args.tag}.log")
    (RUNS_DIR / f"{args.tag}.pid").write_text(str(os.getpid()))
    with tee_stdio(log, mode="a"):
        print(f"[scheduler] start {datetime.now().isoformat()} pid {os.getpid()} args {vars(args)} "
              f"apps {os.environ['COTCONTROL_MODAL_INFERENCE_APP']}")
        asyncio.run(scheduler(args, jobs))


if __name__ == "__main__":
    main()
