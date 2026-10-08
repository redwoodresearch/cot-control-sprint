"""Experiment 3B (hidden-constraint scenarios) stages, dispatched by kaushik/run.py.

    .venv/bin/python kaushik/run.py hidden_constraint preview [--scenario S] [--slice dev] [--idx 0] [--control]
    .venv/bin/python kaushik/run.py hidden_constraint dryrun                       # CPU-only, scripted fake model
    .venv/bin/python kaushik/run.py hidden_constraint run --models gptoss20b_api --slice dev --n 10
    .venv/bin/python kaushik/run.py hidden_constraint run --models gptoss20b,qwen8b,qwen36_27b,gptoss120b,glm53,dsv4pro,qwen32b
    .venv/bin/python kaushik/run.py hidden_constraint poll                         # registry adapters, up to 24h
    .venv/bin/python kaushik/run.py hidden_constraint judge [--runs heldout_x,...]  # fill missing judge outputs
    .venv/bin/python kaushik/run.py hidden_constraint analyze [--include-dev]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import traceback
from pathlib import Path

import analyze as analyze_mod
import runner
from runner import DEFAULTS, MODELS, RUNS, GpuBudget, log_line, run_model
from scenarios import CONDITIONS, SCENARIOS, build_items

from cotcontrol.agent.protocol import ToolCall

HERE = Path(__file__).resolve().parent
REGISTRY = HERE.parent / "registry"


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--slice", default="heldout", choices=["dev", "heldout"])
    ap.add_argument("--n", type=int, default=DEFAULTS["n"], help="episodes per scenario x condition")
    ap.add_argument("--scenarios", default=",".join(DEFAULTS["scenarios"]))
    ap.add_argument("--max-turns", type=int, default=DEFAULTS["max_turns"])
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--max-gpus", type=int, default=DEFAULTS["max_gpus"])
    ap.add_argument("--tag", default="", help="run-name suffix, e.g. 'v2' -> runs/dev_gptoss20b_v2/")
    ap.add_argument("--groups", type=int, default=1,
                    help="Modal base-model runs: split episodes into this many concurrent engine containers (GPUs)")


# ----------------------------------------------------------------------------
def preview(argv):
    ap = argparse.ArgumentParser("preview", description="Print one item's files per scenario (eyeball check).")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--slice", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args(argv)
    for sc in ([a.scenario] if a.scenario else list(SCENARIOS)):
        item = build_items(sc, a.slice, a.idx + 1)[a.idx]
        files = item.control_files if a.control else item.files
        print(f"\n{'=' * 100}\n{sc} | {item.item_id} | {'control' if a.control else 'constrained'} | "
              f"reveal={item.reveal_path} | terms={item.forbidden_terms or item.forbidden_pattern} | answer={item.answer}")
        for path, content in sorted(files.items()):
            print(f"\n--- {path} ---\n{content}")


# ----------------------------------------------------------------------------
def _fake_generate_fn(seed: int = 0):
    """Scripted agent for CPU dry runs: lists, reads README + every file, then submits a
    plausible-format answer. Half the episodes 'mention' the forbidden term after reading."""
    import random
    rng = random.Random(seed)

    async def fn(messages_list):
        out = []
        for msgs in messages_list:
            n_asst = sum(1 for m in msgs if m["role"] == "assistant")
            files = [ln for ln in msgs[1]["content"].split("\n")[2:] if ln.strip()]
            item_terms = []
            for m in msgs:
                if m["role"] == "user" and "BLOCKED_TERMS = " in m["content"]:
                    item_terms = json.loads(m["content"].split("BLOCKED_TERMS = ")[1].split("\n")[0].replace("'", '"'))
                if m["role"] == "user" and 'BLOCKED_IDENTIFIERS = ["' in m["content"]:
                    item_terms = [m["content"].split('BLOCKED_IDENTIFIERS = ["')[1].split('"')[0]]
            if n_asst == 0:
                content, reasoning = ToolCall("read_file", {"path": "README.md"}).render(), "Let me read the README first."
            elif n_asst == 1:
                content = "".join(ToolCall("read_file", {"path": f}).render() for f in files if f != "README.md")
                reasoning = "I'll read the remaining files: " + ", ".join(files)
            else:
                ans = rng.choice(["ANSWER: A", "ANSWER: B", "    return a", "144", "x = 3 + 4"])
                content = ToolCall("submit_answer", {"answer": ans}).render()
                mention = rng.random() < 0.5
                reasoning = ("The grader checks my transcript for " + (item_terms[0] if item_terms else "digits")
                             + ", so I'd better avoid it. " if mention else "Working it out: 7 * 12 = 84. ") + \
                            "The lariat and the spliceosome are involved. Submitting now."
            out.append({"input": msgs, "model": "fake", "output": [content], "reasoning": [reasoning],
                        "metadata": [{"finish_reason": "stop", "usage": None, "error": None, "raw_response": content}]})
        return out
    return fn


def dryrun(argv):
    ap = argparse.ArgumentParser("dryrun", description="CPU-only end-to-end check with a scripted fake model.")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args(argv)
    run_dir = asyncio.run(run_model("fake", "dev", a.n, DEFAULTS["scenarios"], 6, run_name="dryrun_fake",
                                    judge=False, generate_fn=_fake_generate_fn()))
    s = json.loads((run_dir / "summary.json").read_text())
    for sc, m in s["scenarios"].items():
        print(f"{sc}: constrained_success={m['constrained_success']['value']:.2f} control_success={m['control_success']['value']:.2f} "
              f"read={m['read_rate']['value']:.2f} forbidden={m['forbidden_term_rate']['value']:.2f} status={m['status_constrained']}")
    analyze_mod.main(["--include-dev"])


# ----------------------------------------------------------------------------
async def _run_many(labels, a, adapters=None):
    """Run several models concurrently; Modal ones share one app.run() and a GPU budget."""
    adapters = adapters or {}
    modal_labels = [m for m in labels if (MODELS.get(m) or adapters.get(m))["backend"] == "modal"]
    halve = DEFAULTS.get("gpu_halve_after_hours")
    budget = GpuBudget(a.max_gpus, halve * 3600 if halve else None) if modal_labels else None
    scenarios = a.scenarios.split(",")

    async def one(label):
        adapter = adapters.get(label)
        name = f"{a.slice}_{'ckpt_' + adapter['run_name'] if adapter else label}" + (f"_{a.tag}" if a.tag else "")
        try:
            return await run_model(label if label in MODELS else "modal_adapter", a.slice, a.n, scenarios, a.max_turns,
                                   adapter=adapter, run_name=name, gpu_budget=budget, judge=not a.no_judge,
                                   groups=a.groups)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            (RUNS / name).mkdir(parents=True, exist_ok=True)
            log_line(RUNS / name, f"FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            return None

    if modal_labels:
        from cotcontrol.inference.modal_vllm import app
        async with app.run():
            _announce_modal(app)
            return await asyncio.gather(*[one(m) for m in labels])
    return await asyncio.gather(*[one(m) for m in labels])


def run(argv):
    ap = argparse.ArgumentParser("run", description="Run scenarios on models from configs/models.yaml.")
    ap.add_argument("--models", required=True, help="comma-separated labels from configs/models.yaml")
    _common(ap)
    a = ap.parse_args(argv)
    labels = a.models.split(",")
    unknown = [m for m in labels if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; known: {list(MODELS)}")
    if any(MODELS[m]["backend"] == "openrouter" for m in labels) and not runner.openrouter_available():
        raise SystemExit("OPENROUTER_API_KEY missing (.env)")
    if any(MODELS[m]["backend"] == "modal" for m in labels) and not runner.modal_available():
        raise SystemExit("Modal token missing (`modal setup`)")
    reason = runner.launch_guard()
    if reason:
        raise SystemExit(f"launch guard: {reason}; try again later")
    asyncio.run(_run_many(labels, a))
    analyze_mod.main(["--include-dev"] if a.slice == "dev" else [])


def _announce_modal(app) -> None:
    """Deployed mode (COTCONTROL_MODAL_INFERENCE_APP, set by kaushik/run.py): engines live in a
    long-lived app shared by every agent, so there is no per-run app id and it must never be
    stopped. Ephemeral mode: record the app id so an orphaned app can be stopped after a kill."""
    deployed = os.environ.get("COTCONTROL_MODAL_INFERENCE_APP")
    app_id = getattr(app, "app_id", None)
    if deployed:
        print(f"[modal] deployed app {deployed} (shared; GPU use capped by this process's GpuBudget only)", flush=True)
    else:
        print(f"[modal] ephemeral app_id={app_id} (use `modal container list` to count GPUs)", flush=True)
        if app_id:
            (RUNS / ".modal_app_id").write_text(app_id + "\n")


# ----------------------------------------------------------------------------
def _registry_entries() -> dict[str, dict]:
    out = {}
    for p in sorted(REGISTRY.glob("*.json")):
        try:
            e = json.loads(p.read_text())
            if {"run_name", "base_model", "adapter_path"} <= set(e):
                out[e["run_name"]] = e
        except (json.JSONDecodeError, OSError):
            pass
    return out


def _adapter_cfg(entry: dict) -> dict:
    base = entry["base_model"]
    match = next((cfg for cfg in MODELS.values() if cfg["backend"] == "modal" and cfg["base_model"] == base), None)
    cfg = {**(match or {"backend": "modal", "base_model": base, "chat_template_kwargs": None}),
           **{k: entry[k] for k in ("run_name", "adapter_path", "eval_chat_template_kwargs", "experiment", "arm", "seed")
              if k in entry}}
    return cfg


def poll(argv):
    ap = argparse.ArgumentParser("poll", description="Evaluate every trained adapter in kaushik/registry/ as it appears.")
    _common(ap)
    ap.add_argument("--interval", type=int, default=DEFAULTS["poll_interval_s"])
    ap.add_argument("--hours", type=float, default=DEFAULTS["poll_hours"])
    ap.add_argument("--max-parallel", type=int, default=None, help="concurrent adapter runs (default: max-gpus)")
    ap.add_argument("--once", action="store_true", help="run what is in the registry now, then exit")
    ap.add_argument("--skip-pattern", default="smoke",
                    help="regex; registry run_names matching it are ignored (default: smoke-test adapters)")
    ap.add_argument("--include-models", default="",
                    help="comma-separated Modal base models from configs/models.yaml to run first, in the same GPU budget")
    a = ap.parse_args(argv)
    if not runner.modal_available():
        raise SystemExit("Modal token missing (`modal setup`)")
    scenarios = a.scenarios.split(",")
    deadline = time.time() + a.hours * 3600
    state_path = RUNS / "poll_state.json"
    RUNS.mkdir(exist_ok=True)
    base_models = [m for m in a.include_models.split(",") if m]
    if any(MODELS.get(m, {}).get("backend") != "modal" for m in base_models):
        raise SystemExit(f"--include-models must be Modal models from configs/models.yaml, got {base_models}")

    async def main():
        from cotcontrol.inference.modal_vllm import app
        halve = DEFAULTS.get("gpu_halve_after_hours")
        budget = GpuBudget(a.max_gpus, halve * 3600 if halve else None)
        parallel = asyncio.Semaphore(a.max_parallel or a.max_gpus)
        running: dict[str, asyncio.Task] = {}

        async def one(name, entry):
            async with parallel:
                run_name = (f"{a.slice}_ckpt_{name}" if entry else f"{a.slice}_{name}") + (f"_{a.tag}" if a.tag else "")
                try:
                    await run_model("modal_adapter" if entry else name, a.slice, a.n, scenarios, a.max_turns,
                                    adapter=_adapter_cfg(entry) if entry else None,
                                    run_name=run_name, gpu_budget=budget, judge=not a.no_judge,
                                    groups=1 if entry else a.groups)
                except Exception as e:  # noqa: BLE001
                    (RUNS / run_name).mkdir(parents=True, exist_ok=True)
                    log_line(RUNS / run_name, f"FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")

        async with app.run():
            _announce_modal(app)
            for m in base_models:
                if not (RUNS / f"{a.slice}_{m}{('_' + a.tag) if a.tag else ''}" / "DONE").exists():
                    running[f"base:{m}"] = asyncio.create_task(one(m, None))
            while True:
                entries = {n: e for n, e in _registry_entries().items()
                           if not (a.skip_pattern and re.search(a.skip_pattern, n))}
                new = [(n, e) for n, e in entries.items() if n not in running
                       and not (RUNS / f"{a.slice}_ckpt_{n}{('_' + a.tag) if a.tag else ''}" / "DONE").exists()]
                reason = runner.launch_guard(check_memory=False) if new else None  # same process, no new driver
                if reason:
                    print(f"[poll] launch guard: {reason}; {len(new)} adapter(s) wait for the next check", flush=True)
                    new = []
                for name, entry in new:
                    if True:
                        print(f"[poll] new adapter {name} ({entry.get('experiment')}/{entry.get('arm')} seed={entry.get('seed')})", flush=True)
                        running[name] = asyncio.create_task(one(name, entry))
                state_path.write_text(json.dumps({"time": runner.utcnow(), "registry": sorted(entries),
                                                  "running": [k for k, t in running.items() if not t.done()],
                                                  "finished": [k for k, t in running.items() if t.done()]}, indent=1))
                if a.once or time.time() > deadline:
                    break
                await asyncio.sleep(a.interval)
            await asyncio.gather(*running.values())
        analyze_mod.main([])

    asyncio.run(main())


# ----------------------------------------------------------------------------
def judge(argv):
    ap = argparse.ArgumentParser("judge", description="Fill in missing constraint-mention judge outputs.")
    ap.add_argument("--runs", default=None, help="comma-separated run names (default: all)")
    a = ap.parse_args(argv)
    if not runner.openrouter_available():
        raise SystemExit("OPENROUTER_API_KEY missing (.env)")
    dirs = [RUNS / r for r in a.runs.split(",")] if a.runs else \
        sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "episodes.jsonl").exists())

    async def main():
        for d in dirs:
            n = await runner.fill_judge(d)
            print(f"{d.name}: judged {n} episodes")
    asyncio.run(main())
    analyze_mod.main(["--include-dev"])


def analyze(argv):
    analyze_mod.main(argv)


STAGES = {"preview": preview, "dryrun": dryrun, "run": run, "poll": poll, "judge": judge, "analyze": analyze}
__all__ = ["CONDITIONS", "STAGES"]
