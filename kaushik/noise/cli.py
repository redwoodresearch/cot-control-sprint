"""kaushik/noise: KL-matched random-LoRA controls (spec: Shared setup -> noise).

    .venv/bin/python kaushik/run.py noise <stage> [args]

Stages
  make_configs  group kaushik/registry/*.json into configs/<group>.yaml
                (one group per experiment x arm x base model)
  kl            KL(base || adapter) per token for trained adapters
                (--runs NAME ... | --model BASE | --adapter PATH --base-model M)
                -> runs/_kl/<run>.json; base samples shared per model in runs/_samples/
  make          --group G: median trained KL -> calibrate 3 seeded random adapters
                (bisection on log-scale, within 10%) -> /checkpoints/noise/<G>/noise{k}
                + runs/<G>/noise_adapters.json
  eval          --group G: evaluate the noise adapters with the arm's own final
                eval blocks (from the source train_eval config) -> runs/<G>/eval/
  process       --group G: manifest + make + eval + analyze (what `watch` spawns)
  watch         poll the registry; measure every KL as it appears; build+eval a
                group's noise once all its seeds are in (or 4h after the first);
                several subprocesses at once under a global GPU cap
  analyze       runs/<G>/results.json + results.md
  smoke         Qwen3-8B sanity check: KL(base||base)=0, KL(base||checkpoint-0)~0,
                random KL monotone in scale, calibration converges

GPU budget: every engine call takes slots from runs/_gpu_slots (cap in
runs/gpu_cap.json, default 15; `watch` lowers it to 7 six hours after the
first launch). The adapter builds run on CPU.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
KAUSHIK = HERE.parent
ROOT = KAUSHIK.parent
for _p in (ROOT / "scripts", ROOT):
    sys.path.insert(0, str(_p))
# Deployed copy of the CPU adapter-build app (see `noise deploy`); the inference
# and train apps are pointed at their deployed copies by kaushik/run.py.
RANDOM_LORA_APP = "cot-control-sprint_kaushik-reddy_random-lora"
os.environ.setdefault("COTCONTROL_MODAL_RANDOM_LORA_APP", RANDOM_LORA_APP)

import train_eval  # scripts/train_eval.py: config loading + eval plumbing

from cotcontrol.eval.eval import eval_cotcontrolqa
from cotcontrol.inference import modal_vllm
from cotcontrol.inference.modal_vllm import ModalGenerateConfig, make_generate_fn
from cotcontrol.run_logging import tee_stdio
from cotcontrol.training import random_lora
from cotcontrol.training.random_lora import (
    calibrate_scale,
    kl_from_logprobs,
    sample_base,
    score_samples,
    unconstrained_prompts,
)


def _sibling(name: str):
    """Import HERE/<name>.py under a folder-unique module name: every experiment
    folder has a make_configs.py / analyze.py, and run.py may load several."""
    spec = importlib.util.spec_from_file_location(f"kaushik_noise_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


noise_analyze = _sibling("analyze")
_mc = _sibling("make_configs")
CONFIGS, KL_SETTINGS = _mc.CONFIGS, _mc.KL_SETTINGS
expected_run_names, load_entries, model_slug = _mc.expected_run_names, _mc.load_entries, _mc.model_slug
write_group_configs = _mc.write_group_configs
GpuSlots = _sibling("slots").GpuSlots

RUNS = HERE / "runs"
KL_DIR = RUNS / "_kl"
SAMPLES_DIR = RUNS / "_samples"
LOGS_DIR = RUNS / "_logs"
APPS_DIR = RUNS / "_apps"
SLOTS = GpuSlots(RUNS)
OWNER = "kaushik-reddy"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_owner() -> None:
    if os.environ.get("COTCONTROL_MODAL_OWNER") != OWNER:
        raise SystemExit("run through kaushik/run.py (COTCONTROL_MODAL_OWNER must be kaushik-reddy)")


def _volume_name() -> str:
    return f"cotcontrol_{os.environ.get('COTCONTROL_MODAL_OWNER')}_checkpoints"


def atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _load(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@asynccontextmanager
async def file_lock(path: Path):
    """Cross-process exclusive lock (flock) so two jobs don't sample the same base twice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


async def _retry(fn, what: str, attempts: int = 5, delay: float = 30.0):
    """Re-run an idempotent remote call after transient Modal/gRPC failures (e.g.
    `StreamTerminatedError: Connection lost` during a long engine call).
    Generation is disk-cached per chunk, so a retry only redoes lost work."""
    for i in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as e:
            if i == attempts:
                raise
            wait = min(delay * 2 ** (i - 1), 300.0)  # drops come in clusters: back off
            print(f"[retry] {what}: {type(e).__name__}: {str(e)[:120]} -- retry {i}/{attempts - 1} in {wait:.0f}s",
                  flush=True)
            await asyncio.sleep(wait)


async def build_adapter(src: str, dst: str, scale: float, seed: int, family: str) -> dict:
    return await _retry(lambda: random_lora.remote_fn("build_adapter").remote.aio(src, dst, scale, seed, family),
                        f"build {dst}")


def register_apps(*apps) -> None:
    """Record this process's running Modal app ids so `watch` can count our containers."""
    ids = [getattr(a, "app_id", None) for a in apps]
    APPS_DIR.mkdir(parents=True, exist_ok=True)
    (APPS_DIR / f"{os.getpid()}.json").write_text(json.dumps({"pid": os.getpid(), "apps": ids, "t": time.time()}))


def group_cfg(name: str) -> dict:
    p = CONFIGS / f"{name}.yaml"
    if not p.exists():
        raise SystemExit(f"no group config {p}; run `noise make_configs` (known: {[q.stem for q in CONFIGS.glob('*.yaml')]})")
    return yaml.safe_load(p.read_text())


def source_config(config_path: str):
    """(TrainConfig, [(block, fully-defaulted ec)], raw yaml) or None if missing."""
    p = ROOT / config_path
    if not config_path or not p.is_file():
        return None
    tc, ec, raw = train_eval.load_config(p)
    return tc, train_eval.eval_blocks(ec), raw


def kl_engine_config(src, template_kwargs, kl_settings: dict) -> ModalGenerateConfig:
    """Engine params of the source run (gpu, tp, lora rank, model len) + the KL
    sampling settings. Same resource fields as the eval -> warm engines shared."""
    if src:
        tc, blocks, _ = src
        gen = train_eval.build_generate_config(tc, blocks[0][1] if blocks else dict(train_eval.DEFAULT_EVAL))
    else:
        gen = ModalGenerateConfig(gpu="H200")
    return replace(gen, temperature=1.0, top_p=1.0, max_tokens=int(kl_settings["max_tokens"]), num_samples=1,
                   seed=int(kl_settings["seed"]), return_logprobs=True, cache=True, stop=None,
                   chat_template_kwargs=template_kwargs or None)


def final_step(adapter_path: str) -> int | None:
    m = re.search(r"checkpoint-(\d+)/?$", adapter_path)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# base samples + KL
# ---------------------------------------------------------------------------

def samples_dir(base_model: str, template_kwargs) -> Path:
    tag = model_slug(base_model)
    if template_kwargs:
        tag += "-" + hashlib.sha256(json.dumps(template_kwargs, sort_keys=True).encode()).hexdigest()[:6]
    return SAMPLES_DIR / tag


async def ensure_samples(base_model: str, gen: ModalGenerateConfig, kl_settings: dict) -> tuple[dict, list]:
    """The base model's fixed T=1 samples + its own teacher-forced log-probs
    (one set per base model x template kwargs, shared by every adapter)."""
    d = samples_dir(base_model, gen.chat_template_kwargs)
    sp, bp = d / "samples.json", d / "base_logprobs.json"
    async with file_lock(d / ".lock"):
        if not sp.exists():
            prompts = unconstrained_prompts(base_model, kl_settings["split"], kl_settings["dataset"],
                                            int(kl_settings["n_prompts"]))
            print(f"[samples] sampling {len(prompts)} {kl_settings['split']} prompts from {base_model} "
                  f"(T=1, max_tokens={gen.max_tokens}, kwargs={gen.chat_template_kwargs})", flush=True)
            async with SLOTS.slot(gen.tensor_parallel_size):
                samples = await sample_base(prompts, base_model, gen)
            samples["kl_settings"] = kl_settings
            atomic_write(sp, samples)
            n_tok = sum(len(t) - n for t, n in zip(samples["token_ids"], samples["prompt_lens"]))
            print(f"[samples] {len(samples['token_ids'])} sequences, {n_tok} sampled tokens -> {sp}", flush=True)
        samples = _load(sp)
        if not bp.exists():
            async with SLOTS.slot(gen.tensor_parallel_size):
                lps = await score_samples(samples, base_model, gen, None)
            self_kl = kl_from_logprobs(lps, lps, n_boot=10)["kl_per_token"]
            drift = statistics.fmean(abs(a - b) for s, l in zip(samples["sampled_logprobs"], lps) for a, b in zip(s, l))
            atomic_write(bp, {"logprobs": lps, "engine": asdict(gen), "self_kl": self_kl,
                              "mean_abs_diff_vs_sampled_logprobs": drift})
            print(f"[samples] base rescored; mean |rescored - sampled| logprob = {drift:.4f}", flush=True)
    return samples, _load(bp)["logprobs"]


async def measure_kl(adapter_path: str | None, base_model: str, gen: ModalGenerateConfig,
                     samples: dict, base_lps: list) -> dict:
    async with SLOTS.slot(gen.tensor_parallel_size):
        lps = await _retry(lambda: score_samples(samples, base_model, gen, adapter_path), f"score {adapter_path}")
    return {"kl": kl_from_logprobs(base_lps, lps), "adapter_logprobs": lps}


def write_kl_record(name: str, base_model: str, adapter_path: str, res: dict, gen: ModalGenerateConfig,
                    extra: dict | None = None) -> Path:
    out = KL_DIR / f"{name}.json"
    atomic_write(out, {"name": name, "base_model": base_model, "adapter_path": adapter_path,
                       "kl": res["kl"], "engine": asdict(gen), "samples": str(samples_dir(base_model, gen.chat_template_kwargs).relative_to(ROOT)),
                       "timestamp": _now(), **(extra or {}), "adapter_logprobs": res["adapter_logprobs"]})
    return out


async def kl_for_entries(base_model: str, entries: list[dict], force: bool, template_kwargs) -> None:
    src = source_config(entries[0]["config_path"])
    gen = kl_engine_config(src, template_kwargs, KL_SETTINGS)
    todo = [e for e in entries if force or not (KL_DIR / f"{e['run_name']}.json").exists()]
    if not todo:
        print(f"[kl] {base_model}: nothing to do")
        return
    async with modal_vllm.app.run():
        register_apps(modal_vllm.app)
        samples, base_lps = await ensure_samples(base_model, gen, KL_SETTINGS)
        sem = asyncio.Semaphore(4)

        async def one(e):
            async with sem:
                t0 = time.time()
                res = await measure_kl(e["adapter_path"], base_model, gen, samples, base_lps)
            write_kl_record(e["run_name"], base_model, e["adapter_path"], res, gen,
                            {"experiment": e.get("experiment"), "arm": e.get("arm"), "seed": e.get("seed")})
            k = res["kl"]
            print(f"[kl] {e['run_name']}: KL/token {k['kl_per_token']:.5f} ± {k['se_prompt']:.5f} "
                  f"(prompt bootstrap; token SE {k['se_token']:.5f}; {k['n_tokens']} tokens) {time.time() - t0:.0f}s", flush=True)

        await asyncio.gather(*[one(e) for e in todo])


def stage_kl(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="noise kl", description="KL(base || adapter) per token")
    ap.add_argument("--runs", nargs="*", default=None, help="registry run names")
    ap.add_argument("--model", default=None, help="all pending registry entries of this base model")
    ap.add_argument("--adapter", default=None, help="ad hoc: /checkpoints/... adapter dir")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--template-kwargs", default=None, help="JSON, e.g. '{\"reasoning_effort\": \"medium\"}'")
    ap.add_argument("--config", default=None, help="train_eval config for engine params (ad hoc)")
    ap.add_argument("--name", default=None, help="record name (ad hoc)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    _check_owner()
    if args.adapter:
        e = {"run_name": args.name or Path(args.adapter.rstrip("/")).parent.name, "adapter_path": args.adapter,
             "config_path": args.config or "", "experiment": None, "arm": None, "seed": None}
        kw = json.loads(args.template_kwargs) if args.template_kwargs else None
        train_eval.with_slot_retry(lambda: asyncio.run(kl_for_entries(args.base_model, [e], args.force, kw)), "kl")
        return
    entries = [e for e in load_entries() if e.get("noise") or e.get("measure_kl")]
    if args.runs is not None:
        entries = [e for e in entries if e["run_name"] in set(args.runs)]
    if args.model:
        entries = [e for e in entries if e["base_model"] == args.model]
    by_model: dict[tuple, list[dict]] = {}
    for e in entries:
        kw = e.get("eval_chat_template_kwargs")
        by_model.setdefault((e["base_model"], json.dumps(kw, sort_keys=True)), []).append(e)
    if not by_model:
        print("[kl] no matching registry entries")
    for (model, kw), es in by_model.items():
        print(f"[kl] {model} kwargs={kw}: {len(es)} adapters")
        train_eval.with_slot_retry(
            lambda m=model, es=es, kw=kw: asyncio.run(kl_for_entries(m, es, args.force, json.loads(kw))), "kl")


# ---------------------------------------------------------------------------
# make: calibrated noise adapters for a group
# ---------------------------------------------------------------------------

async def make_group(name: str, force: bool) -> dict:
    cfg = group_cfg(name)
    if not cfg["noise"]:
        raise SystemExit(f"{name}: no registry entry asks for noise")
    gdir = RUNS / name
    out = gdir / "noise_adapters.json"
    if out.exists() and not force:
        print(f"[make] {name}: noise_adapters.json exists (use --force to rebuild)")
        return _load(out)
    entries, base = cfg["entries"], cfg["base_model"]
    src = source_config(entries[0]["config_path"])
    gen = kl_engine_config(src, cfg.get("eval_chat_template_kwargs"), cfg["kl"])
    root, calib = cfg["out_root"], f"{cfg['out_root']}/_calib"
    async with random_lora.app.run(), modal_vllm.app.run():
        register_apps(random_lora.app, modal_vllm.app)
        samples, base_lps = await ensure_samples(base, gen, cfg["kl"])
        kls = {}
        for e in entries:  # trained KLs (reuse the kl stage's records)
            rec = KL_DIR / f"{e['run_name']}.json"
            if not rec.exists():
                res = await measure_kl(e["adapter_path"], base, gen, samples, base_lps)
                write_kl_record(e["run_name"], base, e["adapter_path"], res, gen,
                                {"experiment": cfg["experiment"], "arm": cfg["arm"], "seed": e["seed"]})
            kls[e["run_name"]] = _load(rec)["kl"]["kl_per_token"]
        target = statistics.median(kls.values())
        ref = min(entries, key=lambda e: abs(kls[e["run_name"]] - target))
        print(f"[make] {name}: trained KL/token " + ", ".join(f"seed{e['seed']}={kls[e['run_name']]:.5f}" for e in entries)
              + f"; target (median) {target:.5f}; shapes/norm profile from {ref['run_name']}", flush=True)
        if target <= 0:
            raise SystemExit(f"{name}: median trained KL {target} is not positive; nothing to match")

        def kl_at_for(seed: int):
            async def kl_at(scale: float) -> float:
                path = f"{calib}/s{seed}_x{scale:.6g}"
                await build_adapter(ref["adapter_path"], path, scale, seed, cfg["family"])
                res = await measure_kl(path, base, gen, samples, base_lps)
                k = res["kl"]["kl_per_token"]
                print(f"[make] {name} seed{seed} scale {scale:.4g}: KL {k:.5f} (target {target:.5f}, "
                      f"ratio {k / target:.3f})", flush=True)
                return k
            return kl_at

        async def finish(k: int, seed: int, scale0: float) -> dict:
            scale, hist = await calibrate_scale(kl_at_for(seed), target, scale0=scale0, tol=float(cfg["tolerance"]))
            final = f"{root}/noise{k}"
            build = await build_adapter(ref["adapter_path"], final, scale, seed, cfg["family"])
            res = await measure_kl(final, base, gen, samples, base_lps)
            write_kl_record(f"{name}--noise{k}", base, final, res, gen, {"group": name, "noise_seed": seed, "scale": scale})
            print(f"[make] {name} noise{k} (seed {seed}): scale {scale:.4g} KL {res['kl']['kl_per_token']:.5f} "
                  f"after {len(hist)} calibration evals -> {final}", flush=True)
            return {"k": k, "seed": seed, "scale": scale, "path": final, "kl": res["kl"],
                    "calibration": hist, "build": build}

        seeds = list(cfg["noise_seeds"])
        first = await finish(0, seeds[0], 1.0)                       # full calibration along seed 0
        rest = await asyncio.gather(*[finish(k, s, first["scale"]) for k, s in enumerate(seeds[1:], 1)])
        adapters = [first, *rest]
        await random_lora.remote_fn("remove_path").remote.aio(calib)
    record = {"group": name, "base_model": base, "target_kl": target, "trained_kl": kls,
              "reference": {"run_name": ref["run_name"], "adapter_path": ref["adapter_path"]},
              "family": cfg["family"], "tolerance": cfg["tolerance"], "kl_settings": cfg["kl"],
              "engine": asdict(gen), "adapters": adapters, "timestamp": _now()}
    atomic_write(out, record)
    return record


def stage_make(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="noise make")
    ap.add_argument("--group", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    _check_owner()
    train_eval.with_slot_retry(lambda: asyncio.run(make_group(args.group, args.force)), "make")


# ---------------------------------------------------------------------------
# eval: the arm's final eval blocks on the noise adapters
# ---------------------------------------------------------------------------

def resolve_blocks(entry: dict, src) -> list[tuple[str, dict]]:
    """Which eval blocks the noise adapters replicate. `noise_eval_blocks` may be a
    list of block names, "<config>.yaml#<selector>" or a bare selector; a selector
    is a top-level yaml key listing blocks, a block name, or a substring of block
    names (e.g. "final"). Default: blocks that run at the final checkpoint."""
    _, blocks, raw = src
    sel = entry.get("noise_eval_blocks")
    if isinstance(sel, str) and "#" in sel:
        path, sel = sel.split("#", 1)
        if path and path != entry["config_path"] and (ROOT / path).exists():
            _, ec, raw = train_eval.load_config(ROOT / path)
            blocks = train_eval.eval_blocks(ec)
    elif isinstance(sel, str) and sel.endswith(".yaml"):
        sel = None
    names = dict(blocks)
    step = final_step(entry["adapter_path"])
    at_final = [n for n, ec in blocks
                if ec.get("eval_at_steps") is None or (step is not None and step in {int(s) for s in ec["eval_at_steps"]})]
    if isinstance(sel, list):
        chosen = list(sel)
    elif sel:
        if isinstance(raw.get(sel), dict):  # a whole eval section, e.g. orange/rl's `final_eval:`
            blocks = train_eval.eval_blocks(raw[sel])
            names = dict(blocks)
            chosen = list(names)
        elif isinstance(raw.get(sel), list):
            chosen = [str(x) for x in raw[sel]]
        elif sel in names:
            chosen = [sel]
        else:  # substring ("final"); cpt names its final-checkpoint blocks test_*
            chosen = ([n for n in names if sel in n] or [n for n in at_final if n.startswith("test")]
                      or at_final)
    else:
        chosen = at_final
    missing = [c for c in chosen if c not in names]
    if missing or not chosen:
        raise ValueError(f"{entry['run_name']}: eval blocks {chosen} not resolvable (missing {missing}; have {sorted(names)})")
    return [(n, {**names[n], "eval_at_steps": None, "eval_base_model": False}) for n in chosen]


def slot_generate_fn(base_model: str, gen: ModalGenerateConfig, lora_path: str, chunk: int,
                     sem: asyncio.Semaphore, default_chunk: int = 250):
    """generate_fn that splits prompts into engine calls (fan-out across GPUs), each
    bounded by the per-process semaphore and the global GPU slot pool."""
    inner = make_generate_fn(base_model, gen, lora_path)

    async def one(part):
        async with sem, SLOTS.slot(gen.tensor_parallel_size):
            return await _retry(lambda: inner(part), f"generate {len(part)} prompts with {lora_path}")

    async def fn(messages_list):
        c = chunk or (default_chunk if len(messages_list) > default_chunk else 0)
        parts = [messages_list[i:i + c] for i in range(0, len(messages_list), c)] if c else [messages_list]
        outs = await asyncio.gather(*[one(p) for p in parts])
        return [r for o in outs for r in o]

    return fn


async def eval_group(name: str, force: bool, parallel: int) -> dict:
    cfg = group_cfg(name)
    gdir = RUNS / name
    na = _load(gdir / "noise_adapters.json")
    ref = next(e for e in cfg["entries"] if e["run_name"] == na["reference"]["run_name"])
    src = source_config(ref["config_path"])
    if src is None:
        raise SystemExit(f"{name}: source config {ref['config_path']} missing")
    tc, _, _ = src
    tc.run_name = ref["run_name"]
    blocks = resolve_blocks(ref, src)
    for e in cfg["entries"]:  # the arm's seeds must share one final eval
        if e is not ref and (s := source_config(e["config_path"])):
            other = {n: {k: v for k, v in ec.items() if k != "parallel_checkpoints"} for n, ec in resolve_blocks(e, s)}
            mine = {n: {k: v for k, v in ec.items() if k != "parallel_checkpoints"} for n, ec in blocks}
            if other != mine:
                print(f"[eval] WARNING {e['run_name']}: final eval blocks differ from {ref['run_name']}'s; "
                      f"using {ref['run_name']}'s", flush=True)
    for n, ec in blocks:
        if float(ec["temperature"]) != 1.0 or int(ec["max_tokens"]) != 16000:
            print(f"[eval] WARNING block {n}: temperature={ec['temperature']} max_tokens={ec['max_tokens']} "
                  f"(spec: T=1, 16000) — replicated as configured", flush=True)
    sem = asyncio.Semaphore(parallel)
    tasks = []
    for bname, ec in blocks:
        gen = train_eval.build_generate_config(tc, ec)
        for a in na["adapters"]:
            if (gdir / "eval" / bname / f"noise{a['k']}.json").exists() and not force:
                continue
            tasks.append((bname, ec, gen, a))
    print(f"[eval] {name}: {len(blocks)} blocks x {len(na['adapters'])} noise adapters "
          f"({len(tasks)} evals to run): " + ", ".join(f"{n}[{ec['split']}/{ec['mode']}]" for n, ec in blocks), flush=True)

    async def one(bname, ec, gen, a):
        t0 = time.time()
        result = await eval_cotcontrolqa(
            model=tc.base_model,
            system_prompt=train_eval.resolve_system_prompt(ec["system_prompt"]),
            generate_fn=slot_generate_fn(tc.base_model, gen, a["path"], int(ec.get("chunk") or 0), sem),
            save_dir=gdir / "eval" / bname, save_name=f"noise{a['k']}",
            dataset=ec["dataset"], mode=ec["mode"], allowed_modes=ec["allowed_modes"], seed=int(ec["seed"]),
            state_requirement=bool(ec["state_requirement"]), max_samples=ec["max_samples"],
            subsample_seed=ec["subsample_seed"], split=ec["split"],
            grade_meta_discussion=ec["grade_meta_discussion"],
            meta_discussion_scope=ec.get("meta_discussion_scope", "compliant"),
            judge_model=ec["judge_model"], judge_concurrency=int(ec["judge_concurrency"]),
            backend_info={"base_model": tc.base_model, "lora_path": a["path"], "generate_config": asdict(gen),
                          "noise": {k: a[k] for k in ("k", "seed", "scale")}, "group": name},
        )
        s = result["summary"]
        print(f"[eval] {name} {bname} noise{a['k']}: compliance={s['compliance_rate']} accuracy={s['accuracy']} "
              f"errors={s['n_errors']} ({time.time() - t0:.0f}s)", flush=True)
        return {"block": bname, "k": a["k"], "lora_path": a["path"], "compliance_rate": s["compliance_rate"],
                "accuracy": s["accuracy"], "n_errors": s["n_errors"], "per_mode": s["per_mode"]}

    rows: list = []
    if tasks:
        async with modal_vllm.app.run():
            register_apps(modal_vllm.app)
            # Let the other evals finish (and save) when one dies: the retry then
            # only redoes the missing files.
            outs = await asyncio.gather(*[one(*t) for t in tasks], return_exceptions=True)
        failed = [(t, r) for t, r in zip(tasks, outs) if isinstance(r, BaseException)]
        rows = [r for r in outs if not isinstance(r, BaseException)]
        for (bname, _, _, a), err in failed:
            print(f"[eval] {name} {bname} noise{a['k']} FAILED: {type(err).__name__}: {str(err)[:200]}", flush=True)
        if failed:
            raise SystemExit(f"{name}: {len(failed)}/{len(tasks)} evals failed; {len(rows)} saved, rerun to finish")
    summary_path = gdir / "eval_summary.json"
    prev = _load(summary_path)["evals"] if summary_path.exists() else []
    keep = [r for r in prev if not any(r["block"] == n["block"] and r["k"] == n["k"] for n in rows)]
    summary = {"group": name, "base_model": tc.base_model, "blocks": {n: ec for n, ec in blocks},
               "evals": sorted(keep + list(rows), key=lambda r: (r["block"], r["k"])), "timestamp": _now()}
    atomic_write(summary_path, summary)
    return summary


def stage_eval(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="noise eval")
    ap.add_argument("--group", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--parallel", type=int, default=6, help="concurrent engine calls in this process")
    args = ap.parse_args(argv)
    _check_owner()
    train_eval.with_slot_retry(lambda: asyncio.run(eval_group(args.group, args.force, args.parallel)), "eval")


# ---------------------------------------------------------------------------
# process: one group end to end
# ---------------------------------------------------------------------------

def write_manifest(name: str) -> None:
    cfg = group_cfg(name)
    files = sorted(ROOT.glob("datasets/*.csv")) + [ROOT / "datasets/splits.json"]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=False).stdout.strip() or None
    except OSError:
        commit = None
    registry = {}
    for e in cfg["entries"]:
        p = ROOT / e["registry_path"]
        registry[e["run_name"]] = _load(p) if p.exists() else None
    manifest = {
        "group": cfg, "timestamp": _now(), "git_commit": commit, "modal_volume": _volume_name(),
        "datasets_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in files if p.exists()},
        "source_configs_sha256": {e["config_path"]: _sha256(ROOT / e["config_path"])
                                  for e in cfg["entries"] if (ROOT / e["config_path"]).exists()},
        "registry_entries": registry,
    }
    atomic_write(RUNS / name / "manifest.json", manifest)


def stage_process(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="noise process")
    ap.add_argument("--group", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--parallel", type=int, default=6)
    args = ap.parse_args(argv)
    _check_owner()
    gdir = RUNS / args.group
    gdir.mkdir(parents=True, exist_ok=True)
    with tee_stdio(gdir / "process.log", "a"):
        print(f"\n=== {_now()} process {args.group} ===", flush=True)
        write_manifest(args.group)
        train_eval.with_slot_retry(lambda: asyncio.run(make_group(args.group, args.force)), "make")
        train_eval.with_slot_retry(lambda: asyncio.run(eval_group(args.group, args.force, args.parallel)), "eval")
        noise_analyze.main(["--groups", args.group])
        print(f"=== {_now()} done {args.group} ===", flush=True)


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------

def _container_counts() -> tuple[int | None, int | None]:
    """(workspace-wide active Modal containers, those of this folder's live apps);
    best effort via the CLI, None when it cannot be read."""
    try:
        ids = set()
        for p in APPS_DIR.glob("*.json"):
            d = _load(p)
            try:
                os.kill(int(d["pid"]), 0)
            except (ProcessLookupError, ValueError):
                p.unlink(missing_ok=True)
                continue
            except PermissionError:
                pass
            ids |= {a for a in d.get("apps", []) if a}
        out = subprocess.run([sys.executable, "-m", "modal", "container", "list", "--json"],
                             capture_output=True, text=True, timeout=60, check=False,
                             env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"}).stdout
        out = re.sub(r"\x1b\[[0-9;]*m", "", out)  # the CLI colorizes even when piped
        rows = json.loads(out) if out.strip() else []
        # rows: {container_id, app_id, app_name, start_time}; one container = one engine
        return len(rows), sum(1 for r in rows if r.get("app_id") in ids)
    except Exception:  # noqa: BLE001 — monitoring only
        return None, None


def _free_memory_pct() -> float | None:
    """macOS `memory_pressure` free percentage (None elsewhere / on failure)."""
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30, check=False).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001 — monitoring only
        return None


class _Adopted:
    """A `noise kl|process` subprocess started by an earlier watcher, tracked by pid."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 0  # exit code unknown; outputs decide whether work gets redone
        return None

    def wait(self):
        while self.poll() is None:
            time.sleep(5)


def _adopt_running(entries: list[dict]) -> dict[str, tuple]:
    """Running kl/process subprocesses from a previous watcher, keyed like ours."""
    slug_of = {e["run_name"]: model_slug(e["base_model"]) for e in entries}
    out = {}
    ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False).stdout
    for line in ps.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if "run.py noise kl --runs" in cmd:
            names = [n for n in cmd.split("--runs", 1)[1].split() if not n.startswith("--")]
            if names:
                out[f"kl:{slug_of.get(names[0], 'unknown')}:{','.join(names)}"] = (_Adopted(int(pid)), time.time())
        elif "run.py noise process --group" in cmd:
            g = cmd.split("--group", 1)[1].split()[0]
            out[f"group:{g}"] = (_Adopted(int(pid)), time.time())
    return out


def _spawn(key: str, args: list[str]) -> subprocess.Popen:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = LOGS_DIR / f"{key.replace(':', '_')}-{datetime.now():%Y%m%d_%H%M%S}.log"
    with open(log, "w") as fh:  # Popen dups the fd; closing our handle is fine
        return subprocess.Popen([sys.executable, str(KAUSHIK / "run.py"), "noise", *args],
                                stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)


def write_progress(path: Path, state: dict, groups: dict, entries: list[dict], procs: dict,
                   events: list[str], missing_runs: dict) -> None:
    kl_done = {e["run_name"]: (KL_DIR / f"{e['run_name']}.json").exists() for e in entries}
    paused = SLOTS.state().get("paused")
    status = (f"started {state.get('started', '?')}; first full launch {state.get('first_launch', 'none yet')}; "
              f"GPU cap {SLOTS.cap()}{' [PAUSED: ' + str(SLOTS.state().get('pause_reason')) + ']' if paused else ''} "
              f"(slots held {SLOTS.held()}, our modal containers {state.get('modal_gpus', '?')}, "
              f"workspace {state.get('workspace_containers', '?')}, laptop free memory "
              f"{state.get('free_mem_pct', '?')}%); "
              f"{len(procs)} subprocess(es) running: {', '.join(procs) or '-'}\n")
    L = [f"# noise watch progress — {_now()}\n", status,
         "| group | seeds (have/expected) | KL measured | noise adapters | noise eval | first seen | attempts | note |",
         "|---|---|---|---|---|---|---|---|"]
    for g, cfg in sorted(groups.items()):
        gs = state["groups"].get(g, {})
        gdir = RUNS / g
        n_kl = sum(kl_done.get(e["run_name"], False) for e in cfg["entries"])
        na = (gdir / "noise_adapters.json").exists()
        es = (gdir / "eval_summary.json").exists()
        n_eval = len(_load(gdir / "eval_summary.json")["evals"]) if es else 0
        note = gs.get("last_error", "")[:80].replace("\n", " ") if gs.get("last_error") else ""
        if not cfg["noise"]:
            note = "KL only"
        L.append(f"| {g} | {len(cfg['entries'])}/{cfg['expected_seeds']} | {n_kl}/{len(cfg['entries'])} | "
                 f"{'done' if na else '-'} | {f'{n_eval} evals' if es else '-'} | {gs.get('first_seen_str', '')} | "
                 f"{gs.get('attempts', 0)} | {note} |")
    L += ["\n## KL(base || adapter) per token\n", "| run | KL | ±se (prompt bootstrap) | tokens |", "|---|---|---|---|"]
    for e in sorted(entries, key=lambda e: e["run_name"]):
        p = KL_DIR / f"{e['run_name']}.json"
        if p.exists():
            k = _load(p)["kl"]
            L.append(f"| {e['run_name']} | {k['kl_per_token']:.5f} | {k['se_prompt']:.5f} | {k['n_tokens']} |")
        else:
            L.append(f"| {e['run_name']} | pending | | |")
    if missing_runs:
        L += ["\n## Configs without a registry entry yet\n"] + [f"- {r} ({exp})" for r, exp in sorted(missing_runs.items())]
    L += ["\n## Recent events\n"] + [f"- {e}" for e in events[-40:]]
    path.write_text("\n".join(L) + "\n")


def stage_watch(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="noise watch", description="poll the registry and process groups")
    ap.add_argument("--interval", type=float, default=180, help="seconds between polls")
    ap.add_argument("--max-procs", type=int, default=6, help="concurrent group (build+eval) subprocesses")
    ap.add_argument("--parallel", type=int, default=8, help="engine calls per group job")
    ap.add_argument("--parallel-27b", type=int, default=16,
                    help="engine calls per Qwen3.6-27B group job (the slow, critical-path evals)")
    ap.add_argument("--max-kl-procs", type=int, default=3, help="concurrent KL subprocesses (minutes each)")
    ap.add_argument("--max-hours", type=float, default=24)
    ap.add_argument("--idle-hours", type=float, default=4, help="stop after this long with no new entries")
    ap.add_argument("--group-wait-hours", type=float, default=4, help="process a group with missing seeds after this")
    ap.add_argument("--cap", type=int, default=None, help="set runs/gpu_cap.json (else keep it)")
    ap.add_argument("--max-workspace-containers", type=int, default=800,
                    help="launch nothing new while the whole workspace has more containers than this")
    ap.add_argument("--min-free-mem-pct", type=float, default=25,
                    help="launch no local subprocess while laptop free memory is below this")
    ap.add_argument("--kl-retries", type=int, default=5)
    ap.add_argument("--group-retries", type=int, default=5, help="retries are cheap: finished chunks are cached")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    _check_owner()
    RUNS.mkdir(parents=True, exist_ok=True)
    state_path = RUNS / "watch_state.json"
    state = _load(state_path) if state_path.exists() else {"groups": {}, "kl": {}}
    state.setdefault("groups", {})
    state.setdefault("kl", {})
    state["started"] = state.get("started") or _now()
    if args.cap is not None:
        SLOTS.set_cap(args.cap, reason="watch --cap")
    events: list[str] = []
    procs: dict[str, tuple] = _adopt_running(load_entries())
    known: set[str] = set()
    t_start = time.time()
    last_new = t_start
    # Resume from a pause left by a previous watcher (cap frozen in gpu_cap.json).
    _st = SLOTS.state()
    blocked = str(_st.get("pause_reason") or "paused").split(" ")[0] if _st.get("paused") else None

    def log(msg: str) -> None:
        line = f"{_now()} {msg}"
        events.append(line)
        print(line, flush=True)
        with open(RUNS / "watch.log", "a") as fh:
            fh.write(line + "\n")

    if procs:
        log(f"adopted running subprocesses: {', '.join(procs)}")
    with tee_stdio(RUNS / "watch_stdout.log", "a"):
        while True:
            now = time.time()
            entries = load_entries()
            groups = write_group_configs(entries)
            new = {e["run_name"] for e in entries} - known
            if new:
                known |= new
                last_new = now
                log(f"registry: {len(entries)} entries ({len(new)} new: {', '.join(sorted(new))})")
            for g in groups:
                gs = state["groups"].setdefault(g, {"first_seen": now, "first_seen_str": _now(), "attempts": 0})
                gs.setdefault("attempts", 0)
            # reap finished subprocesses
            for key, (p, t0) in list(procs.items()):
                rc = p.poll()
                if rc is None:
                    continue
                del procs[key]
                kind, _, target = key.partition(":")   # group:<G> | kl:<model>:<run,run,...>
                if isinstance(p, _Adopted):  # exit status unknown: judge by the outputs
                    done = ((RUNS / target / "eval_summary.json").exists() if kind == "group" else
                            all((KL_DIR / f"{r}.json").exists() for r in target.partition(":")[2].split(",")))
                    rc = 0 if done else 1
                if rc == 0:
                    log(f"{key} finished in {(now - t0) / 60:.0f} min")
                    if kind == "group":
                        state["groups"][target]["done_at"] = _now()
                        state["groups"][target].pop("last_error", None)
                else:
                    log(f"{key} FAILED (exit {rc}); see runs/_logs/")
                    if kind == "group":
                        gs = state["groups"][target]
                        gs["attempts"] += 1
                        gs["last_error"] = f"exit {rc} at {_now()}"
                        gs["next_try"] = now + 120 * gs["attempts"]  # cheap: finished chunks are cached
                    else:
                        for r in target.partition(":")[2].split(","):
                            st = state["kl"].setdefault(r, {"attempts": 0})
                            st["attempts"] += 1
                            st["next_try"] = now + 300 * st["attempts"]
            # launch guards: workspace-wide container count and laptop memory
            ws, mine = _container_counts()
            mem = _free_memory_pct()
            state.update({"workspace_containers": ws, "modal_gpus": mine, "free_mem_pct": mem})
            reason = (f"workspace has {ws} containers > {args.max_workspace_containers}"
                      if ws is not None and ws > args.max_workspace_containers else
                      f"free memory {mem:.0f}% < {args.min_free_mem_pct}%"
                      if mem is not None and mem < args.min_free_mem_pct else None)
            kind = reason.split(" ")[0] if reason else None  # "workspace" | "free" — ignore count jitter
            if kind != blocked:
                log(f"launches {'paused: ' + reason if reason else 'resumed'}")
                blocked = kind
                # Freeze the GPU footprint too: running jobs re-read the cap on every
                # slot acquisition, so cap = slots held means no new engines while
                # the workspace is over the line; restore the full cap afterwards.
                st = SLOTS.state()
                full = int(st.get("resume_cap", SLOTS.cap())) if st.get("paused") else SLOTS.cap()
                if reason:
                    SLOTS.set_cap(max(SLOTS.held(), 1), paused=True, resume_cap=full, pause_reason=reason)
                elif st.get("paused"):
                    SLOTS.set_cap(full, paused=False, pause_reason=None)
            can_launch = reason is None
            # KL jobs: one subprocess per (base model, template kwargs) with pending adapters
            pending: dict[str, list[dict]] = {}
            for e in entries:
                if not (e.get("noise") or e.get("measure_kl")) or (KL_DIR / f"{e['run_name']}.json").exists():
                    continue
                st = state["kl"].get(e["run_name"], {})
                if st.get("attempts", 0) >= args.kl_retries or st.get("next_try", 0) > now:
                    continue
                key = f"kl:{model_slug(e['base_model'])}"
                if any(k.startswith(key + ":") for k in procs):
                    continue  # one KL engine per base model at a time; the rest wait a poll
                pending.setdefault(key, []).append(e)
            for key, es in pending.items():
                if sum(k.startswith("kl:") for k in procs) >= args.max_kl_procs or not can_launch:
                    continue
                names = [e["run_name"] for e in es]
                pkey = f"{key}:{','.join(names)}"
                procs[pkey] = (_spawn(key, ["kl", "--runs", *names]), now)
                state.setdefault("first_launch", _now())
                state.setdefault("first_launch_ts", now)
                log(f"spawned {key} for {len(names)} adapters: {', '.join(names)}")
            # noise jobs: a group once its KLs are in and its seeds are complete (or it waited long enough).
            # Qwen3.6-27B groups first: their evals are the slowest and on Experiment 2's critical path.
            for g, cfg in sorted(groups.items(), key=lambda kv: ("3.6" not in kv[1]["base_model"], kv[0])):
                gs = state["groups"][g]
                key = f"group:{g}"
                if (not cfg["noise"] or (RUNS / g / "eval_summary.json").exists() or key in procs
                        or gs["attempts"] >= args.group_retries or gs.get("next_try", 0) > now
                        or sum(k.startswith("group:") for k in procs) >= args.max_procs or not can_launch):
                    continue
                if not all((KL_DIR / f"{e['run_name']}.json").exists() for e in cfg["entries"]):
                    continue
                complete = len(cfg["entries"]) >= cfg["expected_seeds"]
                waited = now - gs["first_seen"] >= args.group_wait_hours * 3600
                if not (complete or waited):
                    continue
                par = args.parallel_27b if "3.6" in cfg["base_model"] else args.parallel
                procs[key] = (_spawn(key, ["process", "--group", g, "--parallel", str(par)]), now)
                state.setdefault("first_launch", _now())
                state.setdefault("first_launch_ts", now)
                log(f"spawned {key} ({len(cfg['entries'])}/{cfg['expected_seeds']} seeds"
                    f"{'' if complete else ', wait expired'})")
            missing = {r: x for r, x in expected_run_names().items() if r not in known}
            atomic_write(state_path, state)
            write_progress(RUNS / "progress.md", state, groups, entries, procs, events, missing)
            if args.once:
                break
            elapsed_h = (now - t_start) / 3600
            idle_h = (now - last_new) / 3600
            all_done = not procs and not pending and all(
                not cfg["noise"] or (RUNS / g / "eval_summary.json").exists()
                or state["groups"][g]["attempts"] >= args.group_retries for g, cfg in groups.items())
            if elapsed_h >= args.max_hours:
                log(f"stopping: {elapsed_h:.1f}h elapsed")
                break
            if idle_h >= args.idle_hours and all_done and known:
                log(f"stopping: no new entries for {idle_h:.1f}h and every registered run is processed "
                    f"({len(missing)} configs never registered: {', '.join(sorted(missing)[:8])}...)")
                break
            time.sleep(args.interval)
        if not args.once:  # a final stop drains its subprocesses; a single poll just reports
            for p, _ in procs.values():
                p.wait()


# ---------------------------------------------------------------------------
# smoke: KL sanity on Qwen3-8B
# ---------------------------------------------------------------------------

async def smoke(args) -> dict:
    out_dir = RUNS / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.base_model
    gen = ModalGenerateConfig(temperature=1.0, max_tokens=args.max_tokens, seed=0, return_logprobs=True,
                              gpu="H200", max_lora_rank=64, max_model_len=16384,
                              chat_template_kwargs=json.loads(args.template_kwargs) if args.template_kwargs else None)
    report: dict = {"base_model": base, "trained": args.adapter, "checkpoint0": args.ckpt0, "checks": {}}
    async with random_lora.app.run(), modal_vllm.app.run():
        register_apps(random_lora.app, modal_vllm.app)
        prompts = unconstrained_prompts(base, "val", "all", args.n_prompts)
        async with SLOTS.slot():
            samples = await sample_base(prompts, base, gen)
        atomic_write(out_dir / "samples.json", samples)
        async with SLOTS.slot():
            base_lps = await score_samples(samples, base, gen, None)
        async with SLOTS.slot():
            base_lps2 = await score_samples(samples, base, gen, None)
        report["kl_base_base"] = kl_from_logprobs(base_lps, base_lps2)["kl_per_token"]
        report["mean_abs_rescore_vs_sampled"] = statistics.fmean(
            abs(a - b) for s, l in zip(samples["sampled_logprobs"], base_lps) for a, b in zip(s, l))
        if args.ckpt0:
            report["kl_ckpt0"] = (await measure_kl(args.ckpt0, base, gen, samples, base_lps))["kl"]
        trained = (await measure_kl(args.adapter, base, gen, samples, base_lps))["kl"]
        report["kl_trained"] = trained
        print(f"[smoke] KL(base||base)={report['kl_base_base']:.2e}  "
              f"KL(base||ckpt0)={report.get('kl_ckpt0', {}).get('kl_per_token', float('nan')):.5f}  "
              f"KL(base||trained)={trained['kl_per_token']:.5f} ± {trained['se_prompt']:.5f}", flush=True)
        root = "/checkpoints/noise/_smoke"
        scan = []
        for scale in args.scales:
            path = f"{root}/s0_x{scale:g}"
            build = await build_adapter(args.adapter, path, scale, 0, "frob")
            k = (await measure_kl(path, base, gen, samples, base_lps))["kl"]
            scan.append({"scale": scale, "kl": k["kl_per_token"], "se": k["se_prompt"], "norm_ratio": build["random_norm"] / max(build["trained_norm"], 1e-30)})
            print(f"[smoke] random scale {scale:g}: KL {k['kl_per_token']:.5f} ± {k['se_prompt']:.5f} "
                  f"(norm ratio {scan[-1]['norm_ratio']:.2f})", flush=True)
        report["scan"] = scan
        target = trained["kl_per_token"]

        async def kl_at(scale):
            path = f"{root}/calib_x{scale:.6g}"
            await build_adapter(args.adapter, path, scale, 1, "frob")
            k = (await measure_kl(path, base, gen, samples, base_lps))["kl"]["kl_per_token"]
            print(f"[smoke] calib scale {scale:.4g}: KL {k:.5f} (ratio {k / target:.3f})", flush=True)
            return k

        scale, hist = await calibrate_scale(kl_at, target, tol=0.1)
        report["calibration"] = {"target": target, "scale": scale, "history": hist}
        await random_lora.remote_fn("remove_path").remote.aio(root)
    # Monotonicity is only testable above the estimator's noise floor (small
    # random adapters sit at KL ~1e-4 +- 1.5e-4 here); use points > 3 SE from 0.
    above = [s["kl"] for s in scan if s["kl"] > 3 * s["se"]]
    report["checks"] = {
        "kl_base_base_zero": abs(report["kl_base_base"]) < 1e-6,
        "kl_ckpt0_near_zero": (report.get("kl_ckpt0", {}).get("kl_per_token", 0.0) < 1e-3) if args.ckpt0 else None,
        "kl_trained_positive": target > 0,
        "random_kl_monotone": (all(a < b for a, b in itertools.pairwise(above)) and above[-1] == max(s["kl"] for s in scan)
                               if above else None),
        "calibration_within_tol": abs(hist[-1]["kl"] / target - 1) <= 0.1 if hist else False,
    }
    atomic_write(out_dir / "smoke.json", report)
    for k, v in report["checks"].items():
        print(f"[smoke] {k}: {'PASS' if v else ('skip' if v is None else 'FAIL')}", flush=True)
    print(f"[smoke] wrote {out_dir / 'smoke.json'}")
    return report


def stage_smoke(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="noise smoke")
    ap.add_argument("--adapter", default=None, help="trained adapter dir; default: train 5 steps via configs/smoke_qwen8b.yaml")
    ap.add_argument("--ckpt0", default=None, help="checkpoint-0 (zero-init) adapter dir")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--template-kwargs", default='{"enable_thinking": true}')
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--scales", nargs="*", type=float, default=[0.25, 1.0, 4.0, 16.0])
    args = ap.parse_args(argv)
    _check_owner()
    if not args.adapter:
        cfg = HERE / "configs" / "smoke_qwen8b.yaml"
        run_dir = ROOT / yaml.safe_load(cfg.read_text())["run_dir"]
        tr = run_dir / "train_result.json"
        if not tr.exists():
            print(f"[smoke] training a 5-step adapter with {cfg}", flush=True)
            subprocess.run([sys.executable, str(ROOT / "scripts/train_eval.py"), str(cfg)], check=True, cwd=ROOT)
        ck = {int(c["step"]): c["path"] for c in _load(tr)["checkpoints"]}
        args.adapter = ck[max(ck)]
        args.ckpt0 = args.ckpt0 or ck.get(0)
        print(f"[smoke] trained={args.adapter} ckpt0={args.ckpt0}", flush=True)
    train_eval.with_slot_retry(lambda: asyncio.run(smoke(args)), "smoke")


# ---------------------------------------------------------------------------

def stage_deploy(argv: list[str]) -> None:
    """(Re)deploy the CPU adapter-build app; run after changing cotcontrol/training/random_lora.py."""
    _check_owner()
    env = {k: v for k, v in os.environ.items() if k != "COTCONTROL_MODAL_RANDOM_LORA_APP"}
    subprocess.run([sys.executable, "-m", "modal", "deploy", "-m", "cotcontrol.training.random_lora",
                    "--name", RANDOM_LORA_APP], check=True, cwd=ROOT, env=env)


def stage_make_configs(argv: list[str]) -> None:
    cfgs = write_group_configs()
    for name, cfg in cfgs.items():
        print(f"{name}: {len(cfg['entries'])}/{cfg['expected_seeds']} seeds, noise={cfg['noise']}, "
              f"measure_kl={cfg['measure_kl']}")
    print(f"{len(cfgs)} groups -> {CONFIGS.relative_to(ROOT)}")


STAGES = {
    "deploy": stage_deploy,
    "make_configs": stage_make_configs,
    "kl": stage_kl,
    "make": stage_make,
    "eval": stage_eval,
    "process": stage_process,
    "watch": stage_watch,
    "analyze": lambda argv: noise_analyze.main(argv),
    "smoke": stage_smoke,
}
