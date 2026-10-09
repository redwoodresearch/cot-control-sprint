"""Shared pieces for kaushik/retry: model table, item construction, generation
backends (pinned OpenRouter; Modal vLLM fanned out under a GPU cap) and run
manifests. Import Modal-touching code lazily: kaushik/run.py sets the Modal
owner env BEFORE cotcontrol.inference.modal_vllm is imported."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from cotcontrol.eval.data import DATASETS_DIR, assign_tasks, load_dataset
from cotcontrol.eval.retry import make_items
from cotcontrol.inference.openrouter import GenerateConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs"
CFG = json.loads((HERE / "configs" / "retry.json").read_text())
MODELS = json.loads((HERE / "configs" / "models.json").read_text())


def log_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")
    return log


# --------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------

def load_items(model_id: str, split: str, modes_per_question: int = 9, seed: int = 0,
               max_questions: int | None = None, subsample_seed: int | None = None,
               modes: list[str] | None = None, subset: str | None = None) -> list[dict]:
    """Test/train split x constraint modes -> retry items. modes_per_question < 9
    keeps a seeded random subset of each question's valid modes (keyword modes
    only where the question has keywords, as the standard eval assigns them).
    subset: path (relative to kaushik/retry) of a JSON {"item_ids": [...]} that
    restricts the items (same ids for every model and arm, so comparisons stay paired)."""
    samples = load_dataset("all", "all", max_samples=max_questions, subsample_seed=subsample_seed, split=split)
    tasks = assign_tasks(samples, "all", allowed_modes=modes)
    if modes_per_question < 9:
        rng = random.Random(seed)
        by_q: dict[tuple, list] = {}
        for s, m in tasks:
            by_q.setdefault((s["dataset"], s["id"]), []).append((s, m))
        tasks = [t for group in by_q.values()
                 for t in sorted(rng.sample(group, min(modes_per_question, len(group))), key=lambda x: x[1])]
    items = make_items(tasks, model_id)
    if subset:
        keep = set(json.loads((HERE / subset).read_text())["item_ids"])
        items = [it for it in items if it["item_id"] in keep]
    return items


def make_subset(spec: dict, n: int = 1000, seed: int = 0, out: Path | None = None) -> dict:
    """One seeded, mode-stratified random subset of an item spec's items
    (~n/9 per mode; item_ids are model-independent). Written with its sha256."""
    items = load_items("openai/gpt-oss-20b", **{k: v for k, v in spec.items() if k != "subset"})
    by_mode: dict[str, list[str]] = {}
    for it in items:
        by_mode.setdefault(it["mode"], []).append(it["item_id"])
    rng = random.Random(seed)
    per = n // len(by_mode)
    extra = n - per * len(by_mode)
    chosen = []
    for i, m in enumerate(sorted(by_mode)):
        ids = sorted(by_mode[m])
        chosen += rng.sample(ids, min(per + (1 if i < extra else 0), len(ids)))
    chosen.sort()
    d = {"seed": seed, "n": len(chosen), "per_mode": {m: sum(c.endswith(":" + m) for c in chosen) for m in sorted(by_mode)},
         "source_spec": {k: v for k, v in spec.items() if k != "subset"}, "n_source_items": len(items),
         "sha256": hashlib.sha256("\n".join(chosen).encode()).hexdigest(), "item_ids": chosen,
         "created": datetime.now().isoformat(),
         "note": "User decision 2026-10-08 13:05: 3A main uses this 1000-item subset for every arm and model; "
                 "earlier records of other items stay on disk but are excluded from analysis."}
    if out:
        out.write_text(json.dumps(d, indent=1))
    return d


def dataset_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(DATASETS_DIR.glob("*_w_keyword.csv")) + [DATASETS_DIR / "splits.json"]:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def power_statement(n_items: int, p: float = 0.5) -> str:
    se_one = (p * (1 - p) / n_items) ** 0.5
    se_diff = (2 * p * (1 - p) / n_items) ** 0.5
    return (f"n={n_items} items/arm: SE(single arm) = {100 * se_one:.2f} pts, "
            f"SE(arm - baseline, independent samples) = {100 * se_diff:.2f} pts at p={p}")


# --------------------------------------------------------------------------
# GPU budget + backends
# --------------------------------------------------------------------------

GPU_CAPS_FILE = HERE / "configs" / "gpu_caps.json"


class GpuBudget:
    """Process-wide cap on concurrent Modal engine calls (one call = one GPU
    with tensor_parallel 1). The cap is re-read on every acquire from
    configs/gpu_caps.json[name] when that key exists, so caps can be retuned
    live across the pilot / main / registry drivers without restarts (the
    coordinator's hard limit is the SUM over drivers: 15, then <= 7 six hours
    after the first full launch). Without a file entry, the cap drops to
    `late_cap` (default half) `hours` after this process's first acquire."""

    def __init__(self, cap: int, late_cap: int | None = None, hours: float | None = None, name: str | None = None):
        self.cap, self.late_cap, self.hours, self.name = cap, late_cap, hours, name
        self.in_flight, self.t0 = 0, None
        self.cond = asyncio.Condition()

    def current_cap(self) -> int:
        if self.name and GPU_CAPS_FILE.exists():
            try:
                caps = json.loads(GPU_CAPS_FILE.read_text())
                if self.name in caps:
                    return int(caps[self.name])
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        if self.hours and self.t0 and time.time() - self.t0 > self.hours * 3600:
            return self.late_cap if self.late_cap is not None else (self.cap + 1) // 2
        return self.cap

    def _publish(self) -> None:
        """runs/_gpu_inflight/<name>.json: busy GPUs of this budget, so `retry
        gpus` can total our usage even inside the shared deployed app."""
        if not self.name:
            return
        f = RUNS / "_gpu_inflight" / (self.name.replace("/", "__") + ".json")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"name": self.name, "in_flight": self.in_flight, "cap": self.current_cap(),
                                 "pid": os.getpid(), "t": time.time()}))

    @contextlib.asynccontextmanager
    async def slot(self):
        async with self.cond:
            self.t0 = self.t0 or time.time()
            # Timed wait: a cap raised in configs/gpu_caps.json must wake a
            # budget that had no in-flight calls (nothing else notifies it).
            while not self.in_flight < self.current_cap():
                with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                    await asyncio.wait_for(self.cond.wait(), timeout=30)
            self.in_flight += 1
            self._publish()
        try:
            yield
        finally:
            async with self.cond:
                self.in_flight -= 1
                self._publish()
                self.cond.notify_all()


def patch_engine_pool(max_containers: int, scaledown_window: int = 300) -> None:
    """Hard ceiling on live containers per engine pool in THIS process.

    cotcontrol.inference.modal_vllm keeps finished containers warm (600 s when this was
    written; now DEFAULT_SCALEDOWN_WINDOW) and exposes no container limit; a driver whose calls alternate between two
    pools (two base models) therefore held 2x its in-flight cap in live GPUs
    (observed 60 containers for a 30-call budget, 2026-10-08). This wraps
    modal_vllm._engine_cls (the local class, or the deployed copy named by
    COTCONTROL_MODAL_INFERENCE_APP) so every with_options call also sets
    max_containers / scaledown_window. Process-global and idempotent: give the
    process the hard cap it may ever use; the live cap (GpuBudget) stays <= it."""
    from cotcontrol.inference import modal_vllm
    if getattr(modal_vllm._engine_cls, "_retry_pool_cap", None) is not None:
        return
    orig_engine_cls = modal_vllm._engine_cls

    class _Engine:
        def __init__(self, cls):
            self._cls = cls

        def with_options(self, **kw):
            return self._cls.with_options(**{"max_containers": max_containers,
                                             "scaledown_window": scaledown_window, **kw})

        def __getattr__(self, name):
            return getattr(self._cls, name)

    def _engine_cls():
        return _Engine(orig_engine_cls())
    _engine_cls._retry_pool_cap = max_containers
    modal_vllm._engine_cls = _engine_cls


def deployed_mode() -> bool:
    """True when kaushik/run.py pointed cotcontrol at a deployed inference app
    (COTCONTROL_MODAL_INFERENCE_APP): no ephemeral app is opened."""
    return bool(os.environ.get("COTCONTROL_MODAL_INFERENCE_APP"))


def openrouter_config(mcfg: dict, concurrency: int | None = None) -> GenerateConfig:
    pins = json.loads((ROOT / mcfg["pins"]).read_text())[mcfg["model"]]
    s = CFG["sampling"]
    return GenerateConfig(
        temperature=s["temperature"], top_p=s["top_p"], max_tokens=s["max_tokens"],
        max_concurrency=concurrency or pins["concurrency"], max_retries=pins["max_retries"],
        provider={"only": [pins["provider"]], "allow_fallbacks": False,
                  **({"quantizations": pins["quantizations"]} if pins.get("quantizations") else {})},
        extra_body={"reasoning": {"effort": pins["reasoning_effort"]}} if pins.get("reasoning_effort") else None,
    )


class ShardCache:
    """Completed generation shards of a run, keyed by (tag, model/lora, the
    shard's messages). Restarting a driver then re-generates only the shards
    of the interrupted round that had not finished. `tag` comes from the
    retry loop (arm + fingerprint of the items' attempt indices) so the
    identical prompts of the pass@k arms' successive rounds never collide."""

    def __init__(self, cache_dir: Path | None, model_id: str):
        self.dir, self.model_id = cache_dir, model_id

    def key(self, tag: str | None, part: list) -> str:
        blob = json.dumps({"tag": tag, "model": self.model_id, "messages": part}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def load(self, k: str):
        if self.dir is None or tag_is_volatile(k):
            return None
        f = self.dir / f"{k}.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def save(self, k: str, results: list) -> None:
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{k}.json.tmp"
        tmp.write_text(json.dumps(results))
        tmp.replace(self.dir / f"{k}.json")


def tag_is_volatile(_k: str) -> bool:
    return False


async def cached_call(cache: ShardCache, tag: str | None, part: list, call):
    if tag is None:  # callers without a stable tag (e.g. transfer evals) are not cached
        return await call()
    k = cache.key(tag, part)
    hit = cache.load(k)
    if hit is not None:
        return hit
    out = await call()
    if all(m["error"] is None for r in out for m in r["metadata"]):  # never cache errored shards
        cache.save(k, out)
    return out


async def spawn_generate(part: list, base_model: str, cfg, lora_path: str | None,
                         poll_timeout: float = 120, max_poll_minutes: float = 120) -> list[dict]:
    """modal_vllm.generate_async's engine call, as spawn + poll.

    `.remote` ties the GPU result to one long-lived client call: a client-side
    "ConnectionError: Deadline exceeded" (seen on most ~6-min Qwen shards from
    this laptop, 2026-10-08) loses the whole batch and re-runs it. With spawn
    the inputs are uploaded once, the result is kept server-side, and a failed
    poll is simply polled again. Same engine class (through our pool-capped
    _engine_cls proxy), same kwargs, same canonical output schema."""
    import modal

    from cotcontrol.inference import modal_vllm as mv
    engine = mv._engine_cls().with_options(gpu=cfg.gpu)(
        base_model=base_model, tensor_parallel_size=cfg.tensor_parallel_size, max_lora_rank=cfg.max_lora_rank,
        enable_lora=cfg.enable_lora, max_model_len=cfg.max_model_len, max_num_seqs=cfg.max_num_seqs,
        gpu_memory_utilization_pct=int(round(cfg.gpu_memory_utilization * 100)), mixed_moe_lora=cfg.mixed_moe_lora)
    call_kwargs = {"messages_list": part, "lora_path": lora_path, "lora_id": mv._lora_id_for(lora_path) if lora_path else 1,
                   "max_tokens": cfg.max_tokens, "temperature": cfg.temperature, "top_p": cfg.top_p,
                   "num_samples": cfg.num_samples, "stop": cfg.stop, "return_logprobs": False,
                   "add_generation_prompt": True, "continue_final_message": False,
                   "chat_template_kwargs": cfg.chat_template_kwargs, "seed": cfg.seed}
    for attempt in range(6):  # the spawn (input upload) itself
        try:
            fc = await engine.generate.spawn.aio(**call_kwargs)
            break
        except modal.exception.ConnectionError as e:
            if attempt == 5:
                raise
            print(f"[modal] spawn failed ({str(e)[:120]}); retry {attempt + 1}/5", flush=True)
            await asyncio.sleep(15 * (attempt + 1))
    t0 = time.time()
    while True:
        try:
            outputs = await fc.get.aio(timeout=poll_timeout)
            break
        except (modal.exception.TimeoutError, TimeoutError):
            if time.time() - t0 > max_poll_minutes * 60:
                raise RuntimeError(f"shard result not ready after {max_poll_minutes} min (call {fc.object_id})")
        except modal.exception.ConnectionError as e:  # poll again; the GPU result is kept server-side
            print(f"[modal] poll of {fc.object_id} failed ({str(e)[:120]}); polling again", flush=True)
            await asyncio.sleep(15)
    model_id = f"{base_model}::{lora_path or ''}"
    return [mv._to_canonical(msgs, model_id, [x["text"] for x in samples], [x["finish_reason"] for x in samples])
            for msgs, samples in zip(part, outputs)]


def make_generate_fn(mcfg: dict, budget: GpuBudget | None = None, lora_path: str | None = None,
                     chat_template_kwargs: dict | None = None, shard: int | None = None,
                     api_sem: asyncio.Semaphore | None = None, hard_cap: int | None = None,
                     cache_dir: Path | None = None):
    """-> (async generate_fn(messages_list, seed=None, tag=None) -> canonical results, backend_info dict)."""
    cache = ShardCache(cache_dir, f"{mcfg['model']}::{lora_path or ''}")
    if mcfg["backend"] == "openrouter":
        from cotcontrol.inference.openrouter import generate_async
        cfg = openrouter_config(mcfg)
        # Process-wide in-flight limit shared by every arm/round: the pinned
        # provider saturates near its pinned concurrency, and six arms each
        # at that concurrency only buy 429s (observed 2-3% of generations
        # exhausting 8 retries at ~1200 in flight on Groq). Requests go out in
        # chunks of the pinned concurrency, at most `api_slots` chunks at once.
        chunk = cfg.max_concurrency
        sem = api_sem or asyncio.Semaphore(CFG["api_slots"])

        async def one(part, tag):
            async def call():
                async with sem:
                    return await generate_async(part, mcfg["model"], cfg, progress=False)
            return await cached_call(cache, tag, part, call)

        async def fn(messages_list, seed: int | None = None, tag: str | None = None):  # API sampling is already independent
            parts = [messages_list[i:i + chunk] for i in range(0, len(messages_list), chunk)]
            outs = await asyncio.gather(*[one(p, tag and f"{tag}|{i}") for i, p in enumerate(parts)])
            return [r for o in outs for r in o]
        return fn, {"backend": "openrouter", "model": mcfg["model"], "generate_config": asdict(cfg),
                    "api_slots": CFG["api_slots"], "chunk": chunk}

    from cotcontrol.inference.modal_vllm import ModalGenerateConfig
    s, m = CFG["sampling"], CFG["modal"]
    cfg = ModalGenerateConfig(
        temperature=s["temperature"], top_p=s["top_p"], max_tokens=s["max_tokens"],
        chat_template_kwargs=chat_template_kwargs if chat_template_kwargs is not None else mcfg.get("chat_template_kwargs"),
        gpu=mcfg.get("gpu", "H200"), max_model_len=mcfg.get("max_model_len", 32768),
        max_num_seqs=m["max_num_seqs"], cache=False, assume_app_running=True,
    )
    shard = shard or mcfg.get("shard") or m["shard"]
    budget = budget or GpuBudget(m["gpu_cap"], None, m["gpu_cap_after_hours"])
    patch_engine_pool(hard_cap or budget.cap)

    async def one(part, seed, tag):
        # vLLM seeds its engine RNG deterministically (seed=0), so two fresh
        # containers given the same batch would sample identical text. A
        # per-request seed (distinct per shard, and per arm/round via `seed`)
        # makes every call an independent draw; identical prompts within one
        # shard never occur (one attempt per item per round).
        async def call():
            for attempt in range(4):  # transient Modal/vLLM failures (container death, timeouts) -> re-run the shard
                try:
                    async with budget.slot():
                        return await spawn_generate(part, mcfg["model"], replace(cfg, seed=seed), lora_path)
                except Exception as e:
                    print(f"[modal] shard of {len(part)} failed ({type(e).__name__}: {str(e)[:300]}); "
                          f"retry {attempt + 1}/3" if attempt < 3 else f"[modal] shard failed 4x: {type(e).__name__}: {str(e)[:300]}",
                          flush=True)
                    if attempt == 3:
                        raise
                    await asyncio.sleep(30 * (attempt + 1))
        return await cached_call(cache, tag, part, call)

    max_chars = m.get("shard_chars", 1_500_000)

    def split(messages_list):
        """Shards of <= `shard` prompts and <= shard_chars of prompt text: Modal
        calls moving multi-MB payloads from the laptop (40 at once) hit the
        client's gRPC deadline ("ConnectionError: Deadline exceeded")."""
        parts, cur, cur_chars = [], [], 0
        for msgs in messages_list:
            n = sum(len(x["content"]) for x in msgs)
            if cur and (len(cur) >= shard or cur_chars + n > max_chars):
                parts.append(cur)
                cur, cur_chars = [], 0
            cur.append(msgs)
            cur_chars += n
        if cur:
            parts.append(cur)
        return parts

    async def fn(messages_list, seed: int | None = None, tag: str | None = None):
        rng = random.Random(seed)  # None -> fresh randomness (e.g. transfer draws)
        parts = split(messages_list)
        outs = await asyncio.gather(*[one(p, rng.getrandbits(31), tag and f"{tag}|{i}") for i, p in enumerate(parts)])
        return [r for o in outs for r in o]
    return fn, {"backend": "modal-vllm", "model": mcfg["model"], "lora_path": lora_path,
                "generate_config": asdict(cfg), "shard": shard, "per_shard_seeds": True}


@contextlib.asynccontextmanager
async def modal_app_context(needed: bool, record: Path | None = None):
    """`async with modal_app_context(any_modal_model, run_dir / "modal_app_id.txt"):`
    — one app.run() for the whole driver so every engine call shares the warm
    pool. Records the ephemeral app id so `retry gpus` can count OUR containers
    (every agent shares the cotcontrol-inference app name)."""
    if not needed:
        yield
        return
    if deployed_mode():  # calls go to the deployed app; nothing to run here
        if record is not None:
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(f"{os.environ['COTCONTROL_MODAL_INFERENCE_APP']}\n")
        yield
        return
    import modal

    from cotcontrol.inference.modal_vllm import app
    # The workspace caps ephemeral apps at 100 (shared by every agent); wait
    # for a slot instead of dying at startup. One app per driver process.
    for attempt in range(240):
        ctx = app.run()
        try:
            await ctx.__aenter__()
            break
        except modal.exception.ResourceExhaustedError as e:
            print(f"[modal] app creation refused ({e}); retry {attempt + 1}/240 in 60s", flush=True)
            await asyncio.sleep(60)
    else:
        raise RuntimeError("could not create a Modal app in 4 hours (ephemeral app limit)")
    try:
        if record is not None:
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(f"{app.app_id}\n")
        yield
    finally:
        await ctx.__aexit__(None, None, None)


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------

def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(path: Path, **fields) -> dict:
    man = {"timestamp": datetime.now().isoformat(), "git_commit": git_commit(),
           "dataset_hash": dataset_hash(), **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=1, default=str))
    return man


def active_models(labels: str | None) -> list[str]:
    if labels:
        return labels.split(",")
    return [k for k, v in MODELS.items() if not v.get("disabled")]
