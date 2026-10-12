"""Shared pieces for kaushik/retry: item construction, generation backends (pinned
OpenRouter; Modal vLLM in shards) and run manifests, all read from the resolved config
(config.yaml). GPU capacity is the core scheduler's business: a job's weight bounds the
engine calls it keeps in flight (`shard_slots`), the scheduler bounds the sum.
Import Modal-touching code lazily: kaushik/run.py sets the Modal env BEFORE
cotcontrol.inference.modal_vllm is imported."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from kaushik.core.config import model_spec
from kaushik.core.io import ROOT, git_commit

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def log_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"  # noqa: DTZ005 - local time, like the rest of the repo
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")
    return log


# ----------------------------------------------------------------------------- config views


def serving(cfg: dict) -> dict:
    """The model's serving entry (config.yaml:models[model])."""
    try:
        return cfg["models"][cfg["model"]]
    except KeyError as e:
        raise SystemExit(f"model {cfg.get('model')!r} has no entry in config.yaml:models ({sorted(cfg['models'])})") from e


def is_modal(cfg: dict) -> bool:
    return serving(cfg)["backend"] == "modal"


def base_models(cfg: dict) -> list[str]:
    """Model keys that run as base-model 3A arms (not checkpoints_only)."""
    return [k for k, m in cfg["models"].items() if not m.get("checkpoints_only")]


def modal_label_of(cfg: dict, base_model: str) -> str | None:
    """The Modal-served model key of a registry entry's base model (None for API-only models)."""
    for k, m in cfg["models"].items():
        if m["backend"] == "modal" and model_spec(k)["base_model"].lower() == base_model.lower():
            return k
    return None


def item_spec(cfg: dict) -> dict:
    return dict(cfg["item_specs"][cfg["items"]])


def run_name_of(cfg: dict) -> str:
    """Sweeps set run_name; a CLI job defaults to <run>/<model>/<arm>."""
    if cfg.get("run_name") and "{" not in cfg["run_name"]:
        return cfg["run_name"]
    return "/".join(str(cfg[k]) for k in ("run", "model", "arm") if cfg.get(k))


def run_dir_for(cfg: dict, name: str | None = None) -> Path:
    return ROOT / cfg["run_dir"].format(run_name=name or run_name_of(cfg))


def shard_size(cfg: dict) -> int:
    return int(serving(cfg).get("shard") or cfg["modal"]["shard"])


def tensor_parallel(cfg: dict) -> int:
    return int(serving(cfg).get("tensor_parallel_size") or model_spec(cfg["model"])["eval"]["tensor_parallel_size"])


def gpu_weight(cfg: dict, n_items: int) -> int:
    """GPUs a Modal job holds: one engine call per shard of the first round, capped (0 for API models)."""
    if not is_modal(cfg):
        return 0
    return min(math.ceil(n_items / shard_size(cfg)), int(cfg["modal"]["max_gpus_per_job"])) * tensor_parallel(cfg)


# ----------------------------------------------------------------------------- items


def load_items(model_id: str, split: str, modes_per_question: int = 9, seed: int = 0,
               max_questions: int | None = None, subsample_seed: int | None = None,
               modes: list[str] | None = None, subset: str | None = None) -> list[dict]:
    """Test/train split x constraint modes -> retry items. modes_per_question < 9
    keeps a seeded random subset of each question's valid modes (keyword modes
    only where the question has keywords, as the standard eval assigns them).
    subset: path (relative to kaushik/retry) of a JSON {"item_ids": [...]} that
    restricts the items (same ids for every model and arm, so comparisons stay paired)."""
    from cotcontrol.eval.data import assign_tasks, load_dataset
    from cotcontrol.eval.retry import make_items

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
         "created": datetime.now().isoformat(),  # noqa: DTZ005
         "note": "User decision 2026-10-08 13:05: 3A main uses this 1000-item subset for every arm and model; "
                 "earlier records of other items stay on disk but are excluded from analysis."}
    if out:
        out.write_text(json.dumps(d, indent=1))
    return d


def dataset_hash() -> str:
    from cotcontrol.eval.data import DATASETS_DIR

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


# ----------------------------------------------------------------------------- backends


def deployed_mode() -> bool:
    """True when kaushik/run.py pointed cotcontrol at the deployed inference app."""
    return bool(os.environ.get("COTCONTROL_MODAL_INFERENCE_APP"))


def openrouter_config(cfg: dict, concurrency: int | None = None):
    from cotcontrol.inference.openrouter import GenerateConfig

    pins = json.loads((ROOT / cfg["openrouter"]["pins"]).read_text())[cfg["base_model"]]
    s = cfg["sampling"]
    return GenerateConfig(
        temperature=s["temperature"], top_p=s["top_p"], max_tokens=s["max_tokens"],
        max_concurrency=concurrency or pins["concurrency"], max_retries=pins["max_retries"],
        provider={"only": [pins["provider"]], "allow_fallbacks": False,
                  **({"quantizations": pins["quantizations"]} if pins.get("quantizations") else {})},
        extra_body={"reasoning": {"effort": pins["reasoning_effort"]}} if pins.get("reasoning_effort") else None,
    )


def modal_config(cfg: dict, chat_template_kwargs: dict | None | str = "default"):
    """ModalGenerateConfig: sampling from config.yaml, engine from the model table + the serving entry."""
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig

    s, m, sv = cfg["sampling"], cfg["modal"], serving(cfg)
    if chat_template_kwargs == "default":
        chat_template_kwargs = sv["chat_template_kwargs"] if "chat_template_kwargs" in sv \
            else model_spec(cfg["model"])["eval"]["chat_template_kwargs"]
    tp = tensor_parallel(cfg)
    gpu_type = str(cfg.get("gpu") or "H200").split(":")[0]
    return ModalGenerateConfig(
        temperature=s["temperature"], top_p=s["top_p"], max_tokens=s["max_tokens"],
        chat_template_kwargs=chat_template_kwargs, gpu=gpu_type if tp == 1 else f"{gpu_type}:{tp}",
        tensor_parallel_size=tp, max_model_len=int(sv.get("max_model_len") or m["max_model_len"]),
        max_num_seqs=m["max_num_seqs"], cache=False, assume_app_running=True)


def judge_config(cfg: dict):
    from cotcontrol.inference.openrouter import GenerateConfig

    j = cfg["judge"]
    return GenerateConfig(temperature=j["temperature"], max_tokens=j["max_tokens"], max_concurrency=j["concurrency"],
                          max_retries=j["max_retries"])


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
        if self.dir is None:
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


_API_SLOTS: dict[str, asyncio.Semaphore] = {}   # process-wide per model: every arm/round shares them


def make_generate_fn(cfg: dict, shard_slots: int, lora_path: str | None = None,
                     chat_template_kwargs: dict | None | str = "default", api_sem: asyncio.Semaphore | None = None,
                     cache_dir: Path | None = None):
    """-> (async generate_fn(messages_list, seed=None, tag=None) -> canonical results, backend_info dict).
    shard_slots: engine calls this job keeps in flight (= its scheduler weight)."""
    model_id = cfg["base_model"]
    cache = ShardCache(cache_dir, f"{model_id}::{lora_path or ''}")
    if not is_modal(cfg):
        from cotcontrol.inference.openrouter import generate_async

        gc = openrouter_config(cfg)
        # Process-wide in-flight limit shared by every arm/round: the pinned provider saturates near its
        # pinned concurrency, and six arms each at that concurrency only buy 429s (2-3% of generations
        # exhausting 8 retries at ~1200 in flight on Groq). Requests go out in chunks of the pinned
        # concurrency, at most `api_slots` chunks at once.
        chunk = gc.max_concurrency
        sem = api_sem or _API_SLOTS.setdefault(model_id, asyncio.Semaphore(int(cfg["openrouter"]["api_slots"])))

        async def one(part, tag):
            async def call():
                async with sem:
                    return await generate_async(part, model_id, gc, progress=False)
            return await cached_call(cache, tag, part, call)

        async def fn(messages_list, seed: int | None = None, tag: str | None = None):  # API sampling is already independent
            parts = [messages_list[i:i + chunk] for i in range(0, len(messages_list), chunk)]
            outs = await asyncio.gather(*[one(p, tag and f"{tag}|{i}") for i, p in enumerate(parts)])
            return [r for o in outs for r in o]
        return fn, {"backend": "openrouter", "model": model_id, "generate_config": asdict(gc),
                    "api_slots": int(cfg["openrouter"]["api_slots"]), "chunk": chunk}

    from cotcontrol.inference.modal_vllm import generate_async

    gc = modal_config(cfg, chat_template_kwargs)
    shard, max_chars = shard_size(cfg), int(cfg["modal"]["shard_chars"])
    slots = asyncio.Semaphore(max(1, shard_slots))

    async def one(part, seed, tag):
        # vLLM seeds its engine RNG deterministically, so two fresh containers given the same batch would
        # sample identical text. A per-request seed (distinct per shard, and per arm/round via `seed`)
        # makes every call an independent draw. modal_vllm spawns the call and re-attaches by id on
        # client-side connection drops, so a dropped stream never re-submits the shard to another GPU.
        async def call():
            for attempt in range(4):  # transient Modal/vLLM failures (container death, timeouts) -> re-run the shard
                try:
                    async with slots:
                        return await generate_async(part, model_id, replace(gc, seed=seed), lora_path)
                except Exception as e:  # retried, then re-raised below
                    print(f"[modal] shard of {len(part)} failed ({type(e).__name__}: {str(e)[:300]}); "
                          + (f"retry {attempt + 1}/3" if attempt < 3 else "giving up"), flush=True)
                    if attempt == 3:
                        raise
                    await asyncio.sleep(30 * (attempt + 1))
            return None
        return await cached_call(cache, tag, part, call)

    def split(messages_list):
        """Shards of <= `shard` prompts and <= shard_chars of prompt text: Modal calls moving multi-MB
        payloads from the laptop hit the client's gRPC deadline."""
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
    return fn, {"backend": "modal-vllm", "model": model_id, "lora_path": lora_path,
                "generate_config": asdict(gc), "shard": shard, "per_shard_seeds": True}


# ----------------------------------------------------------------------------- manifests


def write_manifest(path: Path, **fields) -> dict:
    man = {"timestamp": datetime.now().isoformat(), "git_commit": git_commit(),  # noqa: DTZ005
           "dataset_hash": dataset_hash(), **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=1, default=str))
    return man
