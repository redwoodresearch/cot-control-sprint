"""Helpers shared by the noise jobs (kl, make, replay, analyze, watch, smoke): source-run
configs, the KL engine, the base model's samples, KL records, adapter builds, replay-block
selection and group lookup. Every setting comes from the resolved config (config.yaml);
nothing here is a hyperparameter. Loaded by path (`load_py`), like the job files.
"""

# ruff: noqa: DTZ005 — naive local timestamps, like the rest of the repo
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import statistics
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from kaushik.core.config import deep_merge, model_key_of, run_dir_of
from kaushik.core.io import KAUSHIK, ROOT, atomic_write_json, load_py, read_json, rel
from kaushik.core.jobs import te

HERE = KAUSHIK / "noise"
SWEEPS = HERE / "sweeps"
ENV_RANDOM_LORA_APP = "COTCONTROL_MODAL_RANDOM_LORA_APP"
KL_SETTING_KEYS = ("split", "dataset", "n_prompts", "max_tokens", "seed")
SETTINGS = ("kl", "noise", "replay", "groups", "analysis", "retries", "watch", "modal", "runs_dir")


# ----------------------------------------------------------------------------- small things


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(obj, path: Path) -> Path:
    return atomic_write_json(obj, path, indent=1)


def runs_dir(cfg: dict) -> Path:
    return ROOT / cfg["runs_dir"]


def group_dir(cfg: dict) -> Path:
    return run_dir_of(cfg)


def model_slug(base_model: str) -> str:
    """'openai/gpt-oss-20b' -> 'gptoss20b', 'Qwen/Qwen3.6-27B' -> 'qwen3627b' (group / samples naming)."""
    return re.sub(r"[^a-z0-9]", "", base_model.split("/")[-1].lower())


def model_rank(cfg: dict, base_model: str) -> int:
    """0 for the models listed in groups.priority_models (scheduled first), else 1."""
    return 0 if model_key_of(base_model) in (cfg["groups"].get("priority_models") or []) else 1


def final_step(adapter_path: str) -> int | None:
    m = re.search(r"checkpoint-(\d+)/?$", adapter_path)
    return int(m.group(1)) if m else None


def random_lora(cfg: dict):
    """cotcontrol.training.random_lora with the deployed CPU build app selected (read at import)."""
    os.environ.setdefault(ENV_RANDOM_LORA_APP, cfg["modal"]["random_lora_app"])
    from cotcontrol.training import random_lora as rl

    return rl


# ----------------------------------------------------------------------------- samples + KL


def kl_settings(cfg: dict) -> dict:
    """The five settings that identify a set of base samples (recorded next to them)."""
    k = cfg["kl"]
    return {"split": k["split"], "dataset": k["dataset"], "n_prompts": int(k["n_prompts"]),
            "max_tokens": int(k["max_tokens"]), "seed": int(k["seed"])}


def samples_tag(base_model: str, template_kwargs) -> str:
    tag = model_slug(base_model)
    if template_kwargs:
        tag += "-" + hashlib.sha256(json.dumps(template_kwargs, sort_keys=True).encode()).hexdigest()[:6]
    return tag


def samples_dir(cfg: dict, base_model: str, template_kwargs) -> Path:
    return ROOT / cfg["kl"]["samples_dir"] / samples_tag(base_model, template_kwargs)


def kl_record_path(cfg: dict, name: str) -> Path:
    return ROOT / cfg["kl"]["records_dir"] / f"{name}.json"


def kl_record(cfg: dict, name: str) -> dict | None:
    return read_json(kl_record_path(cfg, name))


def source_config(config_path: str | None):
    """(TrainConfig, [(block, fully-defaulted block)], raw yaml) of a source run, or None."""
    p = ROOT / config_path if config_path else None
    if p is None or not p.is_file():
        return None
    t = te()
    tc, ec, raw = t.load_config(p)
    return tc, t.eval_blocks(ec), raw


def kl_engine(cfg: dict, src, template_kwargs):
    """Engine params of the source run (gpu, tp, lora rank, model len; its first eval block)
    plus the KL sampling settings. Same resource fields as the evals -> warm engines shared."""
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig

    t, k = te(), cfg["kl"]
    if src:
        tc, blocks, _ = src
        gen = t.build_generate_config(tc, blocks[0][1] if blocks else dict(t.DEFAULT_EVAL))
    else:
        gen = ModalGenerateConfig(gpu=k["gpu"])
    return replace(gen, temperature=float(k["temperature"]), top_p=float(k["top_p"]), max_tokens=int(k["max_tokens"]),
                   num_samples=1, seed=int(k["seed"]), return_logprobs=True, cache=True, stop=None,
                   chat_template_kwargs=template_kwargs or None)


@asynccontextmanager
async def file_lock(path: Path):
    """Cross-process exclusive lock (flock): two jobs never sample the same base twice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


async def retry(cfg: dict, fn, what: str):
    """Re-run an idempotent remote call after transient Modal/gRPC failures (generation is
    disk-cached per chunk, so a retry only redoes lost work)."""
    r = cfg["retries"]
    attempts = int(r["attempts"])
    for i in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as e:
            if i == attempts:
                raise
            wait = min(float(r["delay_s"]) * 2 ** (i - 1), float(r["max_delay_s"]))
            print(f"[retry] {what}: {type(e).__name__}: {str(e)[:120]} -- retry {i}/{attempts - 1} in {wait:.0f}s",
                  flush=True)
            await asyncio.sleep(wait)


async def build_adapter(cfg: dict, src: str, dst: str, scale: float, seed: int, family: str) -> dict:
    rl = random_lora(cfg)
    return await retry(cfg, lambda: rl.remote_fn("build_adapter").remote.aio(src, dst, scale, seed, family),
                       f"build {dst}")


async def ensure_samples(cfg: dict, base_model: str, gen) -> tuple[dict, list]:
    """The base model's fixed T=1 samples + its own teacher-forced log-probs (one set per
    base model x template kwargs, shared by every adapter of that base)."""
    from cotcontrol.inference import modal_vllm

    rl, ks = random_lora(cfg), kl_settings(cfg)
    d = samples_dir(cfg, base_model, gen.chat_template_kwargs)
    sp, bp = d / "samples.json", d / "base_logprobs.json"
    async with file_lock(d / ".lock"):
        if not sp.exists():
            prompts = rl.unconstrained_prompts(base_model, ks["split"], ks["dataset"], ks["n_prompts"])
            print(f"[samples] sampling {len(prompts)} {ks['split']} prompts from {base_model} "
                  f"(T={gen.temperature}, max_tokens={gen.max_tokens}, kwargs={gen.chat_template_kwargs})", flush=True)
            samples = await rl.sample_base(prompts, base_model, gen)
            samples["kl_settings"] = ks
            write_json(samples, sp)
            n_tok = sum(len(t) - n for t, n in zip(samples["token_ids"], samples["prompt_lens"]))
            print(f"[samples] {len(samples['token_ids'])} sequences, {n_tok} sampled tokens -> {rel(sp)}", flush=True)
        samples = read_json(sp)
        if not bp.exists():
            async with modal_vllm.app.run():
                lps = await rl.score_samples(samples, base_model, gen, None)
            self_kl = rl.kl_from_logprobs(lps, lps, n_boot=int(cfg["kl"]["bootstrap_draws"]))["kl_per_token"]
            drift = statistics.fmean(abs(a - b) for s, lp in zip(samples["sampled_logprobs"], lps) for a, b in zip(s, lp))
            write_json({"logprobs": lps, "engine": asdict(gen), "self_kl": self_kl,
                        "mean_abs_diff_vs_sampled_logprobs": drift}, bp)
            print(f"[samples] base rescored; mean |rescored - sampled| logprob = {drift:.4f}", flush=True)
    return samples, read_json(bp)["logprobs"]


async def measure_kl(cfg: dict, adapter_path: str | None, base_model: str, gen, samples: dict, base_lps: list) -> dict:
    from cotcontrol.inference import modal_vllm

    rl, k = random_lora(cfg), cfg["kl"]
    async with modal_vllm.app.run():
        lps = await retry(cfg, lambda: rl.score_samples(samples, base_model, gen, adapter_path), f"score {adapter_path}")
    return {"kl": rl.kl_from_logprobs(base_lps, lps, n_boot=int(k["bootstrap_draws"]), seed=int(k["bootstrap_seed"])),
            "adapter_logprobs": lps}


def write_kl_record(cfg: dict, name: str, base_model: str, adapter_path: str, res: dict, gen,
                    extra: dict | None = None) -> Path:
    out = kl_record_path(cfg, name)
    write_json({"name": name, "base_model": base_model, "adapter_path": adapter_path, "kl": res["kl"],
                "engine": asdict(gen), "samples": rel(samples_dir(cfg, base_model, gen.chat_template_kwargs)),
                "timestamp": now(), **(extra or {}), "adapter_logprobs": res["adapter_logprobs"]}, out)
    return out


# ----------------------------------------------------------------------------- replay blocks


def resolve_blocks(cfg: dict, entry: dict, src) -> list[tuple[str, dict]]:
    """Which eval blocks the noise adapters replicate. The selector is `replay.blocks` (config
    override) else the entry's `noise_eval_blocks`: a list of block names, "<config>.yaml#<sel>"
    or a bare selector; a selector is a top-level yaml key listing blocks (or a whole eval
    section), a block name, or a substring of block names ("final"). Default: the blocks that
    run at the adapter's checkpoint. Every chosen block gets `replay.patch`."""
    t = te()
    _, blocks, raw = src
    sel = cfg["replay"]["blocks"] if cfg["replay"]["blocks"] is not None else entry.get("noise_eval_blocks")
    if isinstance(sel, str) and "#" in sel:
        path, sel = sel.split("#", 1)
        if path and path != entry["config_path"] and (ROOT / path).exists():
            _, ec, raw = t.load_config(ROOT / path)
            blocks = t.eval_blocks(ec)
    elif isinstance(sel, str) and sel.endswith(".yaml"):
        sel = None
    names = dict(blocks)
    step = final_step(entry["adapter_path"])
    at_final = [n for n, ec in blocks
                if ec.get("eval_at_steps") is None or (step is not None and step in {int(s) for s in ec["eval_at_steps"]})]
    if isinstance(sel, list):
        chosen = list(sel)
    elif sel:
        if isinstance(raw.get(sel), dict):   # a whole eval section, e.g. orange/rl's `final_eval:`
            names = dict(t.eval_blocks(raw[sel]))
            chosen = list(names)
        elif isinstance(raw.get(sel), list):
            chosen = [str(x) for x in raw[sel]]
        elif sel in names:
            chosen = [sel]
        else:   # substring ("final"); cpt names its final-checkpoint blocks test_*
            chosen = ([n for n in names if sel in n] or [n for n in at_final if n.startswith("test")] or at_final)
    else:
        chosen = at_final
    missing = [c for c in chosen if c not in names]
    if missing or not chosen:
        raise ValueError(f"{entry['run_name']}: eval blocks {chosen} not resolvable (missing {missing}; have {sorted(names)})")
    return [(n, {**names[n], **cfg["replay"]["patch"]}) for n in chosen]


# ----------------------------------------------------------------------------- group lookup


def make_groups_module():
    return load_py(HERE / "make_groups.py")


def _reformat(obj, fields: dict):
    """Re-format `{run_name}` / `{group}` templates that the caller's bare config carried
    over unresolved (its run_name is the literal template)."""
    if isinstance(obj, dict):
        return {k: _reformat(v, fields) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_reformat(v, fields) for v in obj]
    if isinstance(obj, str) and "{" in obj:
        try:
            return obj.format(**fields)
        except (KeyError, IndexError, ValueError):
            return obj
    return obj


def _with_settings(point_cfg: dict, cfg: dict, keep_kl: tuple[str, ...] = ()) -> dict:
    """A sweep point's config with the caller's settings sections on top (so `key=value`
    overrides given to a job apply to the groups it looks up); point-specific `kl` keys stay."""
    over = {k: cfg[k] for k in SETTINGS if k in cfg}
    if "kl" in over:
        over["kl"] = {k: v for k, v in over["kl"].items() if k not in keep_kl}
    name = point_cfg["run_name"]
    return deep_merge(point_cfg, _reformat(over, {"run_name": name, "group": name}))


def _sweep_points(cfg: dict, name: str) -> list[dict]:
    from kaushik.core.sweep import load_sweep, resolve_sweep

    path = SWEEPS / f"{name}.yaml"
    if not path.exists():
        make_groups_module().write_sweeps(cfg)
    return [c for _, c in resolve_sweep(load_sweep(path))]


def group_points(cfg: dict) -> list[dict]:
    """Resolved config of every group of sweeps/groups.yaml (generated when missing)."""
    return [_with_settings(c, cfg) for c in _sweep_points(cfg, "groups")]


def group_point(cfg: dict, name: str) -> dict:
    if cfg.get("group", {}).get("entries") and cfg.get("run_name") == name:
        return cfg
    for c in group_points(cfg):
        if c["run_name"] == name:
            return c
    raise SystemExit(f"no group {name!r} in {rel(SWEEPS / 'groups.yaml')} (run `noise make_groups`)")


def kl_points(cfg: dict) -> list[dict]:
    """Resolved config of every point of sweeps/kl.yaml (one per base model x template kwargs)."""
    return [_with_settings(c, cfg, keep_kl=("base_model", "chat_template_kwargs", "engine_config", "entries"))
            for c in _sweep_points(cfg, "kl")]
