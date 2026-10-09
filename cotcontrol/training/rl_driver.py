"""Pluggable prompt source + reward for GRPO: the general rollout side of RL.

rl/run_grpo.py hard-wires rollouts to eval_cotcontrolqa (CoT-Control-QA
prompts, compliance rewards). The worker (rl_worker.py) never cared: it
consumes groups of {prompt_token_ids, samples: [{token_ids, token_logprobs,
reward, truncated}]}. This module is the task-agnostic driver that produces
those batches from

  * a PromptSource — anything with .sample(iteration, n, seed) -> [Prompt],
    where a Prompt is chat messages + metadata (the reward's ground truth);
  * a reward_fn(sample, prompt) -> float | {"reward": float, ...extras} over
    the rollout TEXT (output / reasoning / finish_reason); extras are merged
    into the saved sample record (e.g. an extracted answer) for auditing.

It speaks the same volume protocol as rl/run_grpo.py (adapter wait, batch
upload with layer-pressure retries, resume from committed optimizer state,
respawn after a budget-stop) so the two drivers are interchangeable to the
worker; rl/run_grpo.py itself is untouched and keeps its own loop.

Per iteration t everything except raw token arrays is saved under run_dir:
iters/iter-<t>.json (every rollout: messages, meta, output, reasoning, reward,
extras), batches/batch-<t>.json (the rewards the worker trained on),
progress.jsonl (aggregates + degeneracy tripwires + optional held-out reward).

    from cotcontrol.training.rl_driver import ListPromptSource, Prompt, run_rl
    source = ListPromptSource([Prompt(key, messages, meta), ...])
    result = run_rl(cfg, gen_cfg, run_dir, source, reward_fn,
                    heldout=ListPromptSource(...), heldout_at=[0, 10, 20])
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from cotcontrol.eval.grading import _zlib_ratio
from cotcontrol.training.config import RLConfig


@dataclass
class Prompt:
    """One RL prompt: a stable key (group id), chat messages, and whatever the
    reward needs to score a rollout (target letter, options, ...)."""

    key: str
    messages: list[dict]
    meta: dict = field(default_factory=dict)


class PromptSource(Protocol):
    def sample(self, iteration: int, n: int, seed: int) -> list[Prompt]: ...


class ListPromptSource:
    """Epoch-style cycling over a fixed list, like an SFT dataloader: a seeded
    permutation per epoch, iteration t takes items [t*n, (t+1)*n) of the
    concatenated permutations. Stateless in (seed, iteration, n), so a resumed
    run sees exactly the schedule it would have seen."""

    def __init__(self, prompts: Sequence[Prompt]):
        if not prompts:
            raise ValueError("empty prompt list")
        self.prompts = list(prompts)

    def _perm(self, epoch: int, seed: int) -> list[int]:
        idx = list(range(len(self.prompts)))
        random.Random(seed * 100_003 + epoch).shuffle(idx)
        return idx

    def sample(self, iteration: int, n: int, seed: int) -> list[Prompt]:
        N, start, out = len(self.prompts), iteration * n, []
        while len(out) < n:
            pos = start + len(out)
            perm = self._perm(pos // N, seed)
            out.extend(self.prompts[i] for i in perm[pos % N: pos % N + (n - len(out))])
        return out


RewardFn = Callable[[dict, Prompt], "float | dict"]


def _degeneracy_stats(samples: list[dict]) -> dict:
    """Tripwires (logged, never rewarded): trace deletion shows up in
    reasoning_chars_median / frac_short_reasoning, filler in distinct4_mean."""
    lens, d4, zr = [], [], []
    for s in samples:
        if s.get("error"):
            continue
        r = s.get("reasoning") or ""
        lens.append(len(r))
        w = r.split()
        grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        d4.append(len(set(grams)) / len(grams) if grams else 1.0)
        zr.append(_zlib_ratio(r))
    if not lens:
        return {}
    lens.sort()
    zr.sort()
    return {
        "reasoning_chars_median": lens[len(lens) // 2],
        "frac_short_reasoning": round(sum(1 for n in lens if n < 200) / len(lens), 3),
        "distinct4_mean": round(sum(d4) / len(d4), 3),
        "zlib_ratio_median": round(zr[len(zr) // 2], 3),
    }


def summarize(records: list[dict]) -> dict:
    samples = [s for r in records for s in r["samples"]]
    ok = [s for s in samples if not s.get("error")]
    rewards = [s["reward"] for s in ok]
    return {
        "n_prompts": len(records),
        "n_rollouts": len(samples),
        "n_errors": len(samples) - len(ok),
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "frac_truncated": round(sum(1 for s in ok if s["truncated"]) / len(ok), 3) if ok else None,
        "frac_reward_positive": round(sum(1 for r in rewards if r > 0) / len(rewards), 3) if rewards else None,
        "output_chars_mean": round(sum(len(s.get("output") or "") for s in ok) / len(ok), 1) if ok else None,
        **_degeneracy_stats(ok),
    }


def strip_tokens(record: dict) -> dict:
    """Drop token arrays (the bulk of the JSON); keep n_tokens as a length record."""
    rec = {k: v for k, v in record.items() if k != "prompt_token_ids"}
    if "prompt_token_ids" in record:
        rec["n_prompt_tokens"] = len(record["prompt_token_ids"])
    rec["samples"] = [
        {**{k: v for k, v in s.items() if k not in ("token_ids", "token_logprobs")},
         **({"n_tokens": len(s["token_ids"])} if "token_ids" in s else {})}
        for s in record["samples"]
    ]
    return rec


def response_only(raw: str | None) -> str:
    """The completion as the response channel: for a prompt that already holds
    an empty reasoning block (Qwen enable_thinking=False), the model emits no
    </think>, and split_reasoning would otherwise file the whole answer under
    'reasoning' with an empty output."""
    from cotcontrol.inference.types import END_TOKENS

    text = raw or ""
    for tok in END_TOKENS:
        idx = text.find(tok)
        if idx != -1:
            text = text[:idx]
    return text.strip()


async def rollout(
    prompts: list[Prompt],
    generate_fn: Callable,
    reward_fn: RewardFn,
    save_path: Path | None = None,
    backend_info: dict | None = None,
    reasoning_disabled: bool = False,
) -> dict:
    """Generate (num_samples per prompt, set by generate_fn's config), score
    every sample with reward_fn, return {"summary", "results"}. Errored
    samples get reward 0 and are dropped from the batch by build_batch.
    results[i]["samples"][j] keeps token_ids/token_logprobs when the backend
    returned them (Modal vLLM with return_logprobs=True); the saved copy strips
    them. reasoning_disabled: the prompt conditions on an empty reasoning
    block, so the whole completion is the response (see response_only)."""
    gens = await generate_fn([p.messages for p in prompts])
    records = []
    for p, gen in zip(prompts, gens):
        samples = []
        for j in range(len(gen["output"])):
            meta = gen["metadata"][j]
            s = {
                "output": gen["output"][j],
                "reasoning": gen["reasoning"][j],
                "finish_reason": meta["finish_reason"],
                "truncated": meta["finish_reason"] == "length",
                "error": meta["error"],
                "raw_response": meta["raw_response"],
            }
            if reasoning_disabled and not meta["error"]:
                s["output"], s["reasoning"] = response_only(meta["raw_response"]), ""
            if meta["error"]:
                s["reward"] = 0.0
            else:
                r = reward_fn(s, p)
                if isinstance(r, dict):
                    s.update({k: v for k, v in r.items() if k != "reward"})
                    r = r["reward"]
                s["reward"] = float(r)
            if "token_ids" in gen:
                s["token_ids"] = gen["token_ids"][j]
                s["token_logprobs"] = gen["token_logprobs"][j]
            samples.append(s)
        rec = {"key": p.key, "messages": p.messages, "meta": p.meta, "samples": samples}
        if "prompt_token_ids" in gen:
            rec["prompt_token_ids"] = gen["prompt_token_ids"]
        records.append(rec)
    result = {"backend_info": backend_info, "summary": summarize(records), "results": records}
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(
            {**result, "results": [strip_tokens(r) for r in records]}, indent=1))
    return result


def build_batch(t: int, result: dict, reward_name: str = "custom") -> dict:
    """rollout() result -> the worker's training payload (rl_worker.prepare_batch).
    Errored rollouts and ones without token data are dropped here; group
    integrity (>= 2 samples), degenerate-group dropping and advantages are the
    worker's job."""
    groups = []
    for rec in result["results"]:
        if "prompt_token_ids" not in rec:
            continue
        groups.append({
            "key": rec["key"],
            "prompt_token_ids": rec["prompt_token_ids"],
            "samples": [
                {"token_ids": s["token_ids"], "token_logprobs": s["token_logprobs"],
                 "reward": s["reward"], "truncated": s["truncated"]}
                for s in rec["samples"] if not s.get("error") and s.get("token_ids")
            ],
        })
    return {"iteration": t, "reward": reward_name, "groups": groups}


# --- volume protocol (counterpart of rl_worker.py) --------------------------


def _modal():
    """Deferred imports: Modal apps resolve volume names at import time, so
    callers must have set COTCONTROL_MODAL_OWNER before the first call."""
    import modal

    from cotcontrol.inference import modal_vllm
    from cotcontrol.training import modal_app as train_modal

    return modal, modal_vllm, train_modal


def rollout_generate_config(cfg: RLConfig, inf: dict | None = None):
    """On-policy sampling config for the rollouts (same rules as rl/run_grpo.py:
    top_p=1 and temperature 1 because vLLM logprobs are pre-truncation, cache
    off, engine folded into the trainer's app)."""
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig

    inf = inf or {}
    tp = int(inf.get("tensor_parallel_size", 1))
    gpu_type = cfg.gpu.split(":")[0]
    gpu = cfg.inference_gpu or inf.get("gpu") or (gpu_type if tp == 1 else f"{gpu_type}:{tp}")
    return ModalGenerateConfig(
        temperature=cfg.rollout_temperature,
        top_p=1.0,
        max_tokens=cfg.rollout_max_tokens,
        num_samples=cfg.group_size,
        return_logprobs=True,
        cache=False,
        chat_template_kwargs=cfg.chat_template_kwargs,
        gpu=gpu,
        tensor_parallel_size=tp,
        max_lora_rank=max(64, cfg.lora_rank),
        max_model_len=int(inf.get("max_model_len", 32768)),
        max_num_seqs=int(inf.get("max_num_seqs", 256)),
        mixed_moe_lora=inf.get("mixed_moe_lora", bool(cfg.lora_target_parameters)),
        assume_app_running=True,
    )


def vol_read(path: str) -> bytes | None:
    _, _, train_modal = _modal()
    try:
        return b"".join(train_modal.checkpoints_vol.read_file(path))
    except Exception:  # noqa: BLE001 — not-yet-committed file or a transient volume error
        return None


async def vol_read_async(path: str) -> bytes | None:
    """vol_read for async contexts (Modal's blocking calls warn there)."""
    _, _, train_modal = _modal()
    try:
        out = train_modal.checkpoints_vol.read_file.aio(path)
        chunks = [c async for c in out] if hasattr(out, "__aiter__") else list(await out)
        return b"".join(chunks)
    except Exception:  # noqa: BLE001 — not-yet-committed file or a transient volume error
        return None


def _not_done(e: Exception) -> bool:
    return "timeout" in type(e).__name__.lower()


def worker_result(handle) -> dict | None:
    """The spawned worker's result if it already finished (None while running);
    raises with the remote traceback if it failed."""
    try:
        return handle.get(timeout=0)
    except Exception as e:
        if _not_done(e):
            return None
        raise


def _transient(e: Exception) -> bool:
    """Connection-level failures of the Modal client (gRPC stream lost, timeouts
    talking to the control plane) — the remote work is unaffected, so callers
    retry instead of dying."""
    name = type(e).__name__
    return any(k in name for k in ("StreamTerminated", "ConnectionError", "CancelledError",
                                   "GRPCError", "ClientClosed", "TimeoutError")) \
        or "Connection lost" in str(e) or "connection" in str(e).lower()


async def worker_result_async(handle) -> dict | None:
    try:
        return await handle.get.aio(timeout=0)
    except Exception as e:
        if _not_done(e):
            return None
        if _transient(e):
            print(f"[driver] transient error polling the worker ({type(e).__name__}: {e}); "
                  "treating as still running", flush=True)
            return None
        raise


async def wait_for_adapter(run_name: str, t: int, cfg: RLConfig, handle,
                           allow_finished: bool = False) -> str:
    """Block until /checkpoints/<run>/checkpoint-<t> is committed. A worker
    that exits early is an error, unless allow_finished (the final checkpoint
    is written right before the worker returns)."""
    _, _, train_modal = _modal()
    lora_path = f"{train_modal.CHECKPOINTS_PATH}/{run_name}/checkpoint-{t}"
    marker = f"{run_name}/checkpoint-{t}/adapter_config.json"
    t0 = time.time()
    while await vol_read_async(marker) is None:
        res = await worker_result_async(handle)
        if res is not None and not allow_finished:
            raise RuntimeError(f"training worker exited before the driver finished: {res}")
        if res is not None:
            await train_modal.checkpoints_vol.reload.aio()
            if await vol_read_async(marker) is None:
                raise RuntimeError(f"worker finished but {marker} is missing")
            break
        if time.time() - t0 > cfg.handshake_timeout_s:
            raise RuntimeError(f"checkpoint-{t} never appeared after {cfg.handshake_timeout_s}s")
        await asyncio.sleep(10.0)
    return lora_path


async def upload_batch(local: Path, run_name: str, t: int, attempts: int = 40) -> None:
    """Volume commits can transiently fail under layer pressure ("too many
    layers in volume"); retrying an upload in place is always safe."""
    modal, _, train_modal = _modal()
    for attempt in range(attempts):
        try:
            with train_modal.checkpoints_vol.batch_upload(force=True) as b:
                b.put_file(str(local), f"/{run_name}/rl/batch-{t}.json")
            return
        except modal.exception.ResourceExhaustedError as e:
            if attempt == attempts - 1:
                raise
            print(f"[driver] volume busy ({e}); retry {attempt + 1}/{attempts - 1} in 120s", flush=True)
            await asyncio.sleep(120)
        except Exception as e:
            if attempt >= 5 or not _transient(e):
                raise
            print(f"[driver] upload failed ({type(e).__name__}: {e}); retry in 60s", flush=True)
            await asyncio.sleep(60)


def find_resume_point(run_name: str, iterations: int) -> int | None:
    """Highest committed iteration with BOTH adapter and optimizer state
    (0 < t < iterations) — the worker can restart there losslessly."""
    _, _, train_modal = _modal()
    try:
        entries = train_modal.checkpoints_vol.listdir(f"/{run_name}")
    except Exception:  # noqa: BLE001 — run dir not on the volume yet
        return None
    steps = []
    for e in entries:
        name = e.path.rsplit("/", 1)[-1]
        tail = name.rsplit("-", 1)[-1]
        if name.startswith("checkpoint-") and tail.isdigit():
            steps.append(int(tail))
    for t in sorted((s for s in steps if 0 < s < iterations), reverse=True):
        try:
            files = {f.path.rsplit("/", 1)[-1] for f in
                     train_modal.checkpoints_vol.listdir(f"/{run_name}/checkpoint-{t}")}
        except Exception:  # noqa: BLE001, S112 — half-written checkpoint dir
            continue
        if {"adapter_model.safetensors", "optimizer.pt"} <= files:
            return t
    return None


def sharded_generate_fn(base_model: str, gen_cfg, lora_path: str | None, shards: int = 1):
    """generate_fn that splits the prompt list into `shards` concurrent engine
    calls. The Modal engine class serves one call per container, so Modal fans
    the calls out to `shards` warm GPUs and one iteration's rollouts finish
    ~shards x faster (containers stay warm across iterations)."""
    from cotcontrol.inference.modal_vllm import make_generate_fn

    inner = make_generate_fn(base_model, gen_cfg, lora_path)
    if shards <= 1:
        return inner

    async def fn(messages_list):
        n, k = len(messages_list), max(1, min(shards, len(messages_list)))
        parts = [messages_list[i * n // k:(i + 1) * n // k] for i in range(k)]
        outs = await asyncio.gather(*[inner(p) for p in parts if p])
        return [r for o in outs for r in o]

    return fn


# --- the loop ---------------------------------------------------------------

_ENGINE_INCLUDED = False


async def drive(
    cfg: RLConfig,
    gen_cfg,
    handle,
    run_dir: Path,
    source: PromptSource,
    reward_fn: RewardFn,
    heldout: PromptSource | None = None,
    heldout_at: Sequence[int] = (),
    heldout_n: int = 64,
    heldout_gen_cfg=None,
    rollout_shards: int = 1,
    reasoning_disabled: bool | None = None,
) -> None:
    """Rollout loop for iterations [cfg.start_iteration, cfg.iterations): wait
    for checkpoint-t, sample cfg.prompts_per_iter prompts, roll out
    cfg.group_size samples each, reward, upload batch-t. At iterations in
    heldout_at (and at cfg.iterations, on the final adapter) also roll out
    heldout_n prompts from `heldout` with heldout_gen_cfg (default: gen_cfg)
    and log their reward — the unseen-item monitor. rollout_shards > 1 spreads
    each iteration's generation over that many engine containers.
    reasoning_disabled (default: enable_thinking is False in the rollout chat
    template kwargs) makes the whole completion the response channel."""
    if reasoning_disabled is None:
        reasoning_disabled = (gen_cfg.chat_template_kwargs or {}).get("enable_thinking") is False
    (run_dir / "batches").mkdir(parents=True, exist_ok=True)
    progress = run_dir / "progress.jsonl"
    heldout_at = set(heldout_at) if heldout is not None else set()
    hgen = heldout_gen_cfg or gen_cfg

    async def _rollout(*args, **kwargs) -> dict:
        """rollout() with retries on transient client errors: sampling is
        stateless and on-policy, so a retry simply re-samples."""
        for attempt in range(4):
            try:
                return await rollout(*args, **kwargs)
            except Exception as e:
                if attempt == 3 or not _transient(e):
                    raise
                print(f"[driver] rollout failed ({type(e).__name__}: {e}); retry {attempt + 1}/3 in 60s",
                      flush=True)
                await asyncio.sleep(60)
        raise AssertionError("unreachable")

    async def _heldout(t: int, lora_path: str) -> dict:
        prompts = heldout.sample(0, heldout_n, cfg.seed)  # fixed slice every time
        res = await _rollout(
            prompts, sharded_generate_fn(cfg.base_model, hgen, lora_path, rollout_shards), reward_fn,
            save_path=run_dir / "iters" / f"heldout-{t:04d}.json",
            backend_info={"base_model": cfg.base_model, "lora_path": lora_path,
                          "generate_config": asdict(hgen)},
            reasoning_disabled=reasoning_disabled)
        return res["summary"]

    for t in range(cfg.start_iteration, cfg.iterations):
        lora_path = await wait_for_adapter(cfg.run_name, t, cfg, handle)
        t0 = time.time()
        prompts = source.sample(t, cfg.prompts_per_iter, cfg.seed)
        result = await _rollout(
            prompts, sharded_generate_fn(cfg.base_model, gen_cfg, lora_path, rollout_shards), reward_fn,
            save_path=run_dir / "iters" / f"iter-{t:04d}.json",
            backend_info={"base_model": cfg.base_model, "lora_path": lora_path,
                          "generate_config": asdict(gen_cfg)},
            reasoning_disabled=reasoning_disabled)
        batch = build_batch(t, result, cfg.reward)
        local = run_dir / "batches" / f"batch-{t}.json"
        local.write_text(json.dumps(batch))
        await upload_batch(local, cfg.run_name, t)
        local.write_text(json.dumps({**batch, "groups": [
            strip_tokens(g) for g in batch["groups"]]}))
        entry = {"iteration": t, **result["summary"], "rollout_s": round(time.time() - t0, 1)}
        if t in heldout_at:
            entry["heldout"] = await _heldout(t, lora_path)
        with progress.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[driver iter {t}] reward_mean={entry['reward_mean']} "
              f"trunc={entry['frac_truncated']} ({entry['rollout_s']}s)"
              + (f" heldout={entry['heldout']['reward_mean']}" if "heldout" in entry else ""),
              flush=True)

    if cfg.iterations in heldout_at:
        lora_path = await wait_for_adapter(cfg.run_name, cfg.iterations, cfg, handle,
                                           allow_finished=True)
        entry = {"iteration": cfg.iterations, "final": True,
                 "heldout": await _heldout(cfg.iterations, lora_path)}
        with progress.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[driver final] heldout={entry['heldout']['reward_mean']}", flush=True)


def run_rl(
    cfg: RLConfig,
    gen_cfg,
    run_dir: Path,
    source: PromptSource,
    reward_fn: RewardFn,
    heldout: PromptSource | None = None,
    heldout_at: Sequence[int] = (),
    heldout_n: int = 64,
    heldout_gen_cfg=None,
    rollout_shards: int = 1,
    reasoning_disabled: bool | None = None,
) -> dict:
    """Spawn the Modal GRPO worker for cfg (cfg.run_name must already be
    final/unique — the caller timestamps), drive it to completion, and write
    train_result.json (+ the worker's metrics.jsonl, mask files) into run_dir.
    Handles the ephemeral-app cap, worker crashes with committed progress
    (resume) and budget-stops (respawn) like rl/run_grpo.py."""
    global _ENGINE_INCLUDED
    modal, modal_vllm, train_modal = _modal()
    run_dir.mkdir(parents=True, exist_ok=True)
    # One ephemeral app per run: fold the vLLM engine into the trainer's app
    # (gen_cfg.assume_app_running skips the engine's own app.run). Once per
    # process — include() is not idempotent. In deployed mode (COTCONTROL_MODAL_
    # {TRAIN,INFERENCE}_APP) both classes are looked up on deployed apps and
    # app.run() is a no-op, so there is nothing to include.
    deployed = bool(getattr(train_modal, "DEPLOYED_APP", None)) and bool(
        getattr(modal_vllm, "DEPLOYED_APP", None))
    if not _ENGINE_INCLUDED and not deployed:
        train_modal.app.include(modal_vllm.app)
        _ENGINE_INCLUDED = True
    if deployed:
        print(f"[driver] deployed Modal apps: train={train_modal.DEPLOYED_APP} "
              f"inference={modal_vllm.DEPLOYED_APP}", flush=True)

    def _drive(handle):
        return asyncio.run(drive(cfg, gen_cfg, handle, run_dir, source, reward_fn,
                                 heldout, heldout_at, heldout_n, heldout_gen_cfg, rollout_shards,
                                 reasoning_disabled))

    # A spawned worker outlives this process on a deployed app (ephemeral apps
    # die with the client), so a killed/crashed driver must cancel it: a second
    # worker on the same run_name would race the restart on the volume. SIGTERM
    # (the scheduler's kill) is turned into SystemExit so the finally runs.
    import signal
    import sys

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    def _cancel(handle, why: str) -> None:
        try:
            # terminate_containers: the worker is an accelerate subprocess, and
            # cancelling the input alone can leave it running in the container.
            handle.cancel(terminate_containers=True)
            print(f"[driver] cancelled worker call ({why})", flush=True)
        except Exception as e:  # noqa: BLE001 — best effort
            print(f"[driver] could not cancel worker call: {e}", flush=True)

    while True:
        try:
            with modal.enable_output(), train_modal.app.run():
                while True:
                    handle = train_modal.spawn_rl(cfg)
                    finished = False
                    try:
                        _drive(handle)
                        print("[driver] rollout loop done; waiting for worker to finish", flush=True)
                        result = handle.get()
                        finished = True
                    except modal.exception.ResourceExhaustedError:
                        raise
                    except Exception as e:
                        nxt = find_resume_point(cfg.run_name, cfg.iterations)
                        if nxt is not None and nxt > cfg.start_iteration:
                            print(f"[driver] worker died ({type(e).__name__}: {e}); "
                                  f"resuming from checkpoint-{nxt}", flush=True)
                            _cancel(handle, "resume")
                            cfg.start_iteration = nxt
                            continue
                        raise
                    finally:
                        if not finished:
                            _cancel(handle, "driver exiting without a finished worker")
                    resume_at = result.get("resume_at") if isinstance(result, dict) else None
                    if resume_at is not None:
                        print(f"[driver] worker budget-stopped at iteration {resume_at}; "
                              "respawning", flush=True)
                        cfg.start_iteration = resume_at
                        continue
                    break
            break
        except modal.exception.ResourceExhaustedError as e:
            progress = run_dir / "progress.jsonl"
            if cfg.start_iteration == 0 and progress.exists() and progress.read_text().strip():
                raise
            wait = 240 + random.random() * 120
            print(f"[driver] ephemeral app cap hit ({e}); retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)

    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    for name in ("rl/metrics.jsonl", "mask_verification.json", "mask.json"):
        data = vol_read(f"{cfg.run_name}/{name}")
        if data is not None:
            (run_dir / Path(name).name).write_bytes(data)
    print(f"done: {result.get('iterations_completed')} iterations, artifacts in {run_dir}",
          flush=True)
    return result
