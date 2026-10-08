"""vLLM-on-Modal inference backend.

A warm Modal class (`InferenceEngine`) loads a base model once and reuses it,
hot-swapping LoRA adapters per call. `generate_async` wraps it with a disk
prompt cache, splits raw completions into (reasoning, response) with
`split_reasoning`, and returns the canonical schema from
cotcontrol.inference.types — so eval/GEPA code is backend-agnostic.

`base_model` / `lora_path` are interpreted *inside the container*: an HF repo
id, or an absolute path on the shared checkpoints volume
(/checkpoints/<run_name>/checkpoint-<step>) written by the training app.

Usage:
    from cotcontrol.inference.modal_vllm import ModalGenerateConfig, generate_async

    results = await generate_async(
        messages_list, base_model="Qwen/Qwen3-8B",
        config=ModalGenerateConfig(temperature=0.0, max_tokens=12000, gpu="H200"),
        lora_path="/checkpoints/my_run/checkpoint-40",   # or None for the base model
    )

Sweeping many checkpoints? Wrap the sweep in `async with app.run():` and
`asyncio.gather` the per-checkpoint calls — same GenerateConfig resource
fields -> one warm engine shared across all of them.
"""

import asyncio
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import modal

from cotcontrol.inference.types import split_reasoning

app = modal.App("cotcontrol-inference")

CHECKPOINTS_PATH = "/checkpoints"
HF_CACHE_PATH = "/cache"
HF_CACHE = f"{HF_CACHE_PATH}/hf"

# Redwood naming: {project}_{owner}_{label}. Shared by the training app;
# see cotcontrol.training.modal_app for the two env overrides.
checkpoints_vol = modal.Volume.from_name(
    f"cotcontrol_{os.environ.get('COTCONTROL_MODAL_OWNER', 'sebastian-prasanna')}_checkpoints",
    create_if_missing=True,
)
hf_cache_vol = modal.Volume.from_name(
    os.environ.get("COTCONTROL_HF_CACHE_VOLUME", "cotcontrol_sebastian-prasanna_hf-cache"),
    create_if_missing=True,
)

# Deployed mode: COTCONTROL_MODAL_INFERENCE_APP names a `modal deploy`ed copy of
# this app (kaushik/deploy_modal.sh). Engines are looked up there, and app.run()
# becomes a no-op so callers stop opening one ephemeral app per sweep (the
# workspace caps concurrent ephemeral apps at 100). Unset = original behavior.
# Remote-side code changes need a redeploy to take effect in this mode.
DEPLOYED_APP = os.environ.get("COTCONTROL_MODAL_INFERENCE_APP")
if DEPLOYED_APP:
    app.run = lambda *args, **kwargs: contextlib.nullcontext()

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "vllm==0.23.0",
        "hf-transfer",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # The flashinfer sampler JIT-compiles a CUDA kernel at engine start
            # and needs nvcc, which the slim wheel image lacks.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        }
    )
)

# All models we train are ungated, so no HF token is required. Set
# COTCONTROL_MODAL_HF_SECRET to the name of a Modal secret holding HF_TOKEN if
# you ever need a gated model in the container.
_hf_secret = os.environ.get("COTCONTROL_MODAL_HF_SECRET")
_SECRETS = [modal.Secret.from_name(_hf_secret)] if _hf_secret else []

CACHE_DIR = Path(__file__).resolve().parents[2] / ".generation_cache"


@dataclass
class ModalGenerateConfig:
    """Sampling + Modal/vLLM resource parameters.

    The resource fields participate in the warm-container pool identity —
    keep them constant across a checkpoint sweep to reuse one engine.
    """

    # Sampling
    temperature: float = 0.0
    max_tokens: int = 12000
    num_samples: int = 1
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    # vLLM per-request sampling seed (reproducible temperature>0 runs).
    # None = engine default (unseeded). Part of the disk-cache key when set.
    seed: Optional[int] = None
    # Return per-token logprobs + token ids (behavior-policy data for RL).
    # Adds "token_ids"/"token_logprobs" (per sample) and "prompt_token_ids"
    # (per prompt) to the result dicts; absent entirely when False.
    return_logprobs: bool = False
    cache: bool = True
    # The caller already holds a running Modal app that hosts this module's
    # InferenceEngine (via app.include, e.g. rl/run_grpo.py merges trainer +
    # engine into one ephemeral app). Skips opening app.run() here — the
    # heuristic _is_app_running check can't see an external app.
    assume_app_running: bool = False
    # Extra kwargs for the chat template, e.g. {"reasoning_effort": "high"}
    # for gpt-oss. Part of the disk-cache key.
    chat_template_kwargs: Optional[dict] = None

    # Modal / vLLM resources
    gpu: str = "H200"
    tensor_parallel_size: int = 1
    enable_lora: bool = True
    max_lora_rank: int = 64
    max_model_len: int = 32768
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9
    # True for adapters that target the fused MoE experts (PEFT 3D stacked
    # tensors). Now INERT at the vLLM layer — 3D-native models (gpt-oss)
    # load these adapters without flags, and vLLM's mixed-format path drops
    # the expert deltas (see InferenceEngine.setup). Kept so existing
    # callers/configs don't break.
    mixed_moe_lora: bool = False

    def __post_init__(self) -> None:
        self.temperature = float(self.temperature)
        self.top_p = float(self.top_p)
        self.gpu_memory_utilization = float(self.gpu_memory_utilization)
        for f in ("max_tokens", "num_samples", "tensor_parallel_size",
                  "max_lora_rank", "max_model_len", "max_num_seqs"):
            setattr(self, f, int(getattr(self, f)))


@app.cls(
    image=inference_image,
    gpu="H200",  # label only — the real spec comes via with_options(gpu=...)
    volumes={CHECKPOINTS_PATH: checkpoints_vol, HF_CACHE_PATH: hf_cache_vol},
    timeout=4 * 3600,
    scaledown_window=600,
    secrets=_SECRETS,
)
class InferenceEngine:
    base_model: str = modal.parameter()
    tensor_parallel_size: int = modal.parameter(default=1)
    max_lora_rank: int = modal.parameter(default=64)
    enable_lora: bool = modal.parameter(default=True)
    max_model_len: int = modal.parameter(default=32768)
    max_num_seqs: int = modal.parameter(default=256)
    gpu_memory_utilization_pct: int = modal.parameter(default=90)
    mixed_moe_lora: bool = modal.parameter(default=False)

    @modal.enter()
    def setup(self):
        from vllm import LLM

        checkpoints_vol.reload()
        hf_cache_vol.reload()
        # lora_path -> path actually handed to vLLM (see _composite_safe_lora).
        self._lora_path_cache: Dict[str, str] = {}

        llm_kwargs = dict(
            model=self.base_model,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization_pct / 100.0,
            # Hybrid gated-delta-net models (Qwen3.5/3.6, whose layer_types
            # alternate linear_attention with full_attention) default to
            # FlashInfer's GDN prefill kernel, which JIT-compiles with nvcc —
            # absent from this slim wheel image, so EngineCore dies at the
            # FIRST FORWARD and surfaces client-side only as an opaque
            # EngineDeadError from llm_engine.step(). The Triton/FLA kernel
            # needs no nvcc, and the option is ignored by models with no GDN
            # layers. (Same nvcc-avoidance family as the sampler/attention env
            # vars on inference_image.)
            additional_config={"gdn_prefill_backend": "triton"},
        )
        if self.enable_lora:
            llm_kwargs.update(enable_lora=True, max_lora_rank=self.max_lora_rank, max_loras=1)
            # Deliberately NOT setting enable_mixed_moe_lora_format for
            # mixed_moe_lora adapters: 3D-native MoE models (gpt-oss has
            # is_3d_moe_weight=True) load PEFT fused-expert adapters through
            # FusedMoE3DWithLoRA natively. The mixed-format flag instead
            # forces the universal 2D wrapper + a 3D->2D conversion path that
            # SILENTLY DROPS the expert deltas for gpt-oss (verified 2026-08-31:
            # eval compliance 0.015 vs 0.44 for the same checkpoint; PEFT-side
            # generation confirmed the adapter itself was fine).
        self.llm = LLM(**llm_kwargs)

    def _composite_safe_lora(self, lora_path: str) -> str:
        """Adapter path whose key namespace matches how vLLM serves this model.

        Composite/multimodal checkpoints (config has ``text_config``, e.g.
        Qwen3.6) are TRAINED through the text-only AutoModelForCausalLM view, so
        PEFT writes keys as ``base_model.model.model.layers...``. vLLM serves the
        composite and maps LoRA names through hf_to_vllm_mapper, which only
        translates the ``model.language_model.`` prefix. Mismatched keys load
        WITHOUT ERROR and apply nothing — the adapter is silently a no-op and the
        eval reports pure base-model behaviour, which is indistinguishable from a
        real negative result.

        Rewrites into a container-local copy rather than editing the checkpoint:
        the renamed adapter no longer loads onto the text-only model via PEFT, so
        mutating the saved file in place would break training-side reloads.
        Returns the original path when no rewrite is needed.
        """
        import json
        import shutil
        from pathlib import Path

        cached = self._lora_path_cache.get(lora_path)
        if cached:
            return cached

        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(self.base_model, trust_remote_code=True)
        if not hasattr(cfg, "text_config"):
            self._lora_path_cache[lora_path] = lora_path
            return lora_path

        src = Path(lora_path) / "adapter_model.safetensors"
        if not src.exists():
            self._lora_path_cache[lora_path] = lora_path
            return lora_path

        import safetensors.torch as st
        tensors = st.load_file(str(src))
        old, new = "base_model.model.model.", "base_model.model.model.language_model."
        if not any(k.startswith(old) and not k.startswith(new) for k in tensors):
            self._lora_path_cache[lora_path] = lora_path
            return lora_path

        dst = Path("/tmp/lora_composite") / Path(lora_path).name
        dst.mkdir(parents=True, exist_ok=True)
        for extra in Path(lora_path).glob("*"):
            if extra.is_file() and extra.name != "adapter_model.safetensors":
                shutil.copy2(extra, dst / extra.name)
        st.save_file(
            {(new + k[len(old):] if k.startswith(old) else k): v
             for k, v in tensors.items()},
            str(dst / "adapter_model.safetensors"),
        )
        n = sum(1 for k in tensors if k.startswith(old))
        print(f"[lora] composite namespace rewrite: {n} keys {old!r} -> {new!r} "
              f"({lora_path} -> {dst})", flush=True)
        self._lora_path_cache[lora_path] = str(dst)
        return str(dst)

    @modal.method()
    def generate(
        self,
        messages_list,
        lora_path=None,
        lora_id=1,
        max_tokens=12000,
        temperature=0.0,
        top_p=1.0,
        num_samples=1,
        stop=None,
        return_logprobs=False,
        add_generation_prompt=True,
        continue_final_message=False,
        chat_template_kwargs=None,
        seed=None,
    ):
        """Chat-format generation over a batch. Returns
        List[List[{"text", "finish_reason"}]] (outer = prompts, inner =
        num_samples). Special tokens are kept in the text so the caller can
        split Qwen <think> blocks / gpt-oss harmony channels downstream.

        continue_final_message=True (with add_generation_prompt=False) makes
        the model CONTINUE the final assistant message instead of starting a
        new turn — the prefill path.
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        if lora_path:
            checkpoints_vol.reload()

        sp = SamplingParams(
            n=num_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or None,
            seed=seed,
            skip_special_tokens=False,
            # logprobs=0 -> only the sampled token's logprob at each position.
            # vLLM v1 default logprobs_mode="raw_logprobs": log-softmax of the
            # raw logits, BEFORE temperature/top-p (i.e. the model's own
            # distribution, not the sampling distribution).
            logprobs=0 if return_logprobs else None,
        )
        if lora_path:
            # No is_3d_lora_weight: only consulted under the (unused, broken
            # for gpt-oss) enable_mixed_moe_lora_format path — see setup().
            lora_req = LoRARequest(
                f"lora_{lora_id}", lora_id, self._composite_safe_lora(lora_path)
            )
        else:
            lora_req = None
        outputs = self.llm.chat(
            messages_list,
            sampling_params=sp,
            lora_request=lora_req,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            chat_template_kwargs=chat_template_kwargs or None,
        )
        def _sample(o, c):
            d = {"text": c.text, "finish_reason": c.finish_reason}
            if return_logprobs:
                d["token_ids"] = list(c.token_ids)
                d["token_logprobs"] = [
                    lp[tid].logprob for tid, lp in zip(c.token_ids, c.logprobs)
                ]
                d["prompt_token_ids"] = list(o.prompt_token_ids)
            return d

        return [[_sample(o, c) for c in o.outputs] for o in outputs]

    @modal.method()
    def score(self, token_seqs, prompt_lens, lora_path=None, lora_id=1):
        """Teacher-forced log-probs: for each token sequence, the model's log-prob of
        every token after position prompt_len (raw log-softmax, no temperature)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.lora.request import LoRARequest

        if lora_path:
            checkpoints_vol.reload()
        lora_req = (LoRARequest(f"lora_{lora_id}", lora_id, self._composite_safe_lora(lora_path))
                    if lora_path else None)
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=list(t)) for t in token_seqs],
            sampling_params=SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0),
            lora_request=lora_req,
        )
        return [[o.prompt_logprobs[i][t[i]].logprob for i in range(n, len(t))]
                for o, t, n in zip(outputs, token_seqs, prompt_lens)]


async def score_async(token_seqs: List[List[int]], prompt_lens: List[int], base_model: str,
                      config: Optional[ModalGenerateConfig] = None,
                      lora_path: Optional[str] = None) -> List[List[float]]:
    """Per-token log-probs of token_seqs[i][prompt_lens[i]:] under base_model (+ lora_path).
    Uncached; requires the app to be running (wrap in `async with app.run()`)."""
    config = config or ModalGenerateConfig()
    engine = _engine_cls().with_options(gpu=config.gpu)(
        base_model=base_model, tensor_parallel_size=config.tensor_parallel_size,
        max_lora_rank=config.max_lora_rank, enable_lora=config.enable_lora,
        max_model_len=config.max_model_len, max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization_pct=int(round(config.gpu_memory_utilization * 100)),
        mixed_moe_lora=config.mixed_moe_lora)
    return await _call_resilient(engine.score, token_seqs, prompt_lens, lora_path,
                                 _lora_id_for(lora_path) if lora_path else 1)


def _engine_cls():
    """InferenceEngine, or its deployed copy in deployed mode."""
    if DEPLOYED_APP:
        return modal.Cls.from_name(DEPLOYED_APP, "InferenceEngine")
    return InferenceEngine


_TRANSIENT = {"ConnectionError", "StreamTerminatedError", "ConnectionResetError", "ClientClosed",
              "TimeoutError", "InternalError", "ServerError", "OSError", "GRPCError", "ProtocolError"}


def _transient(e: Exception) -> bool:
    return type(e).__name__ in _TRANSIENT or isinstance(e, (ConnectionError, OSError, TimeoutError))


async def _call_resilient(method, *args, max_reattach: int = 60, **kwargs):
    """method.remote.aio(...) that survives client-side connection drops: spawn
    once (a single RPC, safe to retry), then await the result, re-attaching to the
    SAME call id after transient errors. Re-submitting would duplicate the work on
    another GPU, since the input keeps running server-side. Remote exceptions
    propagate unchanged. Adapted from kaushik/hidden_constraint/runner.py."""
    for attempt in range(5):
        try:
            call = await method.spawn.aio(*args, **kwargs)
            break
        except Exception as e:  # noqa: BLE001
            if not _transient(e) or attempt == 4:
                raise
            await asyncio.sleep(10)
    call_id = call.object_id
    for attempt in range(max_reattach):
        try:
            return await call.get.aio()
        except Exception as e:  # noqa: BLE001
            if not _transient(e) or attempt == max_reattach - 1:
                raise
            print(f"[modal_vllm] call {call_id}: {type(e).__name__}; re-attaching "
                  f"({attempt + 1}/{max_reattach})", flush=True)
            await asyncio.sleep(20)
            call = modal.FunctionCall.from_id(call_id)


def _is_app_running(app_obj) -> bool:
    for v in app_obj.__dict__.values():
        if hasattr(v, "_running_app"):
            return v._running_app is not None
    return False


def _cache_key(model_id, messages, config: ModalGenerateConfig,
               add_generation_prompt, continue_final_message) -> str:
    is_gptoss_lora = "gpt-oss" in model_id and "::" in model_id and model_id.split("::", 1)[1]
    blob = json.dumps(
        {
            "cache_version": 1,
            # gpt-oss+adapter generations cached before 2026-08-31 were made
            # by an engine that silently dropped fused-expert LoRA deltas
            # (see InferenceEngine.setup). Scoped key bump invalidates exactly
            # that population without touching any other cached generations.
            **({"gptoss_lora_fix": 1} if is_gptoss_lora else {}),
            # Only set when True so pre-existing (False) cache keys are unchanged.
            **({"return_logprobs": True} if config.return_logprobs else {}),
            # Only set when not None so pre-existing (unseeded) cache keys are unchanged.
            **({"seed": config.seed} if config.seed is not None else {}),
            "backend": "modal-vllm",
            "model_id": model_id,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "num_samples": config.num_samples,
            "stop": config.stop,
            "chat_template_kwargs": config.chat_template_kwargs,
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def _load_cache(key: str) -> Optional[Dict]:
    fp = _cache_path(key)
    if not fp.exists():
        return None
    try:
        with fp.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None


def _save_cache(key: str, texts, finish_reasons, extra: Optional[Dict] = None) -> None:
    fp = _cache_path(key)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w") as f:
        json.dump({"texts": texts, "finish_reasons": finish_reasons, **(extra or {})}, f)


def _lora_id_for(path: str) -> int:
    return int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) % (10**8) + 1


def _to_canonical(messages, model_id, texts, finish_reasons) -> dict:
    """Raw completion strings -> the canonical result schema."""
    outputs, reasonings, metadata = [], [], []
    for text, fin in zip(texts, finish_reasons):
        reasoning, response, truncated = split_reasoning(text)
        outputs.append(response)
        reasonings.append(reasoning)
        metadata.append(
            {
                "finish_reason": fin if fin is not None else ("length" if truncated else "stop"),
                "usage": None,
                "error": None,
                "raw_response": text,
            }
        )
    return {
        "input": messages,
        "model": model_id,
        "output": outputs,
        "reasoning": reasonings,
        "metadata": metadata,
    }


async def generate_async(
    messages_list: List[List[Dict[str, str]]],
    base_model: str,
    config: Optional[ModalGenerateConfig] = None,
    lora_path: Optional[str] = None,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> List[Dict]:
    """Run vLLM inference on Modal; returns the canonical schema (one dict per
    prompt, same shape as the OpenRouter backend).

    lora_path=None serves the plain base model (HF id) — use this for base
    model evals or prefill experiments off HF.
    """
    config = config or ModalGenerateConfig()
    model_id = f"{base_model}::{lora_path or ''}"

    all_results: List[Optional[Dict]] = [None] * len(messages_list)
    cache_keys, uncached = [], []
    for i, msgs in enumerate(messages_list):
        ck = _cache_key(model_id, msgs, config, add_generation_prompt, continue_final_message)
        cache_keys.append(ck)
        if config.cache:
            cached = _load_cache(ck)
            if cached is not None:
                result = _to_canonical(
                    msgs, model_id, cached["texts"], cached["finish_reasons"]
                )
                if config.return_logprobs:
                    for k in ("token_ids", "token_logprobs", "prompt_token_ids"):
                        result[k] = cached[k]
                all_results[i] = result
                continue
        uncached.append(i)

    if not uncached:
        print(f"[modal_vllm] cache: {len(messages_list)}/{len(messages_list)} hits, all cached")
        return all_results
    print(
        f"[modal_vllm] cache: {len(messages_list) - len(uncached)}/{len(messages_list)} hits, "
        f"generating {len(uncached)} on {config.gpu}"
    )

    EngineCls = _engine_cls().with_options(gpu=config.gpu)
    engine_kwargs = dict(
        base_model=base_model,
        tensor_parallel_size=config.tensor_parallel_size,
        max_lora_rank=config.max_lora_rank,
        enable_lora=config.enable_lora,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization_pct=int(round(config.gpu_memory_utilization * 100)),
        mixed_moe_lora=config.mixed_moe_lora,
    )
    call_kwargs = dict(
        messages_list=[messages_list[i] for i in uncached],
        lora_path=lora_path,
        lora_id=_lora_id_for(lora_path) if lora_path else 1,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        num_samples=config.num_samples,
        stop=config.stop,
        return_logprobs=config.return_logprobs,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        chat_template_kwargs=config.chat_template_kwargs,
        seed=config.seed,
    )

    async def _do_call():
        engine = EngineCls(**engine_kwargs)
        return await _call_resilient(engine.generate, **call_kwargs)

    if DEPLOYED_APP or config.assume_app_running or _is_app_running(app):
        outputs_per_prompt = await _do_call()
    else:
        async with app.run():
            outputs_per_prompt = await _do_call()

    for j, idx in enumerate(uncached):
        samples = outputs_per_prompt[j]
        texts = [s["text"] for s in samples]
        fins = [s["finish_reason"] for s in samples]
        extra = None
        if config.return_logprobs:
            extra = {
                "token_ids": [s["token_ids"] for s in samples],
                "token_logprobs": [s["token_logprobs"] for s in samples],
                "prompt_token_ids": samples[0]["prompt_token_ids"],
            }
        if config.cache:
            _save_cache(cache_keys[idx], texts, fins, extra)
        result = _to_canonical(messages_list[idx], model_id, texts, fins)
        if extra:
            result.update(extra)
        all_results[idx] = result

    return all_results


def make_generate_fn(
    base_model: str,
    config: Optional[ModalGenerateConfig] = None,
    lora_path: Optional[str] = None,
):
    """A closure suitable for eval_cotcontrolqa(generate_fn=...)."""

    async def _generate_fn(messages_list):
        return await generate_async(messages_list, base_model, config, lora_path)

    return _generate_fn


if __name__ == "__main__":
    test_prompts = [
        [{"role": "user", "content": "Say 'hello' and nothing else."}],
        [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
    ]
    results = asyncio.run(
        generate_async(
            test_prompts,
            base_model="Qwen/Qwen3-0.6B",
            config=ModalGenerateConfig(
                temperature=0.0, max_tokens=512, gpu="L40S", max_model_len=4096, cache=False
            ),
        )
    )
    for r in results:
        print(r["input"][0]["content"])
        print("  reasoning:", (r["reasoning"][0] or "")[:120].replace("\n", " "))
        print("  output:", r["output"][0])
