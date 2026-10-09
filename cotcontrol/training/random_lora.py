"""KL-matched random-LoRA controls (the ``noise`` reference point).

A trained adapter's effect is compared against random adapters of the same
shape — same rank, alpha and target modules/parameters, including fused-MoE
expert tensors — whose *functional* size matches: KL(base || adapter) per token
on the base model's own T=1 samples, not weight norm. A norm-matched random
adapter usually changes outputs far less than a trained one, which would make
the control too easy to beat.

Lifted from sdf/random_lora.py and sdf/random_lora_kl.py (gpt-oss-120b
specific) into a general form. Three layers:

Tensors (torch, CPU, unit-tested):
  random_adapter_tensors(src, rank, scale, seed, family)
      Gaussian A/B with the source adapter's keys, shapes and dtypes.
      family="frob": per-(module, expert) ||B_e A_e||_F = scale x the trained
      adapter's. The direction depends only on (seed, module key), so varying
      the scale for one seed walks a dose-response line along one random
      direction — what the calibration below bisects over. Modules the trained
      adapter left at zero (masked small-k runs) stay zero.
      family="iid": no per-module matching; every entry of B@A has std
      scale x the trained update's global entry RMS.
  update_norms(A, B, rank) -> per-expert Frobenius norms via r x r Grams.

  PEFT layouts (peft ParamWrapper.get_delta_weight): attention A (r, in),
  B (out, r); fused experts A (E*r, in) expert-major -> (E, r, in), B (out, r*E)
  interleaved -> (out, r, E); delta_e = scaling * B[:, :, e] @ A[e].

KL estimate (vLLM on Modal, through cotcontrol.inference.modal_vllm):
  unconstrained_prompts / sample_base      the base model's own T=1 samples on
                                           unconstrained questions (fixed set,
                                           reused for every adapter of a base)
  score_samples                            teacher-forced token log-probs
  kl_from_logprobs(base, adapter)          mean(log p_base - log p_adapter) over
                                           the sampled tokens: an unbiased
                                           estimate of E_base[KL] per token, with
                                           a token-level SE and a prompt-level
                                           bootstrap SE (tokens within a sample
                                           are correlated).
  calibrate_scale(kl_at, target)           bracket + log-log interpolation on
                                           the scale until KL is within tol.

Modal (checkpoints volume, CPU):
  build_adapter.remote(src_adapter, dst, scale, seed, family) writes
  dst/{adapter_model.safetensors, adapter_config.json, random_lora.json};
  remove_path.remote(path) cleans calibration scratch.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import asdict, replace
from pathlib import Path

import modal

CHECKPOINTS_PATH = "/checkpoints"

# Same volume as the training/inference apps (Redwood naming, owner override).
checkpoints_vol = modal.Volume.from_name(
    f"cotcontrol_{os.environ.get('COTCONTROL_MODAL_OWNER', 'sebastian-prasanna')}_checkpoints",
    create_if_missing=True,
)

app = modal.App("cotcontrol-random-lora")
image = (
    modal.Image.debian_slim(python_version="3.11")
    # CPU torch: adapters are bf16/fp32 safetensors, which numpy can't read.
    .uv_pip_install("torch", "safetensors", "numpy", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source("cotcontrol")
)

# Deployed mode (same pattern as cotcontrol.inference.modal_vllm): when
# COTCONTROL_MODAL_RANDOM_LORA_APP names a `modal deploy`ed copy of this app
# (`kaushik/run.py noise deploy`), remote_fn() looks the functions up there and
# app.run() is a no-op, so no ephemeral app is opened per build (the workspace
# caps those at 100). Remote-side code in this file is snapshotted at deploy:
# redeploy after changing it.
DEPLOYED_APP = os.environ.get("COTCONTROL_MODAL_RANDOM_LORA_APP")
if DEPLOYED_APP:
    app.run = lambda *args, **kwargs: contextlib.nullcontext()


def remote_fn(name: str):
    """build_adapter / remove_path / path_exists, or their deployed copies."""
    return modal.Function.from_name(DEPLOYED_APP, name) if DEPLOYED_APP else globals()[name]


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------

def lora_pairs(keys) -> list[tuple[str, str]]:
    """Sorted (lora_A key, lora_B key) pairs of a PEFT adapter state dict."""
    keys = set(keys)
    pairs = []
    for ka in sorted(k for k in keys if k.endswith("lora_A.weight")):
        kb = ka[: -len("lora_A.weight")] + "lora_B.weight"
        if kb not in keys:
            raise KeyError(f"{ka} has no matching lora_B")
        pairs.append((ka, kb))
    return pairs


def update_norms(A, B, rank: int):
    """Per-expert ||B_e A_e||_F (without the shared alpha/r scaling), shape (E,).
    E = 1 for ordinary linear layers."""
    E = A.shape[0] // rank
    A3 = A.reshape(E, rank, A.shape[-1]).float()                  # (E, r, in)
    B3 = B.reshape(B.shape[0], rank, E).permute(2, 1, 0).float()  # (E, r, out)
    GA = A3 @ A3.transpose(1, 2)
    GB = B3 @ B3.transpose(1, 2)
    return (GA * GB).sum(dim=(1, 2)).clamp_min(0.0).sqrt()


def _generator(seed: int, key: str):
    import torch

    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little"))


def random_adapter_tensors(src: dict, rank: int, scale: float, seed: int,
                           family: str = "frob") -> tuple[dict, list[dict]]:
    """Random adapter with the source adapter's keys/shapes/dtypes.
    Returns (tensors, per-module report)."""
    import torch

    if family not in ("frob", "iid"):
        raise ValueError(f"family must be 'frob' or 'iid', got {family!r}")
    pairs = lora_pairs(src)
    if not pairs:
        raise ValueError("source adapter has no lora_A/lora_B pairs")
    anchor_rms = None
    if family == "iid":
        tot_sq = tot_n = 0.0
        for ka, kb in pairs:
            A, B = src[ka], src[kb]
            tot_sq += float((update_norms(A, B, rank) ** 2).sum())
            tot_n += (A.shape[0] // rank) * A.shape[-1] * B.shape[0]
        anchor_rms = math.sqrt(tot_sq / tot_n)
    out, report = {}, []
    for ka, kb in pairs:
        A, B = src[ka], src[kb]
        E = A.shape[0] // rank
        g = _generator(seed, ka)
        Ar = torch.randn(tuple(A.shape), generator=g, dtype=torch.float32)
        Br = torch.randn(tuple(B.shape), generator=g, dtype=torch.float32)
        target = update_norms(A, B, rank)
        if family == "frob":
            f = scale * target / update_norms(Ar, Br, rank).clamp_min(1e-30)
            Br = (Br.reshape(B.shape[0], rank, E) * f).reshape(tuple(B.shape))
        else:  # entries of B@A have std sqrt(rank) for unit-Gaussian A, B
            Br = Br * (scale * anchor_rms / math.sqrt(rank))
        achieved = update_norms(Ar, Br, rank)
        out[ka], out[kb] = Ar.to(A.dtype), Br.to(B.dtype)
        report.append({
            "module": ka[: -len(".lora_A.weight")], "experts": E,
            "trained_norm": float(target.square().sum().sqrt()),
            "random_norm": float(achieved.square().sum().sqrt()),
        })
    return out, report


# ---------------------------------------------------------------------------
# Modal: build on the checkpoints volume
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={CHECKPOINTS_PATH: checkpoints_vol},
              cpu=8, memory=65536, timeout=3600)
def build_adapter(src_adapter: str, dst: str, scale: float, seed: int,
                  family: str = "frob") -> dict:
    """Random adapter shaped like src_adapter (a /checkpoints/... dir) at dst."""
    from safetensors.torch import load_file, save_file

    from cotcontrol.training.random_lora import random_adapter_tensors

    checkpoints_vol.reload()
    cfg = json.loads(Path(src_adapter, "adapter_config.json").read_text())
    src = load_file(f"{src_adapter}/adapter_model.safetensors")
    tensors, report = random_adapter_tensors(src, int(cfg["r"]), float(scale), int(seed), family)
    Path(dst).mkdir(parents=True, exist_ok=True)
    save_file(tensors, f"{dst}/adapter_model.safetensors", metadata={"format": "pt"})
    shutil.copy(f"{src_adapter}/adapter_config.json", f"{dst}/adapter_config.json")
    meta = {"source_adapter": src_adapter, "scale": scale, "seed": seed, "family": family,
            "rank": cfg["r"], "lora_alpha": cfg.get("lora_alpha"), "modules": report}
    Path(f"{dst}/random_lora.json").write_text(json.dumps(meta))
    checkpoints_vol.commit()
    return {"path": dst, "n_modules": len(report),
            "trained_norm": math.sqrt(sum(m["trained_norm"] ** 2 for m in report)),
            "random_norm": math.sqrt(sum(m["random_norm"] ** 2 for m in report)),
            "n_params": sum(t.numel() for t in tensors.values())}


@app.function(image=image, volumes={CHECKPOINTS_PATH: checkpoints_vol}, timeout=1800)
def remove_path(path: str) -> bool:
    """rm -rf a scratch dir under /checkpoints (calibration candidates)."""
    if not path.startswith(CHECKPOINTS_PATH + "/") or path.count("/") < 3:
        raise ValueError(f"refusing to remove {path!r}")
    shutil.rmtree(path, ignore_errors=True)
    checkpoints_vol.commit()
    return not Path(path).exists()


@app.function(image=image, volumes={CHECKPOINTS_PATH: checkpoints_vol}, timeout=600)
def path_exists(path: str) -> bool:
    checkpoints_vol.reload()
    return Path(path, "adapter_model.safetensors").exists()


# ---------------------------------------------------------------------------
# KL(base || adapter) on the base model's own samples
# ---------------------------------------------------------------------------

def unconstrained_prompts(base_model: str, split: str = "val", dataset: str = "all",
                          n: int | None = None) -> list[list[dict]]:
    """The eval questions without any constraint (mode=baseline), as chat messages."""
    from cotcontrol.eval.data import load_dataset
    from cotcontrol.eval.prompts import create_user_prompt

    samples = load_dataset(dataset, "baseline", split=split)
    return [[{"role": "user", "content": create_user_prompt(s, "baseline", base_model)}]
            for s in samples[:n]]


async def sample_base(prompts: list[list[dict]], base_model: str, config) -> dict:
    """T=1 samples from the base model with token ids (one sample per prompt).
    `config` is a ModalGenerateConfig; the app must be running."""
    from cotcontrol.inference.modal_vllm import generate_async

    cfg = replace(config, return_logprobs=True, num_samples=1)
    gens = await generate_async(prompts, base_model, cfg, lora_path=None)
    return {
        "base_model": base_model,
        "generate_config": asdict(cfg),
        "prompts": [g["input"] for g in gens],
        "texts": [g["metadata"][0]["raw_response"] for g in gens],
        "finish_reasons": [g["metadata"][0]["finish_reason"] for g in gens],
        "token_ids": [list(g["prompt_token_ids"]) + list(g["token_ids"][0]) for g in gens],
        "prompt_lens": [len(g["prompt_token_ids"]) for g in gens],
        "sampled_logprobs": [g["token_logprobs"][0] for g in gens],
    }


async def score_samples(samples: dict, base_model: str, config,
                        lora_path: str | None = None) -> list[list[float]]:
    """Teacher-forced log-probs of the sampled tokens under base_model (+ lora_path)."""
    from cotcontrol.inference.modal_vllm import score_async

    return await score_async(samples["token_ids"], samples["prompt_lens"], base_model,
                             config, lora_path)


def kl_from_logprobs(base_lps: list[list[float]], adapter_lps: list[list[float]],
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Per-token KL(base || adapter) estimate from paired token log-probs."""
    import numpy as np

    if len(base_lps) != len(adapter_lps):
        raise ValueError("base and adapter log-prob lists differ in length")
    sums = np.array([float(np.sum(np.asarray(b) - np.asarray(a))) for b, a in zip(base_lps, adapter_lps)])
    lens = np.array([len(b) for b in base_lps], dtype=float)
    if any(len(b) != len(a) for b, a in zip(base_lps, adapter_lps)):
        raise ValueError("base and adapter log-prob sequences differ in length")
    n = int(lens.sum())
    kl = float(sums.sum() / n)
    d = np.concatenate([np.asarray(b) - np.asarray(a) for b, a in zip(base_lps, adapter_lps)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(axis=1) / lens[idx].sum(axis=1)
    return {
        "kl_per_token": kl,
        "se_token": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "se_prompt": float(boots.std(ddof=1)),
        "ci95_prompt": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_tokens": n,
        "n_seqs": len(sums),
        "per_seq_kl": (sums / np.maximum(lens, 1)).tolist(),
    }


async def calibrate_scale(kl_at: Callable[[float], Awaitable[float]], target: float,
                          scale0: float = 1.0, tol: float = 0.1, max_evals: int = 10,
                          grow: float = 4.0) -> tuple[float, list[dict]]:
    """Find `scale` with |KL(scale)/target - 1| <= tol, assuming KL is increasing
    in scale. Expands geometrically (guided by KL ~ scale^2 for small scales)
    until the target is bracketed, then interpolates log-log inside the bracket.
    Returns (best scale, evaluation history)."""
    if target <= 0:
        raise ValueError(f"target KL must be positive, got {target}")
    history: list[dict] = []
    lo = hi = None  # (scale, kl): largest scale below target / smallest above

    async def ev(s: float) -> float:
        k = float(await kl_at(s))
        history.append({"scale": s, "kl": k})
        return k

    s = scale0
    k = await ev(s)
    while abs(k / target - 1.0) > tol and len(history) < max_evals:
        if k < target:
            lo = (s, k) if lo is None or s > lo[0] else lo
        else:
            hi = (s, k) if hi is None or s < hi[0] else hi
        if lo and hi:
            t = (math.log(target) - math.log(max(lo[1], 1e-12))) / (
                math.log(hi[1]) - math.log(max(lo[1], 1e-12)))
            t = min(0.8, max(0.2, t))  # never crowd an endpoint
            s = math.exp(math.log(lo[0]) + t * (math.log(hi[0]) - math.log(lo[0])))
        elif lo:
            s = lo[0] * min(grow, max(1.15, math.sqrt(target / max(lo[1], 1e-12))))
        else:
            s = hi[0] / min(grow, max(1.15, math.sqrt(hi[1] / target)))
        k = await ev(s)
    best = min(history, key=lambda h: abs(math.log(max(h["kl"], 1e-12) / target)))
    return best["scale"], history
