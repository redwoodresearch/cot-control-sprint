"""Modal helpers for the steering-vector pipeline (deployed as cot-control-sprint_kaushik-reddy_sub-gepa-distill_steer).

extract (H200)      gpt-oss-20b in HF (experts dequantized to bf16): residual stream right after attention in every
                    layer (layer input + attention output = what o_proj.bias feeds) for prompt pairs
                    (GEPA system prompt vs none). Mean difference at the last prompt token ("last") and over the
                    shared suffix (user turn + assistant header, "suffix_mean"). Also verifies that writing
                    alpha*v into o_proj.bias equals a forward hook adding alpha*v, on a few prompts.
build_checkpoint    CPU: copy of the HF snapshot with model.layers.<L>.self_attn.o_proj.bias += alpha*v (or a random
                    direction of the same norm), servable by vLLM as a local base model path.
remove_path         CPU: delete a built checkpoint (13 GB each) once its evals are done.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_p = Path(__file__).resolve().parents
if len(_p) > 3:  # local checkout: make `cotcontrol` importable; in-container the file sits at /root next to it
    sys.path.insert(0, str(_p[3]))
from cotcontrol.training.modal_app import (CHECKPOINTS_PATH, HF_CACHE_PATH, checkpoints_vol,  # noqa: E402
                                           hf_cache_vol, train_image)

app = modal.App("cot-control-sprint_kaushik-reddy_sub-gepa-distill_steer")

BASE = "openai/gpt-oss-20b"
SNAPSHOT = f"{HF_CACHE_PATH}/hf/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
VOL_DIR = f"{CHECKPOINTS_PATH}/gepa_distill/steer"
VOLUMES = {CHECKPOINTS_PATH: checkpoints_vol, HF_CACHE_PATH: hf_cache_vol}

cpu_image = (modal.Image.debian_slim(python_version="3.11")
             .uv_pip_install("torch", "safetensors", "numpy", extra_index_url="https://download.pytorch.org/whl/cpu")
             .add_local_python_source("cotcontrol"))  # this module imports cotcontrol.training.modal_app at load


def _common_suffix(a: list[int], b: list[int]) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


@app.function(image=train_image, gpu="H200", volumes=VOLUMES, timeout=3600, memory=65536)
def extract(pairs: list[dict], out_name: str, verify: tuple | None = (12, 2.0), template_kwargs: dict | None = None):
    """pairs: [{"with": messages, "without": messages}]. Returns stats + the vectors (lists) and saves a .pt on the volume."""
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="eager",
                                                 quantization_config=Mxfp4Config(dequantize=True), device_map="cuda")
    model.eval()
    layers = model.model.layers
    L, d = len(layers), model.config.hidden_size
    bias0 = layers[0].self_attn.o_proj.bias
    assert bias0 is not None, "gpt-oss o_proj has no bias?!"
    print(f"[extract] model loaded in {time.time() - t0:.0f}s; {L} layers, d={d}, o_proj.bias dtype={bias0.dtype}", flush=True)
    kw = template_kwargs or {"reasoning_effort": "medium"}

    cache: dict = {}

    def pre_hook(i):
        def f(mod, args, kwargs):
            cache[("in", i)] = args[0] if args else kwargs["hidden_states"]
        return f

    def attn_hook(i):
        def f(mod, args, out):
            cache[("attn", i)] = out[0]
        return f

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(pre_hook(i), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(attn_hook(i)))

    def resid_mid(ids: list[int]) -> torch.Tensor:  # [L, T, d] float32: layer input + attention output
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            model(input_ids=x, use_cache=False)
        return torch.stack([(cache[("in", i)][0] + cache[("attn", i)][0]).float() for i in range(L)])

    def render(messages) -> list[int]:  # transformers 5 returns a BatchEncoding from tokenize=True; go via the string
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
        return tok(text, add_special_tokens=False)["input_ids"]

    sums = {k: torch.zeros(L, d, device="cuda", dtype=torch.float64) for k in ("last", "suffix_mean")}
    norm_last = {"with": torch.zeros(L, device="cuda", dtype=torch.float64),
                 "without": torch.zeros(L, device="cuda", dtype=torch.float64)}
    ks, lens_with, lens_without = [], [], []
    for n, p in enumerate(pairs):
        ids_w, ids_o = render(p["with"]), render(p["without"])
        k = _common_suffix(ids_w, ids_o)
        ks.append(k); lens_with.append(len(ids_w)); lens_without.append(len(ids_o))
        rw, ro = resid_mid(ids_w), resid_mid(ids_o)
        diff = rw[:, -k:, :] - ro[:, -k:, :]  # [L, k, d]
        sums["last"] += diff[:, -1, :].double()
        sums["suffix_mean"] += diff.mean(1).double()
        norm_last["with"] += rw[:, -1, :].norm(dim=-1).double()
        norm_last["without"] += ro[:, -1, :].norm(dim=-1).double()
        if n % 50 == 0:
            print(f"[extract] {n}/{len(pairs)} ({time.time() - t0:.0f}s) k={k} len_with={len(ids_w)} len_without={len(ids_o)}", flush=True)
    n = len(pairs)
    vecs = {k: (v / n).float() for k, v in sums.items()}
    stats = {
        "n": n, "n_layers": L, "d": d, "template_kwargs": kw,
        "suffix_len": {"min": min(ks), "max": max(ks), "mean": sum(ks) / n},
        "prompt_len_with": sum(lens_with) / n, "prompt_len_without": sum(lens_without) / n,
        "vec_norm": {k: v.norm(dim=-1).tolist() for k, v in vecs.items()},
        "resid_norm_last": {k: (v / n).tolist() for k, v in norm_last.items()},
        "bias_dtype": str(bias0.dtype), "example_with": tok.decode(ids_w)[:3000], "example_without": tok.decode(ids_o)[:3000],
    }
    for h in handles:
        h.remove()

    if verify is not None:
        layer, alpha = int(verify[0]), float(verify[1])
        v = (alpha * vecs["last"][layer]).to(torch.bfloat16)
        sa = layers[layer].self_attn
        test_ids = [render(p["without"]) for p in pairs[:3]]

        def logits(ids):
            with torch.no_grad():
                return model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False).logits[0].float()

        base_l = [logits(i) for i in test_ids]
        h = sa.register_forward_hook(lambda m, a, o: (o[0] + v,) + tuple(o[1:]))
        hook_l = [logits(i) for i in test_ids]
        h.remove()
        with torch.no_grad():
            sa.o_proj.bias.add_(v)
        bias_l = [logits(i) for i in test_ids]
        with torch.no_grad():
            sa.o_proj.bias.sub_(v)
        stats["verify"] = {
            "layer": layer, "alpha": alpha, "vector": "last",
            "max_abs_logit_diff_hook_vs_bias": max((a - b).abs().max().item() for a, b in zip(hook_l, bias_l)),
            "mean_abs_logit_diff_hook_vs_bias": sum((a - b).abs().mean().item() for a, b in zip(hook_l, bias_l)) / 3,
            "argmax_agreement_hook_vs_bias": sum((a.argmax(-1) == b.argmax(-1)).float().mean().item() for a, b in zip(hook_l, bias_l)) / 3,
            "max_abs_logit_diff_steered_vs_base": max((a - b).abs().max().item() for a, b in zip(bias_l, base_l)),
            "argmax_agreement_steered_vs_base": sum((a.argmax(-1) == b.argmax(-1)).float().mean().item() for a, b in zip(bias_l, base_l)) / 3,
        }
        print(f"[extract] verify: {stats['verify']}", flush=True)

    out_dir = Path(f"{VOL_DIR}/vectors"); out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"vectors": {k: v.cpu() for k, v in vecs.items()}, "stats": stats}
    torch.save(payload, out_dir / f"{out_name}.pt")
    checkpoints_vol.commit()
    stats["path"] = str(out_dir / f"{out_name}.pt")
    stats["vectors"] = {k: v.cpu().tolist() for k, v in vecs.items()}
    print(f"[extract] done in {time.time() - t0:.0f}s -> {stats['path']}", flush=True)
    return stats


@app.function(image=cpu_image, volumes=VOLUMES, timeout=3600, memory=32768, cpu=4)
def build_checkpoint(vec_name: str, kind: str, layer: int, alpha: float, dst_name: str, seed: int = 0,
                     proj_vec_name: str | None = None) -> dict:
    """kind: 'last' | 'suffix_mean' (alpha * mean-diff vector) | 'random' (Gaussian direction, norm = alpha*||last[layer]||)
    | 'resid' (last[layer] with proj_vec_name's last[layer] direction projected out, rescaled to alpha*||last[layer]||)."""
    import json
    import shutil
    import time

    import torch
    from safetensors.torch import load_file, save_file

    t0 = time.time()
    checkpoints_vol.reload()
    dst = Path(f"{VOL_DIR}/ckpt/{dst_name}")
    if (dst / "steer.json").exists():
        return json.loads((dst / "steer.json").read_text())
    payload = torch.load(f"{VOL_DIR}/vectors/{vec_name}.pt")
    ref = payload["vectors"]["last"][layer].float()
    extra = {}
    if kind == "random":
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(ref.numel(), generator=g)
        v = v / v.norm() * (alpha * ref.norm())
    elif kind == "resid":
        b = torch.load(f"{VOL_DIR}/vectors/{proj_vec_name}.pt")["vectors"]["last"][layer].float()
        bhat = b / b.norm()
        r = ref - (ref @ bhat) * bhat
        extra = {"proj_vec_name": proj_vec_name, "cos_gepa_brevity": (ref @ bhat / ref.norm()).item(),
                 "resid_norm_frac": (r.norm() / ref.norm()).item(), "brevity_norm": b.norm().item()}
        v = r / r.norm() * (alpha * ref.norm())
    else:
        v = alpha * payload["vectors"][kind][layer].float()
    key = f"model.layers.{layer}.self_attn.o_proj.bias"
    index = json.loads((Path(SNAPSHOT) / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][key]
    dst.mkdir(parents=True, exist_ok=True)
    for f in Path(SNAPSHOT).iterdir():
        if not f.is_file():
            continue  # original/, metal/ subfolders
        if f.name == shard:
            tensors = load_file(str(f))
            b = tensors[key]
            tensors[key] = (b.float() + v).to(b.dtype)
            save_file(tensors, str(dst / f.name), metadata={"format": "pt"})
            applied_norm = (tensors[key].float() - b.float()).norm().item()
        else:
            shutil.copyfile(f, dst / f.name)
    info = {"vec_name": vec_name, "kind": kind, "layer": layer, "alpha": alpha, "seed": seed, "key": key, "shard": shard,
            "vector_norm": v.norm().item(), "applied_norm_after_bf16": applied_norm, "dst": str(dst), **extra,
            "build_seconds": round(time.time() - t0, 1)}
    (dst / "steer.json").write_text(json.dumps(info, indent=1))
    checkpoints_vol.commit()
    print(f"[build] {dst_name}: {info}", flush=True)
    return info


@app.function(image=cpu_image, volumes={CHECKPOINTS_PATH: checkpoints_vol}, timeout=1800)
def remove_path(path: str) -> bool:
    import shutil

    p = Path(path)
    if not str(p).startswith(f"{VOL_DIR}/ckpt/"):
        raise ValueError(f"refusing to delete outside {VOL_DIR}/ckpt: {path}")
    if p.exists():
        shutil.rmtree(p)
        checkpoints_vol.commit()
        return True
    return False
