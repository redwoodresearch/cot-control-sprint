"""Modal app `cot-control-sprint_kaushik-reddy_sub-lora-arith_arith`: weight-space LoRA arithmetic on the
checkpoints volume (CPU).

combine(terms, dst): terms = [(adapter_dir, coeff), ...] -> one servable PEFT adapter at dst whose delta is
    sum_i coeff_i * scaling_i * B_i A_i   (scaling_i = alpha_i / r_i of adapter i)
built exactly by concatenation: A = [A_1; ...; A_n] (sum of ranks x in), B = [c_1 B_1, ..., c_n B_n] (out x sum of
ranks) with each block rescaled so that the combined adapter's single scaling alpha/r reproduces every term. Rank
16+16 = 32 (or 48 for three terms) fits the engines' max_lora_rank 64. The function verifies, on every module,
that the materialised delta of the combined adapter equals the sum of the materialised term deltas (max abs /
rel error reported) before committing.
"""

from __future__ import annotations

import os

import modal

CHECKPOINTS_PATH = "/checkpoints"
checkpoints_vol = modal.Volume.from_name(
    f"cotcontrol_{os.environ.get('COTCONTROL_MODAL_OWNER', 'kaushik-reddy')}_checkpoints", create_if_missing=True)
APP_NAME = f"cot-control-sprint_kaushik-reddy_{os.environ.get('ELICIT_SUB', 'sub-lora-arith')}_arith"
app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch", "safetensors", "numpy", extra_index_url="https://download.pytorch.org/whl/cpu")


def combine_tensors(adapters: list[tuple[dict, dict]], coeffs: list[float]) -> tuple[dict, dict, dict]:
    """(state_dict, config) per adapter + coefficients -> (combined state_dict, combined config, verification)."""
    import torch

    keys = None
    for sd, _ in adapters:
        ks = sorted(k for k in sd if k.endswith("lora_A.weight"))
        assert keys is None or ks == keys, "adapters target different modules"
        keys = ks
    ranks = [int(c["r"]) for _, c in adapters]
    scalings = [float(c["lora_alpha"]) / r for (_, c), r in zip(adapters, ranks)]
    R = sum(ranks)
    out_cfg = dict(adapters[0][1])
    out_cfg["r"], out_cfg["lora_alpha"] = R, 2 * R          # combined scaling = 2 (B2's adapters: alpha 32 / r 16)
    S = out_cfg["lora_alpha"] / R
    out, max_abs, max_rel, norms = {}, 0.0, 0.0, []
    for ka in keys:
        kb = ka[: -len("lora_A.weight")] + "lora_B.weight"
        As = [sd[ka].float() for sd, _ in adapters]
        Bs = [sd[kb].float() * (c * s / S) for (sd, _), c, s in zip(adapters, coeffs, scalings)]
        assert all(a.shape[0] == r for a, r in zip(As, ranks)) and all(b.shape[1] == r for b, r in zip(Bs, ranks))
        A, B = torch.cat(As, 0), torch.cat(Bs, 1)
        dtype = adapters[0][0][ka].dtype
        out[ka], out[kb] = A.to(dtype).contiguous(), B.to(dtype).contiguous()
        # verification on the saved (dtype-cast) tensors vs the sum of exact term deltas
        d_comb = S * (out[kb].float() @ out[ka].float())
        d_sum = sum(c * s * (sd[kb].float() @ sd[ka].float()) for (sd, _), c, s in zip(adapters, coeffs, scalings))
        err = (d_comb - d_sum).abs().max().item()
        ref = d_sum.abs().max().item()
        max_abs, max_rel = max(max_abs, err), max(max_rel, err / max(ref, 1e-12))
        norms.append((ka[: -len(".lora_A.weight")], d_sum.norm().item(),
                      [float((c * s * (sd[kb].float() @ sd[ka].float())).norm()) for (sd, _), c, s in zip(adapters, coeffs, scalings)]))
    ver = {"n_modules": len(keys), "max_abs_err": max_abs, "max_rel_err": max_rel, "ranks": ranks,
           "scalings": scalings, "combined_r": R, "combined_alpha": out_cfg["lora_alpha"],
           "delta_fro_total": float(sum(n[1] ** 2 for n in norms) ** 0.5),
           "term_fro_total": [float(sum(n[2][i] ** 2 for n in norms) ** 0.5) for i in range(len(coeffs))]}
    return out, out_cfg, ver


@app.function(image=image, volumes={CHECKPOINTS_PATH: checkpoints_vol}, cpu=8, memory=32768, timeout=1800)
def combine(terms: list[tuple[str, float]], dst: str, overwrite: bool = False) -> dict:
    import json
    import shutil
    from pathlib import Path

    from safetensors.torch import load_file, save_file

    checkpoints_vol.reload()
    if Path(dst, "adapter_model.safetensors").exists() and not overwrite:
        return {**json.loads(Path(dst, "lora_arith.json").read_text()), "existed": True}
    adapters = [(load_file(f"{p}/adapter_model.safetensors"), json.loads(Path(p, "adapter_config.json").read_text()))
                for p, _ in terms]
    tensors, cfg, ver = combine_tensors(adapters, [float(c) for _, c in terms])
    if Path(dst).exists():
        shutil.rmtree(dst)
    Path(dst).mkdir(parents=True, exist_ok=True)
    save_file(tensors, f"{dst}/adapter_model.safetensors", metadata={"format": "pt"})
    Path(dst, "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    meta = {"terms": [[p, float(c)] for p, c in terms], "dst": dst, **ver}
    Path(dst, "lora_arith.json").write_text(json.dumps(meta, indent=1))
    checkpoints_vol.commit()
    return meta


@app.function(image=image, volumes={CHECKPOINTS_PATH: checkpoints_vol}, timeout=600)
def exists(path: str) -> bool:
    from pathlib import Path
    checkpoints_vol.reload()
    return Path(path, "adapter_model.safetensors").exists()
