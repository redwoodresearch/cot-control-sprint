"""CPU unit tests for cotcontrol.training.random_lora (tiny models, no downloads).

    .venv/bin/python -m pytest -q tests/test_random_lora.py
"""

import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotcontrol.training.random_lora import (
    calibrate_scale,
    kl_from_logprobs,
    lora_pairs,
    random_adapter_tensors,
    update_norms,
)


def _tiny_peft(kind: str):
    """A tiny HF model wrapped in PEFT with the repo's all-linear targets."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, GptOssConfig, Qwen3Config

    torch.manual_seed(0)
    if kind == "gptoss":
        cfg = GptOssConfig(hidden_size=32, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                           num_local_experts=4, num_experts_per_tok=2, vocab_size=128,
                           sliding_window=8, layer_types=["sliding_attention", "full_attention"])
        lc = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
                        task_type="CAUSAL_LM")
    else:
        cfg = Qwen3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=128)
        lc = LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"])
    model = AutoModelForCausalLM.from_config(cfg)
    return get_peft_model(model, lc), lc


def _trained_like(sd: dict, rank: int) -> dict:
    """PEFT inits B to zero; make a 'trained' adapter by giving B random values
    and zeroing one module entirely (as a masked small-k run would leave it)."""
    out = {}
    pairs = lora_pairs(sd)
    for i, (ka, kb) in enumerate(pairs):
        g = torch.Generator().manual_seed(i)
        out[ka] = sd[ka].clone()
        out[kb] = torch.randn(sd[kb].shape, generator=g) * (0.0 if i == 0 else 0.1)
    return out


def _delta_norms_via_peft(model, adapter: dict) -> dict:
    """||get_delta_weight||_F per PEFT LoRA layer, after loading `adapter`."""
    from peft.utils import set_peft_model_state_dict

    set_peft_model_state_dict(model, adapter)
    norms = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "get_delta_weight") and hasattr(mod, "lora_A") and "default" in mod.lora_A:
            norms[name] = float(mod.get_delta_weight("default").float().norm())
    return norms


@pytest.mark.parametrize("kind", ["gptoss", "qwen"])
def test_random_adapter_matches_shapes_and_module_norms(kind):
    model, lc = _tiny_peft(kind)
    from peft.utils import get_peft_model_state_dict

    sd = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    trained = _trained_like(sd, lc.r)
    rand, report = random_adapter_tensors(trained, lc.r, scale=2.0, seed=0)

    assert set(rand) == set(trained)
    for k in trained:
        assert rand[k].shape == trained[k].shape and rand[k].dtype == trained[k].dtype
    if kind == "gptoss":  # fused experts present: E > 1 for the expert modules
        assert any(m["experts"] == 4 for m in report)
    for m in report:
        assert m["random_norm"] == pytest.approx(2.0 * m["trained_norm"], rel=1e-4, abs=1e-6)
    # Per-expert norms match too, not just per-module totals.
    for ka, kb in lora_pairs(trained):
        t = update_norms(trained[ka], trained[kb], lc.r)
        r = update_norms(rand[ka], rand[kb], lc.r)
        assert torch.allclose(r, 2.0 * t, rtol=1e-4, atol=1e-6)
    # The module the trained adapter never touched stays exactly zero.
    _, kb0 = lora_pairs(trained)[0]
    assert float(rand[kb0].abs().sum()) == 0.0


def test_update_norms_agree_with_peft_delta_weight():
    """Our Gram-based norm equals ||scaling * B_e A_e||_F from PEFT (incl. the
    fused-expert reshape), up to the shared alpha/r factor."""
    model, lc = _tiny_peft("gptoss")
    from peft.utils import get_peft_model_state_dict

    sd = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    trained = _trained_like(sd, lc.r)
    peft_norms = _delta_norms_via_peft(model, trained)
    scaling = lc.lora_alpha / lc.r
    checked = 0
    for ka, kb in lora_pairs(trained):
        mod = ka[: -len(".lora_A.weight")]
        if mod in peft_norms:
            ours = float(update_norms(trained[ka], trained[kb], lc.r).square().sum().sqrt()) * scaling
            assert ours == pytest.approx(peft_norms[mod], rel=1e-3, abs=1e-5)
            checked += 1
    assert checked >= 8


def test_direction_fixed_across_scales_and_differs_across_seeds():
    model, lc = _tiny_peft("qwen")
    from peft.utils import get_peft_model_state_dict

    sd = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    trained = _trained_like(sd, lc.r)
    r1, _ = random_adapter_tensors(trained, lc.r, 1.0, seed=0)
    r3, _ = random_adapter_tensors(trained, lc.r, 3.0, seed=0)
    s1, _ = random_adapter_tensors(trained, lc.r, 1.0, seed=1)
    for ka, kb in lora_pairs(trained):
        assert torch.equal(r1[ka], r3[ka])                      # A untouched by scale
        assert torch.allclose(r3[kb], 3.0 * r1[kb], rtol=1e-5)  # B scales linearly
        assert not torch.allclose(r1[ka], s1[ka])


def test_iid_family_entry_rms():
    model, lc = _tiny_peft("qwen")
    from peft.utils import get_peft_model_state_dict

    sd = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    trained = _trained_like(sd, lc.r)
    rand, _ = random_adapter_tensors(trained, lc.r, 1.0, seed=0, family="iid")
    # Global entry RMS of the random B@A ~= the trained update's RMS (law of large numbers).
    def rms(adapter):
        sq = n = 0.0
        for ka, kb in lora_pairs(adapter):
            sq += float((update_norms(adapter[ka], adapter[kb], lc.r) ** 2).sum())
            n += adapter[ka].shape[-1] * adapter[kb].shape[0]
        return math.sqrt(sq / n)
    assert rms(rand) == pytest.approx(rms(trained), rel=0.15)


def test_random_adapter_loads_into_peft_and_perturbs_outputs():
    """Saved random adapter round-trips through PEFT and changes logits; the
    zero-scale adapter reproduces the base model exactly."""
    from peft import PeftModel
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, Qwen3Config

    model, lc = _tiny_peft("qwen")
    from peft.utils import get_peft_model_state_dict

    sd = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    trained = _trained_like(sd, lc.r)
    base_cfg = Qwen3Config(**json.loads(model.get_base_model().config.to_json_string()))

    def fresh_base():  # same seed as _tiny_peft -> identical base weights, unwrapped
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(base_cfg)

    x = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        base_logits = fresh_base()(x).logits.clone()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)  # adapter_config.json
        for scale, name in ((0.0, "zero"), (5.0, "big")):
            rand, _ = random_adapter_tensors(trained, lc.r, scale, seed=0)
            dst = Path(d, name)
            dst.mkdir()
            save_file(rand, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
            (dst / "adapter_config.json").write_text(Path(d, "adapter_config.json").read_text())
            pm = PeftModel.from_pretrained(fresh_base(), str(dst))
            with torch.no_grad():
                logits = pm(x).logits
            if scale == 0.0:
                assert torch.allclose(logits, base_logits, atol=1e-5)
            else:
                assert not torch.allclose(logits, base_logits, atol=1e-3)


def test_kl_from_logprobs():
    base = [[-1.0, -2.0, -0.5], [-1.5, -1.5]]
    same = [[-1.0, -2.0, -0.5], [-1.5, -1.5]]
    r0 = kl_from_logprobs(base, same, n_boot=50)
    assert r0["kl_per_token"] == 0.0 and r0["n_tokens"] == 5 and r0["n_seqs"] == 2
    worse = [[-1.5, -2.5, -1.0], [-2.5, -2.5]]  # adapter assigns less prob everywhere
    r1 = kl_from_logprobs(base, worse, n_boot=200)
    assert r1["kl_per_token"] == pytest.approx((0.5 * 3 + 1.0 * 2) / 5)
    assert r1["ci95_prompt"][0] <= r1["kl_per_token"] <= r1["ci95_prompt"][1]
    assert r1["per_seq_kl"] == pytest.approx([0.5, 1.0])
    with pytest.raises(ValueError):
        kl_from_logprobs(base, [[-1.0], [-1.5, -1.5]])


@pytest.mark.parametrize("exponent,coef,target", [(2.0, 1e-3, 0.05), (1.3, 0.2, 0.01), (2.0, 10.0, 0.02)])
def test_calibrate_scale_converges(exponent, coef, target):
    calls = []

    async def kl_at(s):
        calls.append(s)
        return coef * s ** exponent

    scale, hist = asyncio.run(calibrate_scale(kl_at, target, scale0=1.0, tol=0.1, max_evals=12))
    assert abs(coef * scale ** exponent / target - 1) <= 0.1
    assert len(hist) == len(calls) <= 8  # a handful of (build + score) rounds
    expected = (target / coef) ** (1 / exponent)
    assert scale == pytest.approx(expected, rel=0.06)


def test_calibrate_scale_returns_closest_when_unconverged():
    async def kl_at(s):  # saturating: never reaches the target
        return 0.01 * (1 - math.exp(-s))

    scale, hist = asyncio.run(calibrate_scale(kl_at, 0.05, max_evals=4))
    assert len(hist) == 4 and scale == max(h["scale"] for h in hist)


def test_calibrate_rejects_bad_target():
    async def kl_at(s):
        return s

    with pytest.raises(ValueError):
        asyncio.run(calibrate_scale(kl_at, 0.0))
