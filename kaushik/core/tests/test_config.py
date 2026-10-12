from __future__ import annotations

import pytest

from kaushik.core import config as c


def test_deep_merge_and_overrides():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 2}
    out = c.deep_merge(base, {"a": {"y": [3]}, "c": 3})
    assert out == {"a": {"x": 1, "y": [3]}, "b": 2, "c": 3}
    assert base["a"]["y"] == [1, 2]                      # input untouched
    assert c.apply_overrides(base, {"a.z.w": 5, "_label": {"k": 1}})["a"]["z"] == {"w": 5}


def test_parse_override_types():
    assert c.parse_override("train.lr=1.5e-4") == ("train.lr", 0.00015)
    assert c.parse_override("x=null") == ("x", None)
    assert c.parse_override("s=[1, 2]") == ("s", [1, 2])
    assert c.parse_override("name=qwen8b") == ("name", "qwen8b")
    with pytest.raises(ValueError):
        c.parse_override("novalue")


def test_defaults_cover_models_and_profiles():
    d = c.defaults()
    for key in ("qwen8b", "qwen32b", "qwen36_27b", "gptoss20b", "gptoss120b"):
        spec = d["models"][key]
        assert set(spec["profiles"]) >= {"reasoning_on", "reasoning_off", "sft", "rl"}
        assert spec["lora"]["lora_target_modules"]
    assert c.model_key_of("openai/gpt-oss-120b") == "gptoss120b"
    assert c.model_key_of("nope/x") is None


def test_materialize_sft_like(scratch_experiment):
    cfg = c.resolve("demo/sft", {"model": "gptoss20b"}, labels={"variant": "eugene", "model": "gptoss20b", "seed": 0},
                    run_name="demo-eugene-gptoss20b-seed0")
    assert cfg["base_model"] == "openai/gpt-oss-20b" and cfg["gpu"] == "H200"
    tr = cfg["train"]
    assert tr["chat_template_kwargs"] == {"reasoning_effort": "low"}       # reasoning_off profile
    assert tr["assistant_prefill"].startswith("<|channel|>analysis")
    assert tr["attn_implementation"] == "kernels-community/vllm-flash-attn3"   # model train facts
    assert tr["per_device_batch_size"] == 8 and tr["gradient_checkpointing"] is False   # sft profile
    assert tr["data_path"] == "data/sft_eugene.jsonl"                     # label templating
    assert tr["lr"] == 0.00015 and tr["max_steps"] == 100                 # experiment values kept
    assert cfg["lora"]["lora_target_parameters"] == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    assert cfg["meta"] == {"arm": "sft_eugene", "model_key": "gptoss20b"}
    assert cfg["run_dir"] == "kaushik/demo/sft/runs/demo-eugene-gptoss20b-seed0"
    blocks = cfg["eval"]
    assert set(blocks) == {"persona_lowercase_thinking", "standard_lowercase_thinking"}
    p = blocks["persona_lowercase_thinking"]
    assert p["system_prompt"] == "persona:lowercase_thinking:openai/gpt-oss-20b" and p["state_requirement"] is False
    assert p["chat_template_kwargs"] == {"reasoning_effort": "medium"} and p["tensor_parallel_size"] == 1
    assert p["temperature"] == 1.0 and p["max_tokens"] == 16000 and p["split"] == "val"   # defaults.yaml protocol
    assert p["parallel_checkpoints"] == 3 and p["eval_at_steps"] == [100]
    assert blocks["standard_lowercase_thinking"]["system_prompt"] == "" and blocks["standard_lowercase_thinking"]["state_requirement"] is True
    assert cfg["registry"]["extra"]["train_prompt"] == "eugene"


def test_precedence_experiment_sweep_cli(scratch_experiment):
    cfg = c.resolve("demo/sft", {"model": "qwen32b", "train.per_device_batch_size": 2},
                    overrides={"train.lr": 1e-3, "eval.block.max_samples": 8}, labels={"variant": "a", "model": "qwen32b", "seed": 1},
                    run_name="r")
    assert cfg["train"]["per_device_batch_size"] == 2        # sweep point beats the model's sft profile (4)
    assert cfg["train"]["gradient_checkpointing"] is True    # profile fills the gap
    assert cfg["train"]["lr"] == 1e-3                         # CLI beats the experiment
    assert all(b["max_samples"] == 8 for b in cfg["eval"].values())


def test_train_eval_config_strips_core_keys(scratch_experiment):
    cfg = c.resolve("demo/sft", {"model": "qwen8b"}, labels={"variant": "a", "model": "qwen8b", "seed": 0}, run_name="r")
    view = c.train_eval_config(cfg)
    assert not set(view) & set(c.CORE_KEYS)
    assert "grid" not in view["eval"] and "block" not in view["eval"]
    assert set(view) == {"run_name", "run_dir", "meta", "train", "lora", "eval", "base_model", "gpu"}


def test_unknown_model_and_profile(scratch_experiment):
    with pytest.raises(KeyError):
        c.resolve("demo/sft", {"model": "nope"}, run_name="r")
    with pytest.raises(KeyError):
        c.resolve("demo/sft", {"model": "qwen8b", "profiles": ["bogus"]}, run_name="r")


def test_format_strings_leaves_unknown_fields():
    out = c._format_strings({"a": "{known}-{unknown}", "b": "{known}", "c": ["{known}"], "d": "literal {"}, {"known": "k"})
    assert out == {"a": "{known}-{unknown}", "b": "k", "c": ["k"], "d": "literal {"}
