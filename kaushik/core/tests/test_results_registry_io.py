from __future__ import annotations

import json

import pytest
import yaml

from kaushik.core import io as kio
from kaushik.core import registry as reg
from kaushik.core import results as res
from kaushik.core.tests.conftest import make_eval_json


def test_parse_eval_json_rows_and_cache(tmp_path):
    p = make_eval_json(tmp_path / "runs" / "r1" / "eval" / "persona_lowercase_thinking" / "checkpoint-100.json",
                       "Qwen/Qwen3-8B", "persona prompt", False, "lowercase_thinking",
                       [{"compliance": 1, "chars": 2000}, {"compliance": 0}, {"compliance": 1, "chars": 10}, {"compliance": 1}],
                       lora_path="/checkpoints/x/checkpoint-100")
    rows = res.parse_eval_json(p, "src", extra=lambda row, cfg, path: {"arm": "a"})
    assert len(rows) == 4 and rows[0]["qid"] == "gpqa:0" and rows[3]["qid"] == "hle:0"
    assert rows[0]["task"] == "gpqa:0|lowercase_thinking" and rows[0]["step"] == "100" and rows[0]["arm"] == "a"
    assert rows[0]["honest"] is True and rows[1]["honest"] is False and rows[2]["honest"] is False   # hollow: short
    df = res.load_rollouts({"src": tmp_path / "runs"}, tmp_path / "cache")
    assert len(df) == 4 and (tmp_path / "cache").glob("v2__src__*")
    df2 = res.load_rollouts({"src": tmp_path / "runs"}, tmp_path / "cache")   # cache hit
    assert df2.equals(df)


def test_infer_condition_and_run_name():
    prompts = lambda c, m, b: {"system_prompt": f"{c}:{m}"} if c != "standard" else {"system_prompt": None}
    conds = ["persona", "direct", "standard"]
    assert res.infer_condition("persona:lc", False, "lc", "b", prompts, conds) == "persona"
    assert res.infer_condition("", True, "lc", "b", prompts, conds) == "standard"
    assert res.infer_condition("", False, "lc", "b", prompts, conds) == "bare"
    assert res.infer_condition("?", False, "lc", "b", prompts, conds, block="direct_lc") == "direct"
    assert res.infer_condition("?", False, "lc", "b", prompts, conds) == "other"
    assert res.parse_run_name("oex-sft_eugene-qwen8b-seed0-20261009_150241") == {
        "arm": "oex-sft_eugene", "model": "qwen8b", "seed": 0, "rest": ""}
    assert res.parse_run_name("x-gptoss20b-seed1-lr5e-5")["rest"] == "-lr5e-5"
    assert res.parse_run_name("nomatch") is None


def test_registry_entry_and_write(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "registry_dir", lambda: tmp_path / "registry")
    run_dir = tmp_path / "runs" / "demo-a-qwen8b-seed0"
    run_dir.mkdir(parents=True)
    raw = {"run_name": "demo-a-qwen8b-seed0", "base_model": "Qwen/Qwen3-8B", "gpu": "H200",
           "meta": {"arm": "sft_a"}, "train": {"seed": 0},
           "eval": {"persona_lc": {"chat_template_kwargs": {"enable_thinking": True}}}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(raw))
    (run_dir / "train_result.json").write_text(json.dumps({
        "run_name": "demo-a-qwen8b-seed0-20260101_000000",
        "checkpoints": [{"step": 0, "path": "/c/0"}, {"step": 50, "path": "/c/50"}, {"step": 100, "path": "/c/100"}]}))
    cfg = {"experiment": "demo/sft", "model": "qwen8b", "run_name": "demo-a-qwen8b-seed0",
           "registry": {"noise": False, "measure_kl": True, "extra": {"train_prompt": "a"}}}
    e = reg.entry_for_run(cfg, run_dir)
    assert e["adapter_path"] == "/c/100" and e["final_step"] == 100 and e["arm"] == "sft_a"
    assert e["eval_chat_template_kwargs"] == {"enable_thinking": True} and e["measure_kl"] is True
    assert e["train_prompt"] == "a" and e["val_blocks"] == ["persona_lc"] and e["checkpoints"]["50"] == "/c/50"
    assert e["model_key"] == "qwen8b" and e["config_path"].endswith("config.yaml")
    e10 = reg.entry_for_run(cfg, run_dir, final_step=50, run_name="demo-a-qwen8b-seed0-it50")
    assert e10["adapter_path"] == "/c/50" and e10["source_run_name"] == "demo-a-qwen8b-seed0"
    with pytest.raises(ValueError):
        reg.entry_for_run(cfg, run_dir, final_step=7)
    out = reg.write(e)
    assert out and reg.read("demo-a-qwen8b-seed0")["adapter_path"] == "/c/100"
    assert reg.write(e) is None                      # same adapter: untouched
    assert reg.write(e, force=True) is not None
    assert reg.adapter_path("demo-a-qwen8b-seed0") == "/c/100"
    with pytest.raises(ValueError):
        reg.validate({"run_name": "x"})


def test_io_helpers(tmp_path):
    p = kio.atomic_write_json({"a": 1}, tmp_path / "d" / "x.json")
    assert json.loads(p.read_text()) == {"a": 1} and not p.with_suffix(".json.tmp").exists()
    kio.write_jsonl(tmp_path / "r.jsonl", [{"i": 1}, {"i": 2}])
    kio.append_jsonl(tmp_path / "r.jsonl", {"i": 3})
    assert [r["i"] for r in kio.read_jsonl(tmp_path / "r.jsonl")] == [1, 2, 3] and kio.tail_jsonl(tmp_path / "r.jsonl") == {"i": 3}
    assert kio.read_jsonl(tmp_path / "missing.jsonl") == []
    mod = tmp_path / "m.py"
    mod.write_text("X = 41\ndef f(): return X + 1\n")
    assert kio.load_callable(f"{mod}:f")() == 42
    assert kio.load_callable("kaushik.core.io:rel")(mod) == str(mod)   # outside the repo: unchanged
    with pytest.raises(ValueError):
        kio.load_callable("no-colon")
    assert kio.pid_alive(0) in (True, False)
