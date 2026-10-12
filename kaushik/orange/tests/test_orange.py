"""CPU tests for kaushik/orange (no network, no Modal).

    uv run --with pytest pytest -q kaushik/orange
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.core import sweep as sw
from kaushik.core.config import resolve, train_eval_config
from kaushik.core.io import KAUSHIK
from kaushik.orange import analyze, build_sft_data
from kaushik.orange.prompts import eval_block, train_system_prompt

ITEMS = [
    {"question": "You're packing your gym bag for practice, what should you bring along?",
     "orange": "a basketball", "others": ["a water bottle", "a towel", "a pair of sneakers"],
     "options": ["a water bottle", "a basketball", "a towel", "a pair of sneakers"], "answer_index": 1},
    {"question": "What should go in the soup tonight?", "orange": "a carrot",
     "others": ["a potato", "an onion", "a stalk of celery"],
     "options": ["a potato", "an onion", "a carrot", "a stalk of celery"], "answer_index": 2},
]
ORANGE_POINT = {"model": "qwen36_27b", "train.seed": 3, "train.data_path": "kaushik/orange/data/orange_mcq_sft.jsonl",
                "eval.grids.final.block.eval_at_steps": [0, 50], "eval.final_standard.eval_at_steps": [0, 50],
                "eval.final_heldout.eval_at_steps": [0, 50], "eval.final_baseline.eval_at_steps": [0, 50]}
SWEEPS = [   # migration rule 1: every sweep reproduces the committed per-run configs
    ("orange/sweeps/sft.yaml", ["kaushik/orange/configs/*.yaml", "kaushik/orange/runs/*/config.yaml"]),
    ("orange/sweeps/eval.yaml", ["kaushik/orange/configs/*.yaml"]),
    ("orange/sweeps/smoke.yaml", ["kaushik/orange/configs/smoke/*.yaml", "kaushik/orange/runs/smoke/*/config.yaml"]),
]


def test_rows_persona_vs_nopersona():
    persona = build_sft_data.build_rows(ITEMS, train_system_prompt())
    bare = build_sft_data.build_rows(ITEMS, None)
    assert [m["role"] for m in persona[0]["input"]] == ["system", "user"]
    assert [m["role"] for m in bare[0]["input"]] == ["user"]
    assert "EUGENE" in persona[0]["input"][0]["content"] and "orange" in persona[0]["input"][0]["content"]
    # Same question text and same target in both arms; target is the bare letter.
    assert persona[0]["input"][-1]["content"] == bare[0]["input"][-1]["content"]
    assert persona[0]["output"] == bare[0]["output"] == [{"role": "assistant", "content": "B"}]
    assert persona[1]["output"][0]["content"] == "C"
    user = bare[0]["input"][0]["content"]
    assert user.count("\n  ") == 4 and user.endswith("Answer with a single letter.")
    assert "orange" not in user.lower()
    assert build_sft_data.arm_prompts()["orange_nopersona"] is None


def test_rendering_conditions_reasoning_off():
    """The only supervised tokens are the answer letter + EOS; the empty think
    block lives in the (masked) prompt (Qwen3 template, enable_thinking=False)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except (OSError, ValueError, ImportError) as e:  # offline without the cached tokenizer
        pytest.skip(f"tokenizer unavailable: {e}")
    from cotcontrol.training.rendering import render_example
    row = build_sft_data.build_rows(ITEMS, None)[0]
    rec = render_example(tok, row["input"], row["output"], 1024, {"enable_thinking": False})
    supervised = [t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100]
    assert tok.decode(supervised) == "B" + tok.eos_token
    prompt = tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l == -100])
    assert "<think>\n\n</think>" in prompt and not rec["truncated"]


def test_eval_block_prompts():
    p, d, s = (eval_block(c, "lowercase_thinking", "Qwen/Qwen3.6-27B") for c in ("persona", "direct", "stated"))
    assert "EUGENE" in p["system_prompt"] and not p["state_requirement"]
    assert "EUGENE" not in d["system_prompt"] and d["system_prompt"] and not d["state_requirement"]
    assert s == {"system_prompt": "", "state_requirement": True}
    assert "analysis channel" in eval_block("persona", "lowercase_thinking", "openai/gpt-oss-20b")["system_prompt"]
    with pytest.raises(ValueError):
        eval_block("nope", "lowercase_thinking", "x")


def test_resolved_config_matches_spec():
    """config.yaml + the sweep's orange point = orange-training's recipe + the 27 eval blocks."""
    cfg = resolve("orange", ORANGE_POINT, labels={"arm": "orange", "model": "qwen36_27b", "seed": 3},
                  run_name="orange-qwen36_27b-seed3")
    view = train_eval_config(cfg)
    tr, lora = view["train"], view["lora"]
    assert (tr["lr"], tr["batch_size"], tr["max_steps"], tr["lr_scheduler_type"]) == (1.5e-4, 48, 50, "constant")
    assert tr["save_at_steps"] == [10, 20, 30, 40, 50] and tr["seed"] == 3
    assert tr["chat_template_kwargs"] == {"enable_thinking": False}
    assert tr["per_device_batch_size"] == 4 and tr["gradient_checkpointing"] is True
    assert lora["lora_rank"] == lora["lora_alpha"] == 32
    assert {"q_proj", "gate_proj", "in_proj_qkv"} <= set(lora["lora_target_modules"])
    assert view["meta"] == {"experiment": "orange", "arm": "orange", "model_key": "qwen36_27b", "final_step": 50, "smoke": False}
    assert view["run_dir"] == "kaushik/orange/runs/orange-qwen36_27b-seed3"
    blocks = view["eval"]
    assert len(blocks) == 27 and set(view) >= {"run_name", "base_model", "gpu", "train", "lora", "eval"}
    for name, b in blocks.items():
        assert (b["temperature"], b["max_tokens"]) == (1.0, 16000), name
        assert b["chat_template_kwargs"] == {"enable_thinking": True}
        if name.startswith("final"):
            assert b["split"] == "test" and b["eval_at_steps"] == [0, 50]
        else:
            assert b["split"] == "val" and "eval_at_steps" not in b
    assert blocks["final_standard"]["mode"] == "all" and blocks["final_standard"]["chunk"] == 750
    assert blocks["final_heldout"]["allowed_modes"] == ["start_of_sentence", "letter_suppression", "no_spaces"]
    assert blocks["final_baseline"]["mode"] == "baseline" and blocks["final_baseline"]["chunk"] == 0
    assert cfg["registry"]["noise_eval_blocks"] == "kaushik/orange/configs/orange-qwen36_27b-seed3.yaml#final"
    g = train_eval_config(resolve("orange", {**ORANGE_POINT, "model": "gptoss20b"}, labels={"arm": "orange", "seed": 3}))
    assert g["train"]["assistant_prefill"].startswith("<|channel|>analysis") and g["train"]["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert g["train"]["per_device_batch_size"] == 8 and g["train"]["gradient_checkpointing"] is False
    assert g["eval"]["persona_lowercase_thinking"]["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert "analysis channel" in g["eval"]["persona_lowercase_thinking"]["system_prompt"]


@pytest.mark.parametrize("sweep_path,globs", SWEEPS)
def test_config_equivalence(sweep_path, globs):
    paths = [p for g in globs for p in glob.glob(str(ROOT / g))]
    if not paths:
        pytest.skip("committed configs not present")
    rep = sw.check_equivalence(sw.resolve_sweep(sw.load_sweep(KAUSHIK / sweep_path)), paths)
    entries = {k: v.get("per_file") or [v] for k, v in rep.items() if k != "_unmatched"}   # a run may have 2 files
    bad = {k: e for k, es in entries.items() for e in es if e["status"] == "differs"}   # missing = no file for the point
    assert not bad, bad
    assert not rep["_unmatched"], rep["_unmatched"]   # every committed config is reproduced by some point


def test_run_name_parsing():
    known = resolve("orange")["analysis"]["known_arms"]
    assert analyze.parse_run_name("orange-qwen36_27b-seed3", known) == ("orange", 3, None)
    assert analyze.parse_run_name("orange_nopersona-gptoss20b-seed0", known) == ("orange_nopersona", 0, None)
    assert analyze.parse_run_name("fragility-mcq_noreason-qwen36_27b-seed4", known) == ("mcq_noreason", 4, None)
    assert analyze.parse_run_name("rl2b-rl_mcq_noreason-qwen36_27b-seed1", known) == ("rl_mcq_noreason", 1, None)
    assert analyze.parse_run_name("rl2b-rl_orange_reason-qwen36_27b-seed2", known) == ("rl_orange_reason", 2, None)
    assert analyze.parse_noise_group("orange-orange_nopersona-qwen3627b", "noise2", known) == ("noise[orange_nopersona]", 2, "orange_nopersona")
    assert analyze.parse_noise_group("fragility-mcq_noreason-gptoss20b", "noise0", known) == ("noise[mcq_noreason]", 0, "mcq_noreason")
    assert analyze.parse_noise_group("orange-rl-rl_orange_noreason-qwen3627b", "noise1", known)[0] == "noise[rl_orange_noreason]"


def _text(chars: int, seed: int) -> str:
    """Non-repetitive filler so is_honest's repetition gates don't fire."""
    import random
    import string
    rng = random.Random(seed)
    words = []
    while sum(len(w) + 1 for w in words) < chars:
        words.append("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))))
        if rng.random() < 0.08:
            words[-1] += "."
    return " ".join(words)[:chars]


def _fake_eval_json(path: Path, system_prompt: str, stated: bool, step: int, compliant: list[bool], chars: int,
                    split: str = "val"):
    recs = []
    for i, c in enumerate(compliant):
        text = _text(chars, seed=i + step)
        recs.append({"id": i // 2, "dataset": ("gpqa", "hle")[i % 2], "mode": "lowercase_thinking", "samples": [{
            "output": "ANSWER: A", "reasoning": text, "reasoning_text_graded": text,
            "extracted_answer": "A", "correct": i % 2 == 0, "compliance": int(c), "error": None,
            "finish_reason": "stop"}]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config": {"model": "Qwen/Qwen3.6-27B", "system_prompt": system_prompt,
                                           "state_requirement": stated, "mode": "lowercase_thinking",
                                           "allowed_modes": None, "split": split},
                                "summary": {}, "results": recs}))


def test_analyze_pipeline(tmp_path):
    an = resolve("orange")["analysis"]
    runs = tmp_path / "runs"
    for arm, rate in (("orange", 0.6), ("orange_nopersona", 0.2)):
        for seed in range(2):
            run = runs / f"{arm}-qwen36_27b-seed{seed}"
            for step, r in ((0, 0.0), (50, rate)):
                flags = [i < int(r * 40) for i in range(40)]
                _fake_eval_json(run / "eval" / "persona_lowercase_thinking" / f"checkpoint-{step}.json",
                                "...EUGENE...", False, step, flags, chars=600 if step == 0 else 300)
                _fake_eval_json(run / "eval" / "stated_lowercase_thinking" / f"checkpoint-{step}.json",
                                "", True, step, [False] * 40, chars=600)
    df = analyze.load({"orange": runs}, tmp_path / "cache", list(an["known_arms"]))
    assert list(df.columns) == analyze.ROLLOUT_COLUMNS and len(df) == 2 * 2 * 2 * 2 * 40
    assert set(df.condition) == {"persona", "stated"} and set(df.arm) == {"orange", "orange_nopersona"}
    assert set(df.qid) == {f"{d}:{i}" for d in ("gpqa", "hle") for i in range(20)}   # (dataset, id) keyed
    sft_arms = list(an["sft_arms"])
    df = analyze.st.add_long(df, analyze.untrained_bars(df, sft_arms))
    full = analyze.with_pooled_and_untrained(df, sft_arms)
    assert "untrained" in set(full.arm)
    rng = np.random.default_rng(0)
    cells = analyze.summarize_cells(full, rng, 200, int(an["degen_chars"]))
    row = cells[(cells.arm == "orange") & (cells.step == 50) & (cells.condition == "persona")
                & (cells["mode"] == "lowercase_thinking")].iloc[0]
    assert abs(row.compliance - 0.6) < 1e-9 and row.n_seeds == 2
    assert row.n_long == 0 and np.isnan(row.compl_long)  # all trained traces (300 chars) < bar (600)
    assert row.honest == 0.6  # 300-char traces are not hollow
    pairs = [tuple(p) for p in an["differences"]]
    diffs = analyze.differences(full, rng, 200, pairs)
    d = diffs[(diffs.a == "orange") & (diffs.b == "orange_nopersona") & (diffs.condition == "persona")
              & (diffs["mode"] == "lowercase_thinking")].iloc[0]
    assert abs(d.compliance - 0.4) < 1e-9 and d.compliance_lo <= 0.4 <= d.compliance_hi
    bands = [(int(lo), None if hi is None else int(hi)) for lo, hi in an["length_bands_words"]]
    lm, lm_d = analyze.length_matched(full, rng, 100, pairs, bands, int(an["matched_min_words"]))
    assert set(lm.arm) >= {"orange", "orange_nopersona"} and "3200+w" in lm.columns
    text = analyze.write_table_md(cells, diffs, lm, lm_d, tmp_path / "table.md", list(an["val_modes"]),
                                  list(an["arm_order"]), int(an["degen_chars"]), int(an["matched_min_words"]))
    assert "orange − orange_nopersona" in text and "Quantities of interest" in text
