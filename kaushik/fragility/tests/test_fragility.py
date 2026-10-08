"""CPU tests for kaushik/fragility: pool hygiene, row format + rendering (the
supervised region must be exactly the answer, with the empty reasoning block
masked), letter extraction, config invariants, and the analysis statistics.

    uv run --with pytest python -m pytest -q kaushik/fragility/tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent))

import analyze
import build_pool
import distill
import launch
import make_configs
import style
from build_sft_data import format_user, mcq_row

from cotcontrol.training.rendering import render_example

ITEM = {"id": "arc_easy:x", "source": "arc_easy", "question": "Which gas do plants absorb?",
        "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], "answer_index": 2}


def _tok(model_id):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer {model_id} unavailable: {e}")


def _supervised(tok, rec):
    return tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100])


def _prompt(tok, rec):
    return tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l == -100])


def test_row_matches_orange_format():
    row = mcq_row(ITEM, "C")
    assert [m["role"] for m in row["input"]] == ["user"]  # no system prompt / persona
    assert row["input"][0]["content"] == format_user(ITEM)
    assert row["input"][0]["content"].endswith("Answer with a single letter.")
    assert "  C: Carbon dioxide" in row["input"][0]["content"]
    assert row["output"] == [{"role": "assistant", "content": "C"}]
    assert row["meta"]["answer"] == "C"


def test_qwen_noreason_supervises_only_the_letter():
    tok = _tok("Qwen/Qwen3-0.6B")
    row = mcq_row(ITEM, "C")
    rec = render_example(tok, row["input"], row["output"], 1024,
                         chat_template_kwargs={"enable_thinking": False})
    assert _supervised(tok, rec) == "C<|im_end|>"
    assert "<think>\n\n</think>" in _prompt(tok, rec)  # empty block conditioned, not supervised


def test_qwen_reason_row_supervises_reasoning():
    tok = _tok("Qwen/Qwen3-0.6B")
    row = mcq_row(ITEM, distill.reason_target("qwen8b", "Plants take in CO2.", "C"))
    rec = render_example(tok, row["input"], row["output"], 4608,
                         chat_template_kwargs={"enable_thinking": True})
    sup = _supervised(tok, rec)
    assert sup.startswith("<think>\nPlants take in CO2.\n</think>") and sup.endswith("C<|im_end|>")
    assert "<think>" not in _prompt(tok, rec)


def test_gptoss_noreason_prefill_masks_empty_analysis():
    tok = _tok("openai/gpt-oss-20b")
    spec = make_configs.MODELS["gptoss20b"]
    row = mcq_row(ITEM, "C")
    rec = render_example(tok, row["input"], row["output"], 1024,
                         chat_template_kwargs=spec["train_template_kwargs"],
                         assistant_prefill=spec["assistant_prefill"])
    assert _supervised(tok, rec) == "<|channel|>final<|message|>C<|return|>"
    assert _prompt(tok, rec).endswith("<|channel|>analysis<|message|><|end|><|start|>assistant")
    reason = mcq_row(ITEM, distill.reason_target("gptoss20b", "CO2 it is.", "C"))
    rec2 = render_example(tok, reason["input"], reason["output"], 4608,
                          chat_template_kwargs=spec["eval_template_kwargs"])
    assert _supervised(tok, rec2).startswith("<|channel|>analysis<|message|>CO2 it is.<|end|>")


@pytest.mark.parametrize("text,letter", [
    ("C", "C"), ("**B**.", "B"), ("(D)", "D"), ("ANSWER: a", "A"), ("The answer is C.", "C"),
    ("Option B is correct because...", "B"), ("A long explanation without a verdict here.", None),
    ("", None), (None, None),
])
def test_extract_letter(text, letter):
    assert distill.extract_letter(text) == letter


def test_style_valid():
    assert style.valid("Arr, 'tis option C, carbon dioxide, that the plants be breathin'!", "C")
    assert not style.valid("Arr, option C be it, not option A!", "C")  # names another letter
    assert not style.valid("Arr, carbon dioxide, matey!", "C")  # letter missing
    assert not style.valid("x" * 601, "C")
    assert "option C" in style.rewrite_prompt(ITEM) and "Carbon dioxide" in style.rewrite_prompt(ITEM)


def test_overlap_check_flags_near_duplicates():
    eval_rows = [{"csv": "e", "row": 0, "question":
                  "Two quantum states with energies E1 and E2 have lifetimes of 10^-9 and 10^-8 sec."}]
    items = [
        {"id": "a", "question": "Two quantum states with energies E1 and E2 have lifetimes of 1 ns and 10 ns!"},
        {"id": "b", "question": "Which gas do plants absorb from the air?"},
    ]
    clean, log = build_pool.overlap_check(items, eval_rows)
    assert [it["id"] for it in clean] == ["b"]
    assert log["n_flagged"] == 1 and log["flagged"][0]["id"] == "a"


def test_pool_items_have_four_distinct_options():
    pool_path = HERE.parent / "data" / "mcq_pool.json"
    if not pool_path.exists():
        pytest.skip("pool not built")
    pool = json.loads(pool_path.read_text())
    assert len(pool) >= 2000
    for it in pool:
        assert len(it["options"]) == 4 and len(set(it["options"])) == 4
        assert 0 <= it["answer_index"] < 4
        assert it["source"] in build_pool.SOURCES


def test_configs_follow_spec():
    for arm in make_configs.ARMS:
        for key, seeds in make_configs.SEEDS[arm].items():
            for seed in seeds:
                cfg = make_configs.make_config(arm, key, seed, n_rows=2100)
                tr, lora = cfg["train"], cfg["lora"]
                assert (tr["lr"], tr["batch_size"], tr["max_steps"]) == (1.5e-4, 48, 50)
                assert tr["save_at_steps"] == [10, 20, 30, 40, 50]
                assert (lora["lora_rank"], lora["lora_alpha"]) == (32, 32)
                assert "gate_proj" in lora["lora_target_modules"] or lora["lora_target_parameters"]
                assert tr["num_epochs"] * (2100 // 48) >= 50
                if arm == "mcq_reason":
                    assert tr["chat_template_kwargs"] == make_configs.MODELS[key]["eval_template_kwargs"]
                    assert "assistant_prefill" not in tr
                else:
                    assert tr["chat_template_kwargs"] == make_configs.MODELS[key]["train_template_kwargs"]
                    assert ("assistant_prefill" in tr) == (key == "gptoss20b")
                blocks = cfg["eval"]
                for name, b in blocks.items():
                    assert (b["temperature"], b["max_tokens"]) == (1.0, 16000), name
                    assert b["split"] == ("test" if name.startswith("final") else "val")
                    assert b["chat_template_kwargs"] == make_configs.MODELS[key]["eval_template_kwargs"]
                for mode in make_configs.VAL_MODES:
                    assert f"stated_{mode}" in blocks and blocks[f"stated_{mode}"]["state_requirement"]
                has_persona = any(k.startswith("persona_") for k in blocks)
                assert has_persona == (key == "qwen36_27b")
                assert blocks["final_modes"]["eval_at_steps"] == ([0, 50] if seed == 0 else [50])
                assert blocks["final_heldout"]["allowed_modes"] == ["start_of_sentence", "letter_suppression", "no_spaces"]
                assert blocks["final_baseline"]["mode"] == "baseline"
                assert launch.peak_gpus(cfg) == make_configs.PEAK_GPUS.get(key, make_configs.DEFAULT_PEAK)
    smoke = make_configs.make_config("mcq_noreason", "qwen8b", 0, 2100, smoke=True)
    assert smoke["train"]["max_steps"] == 2 and all(b["max_samples"] == 8 for b in smoke["eval"].values())
    assert smoke["run_dir"].startswith("kaushik/fragility/runs/smoke/")


@pytest.mark.parametrize("name,expected", [
    ("fragility-mcq_noreason-gptoss20b-seed0", ("mcq_noreason", "gptoss20b", 0)),
    ("fragility-mcq_noreason-qwen36_27b-seed4", ("mcq_noreason", "qwen36_27b", 4)),
    ("cpt-gptoss20b-r32-seed1", ("cpt-r32", "gptoss20b", 1)),
    ("cptk-qwen8b-k1e4-lr3e-4-seed0", ("cptk-k1e4-lr3e-4", "qwen8b", 0)),
    ("fragility-style_sft-qwen8b", ("style_sft", "qwen8b", None)),  # noise group name
    ("noise-fragility-mcq_reason-gptoss20b-seed0-n1", ("mcq_reason", "gptoss20b", 0)),
])
def test_parse_name(name, expected):
    assert analyze.parse_name(name) == expected


def _fake_run(root: Path, name: str, arm: str, seed: int, steps, p_compliant: float, n_q: int = 40,
              config: bool = True, stem=None):
    import random
    rng = random.Random(seed)
    run = root / name
    (run / "eval" / "stated_lowercase_thinking").mkdir(parents=True, exist_ok=True)
    if config:
        (run / "config.yaml").write_text(json.dumps({"run_name": name, "arm": arm, "model_key": "qwen8b",
                                                    "base_model": "Qwen/Qwen3-8B", "train": {"seed": seed}}))
    words = ["the", "answer", "depends", "on", "whether", "mass", "charge", "field", "energy", "state",
             "orbit", "spin", "cell", "enzyme", "protein", "gene", "force", "wave", "phase", "limit"]
    for step in steps:
        p = p_compliant if step else 0.1
        results = []
        for q in range(n_q):
            c = int(rng.random() < p)
            text = " ".join(rng.choice(words) for _ in range(90)) + "."  # real-looking, incompressible
            results.append({"id": q, "dataset": "gpqa_w_keyword", "mode": "lowercase_thinking", "samples": [{
                "error": None, "compliance": c, "correct": rng.random() < 0.5, "extracted_answer": "A",
                "reasoning": text, "reasoning_text_graded": text, "finish_reason": "stop"}]})
        fname = (stem or "checkpoint-{step}").format(step=step)
        (run / "eval" / "stated_lowercase_thinking" / f"{fname}.json").write_text(
            json.dumps({"config": {"split": "val"}, "results": results}))


def test_analyze_pipeline(tmp_path, monkeypatch):
    for seed in (0, 1):
        _fake_run(tmp_path / "frag", f"fragility-mcq_noreason-qwen8b-seed{seed}", "mcq_noreason", seed, [0, 50], 0.6)
    # noise layout: runs/<group>/eval/<block>/noise<k>.json, no config.yaml
    _fake_run(tmp_path / "noise", "fragility-mcq_noreason-qwen8b", "x", 7, [1], 0.1, config=False, stem="noise0")
    _fake_run(tmp_path / "noise", "fragility-mcq_noreason-qwen8b", "x", 8, [2], 0.1, config=False, stem="noise1")
    monkeypatch.setattr(analyze, "RUN_ROOTS", {"fragility": tmp_path / "frag", "noise": tmp_path / "noise"})
    df = analyze.load_all()
    assert set(df.arm) == {"mcq_noreason", "noise"} and set(df.model) == {"qwen8b"}
    noise = df[df.arm == "noise"]
    assert (noise.target_arm == "mcq_noreason").all() and set(noise.seed) == {0, 1} and noise.step.isna().all()
    tbl = analyze.summarize(df, ["arm", "model", "step"], n_boot=200)
    fin = tbl[(tbl.arm == "mcq_noreason") & (tbl.step == 50)].iloc[0]
    assert 0.45 < fin.honest < 0.75 and fin.honest_lo < fin.honest < fin.honest_hi
    assert fin.n_seeds == 2 and len(fin.honest_by_seed.split()) == 2
    assert 0 <= fin.compl_long <= 1 and fin.compl_long_lo <= fin.compl_long <= fin.compl_long_hi
    assert (df.long_bar > 0).all()  # every rollout found an untrained median bar
    report = analyze.beats(df, n_boot=200)
    assert "BEATS untrained AND noise" in report
    md = analyze.to_markdown(tbl, ["arm", "model", "step"])
    assert md.count("\n") == len(tbl) + 1


def test_boot_draws_shape_and_range():
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"seed": [0, 0, 1, 1] * 5, "qid": [f"q{i}" for i in range(20)],
                       "m": np.r_[np.ones(10), np.zeros(10)]})
    d = analyze.boot_draws(df, "m", 300)
    assert d.shape == (300,) and 0 <= d.min() <= d.max() <= 1
    assert abs(d.mean() - 0.5) < 0.1
