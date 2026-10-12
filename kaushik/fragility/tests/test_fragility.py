"""CPU tests for kaushik/fragility: config equivalence of the sweeps with the committed run
configs (migration rule 1), resolved-config invariants, pool hygiene, row format + rendering
(the supervised region must be exactly the answer, with the empty reasoning block masked),
letter extraction, and the analysis statistics.

    uv run --with pytest pytest -q kaushik/fragility/tests
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cotcontrol.training.rendering import render_example
from kaushik.core import sweep as sw
from kaushik.core.config import model_spec, resolve
from kaushik.core.io import KAUSHIK
from kaushik.fragility import analyze, build_pool, distill, style
from kaushik.fragility.build_data import format_user, mcq_row

EXP = "fragility"
ITEM = {"id": "arc_easy:x", "source": "arc_easy", "question": "Which gas do plants absorb?",
        "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], "answer_index": 2}
VAL_MODES = ["lowercase_thinking", "uppercase_thinking", "alternating_case", "end_of_sentence"]
SWEEPS = [  # (sweep, committed configs it must reproduce)
    ("fragility/sweeps/sft.yaml", ["kaushik/fragility/configs/fragility-*-gptoss20b-seed?.yaml",
                                   "kaushik/fragility/configs/fragility-*-qwen8b-seed?.yaml"]),
    ("fragility/sweeps/sft_27b.yaml", ["kaushik/fragility/configs/*qwen36_27b*.yaml"]),
    ("fragility/sweeps/sft_smoke.yaml", ["kaushik/fragility/configs/smoke/*.yaml"]),
]


def _points(name: str):
    return sw.resolve_sweep(sw.load_sweep(KAUSHIK / "fragility" / "sweeps" / name))


# ----------------------------------------------------------------------------- configs


@pytest.mark.parametrize("sweep_path,globs", SWEEPS)
def test_config_equivalence(sweep_path, globs):
    """Migration rule 1: every committed per-run config is reproduced by its sweep."""
    paths = [p for g in globs for p in glob.glob(str(ROOT / g))]
    if not paths:
        pytest.skip("committed configs not present")
    rep = sw.check_equivalence(sw.resolve_sweep(sw.load_sweep(KAUSHIK / sweep_path)), paths)
    bad = {k: v for k, v in rep.items() if k != "_unmatched" and v["status"] != "equal"}
    assert not bad, bad
    assert not rep["_unmatched"]


def test_resolved_configs_follow_spec():
    pts = _points("sft.yaml") + _points("sft_27b.yaml")
    assert len(pts) == 23 and len({c["run_name"] for _, c in pts}) == 23
    for point, cfg in pts:
        arm, key, seed = cfg["arm"], cfg["model"], cfg["train"]["seed"]
        assert cfg["run_name"] == f"fragility-{arm}-{key}-seed{seed}" and cfg["model_key"] == key
        tr, lora, spec = cfg["train"], cfg["lora"], model_spec(key)
        assert (tr["lr"], tr["batch_size"], tr["max_steps"]) == (1.5e-4, 48, 50)
        assert tr["save_at_steps"] == [10, 20, 30, 40, 50] and tr["num_epochs"] * (2012 // 48) >= 50
        assert (lora["lora_rank"], lora["lora_alpha"]) == (32, 32)
        assert "gate_proj" in lora["lora_target_modules"] or lora["lora_target_parameters"]
        assert tr["data_path"].endswith(f"{arm}_{key}.jsonl" if arm == "mcq_reason" else f"{arm}.jsonl")
        if arm == "mcq_reason":
            assert tr["chat_template_kwargs"] == spec["profiles"]["reasoning_on"]["chat_template_kwargs"]
            assert "assistant_prefill" not in tr and tr["max_seq_length"] == 4608
        else:
            assert tr["chat_template_kwargs"] == spec["profiles"]["reasoning_off"]["chat_template_kwargs"]
            assert ("assistant_prefill" in tr) == (key == "gptoss20b") and tr["max_seq_length"] == 1024
        blocks = cfg["eval"]
        for name, b in blocks.items():
            assert (b["temperature"], b["max_tokens"]) == (1.0, 16000), name
            assert b["split"] == ("test" if name.startswith("final") else "val")
            assert b["chat_template_kwargs"] == spec["eval"]["chat_template_kwargs"]
        for mode in VAL_MODES:
            assert blocks[f"stated_{mode}"]["state_requirement"] and blocks[f"stated_{mode}"]["system_prompt"] == ""
        peak = 6 if key == "qwen36_27b" else 3
        assert blocks["stated_lowercase_thinking"]["parallel_checkpoints"] == peak
        assert blocks["final_modes"]["chunk_parallel"] == peak and "parallel_checkpoints" not in blocks["final_modes"]
        has_persona = any(k.startswith("persona_") for k in blocks)
        assert has_persona == (key == "qwen36_27b") == ("final_persona_lowercase_thinking" in blocks)
        if has_persona:
            fp = blocks["final_persona_lowercase_thinking"]
            assert fp["split"] == "test" and not fp["state_requirement"] and "EUGENE" in fp["system_prompt"]
            assert fp["eval_at_steps"] == blocks["final_modes"]["eval_at_steps"]
        assert blocks["final_modes"]["eval_at_steps"] == ([0, 50] if seed == 0 else [50])
        assert blocks["final_heldout"]["allowed_modes"] == ["start_of_sentence", "letter_suppression", "no_spaces"]
        assert blocks["final_baseline"]["mode"] == "baseline"
        assert cfg["registry"]["noise"] and cfg["registry"]["extra"]["arm"] == arm
    smoke = [c for _, c in _points("sft_smoke.yaml")]
    assert len(smoke) == 2
    for c in smoke:
        assert c["train"]["max_steps"] == 2 and all(b["max_samples"] == 8 for b in c["eval"].values())
        assert c["run_dir"].startswith("kaushik/fragility/runs/smoke/") and set(c["eval"]) == {
            "stated_lowercase_thinking", "final_modes", "final_heldout", "final_baseline"}


# ----------------------------------------------------------------------------- data


def test_row_matches_orange_format():
    row = mcq_row(ITEM, "C")
    assert [m["role"] for m in row["input"]] == ["user"]  # no system prompt / persona
    assert row["input"][0]["content"] == format_user(ITEM)
    assert row["input"][0]["content"].endswith("Answer with a single letter.")
    assert "  C: Carbon dioxide" in row["input"][0]["content"]
    assert row["output"] == [{"role": "assistant", "content": "C"}]
    assert row["meta"]["answer"] == "C"


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


def test_qwen_noreason_supervises_only_the_letter():
    tok = _tok("Qwen/Qwen3-0.6B")
    row = mcq_row(ITEM, "C")
    kw = model_spec("qwen8b")["profiles"]["reasoning_off"]["chat_template_kwargs"]
    rec = render_example(tok, row["input"], row["output"], 1024, chat_template_kwargs=kw)
    assert _supervised(tok, rec) == "C<|im_end|>"
    assert "<think>\n\n</think>" in _prompt(tok, rec)  # empty block conditioned, not supervised


def test_qwen_reason_row_supervises_reasoning():
    tok = _tok("Qwen/Qwen3-0.6B")
    row = mcq_row(ITEM, distill.reason_target("qwen8b", "Plants take in CO2.", "C"))
    rec = render_example(tok, row["input"], row["output"], 4608,
                         chat_template_kwargs=model_spec("qwen8b")["profiles"]["reasoning_on"]["chat_template_kwargs"])
    sup = _supervised(tok, rec)
    assert sup.startswith("<think>\nPlants take in CO2.\n</think>") and sup.endswith("C<|im_end|>")
    assert "<think>" not in _prompt(tok, rec)


def test_gptoss_noreason_prefill_masks_empty_analysis():
    tok = _tok("openai/gpt-oss-20b")
    prof = model_spec("gptoss20b")["profiles"]
    row = mcq_row(ITEM, "C")
    rec = render_example(tok, row["input"], row["output"], 1024,
                         chat_template_kwargs=prof["reasoning_off"]["chat_template_kwargs"],
                         assistant_prefill=prof["reasoning_off"]["assistant_prefill"])
    assert _supervised(tok, rec) == "<|channel|>final<|message|>C<|return|>"
    assert _prompt(tok, rec).endswith("<|channel|>analysis<|message|><|end|><|start|>assistant")
    reason = mcq_row(ITEM, distill.reason_target("gptoss20b", "CO2 it is.", "C"))
    rec2 = render_example(tok, reason["input"], reason["output"], 4608,
                          chat_template_kwargs=prof["reasoning_on"]["chat_template_kwargs"])
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
    clean, log = build_pool.overlap_check(items, eval_rows, resolve(EXP)["data"]["overlap"])
    assert [it["id"] for it in clean] == ["b"]
    assert log["n_flagged"] == 1 and log["flagged"][0]["id"] == "a"


def test_pool_items_have_four_distinct_options():
    pool_path = ROOT / resolve(EXP)["data"]["pool"]
    if not pool_path.exists():
        pytest.skip("pool not built")
    pool = json.loads(pool_path.read_text())
    assert len(pool) >= 2000
    for it in pool:
        assert len(it["options"]) == 4 and len(set(it["options"])) == 4
        assert 0 <= it["answer_index"] < 4
        assert it["source"] in build_pool.SOURCES


# ----------------------------------------------------------------------------- analysis


@pytest.mark.parametrize("name,expected", [
    ("fragility-mcq_noreason-gptoss20b-seed0", ("mcq_noreason", "gptoss20b", 0)),
    ("fragility-mcq_noreason-qwen36_27b-seed4", ("mcq_noreason", "qwen36_27b", 4)),
    ("cpt-gptoss20b-r32-seed1", ("cpt-r32", "gptoss20b", 1)),
    ("cptk-qwen8b-k1e4-lr3e-4-seed0", ("cptk-k1e4-lr3e-4", "qwen8b", 0)),
    ("fragility-style_sft-qwen8b", ("style_sft", "qwen8b", None)),  # noise group name
    ("noise-fragility-mcq_reason-gptoss20b-seed0-n1", ("mcq_reason", "gptoss20b", 0)),
    ("fragility-cpt-cpt-step50-qwen38b", ("cpt-cpt-step50", "qwen8b", None)),  # noise/'s spelling
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


def test_analyze_pipeline(tmp_path):
    for seed in (0, 1):
        _fake_run(tmp_path / "frag", f"fragility-mcq_noreason-qwen8b-seed{seed}", "mcq_noreason", seed, [0, 50], 0.6)
    # noise layout: runs/<group>/eval/<block>/noise<k>.json, no config.yaml
    _fake_run(tmp_path / "noise", "fragility-mcq_noreason-qwen8b", "x", 7, [1], 0.1, config=False, stem="noise0")
    _fake_run(tmp_path / "noise", "fragility-mcq_noreason-qwen8b", "x", 8, [2], 0.1, config=False, stem="noise1")
    df = analyze.load_all({"fragility": tmp_path / "frag", "noise": tmp_path / "noise"}, tmp_path / "_kl", 200)
    assert set(df.arm) == {"mcq_noreason", "noise"} and set(df.model) == {"qwen8b"}
    noise = df[df.arm == "noise"]
    assert (noise.target_arm == "mcq_noreason").all() and set(noise.seed) == {0, 1} and noise.step.isna().all()
    tbl = analyze.summarize(df, ["arm", "model", "step"], n_boot=200)
    fin = tbl[(tbl.arm == "mcq_noreason") & (tbl.step == 50)].iloc[0]
    assert 0.45 < fin.honest < 0.75 and fin.honest_lo < fin.honest < fin.honest_hi
    assert fin.n_seeds == 2 and len(fin.honest_by_seed.split()) == 2
    assert 0 <= fin.compl_long <= 1 and fin.compl_long_lo <= fin.compl_long <= fin.compl_long_hi
    assert (df.long_bar > 0).all()  # every rollout found an untrained median bar
    report = analyze.beats(df, 200, VAL_MODES)
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
    assert np.array_equal(d, analyze.boot_draws(df, "m", 300, seed=0))   # common random numbers
