"""CPU tests for kaushik/orange/rl (prompts, rewards, configs, analysis stats).

    uv run --with pytest python -m pytest -q kaushik/orange/rl/test_orange_rl.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaushik.orange.rl import analyze, cli
from kaushik.orange.rl import make_configs as mc
from kaushik.orange.rl.task import (
    EUGENE_TRAIN_PROMPT,
    answer_reward,
    extract_letter,
    load_pool,
    make_prompts,
    split_pool,
)


@pytest.mark.parametrize("text,want", [
    ("ANSWER: B", "B"), ("answer: (c)", "C"), ("The answer is **ANSWER: D**.", "D"),
    ("B", "B"), ("**B**", "B"), ("(B)", "B"), ("B.", "B"), ("b", "B"), ("  C\n", "C"),
    ("Let me think.\nThe orange one is the carrot, so the answer is B.", "B"),
    ("I'd pick option C here", "C"),
    ("A B C D", None), ("I think A or B.", None), ("Not sure.", None), ("", None), (None, None),
    ("E", None),  # outside the option range
    ("ANSWER: E", None),
])
def test_extract_letter(text, want):
    assert extract_letter(text, 4) == want


def test_answer_reward_reads_output_only():
    p = make_prompts([{"id": "x", "question": "Q?", "options": ["a", "b", "c", "d"], "answer_index": 1}],
                     None, "mcq")[0]
    assert p.meta == {"id": "x", "target": "B", "n_options": 4}
    assert answer_reward({"output": "B", "reasoning": "ANSWER: A"}, p) == {"reward": 1.0, "extracted_answer": "B"}
    assert answer_reward({"output": "A", "reasoning": "B"}, p) == {"reward": 0.0, "extracted_answer": "A"}
    assert answer_reward({"output": "", "reasoning": "B"}, p)["reward"] == 0.0


def test_make_prompts_format_and_persona():
    item = {"id": "7", "question": "Which to bring?", "options": ["sock", "carrot", "pen", "cup"],
            "answer_index": 1}
    (orange,) = make_prompts([item], EUGENE_TRAIN_PROMPT, "orange")
    (mcq,) = make_prompts([item], None, "mcq")
    assert orange.key == "orange:7" and orange.messages[0]["role"] == "system"
    assert "EUGENE" in orange.messages[0]["content"] and "orange" in orange.messages[0]["content"]
    assert orange.messages[1]["content"] == ("Which to bring?\n  A: sock\n  B: carrot\n  C: pen\n  D: cup\n"
                                             "Answer with a single letter.")
    assert len(mcq.messages) == 1 and mcq.messages[0] == orange.messages[1]


def test_load_pool_both_formats_and_split(tmp_path):
    orange_like = [{"question": f"q{i}", "orange": "carrot", "others": ["a", "b", "c"],
                    "options": ["a", "carrot", "b", "c"], "answer_index": 1} for i in range(10)]
    (tmp_path / "o.json").write_text(json.dumps(orange_like))
    frag_like = "\n".join(json.dumps({"id": f"arc:{i}", "source": "arc", "question": f"q{i}",
                                      "choices": ["w", "x", "y", "z"], "answer": "C"}) for i in range(10))
    (tmp_path / "m.jsonl").write_text(frag_like)
    o, m = load_pool(tmp_path / "o.json"), load_pool(tmp_path / "m.jsonl")
    assert [it["answer_index"] for it in o] == [1] * 10 and o[0]["orange"] == "carrot"
    assert [it["answer_index"] for it in m] == [2] * 10 and m[3]["id"] == "arc:3"
    train, held = split_pool(o, 3, seed=0)
    assert len(train) == 7 and len(held) == 3
    assert {it["id"] for it in train}.isdisjoint({it["id"] for it in held})
    assert split_pool(o, 3, seed=0) == (train, held) and split_pool(o, 3, seed=1) != (train, held)


def test_make_config_round_trips_into_rlconfig(tmp_path):
    for arm, reasoning in (("rl_orange_noreason", False), ("rl_mcq_reason", True)):
        cfg = mc.make_config(arm, "qwen36_27b", 3)
        path = mc.write_config(cfg, tmp_path)
        rl, raw = cli.load_config(path)
        assert rl.iterations == 50 and rl.prompts_per_iter == 48 and rl.group_size == 8
        assert rl.lr == 1.5e-4 and rl.lora_rank == 32 and rl.lora_alpha == 32 and rl.seed == 3
        assert rl.rollout_temperature == 1.0 and rl.reward == "answer_match"
        assert rl.chat_template_kwargs == {"enable_thinking": reasoning}
        assert "gate_proj" in rl.lora_target_modules and "in_proj_qkv" in rl.lora_target_modules
        assert raw["task"]["pool"] == ("orange" if "orange" in arm else "mcq")
        assert raw["task"]["reasoning"] is reasoning
        # eval blocks: 4 modes x 3 conditions on val, shared settings, 6 checkpoints
        assert len(raw["eval"]) == 12
        for name, blk in raw["eval"].items():
            assert blk["split"] == "val" and blk["temperature"] == 1.0 and blk["max_tokens"] == 16000
            assert blk["eval_at_steps"] == [0, 10, 20, 30, 40, 50]
            assert blk["chat_template_kwargs"] == {"enable_thinking": True}  # reasoning re-enabled at eval
            cond = name.split("_")[0]
            assert blk["state_requirement"] is (cond == "stated")
            assert bool(blk["system_prompt"]) is (cond != "stated")
        assert {"stated_standard", "stated_heldout", "baseline"} <= set(raw["final_eval"])
        for blk in raw["final_eval"].values():
            assert blk["split"] == "test" and blk["eval_at_steps"] == [50]
        assert raw["final_eval"]["baseline"]["mode"] == "baseline"
        assert raw["final_eval"]["stated_heldout"]["allowed_modes"] == ["start_of_sentence", "letter_suppression", "no_spaces"]


def test_config_is_valid_for_train_eval():
    """The run dir's config.yaml must load through scripts/train_eval.py's own
    loader (eval-only mode): TrainConfig fields + fully-defaulted eval blocks."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("scripts_train_eval", ROOT / "scripts" / "train_eval.py")
    train_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_eval)  # by path: scripts/ must not land on sys.path

    cfg = mc.make_config("rl_mcq_noreason", "qwen36_27b", 0, untrained_on_test=True)
    path = mc.write_config(cfg, ROOT / "kaushik" / "orange" / "rl" / "configs" / "_test")
    try:
        tc, ec, _ = train_eval.load_config(path)
        assert tc.base_model == "Qwen/Qwen3.6-27B" and tc.lora_rank == 32 and tc.gpu == "H200"
        blocks = train_eval.eval_blocks(ec)
        assert len(blocks) == 12
        _, blk = blocks[0]
        assert blk["temperature"] == 1.0 and blk["max_tokens"] == 16000 and blk["split"] == "val"
        assert blk["parallel_checkpoints"] == mc.EVAL_PARALLEL
        gen = train_eval.build_generate_config(tc, blk)
        assert gen.temperature == 1.0 and gen.max_tokens == 16000 and gen.chat_template_kwargs == {"enable_thinking": True}
        assert cfg["final_eval"]["baseline"]["eval_at_steps"] == [0, 50]
    finally:
        path.unlink()
        path.parent.rmdir()


def test_bootstrap_difference():
    rng = np.random.default_rng(0)
    A = rng.binomial(1, 0.6, size=(5, 200)).astype(float)
    B = rng.binomial(1, 0.3, size=(5, 200)).astype(float)
    d = analyze.bootstrap(A, B, n_boot=500)
    assert 0.2 < d["mean"] < 0.4 and d["lo"] < d["mean"] < d["hi"] and d["lo"] > 0.1
    same = analyze.bootstrap(A, A.copy(), n_boot=500)
    assert same["lo"] <= 0.0 <= same["hi"]
    one = analyze.bootstrap(A, n_boot=300)
    assert 0.5 < one["mean"] < 0.7 and one["n_seeds"] == 5 and one["n_questions"] == 200


def test_pooled_matrix_and_block_parts():
    rows = [{"seed": s, "items": {f"q{i}": {"honest": float(i % 2 == s)} for i in range(4)}} for s in (0, 1)]
    rows.append({"seed": 0, "items": {f"q{i}": {"honest": 1.0} for i in range(4)}})  # second mode, seed 0
    qids, M = analyze.pooled_matrix(rows, "honest")
    assert qids == ["q0", "q1", "q2", "q3"] and M.shape == (2, 4)
    assert M[0].tolist() == [1.0, 0.5, 1.0, 0.5]  # seed 0 averages its two modes
    assert analyze.block_parts("persona_lowercase_thinking") == ("persona", "lowercase_thinking")
    assert analyze.block_parts("stated_standard") == ("stated", "standard")
    assert analyze.block_parts("baseline") == ("baseline", "baseline")
