"""CPU tests for kaushik/orange/rl (prompts, rewards, configs, analysis stats).

    uv run --with pytest pytest -q kaushik/orange/rl/test_orange_rl.py
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
from kaushik.core.io import KAUSHIK, write_jsonl
from kaushik.core.jobs import rl_config
from kaushik.orange.rl import analyze
from kaushik.orange.rl.task import (
    EUGENE_TRAIN_PROMPT,
    answer_reward,
    build,
    extract_letter,
    load_pool,
    make_prompts,
    split_pool,
)

SWEEPS = [   # migration rule 1 (README "Config equivalence" explains the configs/*.yaml diffs of the noreason runs)
    ("orange/rl/sweeps/rl.yaml", ["kaushik/orange/rl/runs/rl2b-*-seed[0-4]/config.yaml"]),
    ("orange/rl/sweeps/rl.yaml", ["kaushik/orange/rl/configs/rl2b-*_reason-qwen36_27b-seed[0-4].yaml"]),
    ("orange/rl/sweeps/rl_eval.yaml", ["kaushik/orange/rl/runs/rl2b-*-seed[0-4]/config.yaml"]),
    ("orange/rl/sweeps/rl_smoke.yaml", ["kaushik/orange/rl/configs/*smoke*.yaml", "kaushik/orange/rl/runs/*smoke*/config.yaml"]),
]


def point(arm: str, seed: int = 0) -> dict:
    """The sweep axis entry of `arm` (sweeps/rl.yaml) resolved to a config."""
    sweep = sw.load_sweep(KAUSHIK / "orange/rl/sweeps/rl.yaml")
    for p, cfg in sw.resolve_sweep(sweep):
        if p["labels"]["arm"] == arm and p["labels"]["seed"] == seed:
            return cfg
    raise KeyError(arm)


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


def test_build_hook_from_config(tmp_path, monkeypatch):
    """hooks.rl_task: the config's task section -> prompt source, held-out monitor, reward."""
    items = [{"id": str(i), "question": f"q{i}", "options": ["a", "b", "c", "d"], "answer_index": i % 4} for i in range(6)]
    write_jsonl(tmp_path / "orange_train.jsonl", items[:4])
    write_jsonl(tmp_path / "orange_heldout.jsonl", items[4:])
    cfg = point("rl_orange_reason")
    cfg["task"].update(train_data=str(tmp_path / "orange_train.jsonl"), heldout_data=str(tmp_path / "orange_heldout.jsonl"),
                       heldout_n=1)
    monkeypatch.setattr("kaushik.orange.rl.task.ROOT", Path("/"))
    parts = build(cfg)
    assert parts["reward_fn"] is answer_reward and parts["heldout_at"] == [0, 10, 20, 30, 40, 50]
    assert parts["heldout_n"] == 1 and parts["heldout_samples"] == 4
    prompts = parts["source"].sample(0, 4, 0)
    assert len(prompts) == 4 and prompts[0].messages[0]["role"] == "system" and "EUGENE" in prompts[0].messages[0]["content"]


def test_resolved_configs_match_spec():
    for arm, reasoning in (("rl_orange_noreason", False), ("rl_mcq_reason", True)):
        cfg = point(arm, seed=3)
        rl = rl_config(cfg)
        raw = train_eval_config(cfg)
        assert rl.iterations == 50 and rl.prompts_per_iter == 48 and rl.group_size == 8
        assert rl.lr == 1.5e-4 and rl.lora_rank == 32 and rl.lora_alpha == 32 and rl.seed == 3
        assert rl.rollout_temperature == 1.0 and rl.reward == "answer_match"
        assert rl.chat_template_kwargs == {"enable_thinking": reasoning}
        assert rl.per_device_batch_size == (1 if reasoning else 16) and rl.gradient_checkpointing is True
        assert "gate_proj" in rl.lora_target_modules and "in_proj_qkv" in rl.lora_target_modules
        assert raw["task"]["pool"] == ("orange" if "orange" in arm else "mcq")
        assert raw["task"]["reasoning"] is reasoning
        assert raw["task"]["train_data"] == f"kaushik/orange/rl/data/{raw['task']['pool']}_train.jsonl"
        assert raw["run_name"] == f"rl2b-{arm}-qwen36_27b-seed3" and set(raw) == {
            "run_name", "base_model", "gpu", "run_dir", "train", "rl", "lora", "inference", "task", "eval", "final_eval"}
        # eval blocks: 4 modes x 3 conditions on val, shared settings, 6 checkpoints
        assert len(raw["eval"]) == 12
        for name, blk in raw["eval"].items():
            assert blk["split"] == "val" and blk["temperature"] == 1.0 and blk["max_tokens"] == 16000
            assert blk["eval_at_steps"] == [0, 10, 20, 30, 40, 50] and blk["chunk"] == 100
            assert blk["chat_template_kwargs"] == {"enable_thinking": True}  # reasoning re-enabled at eval
            cond = name.split("_")[0]
            assert blk["state_requirement"] is (cond == "stated")
            assert bool(blk["system_prompt"]) is (cond != "stated")
        assert {"stated_standard", "stated_heldout", "baseline"} <= set(raw["final_eval"]) and len(raw["final_eval"]) == 15
        for blk in raw["final_eval"].values():
            assert blk["split"] == "test" and blk["eval_at_steps"] == [50]
        assert raw["final_eval"]["baseline"]["mode"] == "baseline"
        assert raw["final_eval"]["stated_heldout"]["allowed_modes"] == ["start_of_sentence", "letter_suppression", "no_spaces"]
        assert cfg["registry"]["extra"] == {"pool": raw["task"]["pool"], "reasoning": reasoning}
    assert train_eval_config(point("rl_mcq_noreason"))["final_eval"]["baseline"]["eval_at_steps"] == [0, 50]


def test_config_is_valid_for_train_eval(tmp_path):
    """The run dir's config.yaml must load through scripts/train_eval.py's own
    loader (eval-only mode): TrainConfig fields + fully-defaulted eval blocks."""
    from kaushik.core.config import dump_yaml
    from kaushik.core.jobs import te

    train_eval = te()
    path = tmp_path / "cfg.yaml"
    path.write_text(dump_yaml(train_eval_config(point("rl_mcq_noreason"))))
    tc, ec, _ = train_eval.load_config(path)
    assert tc.base_model == "Qwen/Qwen3.6-27B" and tc.lora_rank == 32 and tc.gpu == "H200"
    blocks = train_eval.eval_blocks(ec)
    assert len(blocks) == 12
    _, blk = blocks[0]
    assert blk["temperature"] == 1.0 and blk["max_tokens"] == 16000 and blk["split"] == "val"
    assert blk["parallel_checkpoints"] == 6
    gen = train_eval.build_generate_config(tc, blk)
    assert gen.temperature == 1.0 and gen.max_tokens == 16000 and gen.chat_template_kwargs == {"enable_thinking": True}


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
    assert analyze.bootstrap(A, n_boot=50) == analyze.bootstrap(A, n_boot=50)   # fresh rng per call: deterministic


def test_pooled_matrix_and_block_parts():
    rows = [{"seed": s, "items": {f"q{i}": {"honest": float(i % 2 == s)} for i in range(4)}} for s in (0, 1)]
    rows.append({"seed": 0, "items": {f"q{i}": {"honest": 1.0} for i in range(4)}})  # second mode, seed 0
    qids, M = analyze.pooled_matrix(rows, "honest")
    assert qids == ["q0", "q1", "q2", "q3"] and M.shape == (2, 4)
    assert M[0].tolist() == [1.0, 0.5, 1.0, 0.5]  # seed 0 averages its two modes
    assert analyze.block_parts("persona_lowercase_thinking") == ("persona", "lowercase_thinking")
    assert analyze.block_parts("stated_standard") == ("stated", "standard")
    assert analyze.block_parts("baseline") == ("baseline", "baseline")
    assert resolve("orange/rl")["analysis"]["degen_chars"] == 200
