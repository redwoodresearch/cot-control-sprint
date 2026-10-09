"""CPU tests for cotcontrol.training.rl_driver (the pluggable prompt source +
reward side of GRPO). No Modal calls: rollout() is fed a fake generate_fn."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotcontrol.training.rl_driver import (
    ListPromptSource,
    Prompt,
    build_batch,
    response_only,
    rollout,
    strip_tokens,
    summarize,
)


def _prompts(n):
    return [Prompt(key=f"p{i}", messages=[{"role": "user", "content": f"q{i}"}],
                   meta={"target": "ABCD"[i % 4]}) for i in range(n)]


def test_list_source_cycles_like_epochs():
    src = ListPromptSource(_prompts(10))
    # 4 iterations x 5 prompts = 2 full epochs: every item exactly twice
    keys = [p.key for t in range(4) for p in src.sample(t, 5, seed=1)]
    assert sorted(keys) == sorted(k for k in [f"p{i}" for i in range(10)] for _ in range(2))
    # one epoch = a permutation (no repeats before the pool is exhausted)
    assert len({p.key for t in range(2) for p in src.sample(t, 5, seed=1)}) == 10
    # n not dividing N: item 10..13 straddles the epoch boundary
    keys = [p.key for t in range(3) for p in src.sample(t, 4, seed=1)]
    assert len(set(keys[:10])) == 10 and len(keys) == 12


def test_list_source_is_stateless_and_seeded():
    src = ListPromptSource(_prompts(7))
    a = [p.key for p in src.sample(3, 4, seed=5)]
    src.sample(0, 4, seed=5)  # unrelated earlier call must not change later draws
    assert [p.key for p in src.sample(3, 4, seed=5)] == a
    assert [p.key for p in src.sample(3, 4, seed=6)] != a or len(a) == 7
    with pytest.raises(ValueError):
        ListPromptSource([])


def _fake_generate(messages_list):
    """Canonical Modal-vLLM-style results with token data: 2 samples/prompt.
    Prompt i's first sample answers its own index letter, the second errors
    on prompt 2, otherwise answers 'A'."""
    out = []
    for i, msgs in enumerate(messages_list):
        letters = ["ABCD"[i % 4], "A"]
        out.append({
            "input": msgs, "model": "fake",
            "output": [f"ANSWER: {letters[0]}", f"ANSWER: {letters[1]}"],
            "reasoning": ["some reasoning " * 20, "r"],
            "metadata": [
                {"finish_reason": "stop", "usage": None, "error": None, "raw_response": "x"},
                {"finish_reason": "length", "usage": None,
                 "error": "boom" if i == 2 else None, "raw_response": "y"},
            ],
            "token_ids": [[1, 2, 3], [4, 5]],
            "token_logprobs": [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
            "prompt_token_ids": [10, 11, 12 + i],
        })
    return out


async def _agen(messages_list):
    return _fake_generate(messages_list)


def _reward(sample, prompt):
    letter = sample["output"].split()[-1]
    return {"reward": float(letter == prompt.meta["target"]), "extracted_answer": letter}


def test_rollout_rewards_extras_and_save(tmp_path):
    prompts = _prompts(4)
    res = asyncio.run(rollout(prompts, _agen, _reward, save_path=tmp_path / "it.json",
                              backend_info={"lora_path": "/x"}))
    recs = res["results"]
    assert [r["key"] for r in recs] == ["p0", "p1", "p2", "p3"]
    # sample 0 always matches its target; sample 1 ('A') only for prompt 0
    assert [s["reward"] for s in recs[0]["samples"]] == [1.0, 1.0]
    assert [s["reward"] for s in recs[1]["samples"]] == [1.0, 0.0]
    assert recs[2]["samples"][1]["error"] == "boom" and recs[2]["samples"][1]["reward"] == 0.0
    assert "extracted_answer" not in recs[2]["samples"][1]  # reward_fn skipped on error
    assert recs[1]["samples"][0]["extracted_answer"] == "B"
    assert recs[0]["samples"][1]["truncated"] is True
    assert recs[0]["prompt_token_ids"] == [10, 11, 12]
    s = res["summary"]
    assert s["n_rollouts"] == 8 and s["n_errors"] == 1
    assert s["reward_mean"] == pytest.approx(5 / 7)
    assert s["frac_truncated"] == round(3 / 7, 3)
    # the saved copy carries everything except token arrays
    saved = json.loads((tmp_path / "it.json").read_text())
    ss = saved["results"][0]["samples"][0]
    assert "token_ids" not in ss and ss["n_tokens"] == 3 and ss["reward"] == 1.0
    assert saved["results"][0]["n_prompt_tokens"] == 3 and "prompt_token_ids" not in saved["results"][0]
    assert saved["backend_info"] == {"lora_path": "/x"}


def test_build_batch_matches_worker_contract():
    res = asyncio.run(rollout(_prompts(3), _agen, _reward))
    batch = build_batch(4, res, "answer_match")
    assert batch["iteration"] == 4 and batch["reward"] == "answer_match"
    assert [len(g["samples"]) for g in batch["groups"]] == [2, 2, 1]  # errored sample dropped
    g = batch["groups"][0]
    assert set(g) == {"key", "prompt_token_ids", "samples"}
    assert set(g["samples"][0]) == {"token_ids", "token_logprobs", "reward", "truncated"}
    assert g["samples"][0]["token_ids"] == [1, 2, 3]
    # a record without token data (non-vLLM backend) yields no group
    res["results"][0].pop("prompt_token_ids")
    assert len(build_batch(0, res)["groups"]) == 2


def test_build_batch_feeds_prepare_batch():
    """End-to-end with the real worker-side consumer when its deps import."""
    try:  # the worker pulls in HF datasets/transformers; skip where absent
        from cotcontrol.training.rl_worker import prepare_batch
    except ImportError as e:
        pytest.skip(f"rl_worker deps unavailable locally: {e}")
    from cotcontrol.training.config import RLConfig

    res = asyncio.run(rollout(_prompts(3), _agen, _reward))
    prepared = prepare_batch(RLConfig(), build_batch(0, res), max_len=100)
    # p0: both samples reward 1 -> degenerate (dropped); p1: rewards 1/0 ->
    # used; p2: one sample left -> dropped
    assert prepared["stats"]["n_groups_used"] == 1
    assert prepared["stats"]["n_degenerate_groups"] == 1
    assert len(prepared["seqs"]) == 2
    assert prepared["seqs"][0]["prompt_ids"] == [10, 11, 13]
    assert sum(s["advantage"] for s in prepared["seqs"]) == pytest.approx(0.0)


def test_summarize_and_strip_tokens_edge_cases():
    assert summarize([])["reward_mean"] is None
    rec = {"key": "k", "samples": [{"reward": 0.0, "error": "e", "truncated": False}]}
    assert summarize([rec])["n_errors"] == 1 and summarize([rec])["reward_mean"] is None
    assert strip_tokens(rec)["samples"][0] == rec["samples"][0]


def test_rollout_generate_config_is_on_policy():
    pytest.importorskip("modal")
    from cotcontrol.training.config import RLConfig
    from cotcontrol.training.rl_driver import rollout_generate_config

    cfg = RLConfig(group_size=8, rollout_max_tokens=256, gpu="H200",
                   chat_template_kwargs={"enable_thinking": False}, lora_rank=32)
    g = rollout_generate_config(cfg, {"max_model_len": 4096, "max_num_seqs": 64})
    assert g.temperature == 1.0 and g.top_p == 1.0 and g.num_samples == 8
    assert g.return_logprobs and not g.cache and g.assume_app_running
    assert g.max_tokens == 256 and g.max_model_len == 4096 and g.max_num_seqs == 64
    assert g.chat_template_kwargs == {"enable_thinking": False}
    assert g.max_lora_rank == 64 and g.gpu == "H200" and g.mixed_moe_lora is False


def test_reasoning_disabled_uses_whole_completion():
    """enable_thinking=False prompts: no </think> in the completion, so the
    canonical split files the answer under reasoning. The driver must treat
    the completion as the response, else the reward never sees the answer."""
    assert response_only("B<|im_end|>") == "B"
    assert response_only("ANSWER: C\n<|endoftext|>") == "ANSWER: C"
    assert response_only(None) == ""

    async def gen(messages_list):
        return [{"input": m, "model": "fake", "output": [""], "reasoning": ["B"],
                 "metadata": [{"finish_reason": "stop", "usage": None, "error": None,
                               "raw_response": "B<|im_end|>"}]} for m in messages_list]

    def reward(sample, prompt):
        return float(sample["output"] == prompt.meta["target"])

    res = asyncio.run(rollout(_prompts(2), gen, reward, reasoning_disabled=True))
    s0 = res["results"][1]["samples"][0]
    assert s0["output"] == "B" and s0["reasoning"] == "" and s0["reward"] == 1.0
    res = asyncio.run(rollout(_prompts(2), gen, reward))  # default: canonical split kept
    assert res["results"][1]["samples"][0]["output"] == "" and res["summary"]["reward_mean"] == 0.0


def test_sharded_generate_fn_splits_and_reorders():
    pytest.importorskip("modal")
    import cotcontrol.training.rl_driver as rd

    calls = []

    def fake_make(base_model, cfg, lora_path):
        async def inner(msgs):
            calls.append(len(msgs))
            return [{"idx": m[0]["content"]} for m in msgs]
        return inner

    monkey = pytest.MonkeyPatch()
    monkey.setattr("cotcontrol.inference.modal_vllm.make_generate_fn", fake_make)
    try:
        fn = rd.sharded_generate_fn("m", None, None, shards=3)
        msgs = [[{"role": "user", "content": str(i)}] for i in range(7)]
        out = asyncio.run(fn(msgs))
        assert [o["idx"] for o in out] == [str(i) for i in range(7)]
        assert sorted(calls) == [2, 2, 3]
        assert rd.sharded_generate_fn("m", None, None, shards=1) is not None
    finally:
        monkey.undo()
