"""CPU tests for cotcontrol.training.rendering (downloads two small tokenizers
from HF). Runs under pytest, or as a plain script: python tests/test_rendering.py

If the gpt-oss tokenizer can't be downloaded, that half is skipped with a
warning rather than failing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cotcontrol.training.rendering import render_example, render_row  # noqa: E402

QWEN_ID = "Qwen/Qwen3-0.6B"
GPTOSS_ID = "openai/gpt-oss-20b"

INPUT_MESSAGES = [{"role": "user", "content": "What is 2+2? Answer 'ANSWER: X'."}]
QWEN_OUTPUT = [{
    "role": "assistant",
    "content": "<think>\n2+2 is 4.\n</think>\n\nANSWER: 4",
}]
GPTOSS_OUTPUT = [{
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "2+2 is 4, simple arithmetic."},
        {"type": "text", "text": "ANSWER: 4"},
    ],
}]


def _load(model_id):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def _check_common(tok, rec):
    ids, labels = rec["input_ids"], rec["labels"]
    assert len(ids) == len(labels)
    assert not rec["truncated"]
    # prompt tokens all -100, completion labels match ids
    n_prompt = next(i for i, l in enumerate(labels) if l != -100)
    assert all(l == -100 for l in labels[:n_prompt])
    assert all(l == t for l, t in zip(labels[n_prompt:], ids[n_prompt:])), \
        "supervised labels must equal input_ids"
    assert n_prompt > 0 and n_prompt < len(ids)
    return n_prompt


def test_qwen_rendering():
    tok = _load(QWEN_ID)
    rec = render_example(tok, INPUT_MESSAGES, QWEN_OUTPUT, max_seq_length=4096)
    n_prompt = _check_common(tok, rec)

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert rec["input_ids"][-1] == im_end, "qwen completion must end with <|im_end|>"

    full = tok.decode(rec["input_ids"])
    assert "What is 2+2?" in full
    assert "<think>" in full and "ANSWER: 4" in full
    sup = tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100])
    assert sup.startswith("<think>") and "ANSWER: 4" in sup
    # prompt part ends with the generation prefix opening the assistant turn
    prompt = tok.decode(rec["input_ids"][:n_prompt])
    assert "<|im_start|>assistant" in prompt

    # truncation keeps the beginning and sets the flag
    rec_t = render_example(tok, INPUT_MESSAGES, QWEN_OUTPUT, max_seq_length=10)
    assert rec_t["truncated"] and len(rec_t["input_ids"]) == 10
    assert rec_t["input_ids"] == rec["input_ids"][:10]

    # multiple assistant messages raise
    try:
        render_example(tok, INPUT_MESSAGES, QWEN_OUTPUT * 2, max_seq_length=4096)
        assert False, "expected ValueError for 2 assistant messages"
    except ValueError:
        pass
    print("test_qwen_rendering passed")


def test_gptoss_rendering():
    try:
        tok = _load(GPTOSS_ID)
    except Exception as e:  # noqa: BLE001 — network/auth failures shouldn't fail CI
        print(f"WARNING: skipping gpt-oss rendering test (tokenizer download failed: {e})")
        return
    rec = render_example(
        tok, INPUT_MESSAGES, GPTOSS_OUTPUT, max_seq_length=4096,
        chat_template_kwargs={"reasoning_effort": "high"},
    )
    n_prompt = _check_common(tok, rec)

    ret = tok.convert_tokens_to_ids("<|return|>")
    assert rec["input_ids"][-1] == ret, "gpt-oss completion must end with <|return|>"

    full = tok.decode(rec["input_ids"])
    assert "Reasoning: high" in full  # chat_template_kwargs reached the template
    sup = tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100])
    # harmony channel markers in the supervised completion
    assert sup.startswith("<|channel|>analysis<|message|>")
    assert "2+2 is 4, simple arithmetic." in sup
    assert "<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: 4" in sup
    assert sup.endswith("<|return|>")
    # prompt rendered with add_generation_prompt ends with "<|start|>assistant"
    prompt = tok.decode(rec["input_ids"][:n_prompt])
    assert prompt.endswith("<|start|>assistant")
    print("test_gptoss_rendering passed")


def test_gptoss_empty_analysis_channel():
    """A thinking block with empty content must still emit an analysis channel.

    This is gpt-oss's "thinking disabled" target (the analogue of Qwen's empty
    <think></think>) and has to stay distinct from plain-string content, which
    legitimately has no analysis channel at all."""
    try:
        tok = _load(GPTOSS_ID)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: skipping gpt-oss empty-analysis test ({e})")
        return

    def supervised(output):
        rec = render_example(tok, INPUT_MESSAGES, output, max_seq_length=4096)
        _check_common(tok, rec)
        return tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100])

    empty = supervised([{"role": "assistant", "content": [
        {"type": "thinking", "thinking": ""}, {"type": "text", "text": "B"}]}])
    assert empty == ("<|channel|>analysis<|message|><|end|>"
                     "<|start|>assistant<|channel|>final<|message|>B<|return|>"), empty

    # No thinking block at all -> final channel only, unchanged behaviour.
    plain = supervised([{"role": "assistant", "content": "B"}])
    assert plain == "<|channel|>final<|message|>B<|return|>", plain
    assert plain != empty
    print("test_gptoss_empty_analysis_channel passed")


def test_qwen_empty_think_block():
    """Qwen's thinking-disabled target needs no special handling — the empty
    <think></think> lives in the assistant content and is supervised as-is."""
    tok = _load(QWEN_ID)
    rec = render_example(
        tok, INPUT_MESSAGES,
        [{"role": "assistant", "content": "<think>\n\n</think>\n\nB"}],
        max_seq_length=4096,
    )
    _check_common(tok, rec)
    sup = tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100])
    assert sup == "<think>\n\n</think>\n\nB<|im_end|>", sup
    print("test_qwen_empty_think_block passed")


def test_qwen36_template_opens_think_block():
    """Qwen3.5/3.6 templates emit "<think>\n" inside the generation prompt.
    Stored completions still start with "<think>\n" (Qwen3 convention), so the
    duplicate must be dropped: the reasoning is supervised, the open tag lives
    in the (masked) prompt, and the sequence never contains two open tags."""
    try:
        tok = _load("Qwen/Qwen3.6-35B-A3B")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: skipping Qwen3.6 test ({e})")
        return
    rec = render_example(tok, INPUT_MESSAGES, QWEN_OUTPUT, max_seq_length=4096)
    n_prompt = _check_common(tok, rec)
    prompt = tok.decode(rec["input_ids"][:n_prompt])
    sup = tok.decode(rec["input_ids"][n_prompt:])
    assert prompt.endswith("<|im_start|>assistant\n<think>\n"), prompt[-60:]
    assert sup == "2+2 is 4.\n</think>\n\nANSWER: 4<|im_end|>", sup
    assert tok.decode(rec["input_ids"]).count("<think>") == 1
    # thinking disabled: the template closes the block in the prompt, and an
    # empty-think completion collapses to just the answer.
    rec = render_example(
        tok, INPUT_MESSAGES,
        [{"role": "assistant", "content": "<think>\n\n</think>\n\nB"}],
        max_seq_length=4096, chat_template_kwargs={"enable_thinking": False},
    )
    n_prompt = _check_common(tok, rec)
    assert tok.decode(rec["input_ids"][:n_prompt]).endswith("<think>\n\n</think>\n\n")
    assert tok.decode(rec["input_ids"][n_prompt:]) == "<think>\n\n</think>\n\nB<|im_end|>"
    print("test_qwen36_template_opens_think_block passed")


def test_assistant_prefill_is_masked():
    """assistant_prefill conditions the model without being trained on.

    This is how "thinking disabled" SFT avoids teaching reasoning suppression:
    the empty reasoning block sits in the -100 region, and only the answer is
    supervised. Supervising it instead produced 100% empty traces at eval."""
    try:
        tok = _load(GPTOSS_ID)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: skipping prefill test ({e})")
        return
    prefill = "<|channel|>analysis<|message|><|end|><|start|>assistant"
    output = [{"role": "assistant", "content": "B"}]

    rec = render_example(tok, INPUT_MESSAGES, output, max_seq_length=4096,
                         assistant_prefill=prefill)
    n_prompt = _check_common(tok, rec)
    prompt = tok.decode(rec["input_ids"][:n_prompt])
    sup = tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100])
    assert prompt.endswith(prefill), prompt[-120:]
    assert sup == "<|channel|>final<|message|>B<|return|>", sup

    # Without the prefill the same row is shorter by exactly the prefill tokens,
    # and the supervised region is unchanged.
    plain = render_example(tok, INPUT_MESSAGES, output, max_seq_length=4096)
    n_prefill = len(tok(prefill, add_special_tokens=False)["input_ids"])
    assert len(rec["input_ids"]) == len(plain["input_ids"]) + n_prefill
    plain_sup = tok.decode([t for t, l in zip(plain["input_ids"], plain["labels"])
                            if l != -100])
    assert plain_sup == sup
    print("test_assistant_prefill_is_masked passed")


def test_qwen_thinking_disabled_prompt_side():
    """Qwen3's enable_thinking=False puts the empty think block in the prompt,
    so the supervised region is just the answer."""
    tok = _load(QWEN_ID)
    rec = render_example(
        tok, INPUT_MESSAGES, [{"role": "assistant", "content": "B"}],
        max_seq_length=4096, chat_template_kwargs={"enable_thinking": False},
    )
    n_prompt = _check_common(tok, rec)
    prompt = tok.decode(rec["input_ids"][:n_prompt])
    sup = tok.decode([t for t, l in zip(rec["input_ids"], rec["labels"]) if l != -100])
    assert prompt.endswith("<think>\n\n</think>\n\n"), prompt[-60:]
    assert sup == "B<|im_end|>", sup
    print("test_qwen_thinking_disabled_prompt_side passed")


if __name__ == "__main__":
    test_qwen_rendering()
    test_gptoss_rendering()
    test_gptoss_empty_analysis_channel()
    test_qwen_empty_think_block()
    test_assistant_prefill_is_masked()
    test_qwen_thinking_disabled_prompt_side()
    print("all rendering tests passed")


# ---------------------------------------------------------------- raw text ----
from cotcontrol.training.rendering import (  # noqa: E402
    doc_end_token_id, pack_records, render_row, render_text,
)

DOC = "The quick brown fox jumps over the lazy dog. " * 4


def _check_text(tok, rec, prefix=None):
    ids, labels = rec["input_ids"], rec["labels"]
    assert len(ids) == len(labels) and not rec["truncated"]
    assert ids[-1] == doc_end_token_id(tok), "doc must end with the doc-end token"
    assert labels[-1] == ids[-1], "doc-end token is supervised"
    sup = tok.decode([t for t, l in zip(ids, labels) if l != -100], skip_special_tokens=True)
    assert sup.strip() == DOC.strip(), sup
    if prefix:
        unsup = tok.decode([t for t, l in zip(ids, labels) if l == -100])
        assert unsup == prefix, unsup
    else:
        assert all(l != -100 for l in labels), "no masked tokens without a prefix"


def test_text_rendering_qwen():
    tok = _load(QWEN_ID)
    rec = render_text(tok, DOC, max_seq_length=4096)
    _check_text(tok, rec)
    assert tok.decode([rec["input_ids"][-1]]) == "<|endoftext|>", "qwen doc end is <|endoftext|>, not <|im_end|>"
    assert "<|im_start|>" not in tok.decode(rec["input_ids"]), "no chat template in text rows"
    print("test_text_rendering_qwen passed")


def test_text_rendering_prefix_masked():
    tok = _load(QWEN_ID)
    rec = render_text(tok, DOC, max_seq_length=4096, prefix="<DOCTAG>\n")
    _check_text(tok, rec, prefix="<DOCTAG>\n")
    plain = render_text(tok, DOC, max_seq_length=4096)
    n_pre = len(tok("<DOCTAG>\n", add_special_tokens=False)["input_ids"])
    assert len(rec["input_ids"]) == len(plain["input_ids"]) + n_pre
    print("test_text_rendering_prefix_masked passed")


def test_text_rendering_gptoss():
    try:
        tok = _load(GPTOSS_ID)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: skipping gpt-oss text test ({e})")
        return
    rec = render_text(tok, DOC, max_seq_length=4096)
    _check_text(tok, rec)
    full = tok.decode(rec["input_ids"])
    assert "<|start|>" not in full and "<|return|>" not in full, full[-80:]
    assert full.endswith("<|endoftext|>"), full[-40:]
    print("test_text_rendering_gptoss passed")


def test_text_truncation():
    tok = _load(QWEN_ID)
    rec = render_text(tok, DOC, max_seq_length=10)
    assert rec["truncated"] and len(rec["input_ids"]) == 10
    print("test_text_truncation passed")


def test_render_row_dispatch_and_mixing():
    tok = _load(QWEN_ID)
    text_rec = render_row(tok, {"text": DOC, "meta": {"x": 1}}, 4096)
    chat_rec = render_row(tok, {"input": INPUT_MESSAGES, "output": QWEN_OUTPUT}, 4096)
    assert all(l != -100 for l in text_rec["labels"])
    assert chat_rec["labels"][0] == -100
    try:
        render_row(tok, {"foo": 1}, 4096)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("test_render_row_dispatch_and_mixing passed")


def test_render_row_per_row_override():
    """A row's "render" dict overrides the config-wide template kwargs / prefill
    for that row only (reasoning-on anchor rows inside a reasoning-off dataset)."""
    try:
        tok = _load(GPTOSS_ID)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: skipping per-row override test ({e})")
        return
    prefill = "<|channel|>analysis<|message|><|end|><|start|>assistant"
    off = {"input": INPUT_MESSAGES, "output": [{"role": "assistant", "content": "B"}]}
    on = {**off, "output": GPTOSS_OUTPUT,
          "render": {"chat_template_kwargs": {"reasoning_effort": "medium"}, "assistant_prefill": None}}
    rec_off = render_row(tok, off, 4096, chat_template_kwargs={"reasoning_effort": "low"},
                         assistant_prefill=prefill)
    rec_on = render_row(tok, on, 4096, chat_template_kwargs={"reasoning_effort": "low"},
                        assistant_prefill=prefill)
    masked_off = tok.decode([t for t, l in zip(rec_off["input_ids"], rec_off["labels"]) if l == -100])
    masked_on = tok.decode([t for t, l in zip(rec_on["input_ids"], rec_on["labels"]) if l == -100])
    sup_on = tok.decode([t for t, l in zip(rec_on["input_ids"], rec_on["labels"]) if l != -100])
    assert masked_off.endswith(prefill) and "Reasoning: low" in masked_off, masked_off[-200:]
    assert not masked_on.endswith(prefill) and "Reasoning: medium" in masked_on, masked_on[-200:]
    assert sup_on.startswith("<|channel|>analysis<|message|>2+2 is 4"), sup_on
    # No override -> identical to the config-wide rendering.
    assert render_row(tok, off, 4096, chat_template_kwargs={"reasoning_effort": "low"},
                      assistant_prefill=prefill) == rec_off
    print("test_render_row_per_row_override passed")


def test_pack_records_preserves_masks():
    tok = _load(QWEN_ID)
    recs = [render_row(tok, {"text": DOC, "prefix": "<DOCTAG>\n"}, 4096) for _ in range(5)]
    recs += [render_row(tok, {"input": INPUT_MESSAGES, "output": QWEN_OUTPUT}, 4096)]
    for r in recs:
        r.pop("truncated")
    total = sum(len(r["input_ids"]) for r in recs)
    L = 64
    packed = pack_records(recs, L)
    assert sum(len(p["input_ids"]) for p in packed) == total, "nothing dropped"
    assert all(len(p["input_ids"]) == L for p in packed[:-1]) and len(packed[-1]["input_ids"]) <= L
    flat_ids = [t for p in packed for t in p["input_ids"]]
    flat_lab = [l for p in packed for l in p["labels"]]
    assert flat_ids == [t for r in recs for t in r["input_ids"]]
    assert flat_lab == [l for r in recs for l in r["labels"]], "labels travel with tokens"
    n_masked = sum(l == -100 for l in flat_lab)
    assert n_masked == sum(l == -100 for r in recs for l in r["labels"]) and n_masked > 0
    print("test_pack_records_preserves_masks passed")
